"""
backend/tests/test_issue_041_topology.py — Unit tests for Issue 041
per-call task topology consolidation.

Test plan
─────────

Supervisor owns all 5 tasks
  T1  After ``__init__``: reader_task, sender_task, matcher_task,
      mercury_task, stt_task, _matcher_internal_task are all None
      (not yet spawned).  ``attach_media_stream`` and
      ``start_background_tasks`` are the spawn entry points.
  T2  ``attach_media_stream(websocket, stream_sid)`` spawns both the
      inbound reader task AND the outbound sender task and stores
      them on the supervisor.  Subsequent calls are no-ops
      (idempotent).
  T3  ``start_background_tasks()`` spawns the STT task, Mercury
      driver task, AND the matcher consumer.  ``matcher_task`` is
      a ``MatcherConsumer`` (not None); ``_matcher_internal_task``
      is the underlying ``asyncio.Task``.

Teardown cancels all 5 tasks
  T4  ``trigger_teardown`` cancels mercury_task (aggressive), but
      skips the current task (self-cancellation guard).  The driver
      task set to None after cancel.
  T5  ``trigger_teardown`` cancels reader_task and sender_task
      (the WS bridge halves) aggressively.
  T6  ``trigger_teardown`` closes the matcher via ``MatcherConsumer.close()``
      — graceful path.
  T7  All 5 task handles are None after teardown (released).

Conversation driver inner-task seam (Issue 042 barge-in enabler)
  T8  The outer driver loop wraps ``run_turn`` in an
      ``asyncio.create_task`` and catches ``CancelledError`` so the
      outer loop survives a mid-turn cancellation.  When the INNER
      task is cancelled, the outer loop pulls the next turn from
      the queue.
  T9  If the OUTER driver task itself is cancelled, ``CancelledError``
      propagates (the supervisor teardown path expects this so it
      can take over).

Matcher fan-out in STT
  T10 ``dispatch_event`` on ``EndOfTurn`` calls
       ``supervisor.matcher_task.enqueue_turn(transcript)`` in
       addition to ``completed_turn_queue.put_nowait``.
  T11 ``dispatch_event`` does NOT call enqueue_turn when
       ``matcher_task`` is None (test fixtures without a matcher).
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import AsyncGenerator
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

from backend.call_supervisor import (
    CallSupervisor,
    _active_calls,
    register_call,
    unregister_call,
)
from backend.concurrency import ConcurrencyManager


# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clear_supervisor_registry() -> None:
    _active_calls.clear()
    yield
    _active_calls.clear()


def _schema_path() -> Path:
    return (
        Path(__file__).parent.parent.parent / "database" / "app_schema.sql"
    ).resolve()


def _create_temp_db() -> Path:
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".db", prefix="car_test_i041_"
    ) as fh:
        tmp = Path(fh.name)
    conn = sqlite3.connect(str(tmp))
    conn.execute("PRAGMA foreign_keys = ON;")
    with open(_schema_path(), encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.close()
    return tmp


@pytest_asyncio.fixture
async def temp_db() -> Path:
    db_path = _create_temp_db()
    yield db_path
    try:
        db_path.unlink()
    except OSError:
        pass


@pytest_asyncio.fixture
async def pool(temp_db: Path):
    from backend.database import create_pool

    p = create_pool(
        db_path=temp_db,
        busy_timeout_seconds=5.0,
        pool_size=5,
    )
    yield p
    await p.close()


def _make_supervisor() -> CallSupervisor:
    manager = ConcurrencyManager(capacity=2)
    manager.acquire()
    return CallSupervisor(
        lead_id=1,
        attempt_id=1,
        call_sid="CA_I041_TEST",
        language="English",
        concurrency_manager=manager,
        pool=None,
    )


# ─── T1, T2, T3: supervisor owns all 5 tasks ────────────────────────────────


class TestSupervisorOwnsAllFiveTasks:
    @pytest.mark.asyncio
    async def test_t1_all_task_handles_none_after_init(self) -> None:
        """After __init__, all 5 task handles are None — spawning is
        explicit via ``attach_media_stream`` and ``start_background_tasks``."""
        sup = _make_supervisor()
        assert sup.reader_task is None
        assert sup.sender_task is None
        assert sup.matcher_task is None
        assert sup.mercury_task is None
        assert sup.stt_task is None
        assert sup._matcher_internal_task is None

    @pytest.mark.asyncio
    async def test_t2_attach_media_stream_spawns_reader_and_sender(
        self,
        monkeypatch,
    ) -> None:
        """attach_media_stream spawns both halves of the WebSocket bridge
        and stores them on the supervisor."""
        sup = _make_supervisor()

        # Stub the WS bridge functions to no-op async functions so the
        # task body returns immediately (no real Twilio connection).
        async def _noop_reader(websocket, call_sid, supervisor, stream_sid):
            await asyncio.sleep(0)  # yield once

        async def _noop_sender(supervisor, websocket, stream_sid):
            await asyncio.sleep(0)

        monkeypatch.setattr(
            "backend.api.media_stream._inbound_reader_loop", _noop_reader
        )
        monkeypatch.setattr(
            "backend.audio.outbound_sender.run_outbound_sender", _noop_sender
        )

        fake_ws = MagicMock(name="websocket")
        sup.attach_media_stream(fake_ws, "MZ_TEST")

        assert sup.reader_task is not None
        assert sup.sender_task is not None
        assert sup.reader_task.get_name().startswith("media-stream-reader-")
        assert sup.sender_task.get_name().startswith("media-stream-sender-")

        # Cleanup: cancel the no-op tasks.
        for t in (sup.reader_task, sup.sender_task):
            t.cancel()
        await asyncio.gather(
            sup.reader_task, sup.sender_task, return_exceptions=True
        )

    @pytest.mark.asyncio
    async def test_t2b_attach_media_stream_is_idempotent(
        self,
        monkeypatch,
    ) -> None:
        """A second call to attach_media_stream does NOT spawn duplicates."""
        sup = _make_supervisor()

        async def _noop(*args, **kwargs):
            await asyncio.sleep(0)

        monkeypatch.setattr(
            "backend.api.media_stream._inbound_reader_loop", _noop
        )
        monkeypatch.setattr(
            "backend.audio.outbound_sender.run_outbound_sender", _noop
        )

        fake_ws = MagicMock(name="websocket")
        sup.attach_media_stream(fake_ws, "MZ_TEST")
        first_reader = sup.reader_task
        first_sender = sup.sender_task

        # Second call: should not overwrite the running tasks.
        sup.attach_media_stream(fake_ws, "MZ_TEST")
        assert sup.reader_task is first_reader
        assert sup.sender_task is first_sender

        for t in (sup.reader_task, sup.sender_task):
            t.cancel()
        await asyncio.gather(
            sup.reader_task, sup.sender_task, return_exceptions=True
        )

    @pytest.mark.asyncio
    async def test_t3_start_background_tasks_spawns_stt_driver_matcher(
        self,
        pool,
        monkeypatch,
    ) -> None:
        """start_background_tasks spawns STT, Mercury driver, AND the
        matcher consumer (Issue 041's fifth task)."""
        sup = _make_supervisor()

        # Stub STT and Mercury driver to no-op so we don't open live
        # WebSockets or HTTP connections in the test.
        async def _noop(*args, **kwargs):
            await asyncio.sleep(0)

        monkeypatch.setattr(
            "backend.audio.deepgram_stt.run_deepgram_reader", _noop
        )
        monkeypatch.setattr(
            "backend.conversation_driver.run_mercury_conversation_driver", _noop
        )

        # Stub the catalogue import inside start_background_tasks
        # so the matcher has a non-empty vehicles list (the test
        # environment doesn't load the real catalogue).
        fake_catalogue = [SimpleNamespace(id=1, name="Test Car")]
        monkeypatch.setattr(
            "backend.catalogue.vehicles", fake_catalogue
        )

        sup.start_background_tasks()

        assert sup.stt_task is not None
        assert sup.mercury_task is not None
        assert sup.matcher_task is not None
        assert sup._matcher_internal_task is not None
        # matcher_task should be a MatcherConsumer (not None, not a task).
        from backend.matcher_consumer import MatcherConsumer
        assert isinstance(sup.matcher_task, MatcherConsumer)
        # And its internal task should be running.
        assert not sup._matcher_internal_task.done()

        # Cleanup: cancel + close matcher so the loop exits cleanly.
        await sup.matcher_task.close()
        for t in (sup.stt_task, sup.mercury_task):
            t.cancel()
        await asyncio.gather(
            sup.stt_task, sup.mercury_task, return_exceptions=True
        )


# ─── T4, T5, T6, T7: teardown cancels all 5 tasks ───────────────────────────


class TestTeardownCancelsAllFiveTasks:
    @pytest.mark.asyncio
    async def test_t4_teardown_cancels_mercury_driver(self, pool, monkeypatch) -> None:
        """trigger_teardown aggressively cancels mercury_task (when the
        caller is NOT the driver itself)."""
        lead_id, attempt_id = await _seed_attempt(pool)

        manager = ConcurrencyManager(capacity=2)
        manager.acquire()
        supervisor = CallSupervisor(
            lead_id=lead_id,
            attempt_id=attempt_id,
            call_sid="CA_TEARDOWN_MERCURY",
            language="English",
            concurrency_manager=manager,
            pool=pool,
        )
        register_call(supervisor)

        # Spawn a real but instantly-completing mercury "task" (a fake)
        # so we have a handle to cancel.
        supervisor.mercury_task = asyncio.create_task(asyncio.sleep(60))
        try:
            await supervisor.trigger_teardown("Failed", "no_answer")
            assert supervisor.mercury_task is None, (
                "trigger_teardown did not clear mercury_task after cancel"
            )
        finally:
            # Cleanup if the test fails mid-flight.
            if supervisor.mercury_task is not None:
                supervisor.mercury_task.cancel()

    @pytest.mark.asyncio
    async def test_t5_teardown_cancels_reader_and_sender(
        self, pool, monkeypatch
    ) -> None:
        """trigger_teardown cancels reader_task and sender_task."""
        lead_id, attempt_id = await _seed_attempt(pool)
        manager = ConcurrencyManager(capacity=2)
        manager.acquire()
        sup = CallSupervisor(
            lead_id=lead_id,
            attempt_id=attempt_id,
            call_sid="CA_TEARDOWN_BRIDGE",
            language="English",
            concurrency_manager=manager,
            pool=pool,
        )
        register_call(sup)

        sup.reader_task = asyncio.create_task(asyncio.sleep(60))
        sup.sender_task = asyncio.create_task(asyncio.sleep(60))
        try:
            await sup.trigger_teardown("Failed", "no_answer")
            assert sup.reader_task is None, (
                "trigger_teardown did not clear reader_task after cancel"
            )
            assert sup.sender_task is None, (
                "trigger_teardown did not clear sender_task after cancel"
            )
        finally:
            for t in (sup.reader_task, sup.sender_task):
                if t is not None:
                    t.cancel()

    @pytest.mark.asyncio
    async def test_t6_teardown_closes_matcher_gracefully(
        self, pool, monkeypatch
    ) -> None:
        """trigger_teardown calls MatcherConsumer.close() — graceful path."""
        lead_id, attempt_id = await _seed_attempt(pool)
        manager = ConcurrencyManager(capacity=2)
        manager.acquire()
        sup = CallSupervisor(
            lead_id=lead_id,
            attempt_id=attempt_id,
            call_sid="CA_TEARDOWN_MATCHER",
            language="English",
            concurrency_manager=manager,
            pool=pool,
        )
        register_call(sup)

        # Instantiate a real matcher against a stub catalogue.
        monkeypatch.setattr(
            "backend.catalogue.vehicles", [SimpleNamespace(id=1, name="X")]
        )
        from backend.matcher_consumer import MatcherConsumer

        sup.matcher_task = MatcherConsumer(vehicles=[SimpleNamespace(id=1, name="X")])
        sup.matcher_task.start()
        sup._matcher_internal_task = sup.matcher_task._task

        try:
            await sup.trigger_teardown("Failed", "no_answer")
            # The matcher was closed via the graceful path; its handle
            # is cleared by teardown.
            assert sup.matcher_task is None
            assert sup._matcher_internal_task is None
        finally:
            if sup.matcher_task is not None:
                await sup.matcher_task.close()

    @pytest.mark.asyncio
    async def test_t7_all_task_handles_cleared_after_teardown(
        self, pool, monkeypatch
    ) -> None:
        """After trigger_teardown, every task handle on the supervisor
        is None — guarantees no orphaned task references."""
        lead_id, attempt_id = await _seed_attempt(pool)
        manager = ConcurrencyManager(capacity=2)
        manager.acquire()
        sup = CallSupervisor(
            lead_id=lead_id,
            attempt_id=attempt_id,
            call_sid="CA_TEARDOWN_CLEAR",
            language="English",
            concurrency_manager=manager,
            pool=pool,
        )
        register_call(sup)

        # Stub STT to no-op (returns immediately so the graceful
        # shutdown_event path self-closes the task without the 1.5s
        # backstop). Stub _twilio_hangup_call to avoid the live API.
        async def _instant_stt(supervisor):
            return  # exits immediately — supervisor sees self-close

        monkeypatch.setattr(
            "backend.audio.deepgram_stt.run_deepgram_reader", _instant_stt
        )
        # Spawn all 5 fake tasks directly.
        sup.stt_task = asyncio.create_task(_instant_stt(sup), name="stt")
        sup.mercury_task = asyncio.create_task(
            asyncio.sleep(60), name="mercury"
        )
        sup.reader_task = asyncio.create_task(
            asyncio.sleep(60), name="reader"
        )
        sup.sender_task = asyncio.create_task(
            asyncio.sleep(60), name="sender"
        )
        from backend.matcher_consumer import MatcherConsumer
        sup.matcher_task = MatcherConsumer(vehicles=[SimpleNamespace(id=1)])
        sup.matcher_task.start()
        sup._matcher_internal_task = sup.matcher_task._task

        # Let the STT task self-close (it's instantaneous).
        await asyncio.sleep(0.01)

        try:
            await sup.trigger_teardown("Failed", "no_answer")
            assert sup.stt_task is None, (
                "trigger_teardown did not clear stt_task after self-close"
            )
            assert sup.mercury_task is None
            assert sup.reader_task is None
            assert sup.sender_task is None
            assert sup.matcher_task is None, (
                "trigger_teardown did not clear matcher_task after close"
            )
            assert sup._matcher_internal_task is None
        finally:
            for attr in ("stt_task", "mercury_task", "reader_task",
                         "sender_task"):
                t = getattr(sup, attr, None)
                if t is not None and not t.done():
                    t.cancel()
            if sup._matcher_internal_task is not None and not sup._matcher_internal_task.done():
                sup._matcher_internal_task.cancel()


# ─── T8, T9: conversation driver inner-task seam ───────────────────────────


class TestDriverInnerTaskCancellation:
    """Issue 042 barge-in seam: cancelling the inner task (the one
    wrapping run_turn) must NOT kill the outer driver loop.
    """

    @pytest.mark.asyncio
    async def test_t8_inner_task_cancellation_outer_loop_survives(
        self, pool, monkeypatch
    ) -> None:
        """Cancelling the inner task mid-turn raises CancelledError that
        the OUTER loop catches and continues — it pulls the NEXT turn
        from completed_turn_queue without exiting."""
        from backend.conversation_driver import (
            run_mercury_conversation_driver,
        )

        lead_id, attempt_id = await _seed_attempt(pool)
        manager = ConcurrencyManager(capacity=2)
        manager.acquire()
        sup = CallSupervisor(
            lead_id=lead_id,
            attempt_id=attempt_id,
            call_sid="CA_INNER_CANCEL_TEST",
            language="English",
            concurrency_manager=manager,
            pool=pool,
        )
        register_call(sup)
        # Unblock the kickoff synchronisation.
        sup.mark_disclosure_complete()
        if not sup.amd_verdict.done():
            sup.amd_verdict.set_result("human")

        # Stub run_turn with turn-counting so the KICKOFF (first call)
        # returns immediately — letting the driver reach the main loop —
        # but subsequent calls (the ones we want to cancel) block
        # forever, simulating an in-flight Mercury call.
        turn_calls: list[str] = []

        async def _slow_turn(supervisor, messages, *, new_user_text,
                             http_client, api_key):
            turn_calls.append(new_user_text or "<kickoff>")
            if new_user_text is None:
                # Kickoff: append a fake assistant reply and return.
                messages.append({"role": "assistant", "content": "Hi"})
                return
            # Real customer turn — block (simulating Mercury HTTP).
            await asyncio.sleep(60)

        monkeypatch.setattr(
            "backend.conversation_driver.run_turn", _slow_turn
        )
        # Stub the TTS client (driver creates one before its main loop).
        monkeypatch.setattr(
            "backend.conversation_driver.AsyncSarvamTTSClient",
            _NoOpTTSClient,
        )

        driver_task = asyncio.create_task(
            run_mercury_conversation_driver(sup),
            name="driver-outer",
        )

        # Wait for the kickoff turn to land (first call).
        for _ in range(100):
            if turn_calls:
                break
            await asyncio.sleep(0.02)
        else:  # pragma: no cover
            driver_task.cancel()
            pytest.fail("Driver never reached the kickoff turn")

        # Push a customer transcript so the driver enters the main
        # loop and creates the inner task that we'll cancel.
        await sup.completed_turn_queue.put("hello")

        # Wait for the second turn call (the blocking one).
        for _ in range(200):
            if len(turn_calls) >= 2:
                break
            await asyncio.sleep(0.02)
        else:  # pragma: no cover
            driver_task.cancel()
            pytest.fail("Driver never started the second turn")

        # The driver is now mid-second-turn, awaiting the inner task.
        # Find the inner task by name prefix.
        children = [
            t for t in asyncio.all_tasks()
            if t.get_name().startswith("mercury-inner-")
        ]
        assert children, "Inner mercury-inner-* task not found"
        inner = children[0]

        # Cancel the INNER task (simulating Issue 042 barge-in).
        inner.cancel()
        # Wait for the inner task to settle.
        try:
            await asyncio.wait_for(inner, timeout=1.0)
        except (asyncio.CancelledError, Exception):
            pass

        # Outer driver must NOT exit; the CancelledError is caught and
        # the loop continues to pull the next turn.
        for _ in range(50):
            await asyncio.sleep(0.02)
            if not driver_task.done():
                break
        assert not driver_task.done(), (
            "Outer driver task exited when inner task was cancelled — "
            "the barge-in seam is broken (Issue 042 would lose the "
            "conversation history)"
        )

        # Cleanup: shutdown the driver.
        sup.shutdown_event.set()
        try:
            await asyncio.wait_for(driver_task, timeout=2.0)
        except asyncio.TimeoutError:  # pragma: no cover
            driver_task.cancel()
            try:
                await driver_task
            except (asyncio.CancelledError, Exception):
                pass

    @pytest.mark.asyncio
    async def test_t9_outer_driver_cancellation_propagates(
        self, pool, monkeypatch
    ) -> None:
        """Cancelling the OUTER driver task propagates CancelledError —
        this is what the supervisor teardown relies on."""
        from backend.conversation_driver import (
            run_mercury_conversation_driver,
        )

        lead_id, attempt_id = await _seed_attempt(pool)
        manager = ConcurrencyManager(capacity=2)
        manager.acquire()
        sup = CallSupervisor(
            lead_id=lead_id,
            attempt_id=attempt_id,
            call_sid="CA_OUTER_CANCEL_TEST",
            language="English",
            concurrency_manager=manager,
            pool=pool,
        )
        register_call(sup)
        sup.mark_disclosure_complete()
        if not sup.amd_verdict.done():
            sup.amd_verdict.set_result("human")

        async def _hang(*a, **k):
            await asyncio.sleep(60)

        monkeypatch.setattr(
            "backend.conversation_driver.run_turn", _hang
        )
        monkeypatch.setattr(
            "backend.conversation_driver.AsyncSarvamTTSClient",
            _NoOpTTSClient,
        )

        driver_task = asyncio.create_task(
            run_mercury_conversation_driver(sup), name="driver-outer"
        )
        await asyncio.sleep(0.05)

        # Cancel the OUTER driver task — should propagate.
        driver_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await driver_task


class _NoOpTTSClient:
    """Stand-in for AsyncSarvamTTSClient: connect/synthesize/close are no-ops."""

    def __init__(self):
        pass

    async def connect(self, supervisor):
        pass

    async def synthesize_turn(self, *args, **kwargs):
        pass

    async def close(self):
        pass


# ─── T10, T11: matcher fan-out in STT ────────────────────────────────────────


class TestMatcherFanOutInSTT:
    """dispatch_event fans out completed transcripts to the matcher
    consumer in parallel with the driver's queue."""

    @pytest.mark.asyncio
    async def test_t10_end_of_turn_calls_matcher_enqueue(
        self, monkeypatch
    ) -> None:
        """EndOfTurn fans out to BOTH the driver's queue AND the matcher."""
        from backend.audio.deepgram_stt import dispatch_event
        from backend.matcher_consumer import MatcherConsumer

        sup = MagicMock(spec=CallSupervisor)
        sup.call_sid = "CA_FANOUT_TEST"

        # Stub the Gemini extractor (no real API call) AND the scorer
        # (avoid SimpleNamespace attribute mismatch — we only need to
        # prove the matcher was reached, not that it scored correctly).
        from backend.preference_extractor import GeminiClient
        from backend.scorer import RankingResult

        class _FakeExtractor(GeminiClient):
            def extract(self, utterance, current_prefs):
                return {
                    "budget": 800000.0,
                    "min_seats": 5,
                    "fuel_type": None,
                    "body_style": "SUV",
                    "transmission": None,
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }

        def _fake_score(prefs, vehicles):
            # Return a benign RankingResult so _process_one completes.
            # SYNC function — matcher's _process_one calls score_and_rank
            # synchronously and assigns ``result.cars`` directly.
            return RankingResult(cars=(), note=None)

        # Patch the import the matcher uses internally.
        from backend import matcher_consumer as mc_mod
        original_extract = mc_mod.extract_preferences if hasattr(mc_mod, "extract_preferences") else None
        monkeypatch.setattr(
            "backend.preference_extractor.extract_preferences",
            lambda utterance, current_prefs, client: _FakeExtractor().extract(utterance, current_prefs),
        )
        monkeypatch.setattr(
            "backend.scorer.score_and_rank", _fake_score
        )

        sup.matcher_task = MatcherConsumer(
            vehicles=[SimpleNamespace(id=1, name="X")],
            extractor_cls=_FakeExtractor,
        )
        sup.matcher_task.start()

        event = {
            "event": "EndOfTurn",
            "turn_index": 1,
            "transcript": "I want an SUV",
        }

        dispatch_event(event, sup)

        # The transcript was pushed onto completed_turn_queue.
        sup.completed_turn_queue.put_nowait.assert_called_once_with(
            "I want an SUV"
        )

        # The matcher received the turn via its own queue.  We can't
        # race the consumer task to read the queue — it processes
        # items asynchronously — but the matcher tracks the LATEST
        # utterance processed via ``_running_prefs``.  Wait until the
        # extractor's `extract` is called, which proves the turn was
        # enqueued and processed.
        for _ in range(100):
            if sup.matcher_task._running_prefs is not None:
                break
            await asyncio.sleep(0.02)
        else:  # pragma: no cover
            await sup.matcher_task.close()
            pytest.fail("Matcher did not process the EndOfTurn transcript")

        assert sup.matcher_task._running_prefs["body_style"] == "SUV"

        # Cleanup.
        await sup.matcher_task.close()

    @pytest.mark.asyncio
    async def test_t11_end_of_turn_skips_matcher_when_none(
        self, monkeypatch
    ) -> None:
        """When matcher_task is None (test fixture without a matcher),
        dispatch_event still pushes onto completed_turn_queue without
        raising."""
        from backend.audio.deepgram_stt import dispatch_event

        sup = MagicMock(spec=CallSupervisor)
        sup.call_sid = "CA_NO_MATCHER"
        # matcher_task is the default MagicMock auto-attr — set to None
        # explicitly to simulate the no-matcher path.
        sup.matcher_task = None

        event = {"event": "EndOfTurn", "turn_index": 1, "transcript": "hi"}

        # Should NOT raise.
        dispatch_event(event, sup)

        sup.completed_turn_queue.put_nowait.assert_called_once_with("hi")


# ─── Helpers ──────────────────────────────────────────────────────────────────


async def _seed_attempt(pool, call_sid: str = "CA_I041_HELPER") -> tuple[int, int]:
    """Insert a Processing lead + call_attempt; return (lead_id, attempt_id)."""
    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            INSERT INTO leads
                (name, phone, email, city, language, status, attempt_count)
            VALUES ('I041 Lead', '+919999999999', 'i@x.com',
                    'Mumbai', 'English', 'Processing', 1)
            """,
        )
        lead_id = cursor.lastrowid
        cursor = await conn.execute(
            """
            INSERT INTO call_attempts (lead_id, attempt_number, twilio_call_sid)
            VALUES (?, 1, ?)
            """,
            (lead_id, call_sid),
        )
        attempt_id = cursor.lastrowid
    return lead_id, attempt_id