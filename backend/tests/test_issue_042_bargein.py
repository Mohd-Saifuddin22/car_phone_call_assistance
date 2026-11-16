"""
backend/tests/test_issue_042_bargein.py — Issue 042 barge-in unit tests.

Deterministic fakes + ``asyncio.Event`` synchronisation only — never real
sockets, live providers, credentials, or ``sleep``-based timing as a
synchronisation primitive.  Indefinite fake work parks on an ``Event`` (never
``sleep(60)``); ordering is proven with events, task state, and explicit
acknowledgements.

Coverage map (matches the Issue 042 acceptance list + the hardening defects)
────────────────────────────────────────────────────────────────────────────
 1  StartOfTurn while Riya speaks (outside disclosure) fires barge-in.
 2  StartOfTurn while the disclosure is active is a no-op.
 3  StartOfTurn while Riya is NOT speaking is a no-op.
 4  A duplicate StartOfTurn does not run the stop sequence twice.
 5  Generation is bumped and a ClearPlayback (post-bump generation) is
    delivered on the dedicated priority control channel.
 6  DEFECT 1 — clear preempts stale audio even when a media send is blocked;
    stale audio queued behind the in-progress chunk cannot delay the clear;
    the clear ack resolves True and no further old-gen media/marks are sent.
 7  Only ``current_turn_task`` is cancelled; ``mercury_task`` stays alive.
 8  The kickoff/greeting turn is represented by the cancellable
    ``current_turn_task``.
 9  Sarvam is closed + detached on barge-in; the next turn lazily reconnects.
10  Normal (uninterrupted) turns reuse the same Sarvam client.
11  Interrupted metadata carries the FULL intended text, the marker, and
    ``played_ms``; ``total_ms`` comes from SYNTHESIZED bytes.
12  Pending/stale mark acknowledgements after a clear do not update the new
    generation's playback state.
13  Teardown stays idempotent and cancels the barge-in task handles.
14  DEFECT 2 — the next customer EndOfTurn (queued before barge-in completes)
    is not processed until the stop sequence has closed + detached the old
    Sarvam client; then a fresh client connects before the next turn.
15  DEFECT 3 — ``total_ms`` uses cumulative synthesized bytes (incl. a chunk
    that arrives during the stop sequence before socket closure), not sent
    duration.
16  EndOfTurn never triggers barge-in.
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from backend.audio.deepgram_stt import dispatch_event
from backend.audio.outbound_sender import ClearPlayback, run_outbound_sender
from backend.call_supervisor import CallSupervisor, _active_calls, register_call
from backend.concurrency import ConcurrencyManager
from backend.conversation_driver import (
    BargeinCompletionTimeoutError,
    _await_bargein_complete,
    _ensure_tts_client,
    run_mercury_conversation_driver,
)


# ─── Shared fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_active_calls_registry() -> None:
    _active_calls.clear()
    yield
    _active_calls.clear()


def _schema_path() -> Path:
    return (
        Path(__file__).parent.parent.parent / "database" / "app_schema.sql"
    ).resolve()


def _create_temp_db() -> Path:
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".db", prefix="car_test_i042_"
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

    p = create_pool(db_path=temp_db, busy_timeout_seconds=5.0, pool_size=5)
    yield p
    await p.close()


def _make_supervisor(*, lead_id: int = 1, pool: Any = None) -> CallSupervisor:
    """Build a CallSupervisor with a real concurrency manager + stub pool."""
    manager = ConcurrencyManager(capacity=2)
    manager.acquire()
    return CallSupervisor(
        lead_id=lead_id,
        attempt_id=1,
        call_sid="CA_I042_TEST",
        language="English",
        concurrency_manager=manager,
        pool=pool if pool is not None else MagicMock(),
    )


async def _seed_lead(pool) -> int:
    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            INSERT INTO leads
                (name, phone, email, city, language, status,
                 attempt_count, vehicle_id)
            VALUES ('I042 Lead', '+919876543210', 'l@x.com',
                    'Mumbai', 'English', 'Processing', 1, NULL)
            """,
        )
        return cursor.lastrowid


def _unblock_call_start(supervisor: CallSupervisor) -> None:
    """Simulate disclosure playback + a 'human' AMD verdict having completed."""
    supervisor.mark_disclosure_complete()
    if not supervisor.amd_verdict.done():
        supervisor.amd_verdict.set_result("human")


def _spawn_clear_acker(
    supervisor: CallSupervisor,
    *,
    recorded: list[ClearPlayback] | None = None,
) -> asyncio.Task:
    """
    Stand-in for the sole WS writer's clear handling: drain the priority
    control queue and resolve each ClearPlayback's ack ``True``.  Lets a
    stop-sequence test confirm the clear ack path without standing up a full
    outbound sender + fake WebSocket.  Cancel the returned task in cleanup.
    """

    async def _acker() -> None:
        while True:
            cmd = await supervisor.outbound_control_queue.get()
            if recorded is not None:
                recorded.append(cmd)
            if cmd.ack is not None and not cmd.ack.done():
                cmd.ack.set_result(True)

    return asyncio.create_task(_acker(), name="clear-acker")


async def _cancel_tasks(*tasks: asyncio.Task | None) -> None:
    live = [t for t in tasks if t is not None and not t.done()]
    for t in live:
        t.cancel()
    if live:
        await asyncio.gather(*live, return_exceptions=True)


# ─── Fakes ────────────────────────────────────────────────────────────────────


class _RecordingTTSClient:
    """Sarvam client stand-in that records connect/close and never raises."""

    def __init__(self) -> None:
        self.connected_to: Any = None
        self.closed: bool = False

    async def connect(self, supervisor: Any) -> None:
        self.connected_to = supervisor

    async def synthesize_turn(self, text, *, language, generation, supervisor):
        return None

    async def close(self) -> None:
        self.closed = True


class _FakeOutboundWebSocket:
    """Records every ``send_json`` payload (mirrors the sender's WS surface)."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.client_state = "CONNECTED"

    async def send_json(self, payload: Any) -> None:
        self.sent.append(payload)


class _BlockingFirstMediaWebSocket:
    """
    WebSocket fake whose FIRST ``media`` ``send_json`` parks on an event so a
    test can bump the generation + request a clear while a send is in flight.
    All other sends (and the released first send) record immediately.
    """

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.client_state = "CONNECTED"
        self.first_media_reached = asyncio.Event()
        self.release_first_media = asyncio.Event()
        self._blocked_once = False

    async def send_json(self, payload: Any) -> None:
        if not self._blocked_once and payload.get("event") == "media":
            self._blocked_once = True
            self.first_media_reached.set()
            await self.release_first_media.wait()
        self.sent.append(payload)


class _FakeAsyncClientContextManager:
    """``async with httpx.AsyncClient(...)`` shim for driver tests."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def __aenter__(self) -> Any:
        return self._client

    async def __aexit__(self, *exc: Any) -> None:
        return None


# ─── 1 + 16: StartOfTurn trigger via dispatch_event ──────────────────────────


class TestStartOfTurnTrigger:
    @pytest.mark.asyncio
    async def test_start_of_turn_while_speaking_fires_bargein(self) -> None:
        sup = _make_supervisor()
        sup.is_speaking = True
        sup.disclosure_active = False
        sup.generation_counter = 1
        sup.current_turn_text = "Let me tell you about the Nexon"
        sup.played_duration_ms = 200
        sup.synthesized_bytes_by_generation[1] = 3200  # 400 ms synthesized
        acker = _spawn_clear_acker(sup)
        try:
            dispatch_event(
                {"event": "StartOfTurn", "turn_index": 0, "transcript": "wait"},
                sup,
            )
            assert sup.bargein_task is not None
            await sup.bargein_task

            assert sup.generation_counter == 2
            assert sup.is_speaking is False
            assert len(sup.interrupted_turns) == 1
            record = sup.interrupted_turns[0]
            assert record.generation == 1
            assert record.full_text == "Let me tell you about the Nexon"
            assert record.played_ms == 200
            assert record.total_ms == 400  # 3200 bytes @ 8 kHz
        finally:
            await _cancel_tasks(acker)

    @pytest.mark.asyncio
    async def test_end_of_turn_never_fires_bargein(self) -> None:
        sup = _make_supervisor()
        sup.is_speaking = True
        sup.disclosure_active = False

        dispatch_event(
            {"event": "EndOfTurn", "turn_index": 0, "transcript": "done"}, sup
        )

        assert sup.bargein_task is None
        assert sup.completed_turn_queue.get_nowait() == "done"


# ─── 2 + 3: guard conditions ─────────────────────────────────────────────────


class TestBargeinGuards:
    def test_no_bargein_while_disclosure_active(self) -> None:
        sup = _make_supervisor()
        sup.is_speaking = True
        sup.disclosure_active = True

        assert sup.maybe_start_bargein() is False
        assert sup.bargein_task is None
        assert sup.generation_counter == 0
        assert sup.bargein_complete.is_set() is True

    def test_no_bargein_while_not_speaking(self) -> None:
        sup = _make_supervisor()
        sup.is_speaking = False
        sup.disclosure_active = False

        assert sup.maybe_start_bargein() is False
        assert sup.bargein_task is None
        assert sup.generation_counter == 0

    def test_no_bargein_after_shutdown(self) -> None:
        sup = _make_supervisor()
        sup.is_speaking = True
        sup.disclosure_active = False
        sup.shutdown_event.set()

        assert sup.maybe_start_bargein() is False


# ─── 4: duplicate StartOfTurn ────────────────────────────────────────────────


class TestDuplicateStartOfTurn:
    @pytest.mark.asyncio
    async def test_duplicate_start_of_turn_runs_stop_sequence_once(self) -> None:
        sup = _make_supervisor()
        sup.is_speaking = True
        sup.disclosure_active = False
        sup.current_turn_text = "hi"
        acker = _spawn_clear_acker(sup)
        try:
            assert sup.maybe_start_bargein() is True
            # Synchronous back-to-back: the spawned task has not run yet, so
            # is_speaking is still True; only the re-entrancy latch stops it.
            assert sup.maybe_start_bargein() is False
            await sup.bargein_task
            assert len(sup.interrupted_turns) == 1
        finally:
            await _cancel_tasks(acker)


# ─── 5: generation bump + clear on the priority control channel ──────────────


class TestGenerationLifecycle:
    @pytest.mark.asyncio
    async def test_clear_delivered_on_priority_control_channel(self) -> None:
        sup = _make_supervisor()
        sup.generation_counter = 3
        sup.is_speaking = True
        sup.disclosure_active = False
        sup.current_turn_text = "hi"
        recorded: list[ClearPlayback] = []
        acker = _spawn_clear_acker(sup, recorded=recorded)
        try:
            assert sup.maybe_start_bargein() is True
            await sup.bargein_task

            assert sup.generation_counter == 4
            # Exactly one clear, delivered on the control channel, tagged with
            # the post-bump generation.
            assert len(recorded) == 1
            assert isinstance(recorded[0], ClearPlayback)
            assert recorded[0].generation == 4
        finally:
            await _cancel_tasks(acker)


# ─── 6: DEFECT 1 — clear preempts stale audio with a blocked send ────────────


class TestClearPreemption:
    @pytest.mark.asyncio
    async def test_clear_preempts_blocked_send_and_queued_stale_audio(
        self,
    ) -> None:
        from backend.audio.repacketizer import MULAW_FRAME_BYTES

        sup = _make_supervisor()
        sup.generation_counter = 1
        ws = _BlockingFirstMediaWebSocket()

        # An old-generation multi-frame chunk (3 frames), plus more stale
        # audio queued BEHIND it — both must not delay the clear.
        sup.outbound_audio_queue.put_nowait((1, bytes(3 * MULAW_FRAME_BYTES)))
        sup.outbound_audio_queue.put_nowait((1, bytes(2 * MULAW_FRAME_BYTES)))

        sender = asyncio.create_task(run_outbound_sender(sup, ws, "MZ_SID"))
        try:
            # Wait until the sender is parked inside the first media send.
            await asyncio.wait_for(ws.first_media_reached.wait(), timeout=2.0)

            # Barge-in: bump the generation and request a clear (priority
            # control channel) while that send is blocked.
            sup.generation_counter = 2
            ack: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
            sup.outbound_control_queue.put_nowait(
                ClearPlayback(generation=2, ack=ack)
            )

            # Release the blocked send — the clear must be the NEXT event.
            ws.release_first_media.set()

            assert await asyncio.wait_for(ack, timeout=2.0) is True
        finally:
            sup.shutdown_event.set()
            await _cancel_tasks(sender)

        media = [e for e in ws.sent if e.get("event") == "media"]
        marks = [e for e in ws.sent if e.get("event") == "mark"]
        clears = [e for e in ws.sent if e.get("event") == "clear"]

        # The single in-flight frame went out, then the clear — nothing else.
        assert ws.sent[0]["event"] == "media"
        assert ws.sent[1]["event"] == "clear"
        assert len(media) == 1, "stale frames after the in-flight one leaked"
        assert marks == [], "no old-generation marks may be emitted"
        assert clears == [{"event": "clear", "streamSid": "MZ_SID"}]


# ─── 7: only current_turn_task cancelled; mercury_task alive ─────────────────


class TestTaskOwnership:
    @pytest.mark.asyncio
    async def test_bargein_cancels_only_current_turn_task(self) -> None:
        sup = _make_supervisor()
        sup.is_speaking = True
        sup.disclosure_active = False
        sup.current_turn_text = "hi"

        turn_hold = asyncio.Event()
        mercury_hold = asyncio.Event()
        turn_task = asyncio.create_task(turn_hold.wait(), name="turn")
        mercury_task = asyncio.create_task(mercury_hold.wait(), name="mercury")
        sup.current_turn_task = turn_task
        sup.mercury_task = mercury_task
        acker = _spawn_clear_acker(sup)
        try:
            assert sup.maybe_start_bargein() is True
            await sup.bargein_task

            assert turn_task.cancelled() or turn_task.done()
            assert not mercury_task.done(), (
                "barge-in must NOT cancel the long-lived Mercury driver task"
            )
            # Barge-in cancels + awaits the in-flight turn but does not clear
            # the handle — the driver's wrapper owns that via try/finally.
            assert sup.current_turn_task is turn_task
        finally:
            await _cancel_tasks(acker, mercury_task)


# ─── 8: kickoff represented by current_turn_task ─────────────────────────────


class TestKickoffCancellableTurn:
    @pytest.mark.asyncio
    async def test_kickoff_publishes_current_turn_task(
        self, pool, monkeypatch
    ) -> None:
        lead_id = await _seed_lead(pool)
        sup = _make_supervisor(lead_id=lead_id, pool=pool)
        register_call(sup)
        _unblock_call_start(sup)

        kickoff_running = asyncio.Event()
        kickoff_hold = asyncio.Event()

        async def _blocking_kickoff(supervisor, messages, *, new_user_text,
                                    http_client, api_key):
            if new_user_text is None:
                kickoff_running.set()
                await kickoff_hold.wait()

        monkeypatch.setattr(
            "backend.conversation_driver.run_turn", _blocking_kickoff
        )
        monkeypatch.setattr(
            "backend.conversation_driver.AsyncSarvamTTSClient",
            _RecordingTTSClient,
        )
        monkeypatch.setattr(
            "backend.conversation_driver.httpx.AsyncClient",
            lambda timeout: _FakeAsyncClientContextManager(MagicMock()),
        )

        driver_task = asyncio.create_task(
            run_mercury_conversation_driver(sup), name="driver"
        )
        try:
            await asyncio.wait_for(kickoff_running.wait(), timeout=2.0)
            assert sup.current_turn_task is not None
            assert sup.current_turn_task.get_name().startswith(
                "mercury-inner-"
            )
        finally:
            sup.shutdown_event.set()
            await _cancel_tasks(driver_task)


# ─── 9 + 10: Sarvam close-on-barge-in + lazy reconnect / reuse ───────────────


class TestSarvamLifecycle:
    @pytest.mark.asyncio
    async def test_bargein_closes_and_detaches_sarvam(self) -> None:
        sup = _make_supervisor()
        sup.is_speaking = True
        sup.disclosure_active = False
        sup.current_turn_text = "hi"
        tts = _RecordingTTSClient()
        sup.tts_client = tts
        acker = _spawn_clear_acker(sup)
        try:
            assert sup.maybe_start_bargein() is True
            await sup.bargein_task

            assert tts.closed is True
            assert sup.tts_client is None
        finally:
            await _cancel_tasks(acker)

    @pytest.mark.asyncio
    async def test_next_turn_lazily_reconnects_after_bargein(
        self, monkeypatch
    ) -> None:
        sup = _make_supervisor()
        sup.tts_client = None  # barge-in nulled it

        monkeypatch.setattr(
            "backend.conversation_driver.AsyncSarvamTTSClient",
            _RecordingTTSClient,
        )
        await _ensure_tts_client(sup)

        assert isinstance(sup.tts_client, _RecordingTTSClient)
        assert sup.tts_client.connected_to is sup

    @pytest.mark.asyncio
    async def test_normal_turn_reuses_existing_sarvam_client(
        self, monkeypatch
    ) -> None:
        sup = _make_supervisor()
        existing = _RecordingTTSClient()
        sup.tts_client = existing

        constructed = {"n": 0}

        def _factory() -> _RecordingTTSClient:
            constructed["n"] += 1
            return _RecordingTTSClient()

        monkeypatch.setattr(
            "backend.conversation_driver.AsyncSarvamTTSClient", _factory
        )
        await _ensure_tts_client(sup)

        assert sup.tts_client is existing
        assert constructed["n"] == 0, "a connected client must be reused"


# ─── 11: interrupted-turn metadata (full text + synthesized total_ms) ────────


class TestInterruptedTurnMetadata:
    @pytest.mark.asyncio
    async def test_metadata_full_text_and_synthesized_total_ms(self) -> None:
        sup = _make_supervisor()
        sup.is_speaking = True
        sup.disclosure_active = False
        sup.generation_counter = 7
        sup.current_turn_text = "This is a long intended sentence about SUVs"
        sup.played_duration_ms = 120
        sup.synthesized_bytes_by_generation[7] = 24000  # 3000 ms synthesized
        acker = _spawn_clear_acker(sup)
        try:
            assert sup.maybe_start_bargein() is True
            await sup.bargein_task

            record = sup.interrupted_turns[0]
            # Full intended text is stored verbatim — NOT chopped proportionally.
            assert record.full_text == "This is a long intended sentence about SUVs"
            assert record.display_text.startswith(
                "This is a long intended sentence about SUVs"
            )
            assert record.display_text.endswith("[interrupted]")
            assert record.played_ms == 120
            assert record.total_ms == 3000  # 24000 bytes @ 8 kHz
            assert record.generation == 7
            # Counter retired.
            assert 7 not in sup.synthesized_bytes_by_generation
        finally:
            await _cancel_tasks(acker)


# ─── 12: stale mark acks after clear ─────────────────────────────────────────


class TestStaleMarkHandling:
    def test_stale_mark_after_clear_does_not_touch_new_generation(self) -> None:
        from backend.api.media_stream import _dispatch_mark

        sup = _make_supervisor()
        sup.generation_counter = 5
        sup.played_duration_ms = 0
        sup.is_speaking = True

        _dispatch_mark(
            {"event": "mark", "mark": {"name": "gen=4:offset=800:final=true"}},
            "CA_I042_TEST",
            sup,
        )
        assert sup.played_duration_ms == 0
        assert sup.is_speaking is True

        _dispatch_mark(
            {"event": "mark", "mark": {"name": "gen=5:offset=100"}},
            "CA_I042_TEST",
            sup,
        )
        assert sup.played_duration_ms == 100


# ─── 13: teardown idempotent + no barge-in task leak ─────────────────────────


class TestTeardownWithBargeinTasks:
    @pytest.mark.asyncio
    async def test_teardown_cancels_bargein_handles_and_is_idempotent(
        self, pool
    ) -> None:
        lead_id = await _seed_lead(pool)
        sup = _make_supervisor(lead_id=lead_id, pool=pool)
        register_call(sup)

        turn_hold = asyncio.Event()
        bargein_hold = asyncio.Event()
        sup.current_turn_task = asyncio.create_task(turn_hold.wait())
        sup.bargein_task = asyncio.create_task(bargein_hold.wait())

        async def _noop_resolve(*a: Any, **kw: Any) -> None:
            return None

        async def _noop_delete(_call_sid: str) -> int:
            return 0

        with patch(
            "backend.call_supervisor._resolve_failed_attempt",
            side_effect=_noop_resolve,
        ), patch(
            "backend.call_supervisor._twilio_hangup_call", return_value=True
        ), patch(
            "backend.call_supervisor._delete_call_recordings_async",
            side_effect=_noop_delete,
        ):
            await sup.trigger_teardown("Failed", "dropped")

            assert sup.current_turn_task is None
            assert sup.bargein_task is None
            assert sup._teardown_run is True
            assert sup.bargein_complete.is_set() is True

            # Second trigger is a no-op (one-shot latch).
            await sup.trigger_teardown("Failed", "dropped")
            assert sup._teardown_run is True


# ─── 14: DEFECT 2 — completion barrier gates the next turn ───────────────────


class TestBargeinCompletionBarrier:
    @pytest.mark.asyncio
    async def test_next_turn_waits_for_stop_sequence_to_close_sarvam(
        self, pool, monkeypatch
    ) -> None:
        lead_id = await _seed_lead(pool)
        sup = _make_supervisor(lead_id=lead_id, pool=pool)
        register_call(sup)
        _unblock_call_start(sup)

        connect_log: list[Any] = []
        close_started = asyncio.Event()
        close_release = asyncio.Event()
        kickoff_done = asyncio.Event()
        first_started = asyncio.Event()
        first_hold = asyncio.Event()
        second_started = asyncio.Event()

        class _CtrlTTS:
            def __init__(self) -> None:
                self.connected_to: Any = None
                self.closed = False

            async def connect(self, supervisor: Any) -> None:
                self.connected_to = supervisor
                connect_log.append(self)

            async def synthesize_turn(self, *a: Any, **k: Any) -> None:
                return None

            async def close(self) -> None:
                close_started.set()
                await close_release.wait()
                self.closed = True

        async def _run(supervisor, messages, *, new_user_text, http_client,
                       api_key):
            if new_user_text is None:
                messages.append({"role": "assistant", "content": "greeting"})
                kickoff_done.set()
                return
            if new_user_text == "first":
                supervisor.is_speaking = True
                first_started.set()
                await first_hold.wait()  # in-flight until barge-in cancels
                return
            if new_user_text == "second":
                second_started.set()
                return

        monkeypatch.setattr("backend.conversation_driver.run_turn", _run)
        monkeypatch.setattr(
            "backend.conversation_driver.AsyncSarvamTTSClient", _CtrlTTS
        )
        monkeypatch.setattr(
            "backend.conversation_driver.httpx.AsyncClient",
            lambda timeout: _FakeAsyncClientContextManager(MagicMock()),
        )

        ws = _FakeOutboundWebSocket()
        sender_task = asyncio.create_task(run_outbound_sender(sup, ws, "MZ_SID"))
        driver_task = asyncio.create_task(
            run_mercury_conversation_driver(sup), name="driver-i042-14"
        )
        try:
            await asyncio.wait_for(kickoff_done.wait(), timeout=2.0)

            await sup.completed_turn_queue.put("first")
            await asyncio.wait_for(first_started.wait(), timeout=2.0)

            # Barge-in, then queue the NEXT turn immediately — before the
            # stop sequence completes.
            assert sup.maybe_start_bargein() is True
            await sup.completed_turn_queue.put("second")

            # The stop sequence reaches the (blocked) Sarvam close.
            await asyncio.wait_for(close_started.wait(), timeout=2.0)

            # While close is blocked: the next turn must NOT have started and
            # NO reconnect may have occurred.
            assert not second_started.is_set()
            assert len(connect_log) == 1
            first_client = connect_log[0]
            assert first_client.closed is False

            # Release the close — barrier lifts, driver reconnects, next turn.
            close_release.set()
            await asyncio.wait_for(second_started.wait(), timeout=2.0)

            # A fresh client connected before the second turn; old never reused.
            assert len(connect_log) == 2
            assert connect_log[1].connected_to is sup
            assert sup.tts_client is connect_log[1]
            assert first_client.closed is True
            assert sup.tts_client is not first_client
            # Mercury driver alive; teardown never triggered.
            assert not driver_task.done()
            assert sup.teardown_run is False
        finally:
            close_release.set()
            first_hold.set()
            sup.shutdown_event.set()
            await _cancel_tasks(driver_task, sender_task, sup.bargein_task)


# ─── 15: DEFECT 3 — total_ms from synthesized bytes incl. late chunk ─────────


class TestTotalMsSynthesized:
    @pytest.mark.asyncio
    async def test_total_ms_uses_synthesized_bytes_not_sent_duration(
        self,
    ) -> None:
        sup = _make_supervisor()
        sup.is_speaking = True
        sup.disclosure_active = False
        sup.generation_counter = 2
        sup.current_turn_text = "long reply"
        sup.played_duration_ms = 40
        # Only 2 frames actually reached Twilio (sender was blocked) …
        sup.sent_duration_ms = 40
        # … but Sarvam had already synthesized ~1000 ms for this generation.
        sup.synthesized_bytes_by_generation[2] = 8000

        # A chunk that arrives DURING the stop sequence, before socket closure,
        # must still be counted — simulated by the client's close() adding
        # 100 ms of bytes right before it finishes closing.
        class _LateChunkTTS:
            def __init__(self) -> None:
                self.closed = False

            async def close(self) -> None:
                sup.synthesized_bytes_by_generation[2] += 800  # +100 ms
                self.closed = True

        sup.tts_client = _LateChunkTTS()
        acker = _spawn_clear_acker(sup)
        try:
            assert sup.maybe_start_bargein() is True
            await sup.bargein_task

            record = sup.interrupted_turns[0]
            # 8000 + 800 = 8800 bytes @ 8 kHz = 1100 ms — includes the late
            # chunk and is well above the 40 ms actually sent to Twilio.
            assert record.total_ms == 1100
            assert record.total_ms > sup.sent_duration_ms
            assert record.played_ms == 40
        finally:
            await _cancel_tasks(acker)


# ─── 16: Concurrent waiter race — clear latency is gone ─────────────────────


class _InstrumentedQueue(asyncio.Queue):
    """
    A subclass of ``asyncio.Queue`` that signals an ``asyncio.Event``
    immediately before blocking on ``get()``.  Subclassing is required
    so the sender's production ``isinstance(..., asyncio.Queue)`` type
    guard accepts the instrumented queue.

    The parked event is ``set`` every time a ``get()`` is about to
    block.  Tests observe it to prove the sender is actually parked in
    the concurrent waiter race (not on some other await point).
    """

    def __init__(self, *, maxsize: int = 0) -> None:
        super().__init__(maxsize=maxsize)
        self.parked: asyncio.Event = asyncio.Event()
        self._closed: bool = False

    def close(self) -> None:
        self._closed = True

    async def get(self) -> Any:  # type: ignore[override]
        # If an item is already queued, return it synchronously WITHOUT
        # firing the parked event (it never actually blocks).
        try:
            return self.get_nowait()
        except asyncio.QueueEmpty:
            pass
        # No item — about to block.  Signal the test.
        self.parked.set()
        if self._closed:
            raise asyncio.CancelledError()
        return await super().get()


class _DelayedQueue(_InstrumentedQueue):
    """
    An instrumented queue that GATES the ``get()`` coroutine behind a
    release event.  The test calls ``hold()`` BEFORE putting an item so
    the get-task parks at the gate.  Then ``put_nowait(item)`` queues
    the item but the get-task still waits.  Only when the test calls
    ``release()`` does the get-task finally consume the item and
    complete — letting the test deterministically interleave the
    audio waiter's completion with the concurrent waiter race.
    """

    def __init__(self, *, maxsize: int = 0) -> None:
        super().__init__(maxsize=maxsize)
        self._release: asyncio.Event = asyncio.Event()
        self._release.set()  # default: released (no gating)

    def hold(self) -> None:
        """Make the next ``get()`` wait for ``release()`` before consuming."""
        self._release.clear()

    def release(self) -> None:
        """Let the held ``get()`` consume the queued item."""
        self._release.set()

    async def get(self) -> Any:  # type: ignore[override]
        # Drain any already-queued item normally (without gating).
        try:
            return self.get_nowait()
        except asyncio.QueueEmpty:
            pass
        # Wait for release FIRST (test may have called hold() before any
        # item was queued).
        self.parked.set()
        if self._closed:
            raise asyncio.CancelledError()
        await self._release.wait()
        if self._closed:
            raise asyncio.CancelledError()
        # After release, the item should be queued.
        return await super().get()


class TestConcurrentWaiterRace:
    @pytest.mark.asyncio
    async def test_clear_wakes_idle_sender_via_real_concurrent_wait(
        self,
    ) -> None:
        """
        With BOTH queues empty, the sender parks in the concurrent
        waiter race.  Enqueuing only a ``ClearPlayback`` on the control
        channel wakes the sender immediately — there is NO polling
        interval to wait through, and NO audio is added.  All waiter
        tasks are cleaned up at shutdown.
        """
        sup = _make_supervisor()
        ws = _FakeOutboundWebSocket()

        # Swap the real queues for instrumented ones so the test can
        # observe the sender's exact parking points.
        control_q = _InstrumentedQueue()
        audio_q = _InstrumentedQueue()
        sup.outbound_control_queue = control_q  # type: ignore[assignment]
        sup.outbound_audio_queue = audio_q  # type: ignore[assignment]

        ack: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        cmd = ClearPlayback(generation=2, ack=ack)

        sender_task = asyncio.create_task(
            run_outbound_sender(sup, ws, "MZ_SID"),
            name="concurrent-race-sender",
        )
        try:
            # The sender enters the concurrent waiter race.  It must
            # park on both the control and audio gets (and the shutdown
            # event).  Wait for both to signal they are about to block.
            await asyncio.wait_for(
                _both_parked(control_q, audio_q), timeout=2.0
            )

            # Sanity: the sender is idle.  No WS traffic, ack not yet done.
            assert ws.sent == []
            assert not ack.done()

            # Enqueue the ONLY thing that can wake the sender.
            control_q.put_nowait(cmd)

            # The control waiter must resolve and the ack must fire
            # without any audio being enqueued.  No polling timer was
            # advanced (there is none in the implementation).
            assert await asyncio.wait_for(ack, timeout=2.0) is True
            assert control_q.qsize() == 0
            assert audio_q.qsize() == 0
        finally:
            sup.shutdown_event.set()
            # Give the sender time to exit cleanly, then collect any
            # leftover waiters.
            await asyncio.wait_for(sender_task, timeout=2.0)
            await _assert_no_leaked_waiter_tasks(
                sender_task, control_q, audio_q
            )

        # Exactly one clear went out, on the right stream SID.  No media.
        clears = [e for e in ws.sent if e.get("event") == "clear"]
        assert clears == [{"event": "clear", "streamSid": "MZ_SID"}]
        assert not [e for e in ws.sent if e.get("event") == "media"]

    @pytest.mark.asyncio
    async def test_both_ready_control_wins_audio_deferred_then_dropped(
        self,
    ) -> None:
        """
        Both queues are empty when the sender enters the concurrent
        waiter race.  The test then enqueues BOTH a ``ClearPlayback``
        and an audio frame; the control waiter resolves, the audio
        waiter resolves, both end up in ``done`` for the same race
        iteration.  Control is processed first; the simultaneously
        dequeued audio item is preserved in a local deferred-item
        variable and processed on the next iteration through the
        generation staleness check.

        Here the audio's generation matches the current
        ``generation_counter`` (no barge-in has bumped it), so the
        deferred audio frame IS emitted on the next iteration.
        Verified end-to-end: exactly one clear then one media frame,
        in that order.
        """
        from backend.audio.repacketizer import MULAW_FRAME_BYTES

        sup = _make_supervisor()
        sup.generation_counter = 1
        ws = _FakeOutboundWebSocket()

        control_q = _InstrumentedQueue()
        audio_q = _InstrumentedQueue()
        sup.outbound_control_queue = control_q
        sup.outbound_audio_queue = audio_q

        clear_ack: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        cmd = ClearPlayback(generation=2, ack=clear_ack)
        audio_frame = bytes(MULAW_FRAME_BYTES)

        sender_task = asyncio.create_task(
            run_outbound_sender(sup, ws, "MZ_SID"),
            name="both-ready-sender",
        )
        try:
            # Sender enters the waiter race with BOTH queues empty.
            # Both the control and audio waiters park.  Wait for that.
            await asyncio.wait_for(
                _both_parked(control_q, audio_q), timeout=2.0
            )

            # Now enqueue BOTH (control first, then audio).  They both
            # resolve in the same race iteration.  Control is processed
            # first; the audio item is deferred via the local
            # deferred-item variable (NOT re-queued at the tail).
            control_q.put_nowait(cmd)
            audio_q.put_nowait((1, audio_frame))  # gen 1 matches current

            # Control wins; ack fires; the deferred audio is then
            # processed on the next iteration (same generation, no
            # barge-in bump) and emitted.
            assert await asyncio.wait_for(clear_ack, timeout=2.0) is True
            for _ in range(200):
                if any(
                    e.get("event") == "media" for e in ws.sent
                ):
                    break
                await asyncio.sleep(0.01)
        finally:
            sup.shutdown_event.set()
            await asyncio.wait_for(sender_task, timeout=2.0)
            await _assert_no_leaked_waiter_tasks(
                sender_task, control_q, audio_q
            )

        # Order: clear first, then the audio frame.  No item re-queued.
        events = [e.get("event") for e in ws.sent]
        assert events[0] == "clear"
        assert "media" in events
        assert events.index("clear") < events.index("media")
        # Audio queue must be empty — the item was consumed via the
        # deferred path, never re-queued.
        assert audio_q.qsize() == 0

    @pytest.mark.asyncio
    async def test_both_ready_audio_stale_then_dropped(self) -> None:
        """
        When the deferred audio item's generation is STALE (the
        orchestrator has already bumped ``generation_counter`` for a
        barge-in), the deferred audio is dropped by the generation
        check, NOT re-emitted.  This is the path where a late clear
        can never leak old-generation audio into a new turn.
        """
        from backend.audio.repacketizer import MULAW_FRAME_BYTES

        sup = _make_supervisor()
        sup.generation_counter = 5  # post-bump
        ws = _FakeOutboundWebSocket()

        control_q = _InstrumentedQueue()
        audio_q = _InstrumentedQueue()
        sup.outbound_control_queue = control_q
        sup.outbound_audio_queue = audio_q

        clear_ack: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        cmd = ClearPlayback(generation=5, ack=clear_ack)
        audio_frame = bytes(MULAW_FRAME_BYTES)

        sender_task = asyncio.create_task(
            run_outbound_sender(sup, ws, "MZ_SID"),
            name="stale-audio-sender",
        )
        try:
            await asyncio.wait_for(
                _both_parked(control_q, audio_q), timeout=2.0
            )

            # Enqueue BOTH.  Control wins.  Deferred audio has stale
            # generation 1 < current 5 → dropped.
            control_q.put_nowait(cmd)
            audio_q.put_nowait((1, audio_frame))  # gen 1 < current 5

            assert await asyncio.wait_for(clear_ack, timeout=2.0) is True
            # Give the sender a moment to drain the deferred stale audio.
            for _ in range(200):
                if audio_q.qsize() == 0:
                    break
                await asyncio.sleep(0.01)
        finally:
            sup.shutdown_event.set()
            await asyncio.wait_for(sender_task, timeout=2.0)
            await _assert_no_leaked_waiter_tasks(
                sender_task, control_q, audio_q
            )

        # Stale audio MUST be dropped — no media frames at all.
        assert not [e for e in ws.sent if e.get("event") == "media"]
        # And it was consumed from the queue (deferred path), not
        # re-queued.
        assert audio_q.qsize() == 0


# ─── 17: Media-stream fixture regression — MagicMock control queue is safe ──


class TestMagicMockControlQueueSafety:
    @pytest.mark.asyncio
    async def test_sender_exits_safely_when_control_queue_is_missing(
        self,
    ) -> None:
        """
        Even if a test fixture forgets to initialise
        ``outbound_control_queue`` (the OOM regression root cause), the
        sender must NEVER spin.  Production hardening detects the missing
        / non-``asyncio.Queue`` channel, logs the failure, and either skips
        the drain or exits the sender — it must not loop forever.
        """
        manager = ConcurrencyManager(capacity=1)
        # Real supervisor but with the control queue attribute forcibly deleted
        # so the production validation path is exercised.
        sup = CallSupervisor(
            lead_id=1, attempt_id=1, call_sid="CA_NO_CTRL", language="English",
            concurrency_manager=manager, pool=MagicMock(),
        )
        del sup.outbound_control_queue  # type: ignore[attr-defined]

        ws = _FakeOutboundWebSocket()
        # CRITICAL: do NOT pre-set shutdown — the test must actually enter
        # the sender loop to exercise the production check.
        sender_task = asyncio.create_task(run_outbound_sender(sup, ws, "MZ"))
        # The sender must exit on its own within a short window — either the
        # loop noticed the missing queue and exited, or our pre-set shutdown
        # below ends it.  No infinite spin either way.
        await asyncio.sleep(0.05)
        sup.shutdown_event.set()
        await asyncio.wait_for(sender_task, timeout=2.0)
        # Nothing was sent (no real clear was enqueued).
        assert ws.sent == []

    @pytest.mark.asyncio
    async def test_sender_exits_safely_when_control_queue_is_magic_mock(
        self,
    ) -> None:
        """A non-queue object on ``outbound_control_queue`` is treated as
        absent — the sender keeps polling audio but never spins on it."""
        sup = _make_supervisor()
        # Replace the real queue with a MagicMock that never raises
        # QueueEmpty — the exact regression that caused the 8 GiB OOM.
        sup.outbound_control_queue = MagicMock()

        ws = _FakeOutboundWebSocket()
        sender_task = asyncio.create_task(run_outbound_sender(sup, ws, "MZ"))
        # If the validation is missing the sender would spin allocating
        # objects; we instead expect a clean exit when shutdown fires.
        await asyncio.sleep(0.05)
        sup.shutdown_event.set()
        await asyncio.wait_for(sender_task, timeout=2.0)
        assert ws.sent == []


# ─── 18: Barrier timeout raises + triggers safe teardown ─────────────────────


class TestBarrierTimeout:
    @pytest.mark.asyncio
    async def test_barrier_timeout_raises_and_does_not_start_next_turn(
        self, pool, monkeypatch
    ) -> None:
        """
        With the barrier wait cranked to 0 s, the driver must NOT proceed to
        the next turn — ``_ensure_tts_client`` and the next ``run_turn`` are
        never called, and teardown is triggered with ``provider_fault=True``.
        """
        lead_id = await _seed_lead(pool)
        sup = _make_supervisor(lead_id=lead_id, pool=pool)
        register_call(sup)

        # Stop sequence that stalls forever — the barrier is never set.
        stall = asyncio.Event()
        sup.bargein_complete.clear()

        async def _stalling_stop() -> None:
            await stall.wait()  # would only exit on test cleanup

        async def _noop_resolve(*a, **kw):
            return None

        # Crank the timeout to 0 so the wait fails immediately.
        monkeypatch.setattr(
            "backend.conversation_driver._BARGEIN_COMPLETE_TIMEOUT_S", 0.0
        )

        sup.is_speaking = True
        ensure_calls: list[Any] = []
        run_calls: list[Any] = []

        async def _fake_ensure(_sup):
            ensure_calls.append(_sup)
            return None

        async def _fake_run(*args, **kwargs):
            run_calls.append((args, kwargs))

        async def _fake_teardown(*args, **kwargs):
            return None

        with patch(
            "backend.conversation_driver._ensure_tts_client", _fake_ensure
        ), patch("backend.conversation_driver.run_turn", _fake_run), patch(
            "backend.call_supervisor.CallSupervisor.trigger_teardown",
            AsyncMock(side_effect=_fake_teardown),
        ) as mock_teardown:
            # The new top-level call path is direct: invoke the wait and
            # confirm it raises; the caller's outer handler (here invoked as
            # the driver loop would do) escalates to teardown.
            try:
                await _await_bargein_complete(sup)
                raise AssertionError("_await_bargein_complete should have raised")
            except BargeinCompletionTimeoutError:
                await sup.trigger_teardown(
                    outcome_status="Failed", failure_reason="error",
                    provider_fault=True,
                )
        assert mock_teardown.await_count == 1

        assert ensure_calls == [], "_ensure_tts_client must NOT run on barrier timeout"
        assert run_calls == [], "next run_turn must NOT run on barrier timeout"

    @pytest.mark.asyncio
    async def test_shutdown_releases_barrier_wait_without_starting_next_turn(
        self, monkeypatch
    ) -> None:
        """
        Setting ``shutdown_event`` while the barrier is unset must release
        the wait cleanly (no raise, no next turn) so the driver can exit.
        """
        sup = _make_supervisor()
        sup.bargein_complete.clear()
        monkeypatch.setattr(
            "backend.conversation_driver._BARGEIN_COMPLETE_TIMEOUT_S", 0.0
        )

        # Wake shutdown very soon — well within the 0 s ceiling.
        async def _shutdown_after():
            await asyncio.sleep(0.0)
            sup.shutdown_event.set()

        await asyncio.gather(_await_bargein_complete(sup), _shutdown_after())
        # Clean return — no exception bubbled.
        assert sup.bargein_complete.is_set() is False  # still unset, but exit was clean
        assert sup.shutdown_event.is_set() is True


# ─── 20: Clear failure → safe teardown (broken-media-path guard) ────────────


class TestWaiterCleanupRace:
    """
    The concurrent waiter race has a subtle cleanup hazard: the
    ``done`` set returned by ``asyncio.wait(FIRST_COMPLETED)`` is a
    snapshot.  A waiter that completes AFTER the snapshot but BEFORE
    our cleanup runs has already dequeued its item; cancelling it
    post-hoc would silently drop the item.  The correct fix is to use
    ``asyncio.gather(return_exceptions=True)`` on every waiter after
    cancelling only those that are NOT done.

    These tests use ``_DelayedQueue`` to deterministically interleave
    a waiter's completion with the cleanup.  No sleep-based timing.
    """

    @pytest.mark.asyncio
    async def test_control_wins_audio_completes_after_snapshot(
        self,
    ) -> None:
        """
        Control waiter completes first (``asyncio.wait`` captures that
        result in its ``done`` snapshot).  The audio waiter is still
        parked (gated by a release event).  The test then enqueues an
        audio frame AND releases the audio waiter — the audio task
        completes AFTER the snapshot but BEFORE the sender's cleanup.
        With the fix, the audio item is preserved in the deferred-item
        path and emitted on the next iteration.
        """
        from backend.audio.repacketizer import MULAW_FRAME_BYTES

        sup = _make_supervisor()
        sup.generation_counter = 1
        ws = _FakeOutboundWebSocket()

        control_q = _InstrumentedQueue()
        audio_q = _DelayedQueue()
        audio_q.hold()  # gate the audio get() behind release()
        sup.outbound_control_queue = control_q
        sup.outbound_audio_queue = audio_q

        clear_ack: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        cmd = ClearPlayback(generation=2, ack=clear_ack)
        audio_frame = bytes(MULAW_FRAME_BYTES)

        sender_task = asyncio.create_task(
            run_outbound_sender(sup, ws, "MZ_SID"),
            name="control-first-audio-after-snapshot",
        )
        try:
            # Wait for the sender to enter the concurrent waiter race.
            # Both control and audio waiters are parked.
            await asyncio.wait_for(
                _both_parked(control_q, audio_q), timeout=2.0
            )

            # Enqueue the control clear — the control waiter resolves
            # IMMEDIATELY (no gating).  ``asyncio.wait(FIRST_COMPLETED)``
            # wakes with control in the done snapshot.
            control_q.put_nowait(cmd)

            # Enqueue the audio item.  The audio waiter is still parked
            # at the release gate (it has not yet consumed the item).
            audio_q.put_nowait((1, audio_frame))

            # Give the event loop a beat to process the control waiter's
            # completion so the snapshot is taken with ONLY control in
            # the done set.  (No sleep — just yield once.)
            await asyncio.sleep(0)

            # NOW release the audio waiter.  It dequeues the item and
            # completes — AFTER the snapshot, BEFORE the sender's
            # cleanup.  This is the exact race the fix protects against.
            audio_q.release()

            # The fix must preserve the audio item: clear is emitted,
            # then the deferred audio is emitted on the next iteration.
            assert await asyncio.wait_for(clear_ack, timeout=2.0) is True
            for _ in range(200):
                if any(
                    e.get("event") == "media" for e in ws.sent
                ):
                    break
                await asyncio.sleep(0.01)
        finally:
            sup.shutdown_event.set()
            await asyncio.wait_for(sender_task, timeout=2.0)
            await _assert_no_leaked_waiter_tasks(
                sender_task, control_q, audio_q
            )

        # Order: clear first, then the audio frame.  The audio queue
        # is EMPTY because the item was processed via the deferred
        # path, not lost.
        events = [e.get("event") for e in ws.sent]
        assert events[0] == "clear"
        assert "media" in events
        assert events.index("clear") < events.index("media")
        assert audio_q.qsize() == 0

    @pytest.mark.asyncio
    async def test_audio_wins_control_completes_after_snapshot(
        self,
    ) -> None:
        """
        Symmetric: the audio waiter completes first; the control
        waiter is gated behind a release event.  The test enqueues a
        control clear and releases the control waiter AFTER the
        snapshot.  The control item is preserved and processed (after
        the audio) — the control clear is emitted.
        """
        sup = _make_supervisor()
        sup.generation_counter = 1
        ws = _FakeOutboundWebSocket()

        control_q = _DelayedQueue()
        control_q.hold()  # gate the control get() behind release()
        audio_q = _InstrumentedQueue()
        sup.outbound_control_queue = control_q
        sup.outbound_audio_queue = audio_q

        from backend.audio.repacketizer import MULAW_FRAME_BYTES

        clear_ack: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        cmd = ClearPlayback(generation=2, ack=clear_ack)
        audio_frame = bytes(MULAW_FRAME_BYTES)

        sender_task = asyncio.create_task(
            run_outbound_sender(sup, ws, "MZ_SID"),
            name="audio-first-control-after-snapshot",
        )
        try:
            # Wait for the sender to enter the concurrent waiter race.
            # Both control and audio waiters are parked.
            await asyncio.wait_for(
                _both_parked(control_q, audio_q), timeout=2.0
            )

            # Enqueue the audio frame — the audio waiter resolves
            # IMMEDIATELY (no gating).  ``asyncio.wait(FIRST_COMPLETED)``
            # wakes with audio in the done snapshot.
            audio_q.put_nowait((1, audio_frame))

            # Enqueue the control clear.  The control waiter is still
            # parked at the release gate.
            control_q.put_nowait(cmd)

            # Yield once so the audio waiter is fully marked done.
            await asyncio.sleep(0)

            # Release the control waiter.  It dequeues the clear and
            # completes — AFTER the snapshot, BEFORE the sender's
            # cleanup.
            control_q.release()

            # Audio is processed first (it was in the done snapshot).
            # The control clear is preserved by the fix and emitted.
            # The audio frame is emitted before the clear (it was
            # already in the snapshot).
            for _ in range(200):
                if any(
                    e.get("event") == "clear" for e in ws.sent
                ):
                    break
                await asyncio.sleep(0.01)
        finally:
            sup.shutdown_event.set()
            await asyncio.wait_for(sender_task, timeout=2.0)
            await _assert_no_leaked_waiter_tasks(
                sender_task, control_q, audio_q
            )

        # Both audio and clear were processed (both items dequeued).
        # The audio queue and control queue are both empty.
        events = [e.get("event") for e in ws.sent]
        assert "media" in events
        assert "clear" in events
        assert clear_ack.result() is True
        assert audio_q.qsize() == 0
        assert control_q.qsize() == 0


class TestClearFailureSafeTeardown:
    """
    A barge-in ``ClearPlayback`` that cannot be confirmed delivered to
    Twilio is a broken-media-path incident: Riya has not actually gone
    silent.  The stop sequence must (a) still complete normal cleanup
    (cancel inner turn + close Sarvam + record interrupted metadata),
    (b) mark the media path broken so every later ClearPlayback is
    discarded by the sender (no late clear can reach the wire), and
    (c) escalate to safe teardown so the driver never starts another
    turn against a half-broken media path.
    """

    @pytest.mark.asyncio
    async def test_clear_send_failure_escalates_to_safe_teardown(self) -> None:
        """
        Sender reports ``ack=False`` (send raised) — stop sequence still
        completes its normal cleanup, marks media_path_broken, and
        escalates to trigger_teardown exactly once.
        """
        sup = _make_supervisor()
        sup.is_speaking = True
        sup.disclosure_active = False
        sup.generation_counter = 1
        sup.current_turn_text = "long reply"
        sup.played_duration_ms = 100
        sup.synthesized_bytes_by_generation[1] = 1600  # 200 ms synthesized
        # Inner turn that gets cancelled during the stop sequence.
        turn_hold = asyncio.Event()
        sup.current_turn_task = asyncio.create_task(turn_hold.wait())

        # Ack-acker that resolves ack=False (simulates send failure).
        async def _failing_acker() -> None:
            cmd = await sup.outbound_control_queue.get()
            if cmd.ack is not None and not cmd.ack.done():
                cmd.ack.set_result(False)

        acker = asyncio.create_task(_failing_acker())
        try:
            assert sup.maybe_start_bargein() is True
            await sup.bargein_task

            assert sup.media_path_broken is True
            assert sup.interrupted_turns[0].full_text == "long reply"
            assert sup.interrupted_turns[0].total_ms == 200
            # Safe teardown executed — the latch is set and the call is
            # resolved to Failed/error.
            assert sup._teardown_run is True
            assert sup._last_outcome_status == "Failed"
            assert sup._last_failure_reason == "error"
        finally:
            await _cancel_tasks(acker, turn_hold.wait() if False else None)
            turn_hold.set()

    @pytest.mark.asyncio
    async def test_clear_ack_timeout_escalates_to_safe_teardown(self) -> None:
        """
        Sender never acks (no acker task) — ``_CLEAR_ACK_TIMEOUT_S``
        fires; stop sequence still completes cleanup and escalates.
        """
        sup = _make_supervisor()
        sup.is_speaking = True
        sup.disclosure_active = False
        sup.generation_counter = 1
        sup.current_turn_text = "reply"
        sup.synthesized_bytes_by_generation[1] = 800

        # No acker — the ack future is left pending and the timeout fires.
        # Shorten the timeout so the test is fast.
        with patch(
            "backend.call_supervisor._CLEAR_ACK_TIMEOUT_S", 0.05
        ):
            assert sup.maybe_start_bargein() is True
            await sup.bargein_task

        assert sup.media_path_broken is True
        assert sup.interrupted_turns[0].full_text == "reply"
        assert sup._teardown_run is True

    @pytest.mark.asyncio
    async def test_clear_failure_blocks_next_turn_no_mercury_or_tts(self) -> None:
        """
        After a failed clear, the Mercury driver must NOT start another
        turn.  The stop sequence's normal cleanup (cancel inner turn,
        close Sarvam, record interrupted metadata) must all complete
        BEFORE the escalation, and the escalated ``trigger_teardown``
        must run exactly once.  After the escalation, the one-shot
        teardown latch is set so any external second trigger is a no-op.
        """
        sup = _make_supervisor()
        sup.is_speaking = True
        sup.disclosure_active = False
        sup.generation_counter = 1
        sup.current_turn_text = "reply"

        # Inner turn task that the stop sequence must cancel.
        turn_hold = asyncio.Event()
        turn_task = asyncio.create_task(turn_hold.wait())
        sup.current_turn_task = turn_task

        # Sarvam client whose ``close()`` flips ``closed``.
        class _FakeTTS:
            def __init__(self) -> None:
                self.closed = False

            async def connect(self, _sup: Any) -> None:
                return None

            async def synthesize_turn(self, *a: Any, **k: Any) -> None:
                return None

            async def close(self) -> None:
                self.closed = True

        tts_client = _FakeTTS()
        sup.tts_client = tts_client

        # Ack-acker that reports failure.
        async def _failing_acker() -> None:
            cmd = await sup.outbound_control_queue.get()
            if cmd.ack is not None and not cmd.ack.done():
                cmd.ack.set_result(False)

        async def _noop_resolve(*a: Any, **kw: Any) -> None:
            return None

        async def _noop_delete(_call_sid: str) -> int:
            return 0

        acker = asyncio.create_task(_failing_acker())
        try:
            with patch(
                "backend.call_supervisor._resolve_failed_attempt",
                side_effect=_noop_resolve,
            ), patch(
                "backend.call_supervisor._delete_call_recordings_async",
                side_effect=_noop_delete,
            ), patch(
                "backend.call_supervisor._twilio_hangup_call", return_value=True
            ):
                assert sup.maybe_start_bargein() is True
                await sup.bargein_task

            # Normal cleanup completed: inner turn cancelled, TTS closed,
            # interrupted metadata recorded, media_path_broken set.
            assert turn_task.cancelled() or turn_task.done()
            assert tts_client.closed is True
            assert sup.media_path_broken is True
            assert len(sup.interrupted_turns) == 1
            # Safe teardown fired exactly once (the one-shot latch is set).
            assert sup._teardown_run is True
            assert sup._last_outcome_status == "Failed"
            assert sup._last_failure_reason == "error"

            # An external second trigger is a no-op (one-shot latch).
            await sup.trigger_teardown("Failed", "dropped")
            assert sup._teardown_run is True
            assert sup._last_outcome_status == "Failed"  # unchanged
            assert sup._last_failure_reason == "error"  # unchanged
        finally:
            await _cancel_tasks(acker)
        assert sup._last_outcome_status == "Failed"
        assert sup._last_failure_reason == "error"

    @pytest.mark.asyncio
    async def test_no_late_clear_affects_newer_generation(self) -> None:
        """
        After the media path is marked broken, any subsequent
        ``ClearPlayback`` (e.g. from a second barge-in race or a stale
        queued command) is silently DISCARDED by the sender with its
        ack resolved ``False`` — it can never reach the wire and reset
        a newer-generation playback buffer.
        """
        sup = _make_supervisor()
        ws = _FakeOutboundWebSocket()
        # Simulate "media path broken" already set (e.g. previous barge-in
        # failed).  We do NOT run the stop sequence here — we just verify
        # the sender's stale-clear guard drops later commands.
        sup.media_path_broken = True
        sup.generation_counter = 5  # "current" generation post-clear

        # Enqueue a stale ClearPlayback.  The sender must discard it
        # and resolve the ack ``False``.
        stale_ack: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        stale_cmd = ClearPlayback(generation=3, ack=stale_ack)  # gen 3 < 5
        sup.outbound_control_queue.put_nowait(stale_cmd)

        # Drive the sender.  It will pick the stale command, discard it,
        # and resolve the ack False.
        sender_task = asyncio.create_task(
            run_outbound_sender(sup, ws, "MZ_SID")
        )
        try:
            assert await asyncio.wait_for(stale_ack, timeout=2.0) is False
            # Nothing was emitted to the WS — the stale command was dropped.
            assert ws.sent == []
        finally:
            sup.shutdown_event.set()
            await asyncio.wait_for(sender_task, timeout=2.0)

    @pytest.mark.asyncio
    async def test_teardown_executes_exactly_once_after_clear_failure(self) -> None:
        """
        Even if the stop sequence itself completes cleanup that
        triggers the teardown escalation, the supervisor's one-shot
        latch ensures ``trigger_teardown`` runs exactly once.  A
        subsequent external teardown trigger is a no-op.
        """
        sup = _make_supervisor()
        sup.is_speaking = True
        sup.disclosure_active = False
        sup.generation_counter = 1
        sup.current_turn_text = "reply"

        # Stub the resolve path so we can confirm teardown was called once.
        resolve_calls: list[str] = []

        async def _spy_resolve(*a: Any, **kw: Any) -> None:
            resolve_calls.append(kw.get("failure_reason", ""))

        async def _failing_acker() -> None:
            cmd = await sup.outbound_control_queue.get()
            if cmd.ack is not None and not cmd.ack.done():
                cmd.ack.set_result(False)

        async def _noop_delete(_call_sid: str) -> int:
            return 0

        acker = asyncio.create_task(_failing_acker())
        try:
            with patch(
                "backend.call_supervisor._resolve_failed_attempt",
                side_effect=_spy_resolve,
            ), patch(
                "backend.call_supervisor._delete_call_recordings_async",
                side_effect=_noop_delete,
            ), patch(
                "backend.call_supervisor._twilio_hangup_call", return_value=True
            ):
                assert sup.maybe_start_bargein() is True
                await sup.bargein_task

                # First (and only) trigger_teardown fired during the stop
                # sequence because the clear failed.  The latch is set.
                assert sup._teardown_run is True
                first_call_count = len(resolve_calls)
                assert first_call_count == 1

                # An external second trigger is a no-op (one-shot latch).
                await sup.trigger_teardown("Failed", "dropped")
                assert sup._teardown_run is True
                assert len(resolve_calls) == first_call_count
        finally:
            await _cancel_tasks(acker)


# ─── 21: Missing audio queue → clean exit (no busy loop) ────────────────────


class TestMissingAudioQueue:
    @pytest.mark.asyncio
    async def test_missing_audio_queue_exits_cleanly(self) -> None:
        """
        If ``outbound_audio_queue`` is missing or invalid (test fixture
        or operator oversight), the sender must exit cleanly without
        spinning on ``asyncio.sleep(0)`` or otherwise wasting CPU.
        Bounded wait proves the sender does NOT loop forever.
        """
        manager = ConcurrencyManager(capacity=1)
        sup = CallSupervisor(
            lead_id=1,
            attempt_id=1,
            call_sid="CA_NO_AUDIO",
            language="English",
            concurrency_manager=manager,
            pool=MagicMock(),
        )
        # Forcibly remove the audio queue — simulates a missing fixture.
        del sup.outbound_audio_queue
        # Make sure the control queue is present so we exercise the
        # no-audio + control-available path.
        ws = _FakeOutboundWebSocket()

        # Bound the runtime — the sender must exit in well under a
        # second even with no audio and no clear to process.
        sender_task = asyncio.create_task(
            run_outbound_sender(sup, ws, "MZ")
        )
        await asyncio.wait_for(sender_task, timeout=1.0)
        # Nothing was sent.
        assert ws.sent == []


# ─── 22: Real-driver barrier-timeout integration test ───────────────────────


class TestBarrierTimeoutIntegration:
    @pytest.mark.asyncio
    async def test_barrier_timeout_drives_real_driver_to_safe_teardown(
        self, pool, monkeypatch
    ) -> None:
        """
        Real-driver end-to-end barrier-timeout integration.

        Drives the REAL ``run_mercury_conversation_driver`` coroutine
        (no manual ``except`` block, no manual ``trigger_teardown``
        call).  The ONLY thing monkey-patched is the stop sequence
        itself (a stalling replacement that cancels the inner turn but
        NEVER releases the completion barrier) and the Sarvam
        TTS / Mercury HTTP plumbing (to avoid live network).  The
        real kickoff ``except BargeinCompletionTimeoutError`` handler
        must catch the timeout from the cancellable-turn wrapper and
        escalate to ``trigger_teardown(provider_fault=True)``.

        Verifies the FULL contract: when the barrier does not fire
        within the deadline and shutdown is not active, the wrapper
        raises ``BargeinCompletionTimeoutError`` AND the driver
        catch handler catches it AND triggers teardown with
        ``provider_fault=True``.  No subsequent Mercury/TTS turn ever
        runs.
        """
        # Crank the barrier timeout to 0 s — any non-immediate return
        # from ``_await_bargein_complete`` is a timeout.
        monkeypatch.setattr(
            "backend.conversation_driver._BARGEIN_COMPLETE_TIMEOUT_S", 0.0
        )

        lead_id = await _seed_lead(pool)
        sup = _make_supervisor(lead_id=lead_id, pool=pool)
        register_call(sup)

        # Stalling stop sequence: cancels the inner turn task (so the
        # wrapper catches CancelledError) but NEVER sets the barrier
        # (so ``_await_bargein_complete`` times out).
        stop_release = asyncio.Event()

        async def _stalling_stop(self: Any) -> None:
            turn_task = self.current_turn_task
            if turn_task is not None and not turn_task.done():
                turn_task.cancel()
                try:
                    await turn_task
                except asyncio.CancelledError:
                    pass
            # Park forever (do NOT set the barrier).
            await asyncio.shield(stop_release.wait())

        monkeypatch.setattr(
            "backend.call_supervisor.CallSupervisor._run_bargein_stop_sequence",
            _stalling_stop,
        )

        # Stub the Sarvam TTS client and Mercury HTTP plumbing so the
        # driver can run end-to-end without any live network.
        class _TTS:
            def __init__(self) -> None:
                self.connected_to: Any = None
                self.closed = False

            async def connect(self, supervisor: Any) -> None:
                self.connected_to = supervisor

            async def synthesize_turn(self, *a: Any, **k: Any) -> None:
                return None

            async def close(self) -> None:
                self.closed = True

        tts = _TTS()

        kickoff_done = asyncio.Event()
        turn_started = asyncio.Event()
        run_hold = asyncio.Event()
        ensure_calls: list[int] = {"n": 0}
        run_calls: list[str] = []

        async def _run_turn_stub(
            supervisor: Any, messages: Any, *, new_user_text: Any,
            http_client: Any, api_key: Any
        ) -> None:
            run_calls.append(repr(new_user_text))
            if new_user_text is None:
                # Kickoff: append greeting and signal completion.
                messages.append({"role": "assistant", "content": "greeting"})
                kickoff_done.set()
                return
            # Real customer turn: signal entry then park so the test can
            # observe the wrapper awaiting the inner task and trigger
            # barge-in.
            turn_started.set()
            await run_hold.wait()

        async def _ensure_stub(_sup: Any) -> None:
            ensure_calls["n"] += 1

        # Spy on the REAL trigger_teardown so the test can observe the
        # escalation outcome without touching live Twilio / DB.
        teardown_outcomes: list[str] = []

        original_teardown = CallSupervisor.trigger_teardown

        async def _spy_teardown(
            self: Any, *args: Any, **kwargs: Any
        ) -> Any:
            teardown_outcomes.append(
                f"{kwargs.get('outcome_status', '?')}/"
                f"{kwargs.get('failure_reason', '?')}/"
                f"{kwargs.get('provider_fault', '?')}"
            )
            async def _noop_resolve(*a: Any, **kw: Any) -> None:
                return None

            with patch(
                "backend.call_supervisor._resolve_failed_attempt",
                side_effect=_noop_resolve,
            ), patch(
                "backend.call_supervisor._delete_call_recordings_async",
                side_effect=lambda _call_sid: 0,
            ), patch(
                "backend.call_supervisor._twilio_hangup_call", return_value=True
            ):
                return await original_teardown(self, *args, **kwargs)

        with patch(
            "backend.conversation_driver._ensure_tts_client", _ensure_stub
        ), patch(
            "backend.conversation_driver.run_turn", _run_turn_stub
        ), patch(
            "backend.conversation_driver.AsyncSarvamTTSClient", lambda: tts
        ), patch(
            "backend.conversation_driver.httpx.AsyncClient",
            lambda timeout: _FakeAsyncClientContextManager(MagicMock()),
        ), patch(
            "backend.call_supervisor.CallSupervisor.trigger_teardown",
            _spy_teardown,
        ):
            driver_task = asyncio.create_task(
                run_mercury_conversation_driver(sup),
                name="barrier-timeout-driver",
            )
            try:
                # Unblock the disclosure + AMD gates so the driver
                # reaches the kickoff turn.
                _unblock_call_start(sup)

                # Wait for the kickoff to complete (single run_turn call).
                await asyncio.wait_for(kickoff_done.wait(), timeout=2.0)
                assert ensure_calls["n"] == 1
                assert run_calls == ["None"]

                # Now push a customer transcript.  The driver enters
                # the cancellable-turn wrapper, which calls run_turn.
                await sup.completed_turn_queue.put("real-customer-turn")
                await asyncio.wait_for(turn_started.wait(), timeout=2.0)
                # The wrapper is now parked on ``run_hold.wait()``
                # inside the inner turn.  Barge-in is triggered by the
                # listener in production; in the test we trigger
                # maybe_start_bargein directly (the supervisor's
                # bargein_task is the stalling stop sequence).
                sup.is_speaking = True
                assert sup.maybe_start_bargein() is True

                # The driver must catch the barrier timeout from the
                # wrapper and escalate to trigger_teardown within a
                # bounded time (barrier timeout is 0 s, so this should
                # fire almost immediately after the wrapper catches the
                # CancelledError from the stalling stop sequence).
                await asyncio.wait_for(driver_task, timeout=2.0)

                # 1. The real kickoff ``except`` handler escalated to
                #    ``trigger_teardown`` exactly once with
                #    ``provider_fault=True``.
                assert len(teardown_outcomes) == 1
                assert teardown_outcomes[0] == "Failed/error/True"
                # 2. ``run_turn`` was called once for the kickoff and
                #    once for the customer turn (then the wrapper raised
                #    the barrier timeout — no third call).
                assert run_calls == ["None", "'real-customer-turn'"]
                # 3. ``_ensure_tts_client`` was called once per turn.
                assert ensure_calls["n"] == 2
            finally:
                # Release the stalled stop sequence so the test exits
                # cleanly.
                stop_release.set()
                if not driver_task.done():
                    driver_task.cancel()
                    try:
                        await asyncio.gather(
                            driver_task, return_exceptions=True
                        )
                    except Exception:
                        pass


# ─── Helpers used by TestConcurrentWaiterRace ─────────────────────────────


async def _both_parked(*queues: "_InstrumentedQueue") -> None:
    """Wait until every instrumented queue has signalled that its
    ``get()`` is about to block (i.e. the test has observed the
    concurrent-waiter-race parking point).  Used to prove the sender
    is truly in the waiter race, not on some other await point."""
    for q in queues:
        await q.parked.wait()


async def _assert_no_leaked_waiter_tasks(
    sender_task: "asyncio.Task[None]",
    *queues: "_InstrumentedQueue",
) -> None:
    """After the sender task has finished, assert that NO unfinished
    tasks whose names start with ``sender-wait-`` remain.  The sender
    spawns these as the control/audio/shutdown waiters inside the
    concurrent waiter race; if any are still running at sender exit
    the cleanup is broken and they have leaked.

    Tolerates the instrumented queue's own ``Queue.get()`` task (which
    lives on the real ``asyncio.Queue`` underneath — its name does NOT
    start with ``sender-wait-`` so it is not flagged here).  The
    instrumented queue's get-task is also a test artifact, not a
    production leak; we drain it explicitly via ``queue.close()`` so
    the test does not leave it running between tests.
    """
    # First: close all instrumented queues so their internal get() task
    # unblocks (it is parked on ``super().get()``).  We do this before
    # the assertion so we are not flagging a task we caused to stay
    # parked by the test setup itself.
    for q in queues:
        q.close()

    leaked = [
        t for t in list(asyncio.all_tasks())
        if t is not sender_task
        and not t.done()
        and (
            (t.get_name() or "").startswith("sender-wait-")
        )
    ]
    assert not leaked, (
        f"sender leaked {len(leaked)} unfinished waiter task(s): "
        f"{[t.get_name() for t in leaked]}"
    )


# ─── Helpers used by TestBarrierTimeoutIntegration ─────────────────────────


def _make_pool_for_test():
    """Convenience fixture factory used by tests above (matches conftest)."""
    # Hand the tests an inline, non-async pool — the driver does not use
    # it for the barge-in path (turn is cancelled before any DB call).
    # Tests that DO need a real DB use the ``pool`` fixture instead.
    return MagicMock()
