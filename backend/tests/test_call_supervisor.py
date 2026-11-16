"""
backend/tests/test_call_supervisor.py — Unit tests for CallSupervisor + registry.

These tests cover Issue 032 (Call Supervisor & Idempotent Teardown) at the
unit level — no FastAPI app, no live Twilio, no live DB writes from webhooks.
They verify:

  - The one-shot latch prevents double-execution under concurrent triggers.
  - The slot is released exactly once even with multiple triggers.
  - The fixed teardown sequence (hangup → resolve → release → unregister)
    runs in order.
  - The registry functions (register/get/unregister) work in isolation.
  - The supervisor self-unregisters at the end of teardown.

The arrange_ai_call orphan-window test is in test_arrange.py (it requires
the full FastAPI app).
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from backend.call_supervisor import (
    CallSupervisor,
    _active_calls,
    get_call,
    register_call,
    unregister_call,
)
from backend.concurrency import ConcurrencyManager


# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clear_supervisor_registry() -> None:
    """Clear the in-memory active-calls registry between tests."""
    _active_calls.clear()
    yield
    _active_calls.clear()


def _schema_path() -> Path:
    return (
        Path(__file__).parent.parent.parent / "database" / "app_schema.sql"
    ).resolve()


def _create_temp_db() -> Path:
    """Create a temporary app.db from the application schema."""
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".db", prefix="car_test_supervisor_"
    ) as fh:
        tmp = Path(fh.name)
    conn = sqlite3.connect(str(tmp))
    conn.execute("PRAGMA foreign_keys = ON;")
    with open(_schema_path(), encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.close()
    return tmp


@pytest_asyncio.fixture
async def temp_db() -> AsyncGenerator[Path, None]:
    """Per-test isolated database backed by a temp file."""
    db_path = _create_temp_db()
    yield db_path
    try:
        db_path.unlink()
    except OSError:
        pass


@pytest_asyncio.fixture
async def pool(temp_db: Path):
    """Async DB connection pool backed by the temp DB."""
    from backend.database import create_pool

    p = create_pool(
        db_path=temp_db,
        busy_timeout_seconds=5.0,
        pool_size=5,
    )
    yield p
    await p.close()


async def _insert_processing_lead_with_live_attempt(
    pool,
    *,
    call_sid: str,
    phone: str = "+919911111111",
) -> tuple[int, int]:
    """Insert a Processing lead + a live call_attempt with the given CallSid."""
    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            INSERT INTO leads
                (name, phone, email, city, language, status, attempt_count)
            VALUES ('Supervisor Test Lead', ?, 'sup@example.com',
                    'Mumbai', 'English', 'Processing', 1)
            """,
            (phone,),
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


# ─── Test: Active-calls registry ───────────────────────────────────────────────


class TestActiveCallsRegistry:
    """The module-level registry's accessor functions work in isolation."""

    def test_register_then_get_returns_supervisor(self) -> None:
        """register_call + get_call round-trip."""
        # Use a MagicMock stand-in for the supervisor — registry doesn't
        # introspect the object, just keys by call_sid.
        from unittest.mock import MagicMock

        supervisor = MagicMock()
        supervisor.call_sid = "CA_REG_001"

        register_call(supervisor)
        assert get_call("CA_REG_001") is supervisor
        assert get_call("CA_UNKNOWN") is None

    def test_unregister_removes_entry(self) -> None:
        """unregister_call removes the entry; idempotent on missing."""
        from unittest.mock import MagicMock

        supervisor = MagicMock()
        supervisor.call_sid = "CA_REG_002"

        register_call(supervisor)
        unregister_call("CA_REG_002")
        assert get_call("CA_REG_002") is None

        # Idempotent — no error on missing.
        unregister_call("CA_REG_002")

    def test_register_overwrites_existing_for_same_call_sid(self) -> None:
        """A second register_call for the same CallSid overwrites (last wins).

        This shouldn't happen in production (CallSids are unique), but the
        registry's dict-semantics is documented behavior.
        """
        from unittest.mock import MagicMock

        first = MagicMock(name="first")
        first.call_sid = "CA_REG_003"
        second = MagicMock(name="second")
        second.call_sid = "CA_REG_003"

        register_call(first)
        register_call(second)
        assert get_call("CA_REG_003") is second


# ─── Test: CallSupervisor — single-shot latch ──────────────────────────────────


class TestCallSupervisorLatch:
    """
    The one-shot latch guarantees the teardown sequence runs exactly once,
    regardless of how many triggers fire on the same call.
    """

    @pytest.mark.asyncio
    async def test_single_trigger_runs_teardown_once(
        self,
        pool,
    ) -> None:
        """A single trigger_teardown runs the sequence once and unregisters."""
        manager = ConcurrencyManager(capacity=2)
        manager.acquire()
        assert manager.active == 1

        lead_id, attempt_id = await _insert_processing_lead_with_live_attempt(
            pool, call_sid="CA_LATCH_001",
        )
        supervisor = CallSupervisor(
            lead_id=lead_id,
            attempt_id=attempt_id,
            call_sid="CA_LATCH_001",
            language="English",
            concurrency_manager=manager,
            pool=pool,
        )
        register_call(supervisor)
        assert supervisor.teardown_run is False

        await supervisor.trigger_teardown("Failed", "no_answer")

        # Slot released, supervisor marked, registry cleared.
        assert manager.active == 0
        assert supervisor.teardown_run is True
        assert get_call("CA_LATCH_001") is None

    @pytest.mark.asyncio
    async def test_concurrent_triggers_only_run_teardown_once(
        self,
        pool,
    ) -> None:
        """
        REGRESSION (Issue 032): multiple concurrent triggers (e.g., status
        callback + AMD callback racing) must result in EXACTLY ONE teardown
        execution.  The slot is released exactly once, and DB writes happen
        exactly once.
        """
        manager = ConcurrencyManager(capacity=2)
        manager.acquire()
        assert manager.active == 1

        lead_id, attempt_id = await _insert_processing_lead_with_live_attempt(
            pool, call_sid="CA_RACE_001",
        )

        # Wrap DB resolve + hangup so we can count calls under concurrency.
        call_count = {"resolve": 0, "hangup": 0}

        from backend.call_supervisor import _resolve_failed_attempt

        original_resolve = _resolve_failed_attempt

        async def counting_resolve(*args, **kwargs):
            call_count["resolve"] += 1
            return await original_resolve(*args, **kwargs)

        def counting_hangup(*args, **kwargs):
            call_count["hangup"] += 1
            return True

        supervisor = CallSupervisor(
            lead_id=lead_id,
            attempt_id=attempt_id,
            call_sid="CA_RACE_001",
            language="English",
            concurrency_manager=manager,
            pool=pool,
        )
        register_call(supervisor)

        with patch(
            "backend.call_supervisor._resolve_failed_attempt",
            side_effect=counting_resolve,
        ), patch(
            "backend.call_supervisor._twilio_hangup_call",
            side_effect=counting_hangup,
        ):
            # Fire 10 concurrent triggers with mixed outcomes (simulating
            # status callback + AMD callback + extra noise).
            triggers = [
                supervisor.trigger_teardown("Failed", "no_answer"),
                supervisor.trigger_teardown("Completed", None, False, True),
                supervisor.trigger_teardown("Failed", "dropped"),
                supervisor.trigger_teardown("Completed", None, False, False),
                supervisor.trigger_teardown("Failed", "error"),
                supervisor.trigger_teardown("Completed", None, False, True),
                supervisor.trigger_teardown("Failed", "no_answer"),
                supervisor.trigger_teardown("Completed", None, False, False),
                supervisor.trigger_teardown("Failed", "dropped"),
                supervisor.trigger_teardown("Completed", None, False, True),
            ]
            await asyncio.gather(*triggers)

        # Only ONE trigger won the race; only one DB resolve + one hangup fired.
        assert call_count["resolve"] == 1, (
            f"DB resolve fired {call_count['resolve']} times — should be exactly 1"
        )
        assert call_count["hangup"] == 1, (
            f"Hangup fired {call_count['hangup']} times — should be exactly 1"
        )

        # Slot released exactly once.
        assert manager.active == 0, (
            f"Slot leaked or double-released: active={manager.active}"
        )

        # Supervisor unregistered from registry.
        assert get_call("CA_RACE_001") is None

    @pytest.mark.asyncio
    async def test_sequential_repeat_triggers_are_no_ops(
        self,
        pool,
    ) -> None:
        """Triggers after teardown completes are no-ops (not just concurrent)."""
        manager = ConcurrencyManager(capacity=2)
        manager.acquire()

        lead_id, attempt_id = await _insert_processing_lead_with_live_attempt(
            pool, call_sid="CA_SEQ_001",
        )

        call_count = {"resolve": 0}

        from backend.call_supervisor import _resolve_failed_attempt

        original_resolve = _resolve_failed_attempt

        async def counting_resolve(*args, **kwargs):
            call_count["resolve"] += 1
            return await original_resolve(*args, **kwargs)

        supervisor = CallSupervisor(
            lead_id=lead_id,
            attempt_id=attempt_id,
            call_sid="CA_SEQ_001",
            language="English",
            concurrency_manager=manager,
            pool=pool,
        )
        register_call(supervisor)

        with patch(
            "backend.call_supervisor._resolve_failed_attempt",
            side_effect=counting_resolve,
        ):
            await supervisor.trigger_teardown("Failed", "no_answer")
            # Subsequent triggers — different outcomes — must no-op.
            await supervisor.trigger_teardown("Completed", None, False, True)
            await supervisor.trigger_teardown("Failed", "dropped")
            await supervisor.trigger_teardown("Completed", None, False, False)

        assert call_count["resolve"] == 1
        assert manager.active == 0


# ─── Test: CallSupervisor — DB resolve failure doesn't block teardown ──────────


class TestCallSupervisorPersistenceFailure:
    """
    If the DB persistence step raises, the supervisor must STILL release the
    slot and unregister.  Persistence failure must not leave the supervisor
    in a stuck state.
    """

    @pytest.mark.asyncio
    async def test_db_resolve_failure_still_releases_slot(
        self,
        pool,
    ) -> None:
        """DB failure during teardown → slot still released, registry cleared."""
        manager = ConcurrencyManager(capacity=2)
        manager.acquire()

        lead_id, attempt_id = await _insert_processing_lead_with_live_attempt(
            pool, call_sid="CA_PERSIST_FAIL_001",
        )
        supervisor = CallSupervisor(
            lead_id=lead_id,
            attempt_id=attempt_id,
            call_sid="CA_PERSIST_FAIL_001",
            language="English",
            concurrency_manager=manager,
            pool=pool,
        )
        register_call(supervisor)

        # Force DB resolve to raise.
        async def _explode(*args, **kwargs):
            raise RuntimeError("simulated DB failure")

        with patch(
            "backend.call_supervisor._resolve_failed_attempt",
            side_effect=_explode,
        ):
            # Must NOT raise out of trigger_teardown.
            await supervisor.trigger_teardown("Failed", "no_answer")

        # Slot still released, registry still cleared, latch still set.
        assert manager.active == 0
        assert get_call("CA_PERSIST_FAIL_001") is None
        assert supervisor.teardown_run is True


# ─── Test: CallSupervisor — already-gone heuristic ────────────────────────────


class TestCallSupervisorHangupFlag:
    """
    The supervisor's hangup-step is gated by the explicit
    ``call_already_terminal`` flag, NOT by outcome-based heuristic inference.

    Only triggers that fire AFTER Twilio has already ended the call
    (currently the Twilio status-callback endpoint, which only ever fires
    with terminal CallStatus values) pass ``call_already_terminal=True``.
    All other triggers — AMD verdicts (voicemail / fax), the orphan-window
    handler in arrange_ai_call, future pipeline triggers (Issue 046 clean
    close) — leave it at the default ``False``, meaning "the Twilio line is
    possibly still live, hang up to be safe".

    This replaced the earlier ``outcome_status == "Completed"`` heuristic,
    which was inverted: clean closes via the consultation pipeline have
    outcome="Completed" but the Twilio line is still live at that point
    (we're the ones who must end it).
    """

    @pytest.mark.asyncio
    async def test_default_flag_attempts_hangup(
        self,
        pool,
    ) -> None:
        """Without ``call_already_terminal=True``, the supervisor must hang up
        regardless of outcome.  This pins the new default behavior."""
        manager = ConcurrencyManager(capacity=2)
        manager.acquire()

        lead_id, attempt_id = await _insert_processing_lead_with_live_attempt(
            pool, call_sid="CA_DEFAULT_HANGUP_001",
        )
        supervisor = CallSupervisor(
            lead_id=lead_id,
            attempt_id=attempt_id,
            call_sid="CA_DEFAULT_HANGUP_001",
            language="English",
            concurrency_manager=manager,
            pool=pool,
        )
        register_call(supervisor)

        with patch(
            "backend.call_supervisor._twilio_hangup_call",
            return_value=True,
        ) as mock_hangup:
            # No call_already_terminal — defaults to False, so hangup fires.
            await supervisor.trigger_teardown("Completed", completed_timeout=False)

        mock_hangup.assert_called_once()
        assert manager.active == 0
        assert get_call("CA_DEFAULT_HANGUP_001") is None

    @pytest.mark.asyncio
    async def test_call_already_terminal_skips_hangup(
        self,
        pool,
    ) -> None:
        """``call_already_terminal=True`` must skip the hangup.  This models
        the status-callback endpoint firing after Twilio has already ended
        the call (terminal CallStatus)."""
        manager = ConcurrencyManager(capacity=2)
        manager.acquire()

        lead_id, attempt_id = await _insert_processing_lead_with_live_attempt(
            pool, call_sid="CA_ALREADY_TERMINAL_001",
        )
        supervisor = CallSupervisor(
            lead_id=lead_id,
            attempt_id=attempt_id,
            call_sid="CA_ALREADY_TERMINAL_001",
            language="English",
            concurrency_manager=manager,
            pool=pool,
        )
        register_call(supervisor)

        with patch(
            "backend.call_supervisor._twilio_hangup_call",
            return_value=True,
        ) as mock_hangup:
            await supervisor.trigger_teardown(
                "Completed",
                completed_timeout=False,
                call_already_terminal=True,
            )

        # Hangup was skipped — Twilio already ended the call.
        mock_hangup.assert_not_called()
        # But DB write + slot release still happened.
        assert manager.active == 0
        assert get_call("CA_ALREADY_TERMINAL_001") is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "failure_reason",
        ["no_answer", "dropped", "error"],
    )
    async def test_failed_outcome_default_attempts_hangup(
        self,
        pool,
        failure_reason: str,
    ) -> None:
        """All Failed outcomes with the default flag must attempt the hangup.
        Models the AMD-callback and orphan-window trigger paths."""
        manager = ConcurrencyManager(capacity=2)
        manager.acquire()

        lead_id, attempt_id = await _insert_processing_lead_with_live_attempt(
            pool, call_sid=f"CA_DO_HANGUP_{failure_reason.upper()}",
        )
        supervisor = CallSupervisor(
            lead_id=lead_id,
            attempt_id=attempt_id,
            call_sid=f"CA_DO_HANGUP_{failure_reason.upper()}",
            language="English",
            concurrency_manager=manager,
            pool=pool,
        )
        register_call(supervisor)

        with patch(
            "backend.call_supervisor._twilio_hangup_call",
            return_value=True,
        ) as mock_hangup:
            # Default flag (False) — hangup must fire.
            await supervisor.trigger_teardown("Failed", failure_reason)

        mock_hangup.assert_called_once()
        assert manager.active == 0


# ─── Test: CallSupervisor — fixed teardown sequence ────────────────────────────


class TestCallSupervisorSequence:
    """
    The teardown sequence is fixed by Issue 032:
        1. Hangup (when not already-gone)
        ...
        4. DB persist
        ...
        6. Release slot
        7. Unregister
    """

    @pytest.mark.asyncio
    async def test_full_sequence_hangup_resolve_release_unregister(
        self,
        pool,
    ) -> None:
        """Verify the Issue 032 fixed sequence fires in order."""
        manager = ConcurrencyManager(capacity=2)
        manager.acquire()

        lead_id, attempt_id = await _insert_processing_lead_with_live_attempt(
            pool, call_sid="CA_SEQ_FULL_001",
        )

        events: list[str] = []

        from backend.call_supervisor import _resolve_failed_attempt

        original_resolve = _resolve_failed_attempt

        async def _record_resolve(*args, **kwargs):
            events.append("resolve")
            return await original_resolve(*args, **kwargs)

        def _record_hangup(*args, **kwargs):
            events.append("hangup")
            return True

        supervisor = CallSupervisor(
            lead_id=lead_id,
            attempt_id=attempt_id,
            call_sid="CA_SEQ_FULL_001",
            language="English",
            concurrency_manager=manager,
            pool=pool,
        )
        register_call(supervisor)

        # Wrap the slot release + unregister to capture their order.
        original_release = manager.release

        def _record_release():
            events.append("release")
            original_release()

        manager.release = _record_release  # type: ignore[assignment]

        with patch(
            "backend.call_supervisor._resolve_failed_attempt",
            side_effect=_record_resolve,
        ), patch(
            "backend.call_supervisor._twilio_hangup_call",
            side_effect=_record_hangup,
        ):
            await supervisor.trigger_teardown("Failed", "no_answer")

        # Note: "unregister" itself isn't in events (the registry pop is
        # silent), but the events that ARE recorded must follow the
        # specified sequence.
        assert events == ["hangup", "resolve", "release"], (
            f"Sequence violated. Got: {events}. "
            f"Expected: hangup (Step 1), resolve (Step 4), release (Step 6)."
        )

        # And unregister happened (registry is clear).
        assert get_call("CA_SEQ_FULL_001") is None


# ─── Test: CallSupervisor — slot double-release guard ──────────────────────────


class TestCallSupervisorSlotReleaseGuard:
    """The slot-release call has a RuntimeError guard for stale states."""

    @pytest.mark.asyncio
    async def test_slot_release_raises_runtime_error_is_swallowed(
        self,
        pool,
    ) -> None:
        """If manager.release raises RuntimeError (e.g., counter already 0),
        the supervisor logs a warning but does not crash."""
        manager = ConcurrencyManager(capacity=2)
        manager.acquire()

        lead_id, attempt_id = await _insert_processing_lead_with_live_attempt(
            pool, call_sid="CA_RELEASE_GUARD_001",
        )
        supervisor = CallSupervisor(
            lead_id=lead_id,
            attempt_id=attempt_id,
            call_sid="CA_RELEASE_GUARD_001",
            language="English",
            concurrency_manager=manager,
            pool=pool,
        )
        register_call(supervisor)

        # Force release to raise.
        def _explode_release():
            raise RuntimeError("forced double-release")

        manager.release = _explode_release  # type: ignore[assignment]

        # Must not raise.
        await supervisor.trigger_teardown("Failed", "no_answer")

        # Latch still flipped, registry still cleaned up.
        assert supervisor.teardown_run is True
        assert get_call("CA_RELEASE_GUARD_001") is None


# ─── Test: graceful STT shutdown — CloseStream self-close wins (Issue 035) ────


class TestCallSupervisorSttGracefulShutdown:
    """
    REGRESSION (Issue 035 follow-up): ``trigger_teardown`` must give the
    STT reader a bounded chance to self-close via Deepgram's
    ``CloseStream`` handshake (sending ``{"type": "CloseStream"}`` and
    draining the final transcript) BEFORE falling back to ``cancel()``.

    The previous design called ``cancel()`` immediately after setting
    ``shutdown_event``, which interrupted the reader's ``await ws.recv()``
    with ``CancelledError`` and bypassed the top-of-loop
    ``shutdown_event.is_set()`` check entirely — silently dropping the
    final transcript from every clean teardown.
    """

    @pytest.mark.asyncio
    async def test_clean_self_close_does_not_cancel_stt_task(
        self,
        pool,
    ) -> None:
        """
        When the STT task self-closes promptly after observing
        ``shutdown_event``, the supervisor must NOT cancel it.  We assert
        this by:
          1. Starting a real ``asyncio.Task`` that watches the supervisor's
             ``shutdown_event`` and exits cleanly the moment it is set.
          2. Running ``trigger_teardown``.
          3. Verifying the task was NOT cancelled (``task.cancelled() is
             False``) and DID complete normally.

        The real task, not a mock, exercises the actual ``asyncio.wait_for``
        branch and its cancellation-vs-clean-exit outcome.
        """
        manager = ConcurrencyManager(capacity=2)
        manager.acquire()

        lead_id, attempt_id = await _insert_processing_lead_with_live_attempt(
            pool, call_sid="CA_STT_GRACEFUL_001",
        )

        supervisor = CallSupervisor(
            lead_id=lead_id,
            attempt_id=attempt_id,
            call_sid="CA_STT_GRACEFUL_001",
            language="English",
            concurrency_manager=manager,
            pool=pool,
        )
        register_call(supervisor)

        # Build a real asyncio.Task that mirrors the deepgram reader's
        # self-close behaviour: it polls shutdown_event between cycles and
        # returns as soon as the event is set.  This is exactly the code
        # path in ``run_deepgram_reader`` that the issue is about to test.
        self_closed_cleanly = {"value": False}

        async def _stt_self_close():
            try:
                # Spin until shutdown is signalled — mimics the reader
                # blocked on ``ws.recv()`` with the top-of-loop event check.
                while not supervisor.shutdown_event.is_set():
                    await asyncio.sleep(0.01)
                # Self-close: send CloseStream, drain, return.
                self_closed_cleanly["value"] = True
            except asyncio.CancelledError:
                # If we get cancelled, that's the OLD buggy path.
                self_closed_cleanly["value"] = False
                raise

        supervisor.stt_task = asyncio.create_task(
            _stt_self_close(),
            name=f"fake-stt-{supervisor.call_sid}",
        )
        # Capture the task reference BEFORE teardown — Issue 041 clears
        # ``supervisor.stt_task`` to None in teardown (consistent with
        # the other 4 task handles), so the assertion must operate on
        # the captured reference.
        stt_task_ref = supervisor.stt_task
        # Let the task start.
        await asyncio.sleep(0)

        with patch(
            "backend.call_supervisor._resolve_failed_attempt",
            new=AsyncMock(),
        ), patch(
            "backend.call_supervisor._twilio_hangup_call",
            return_value=True,
        ):
            await supervisor.trigger_teardown("Failed", "no_answer")

        # The task must have self-closed — NOT been cancelled.
        assert self_closed_cleanly["value"] is True, (
            "STT task must self-close on shutdown_event; the supervisor "
            "must NOT cancel it before the grace window expires."
        )
        assert stt_task_ref.cancelled() is False, (
            "STT task was force-cancelled even though it self-closed "
            "within the grace window.  cancel() must be a backstop, "
            "not the primary path."
        )
        assert stt_task_ref.done() is True
        # Issue 041: stt_task is now cleared by teardown (handle release).
        assert supervisor.stt_task is None

    @pytest.mark.asyncio
    async def test_unresponsive_stt_task_falls_back_to_cancel(
        self,
        pool,
    ) -> None:
        """
        When the STT task does NOT self-close within the grace window
        (e.g. it's stuck on a network read), the supervisor must
        ``cancel()`` it as a backstop so the rest of teardown can run
        (DB resolve, slot release, unregister).

        Verifies the timeout backstop: we use a task that ignores
        ``shutdown_event`` and sleeps forever.
        """
        manager = ConcurrencyManager(capacity=2)
        manager.acquire()

        lead_id, attempt_id = await _insert_processing_lead_with_live_attempt(
            pool, call_sid="CA_STT_TIMEOUT_001",
        )

        supervisor = CallSupervisor(
            lead_id=lead_id,
            attempt_id=attempt_id,
            call_sid="CA_STT_TIMEOUT_001",
            language="English",
            concurrency_manager=manager,
            pool=pool,
        )
        register_call(supervisor)

        async def _stt_ignore_shutdown():
            # Deliberately never checks shutdown_event — mirrors a hung
            # reader blocked on a network read.  This task can only be
            # stopped by cancel().
            await asyncio.sleep(60)

        supervisor.stt_task = asyncio.create_task(
            _stt_ignore_shutdown(),
            name=f"hung-stt-{supervisor.call_sid}",
        )
        # Capture reference before teardown — Issue 041 clears
        # supervisor.stt_task to None.
        stt_task_ref = supervisor.stt_task
        await asyncio.sleep(0)

        # Short-circuit the grace window so the test doesn't take 1.5s.
        from backend import call_supervisor as cs_mod

        original = cs_mod._STT_GRACEFUL_SHUTDOWN_TIMEOUT_S
        cs_mod._STT_GRACEFUL_SHUTDOWN_TIMEOUT_S = 0.05
        try:
            with patch(
                "backend.call_supervisor._resolve_failed_attempt",
                new=AsyncMock(),
            ), patch(
                "backend.call_supervisor._twilio_hangup_call",
                return_value=True,
            ):
                await supervisor.trigger_teardown("Failed", "no_answer")
        finally:
            cs_mod._STT_GRACEFUL_SHUTDOWN_TIMEOUT_S = original

        # The task must have been cancelled (forced) by the backstop.
        assert stt_task_ref.cancelled() is True, (
            "Hung STT task must be force-cancelled after the grace window."
        )
        # And the rest of teardown must have completed.
        assert supervisor.teardown_run is True
        assert manager.active == 0
        assert get_call("CA_STT_TIMEOUT_001") is None
        # Issue 041: stt_task handle cleared by teardown.
        assert supervisor.stt_task is None


# ─── Test: CallSupervisor — language plumbing (Issue 035 follow-up) ──────────


class TestCallSupervisorLanguage:
    """
    REGRESSION (Issue 035 follow-up): the supervisor owns the lead's
    preferred language so the Deepgram Flux reader can pick the right
    model and language hints at connect time.  A wrong value here would
    silently misroute every call.
    """

    def test_stores_language_when_english(self) -> None:
        """English is stored verbatim on the supervisor instance."""
        supervisor = CallSupervisor(
            lead_id=1,
            attempt_id=1,
            call_sid="CA_LANG_EN_001",
            language="English",
            concurrency_manager=ConcurrencyManager(),
            pool=None,
        )
        assert supervisor.language == "English"

    def test_stores_language_when_hinglish(self) -> None:
        """Hinglish is stored verbatim on the supervisor instance."""
        supervisor = CallSupervisor(
            lead_id=1,
            attempt_id=1,
            call_sid="CA_LANG_HI_001",
            language="Hinglish",
            concurrency_manager=ConcurrencyManager(),
            pool=None,
        )
        assert supervisor.language == "Hinglish"

    def test_rejects_invalid_language(self) -> None:
        """
        Anything other than 'English' or 'Hinglish' must raise ValueError
        at construction time.  A silent fallback (e.g. defaulting to
        multilingual) would hide caller bugs and corrupt transcription.
        The error must name the bad value AND the valid choices so a
        developer reading a traceback can fix it without reading code.
        """
        with pytest.raises(ValueError) as exc_info:
            CallSupervisor(
                lead_id=1,
                attempt_id=1,
                call_sid="CA_LANG_BAD_001",
                language="French",
                concurrency_manager=ConcurrencyManager(),
                pool=None,
            )
        msg = str(exc_info.value)
        assert "French" in msg
        assert "English" in msg
        assert "Hinglish" in msg

    def test_rejects_empty_string_language(self) -> None:
        """Empty string is not a valid language — fail loudly."""
        with pytest.raises(ValueError):
            CallSupervisor(
                lead_id=1,
                attempt_id=1,
                call_sid="CA_LANG_EMPTY_001",
                language="",
                concurrency_manager=ConcurrencyManager(),
                pool=None,
            )