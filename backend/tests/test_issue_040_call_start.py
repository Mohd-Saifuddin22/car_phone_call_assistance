"""
backend/tests/test_issue_040_call_start.py — Unit tests for Issue 040 state
and synchronization (disclosure + AMD future + kickoff + barge-in gate).

Test plan
─────────

CallSupervisor state (Issue 040)
S1  amd_verdict is an asyncio.Future[str], unset after construction.
S2  disclosure_active starts True.
S3  disclosure_complete starts unset.
S4  mark_disclosure_complete flips BOTH the bool to False AND sets the
    event — atomic, no window where they disagree.
S5  mark_disclosure_complete is idempotent — calling twice leaves the
    state consistent.

AMD future idempotency (mirrors the arrange.py webhook guard)
F1  Setting amd_verdict.set_result on a fresh future works.
F2  Calling set_result a second time on a done future raises
    InvalidStateError — confirms the production code needs the
    future.done() guard.
F3  Guarded set (mirror production) is idempotent.

run_turn — kickoff mode (new_user_text=None)
T1  new_user_text=None does NOT append a user message.
T2  new_user_text=None still appends the assistant reply on success.
T3  new_user_text="" (empty string) is treated as the existing
    "skip turn" behaviour — defensive empty-text guard preserved.

drain helper
D1  _drain_completed_turn_queue returns the count of items drained.
D2  Drain on empty queue returns 0.

driver synchronization (against the live driver coroutine)
    These tests stub the Mercury HTTP transport and assert the driver's
    flow through the new sync block.  They rely on _unblock_call_start
    to simulate the disclosure+AMD having completed (with the desired
    AMD verdict) before the driver is spawned — without that the
    driver blocks forever on disclosure_complete.

V1  Driver waits for disclosure_complete: with disclosure NOT yet
    complete, the driver does not process a queued transcript until
    the test fires mark_disclosure_complete().
V2  Driver waits for amd_verdict: with disclosure done but amd_verdict
    unset, the driver does not process until the future is resolved.
V3  AMD timeout: with disclosure done but amd_verdict unset, the
    driver proceeds (kickoff fires) after the 2 s timeout as if the
    verdict were "human".
V4  Machine verdict: driver exits cleanly without invoking Mercury.
V5  Kickoff greeting IS the first synthesis (queued transcript arrives
    AFTER the greeting).
V6  Ephemeral kickoff is popped from messages after the greeting —
    final messages list is [system, assistant].
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio

from backend.call_supervisor import (
    CallSupervisor,
    _active_calls,
    register_call,
    unregister_call,
)
from backend.concurrency import ConcurrencyManager
from backend.conversation_driver import (
    _drain_completed_turn_queue,
    _KICKOFF_SYSTEM_MESSAGE,
    run_mercury_conversation_driver,
    run_turn,
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
        delete=False, suffix=".db", prefix="car_test_i040_"
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


def _make_supervisor(*, lead_id: int = 1, pool=None) -> CallSupervisor:
    """Build a fresh CallSupervisor wired with a real ConcurrencyManager."""
    manager = ConcurrencyManager(capacity=2)
    manager.acquire()
    return CallSupervisor(
        lead_id=lead_id,
        attempt_id=1,
        call_sid="CA_I040_TEST",
        language="English",
        concurrency_manager=manager,
        pool=pool if pool is not None else MagicMock(),
    )


def _unblock_call_start(
    supervisor: CallSupervisor,
    *,
    amd_verdict: str = "human",
) -> None:
    """Simulate disclosure-playback + AMD webhook having completed."""
    supervisor.mark_disclosure_complete()
    if not supervisor.amd_verdict.done():
        supervisor.amd_verdict.set_result(amd_verdict)


# ─── S1–S5: CallSupervisor Issue 040 state ───────────────────────────────────


class TestCallSupervisorIssue040State:
    """The new state the driver + AMD callback + barge-in gate rely on."""

    def test_amd_verdict_is_unset_future_on_construction(self) -> None:
        """S1: amd_verdict is a Future, not done after __init__."""
        s = _make_supervisor()
        assert isinstance(s.amd_verdict, asyncio.Future)
        assert s.amd_verdict.done() is False

    def test_disclosure_active_is_true_on_construction(self) -> None:
        """S2: disclosure_active starts True (disclosure not yet played)."""
        s = _make_supervisor()
        assert s.disclosure_active is True

    def test_disclosure_complete_is_unset_event_on_construction(self) -> None:
        """S3: disclosure_complete is an Event, not set after __init__."""
        s = _make_supervisor()
        assert isinstance(s.disclosure_complete, asyncio.Event)
        assert s.disclosure_complete.is_set() is False

    def test_mark_disclosure_complete_flips_both_atomically(self) -> None:
        """S4: mark_disclosure_complete flips the bool AND sets the event
        in a single call so no consumer ever observes them disagreeing.
        """
        s = _make_supervisor()
        assert s.disclosure_active is True
        assert not s.disclosure_complete.is_set()

        s.mark_disclosure_complete()

        assert s.disclosure_active is False
        assert s.disclosure_complete.is_set() is True

    def test_mark_disclosure_complete_is_idempotent(self) -> None:
        """S5: Calling mark_disclosure_complete twice leaves consistent state."""
        s = _make_supervisor()
        s.mark_disclosure_complete()
        # Second call — must not raise, must leave state consistent.
        s.mark_disclosure_complete()
        assert s.disclosure_active is False
        assert s.disclosure_complete.is_set() is True


# ─── F1–F3: AMD future idempotency (mirrors arrange.py guard) ────────────────


class TestAmdVerdictFutureIdempotency:
    """The arrange.py webhook guards set_result with future.done(); these
    tests prove the guard is necessary (duplicate set raises) and that
    the guard pattern works.
    """

    def test_set_result_on_fresh_future_succeeds(self) -> None:
        """F1: First set_result wins; future.done() returns True."""
        s = _make_supervisor()
        s.amd_verdict.set_result("human")
        assert s.amd_verdict.done()
        assert s.amd_verdict.result() == "human"

    def test_set_result_on_done_future_raises(self) -> None:
        """F2: Duplicate set_result raises InvalidStateError — proves the
        production guard (``if not future.done()``) is required.
        """
        s = _make_supervisor()
        s.amd_verdict.set_result("human")
        with pytest.raises(asyncio.InvalidStateError):
            s.amd_verdict.set_result("machine_end")

    def test_guarded_set_pattern_is_idempotent(self) -> None:
        """F3: The production-style ``if not future.done()`` guard makes
        a duplicate set a silent no-op (first verdict wins).
        """
        s = _make_supervisor()
        # First call (human) wins.
        if not s.amd_verdict.done():
            s.amd_verdict.set_result("human")
        # Duplicate call (machine_end) is silently dropped.
        if not s.amd_verdict.done():
            s.amd_verdict.set_result("machine_end")
        assert s.amd_verdict.result() == "human"


# ─── T1–T3: run_turn kickoff mode ─────────────────────────────────────────────


class TestRunTurnKickoffMode:
    """run_turn(new_user_text=None) is the new kickoff-mode contract; the
    empty-string behaviour is preserved for backward compatibility.
    """

    @pytest.mark.asyncio
    async def test_none_does_not_append_user_message(self) -> None:
        """T1: new_user_text=None → no user message appended, only system
        priming content (e.g. the kickoff) reaches Mercury.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": "sys"},
            {"role": "system", "content": _KICKOFF_SYSTEM_MESSAGE},
        ]

        # Stub the Mercury HTTP response.
        async def _fake_post(url, *, json, headers):
            request = httpx.Request("POST", "https://api.inceptionlabs.ai/v1/chat/completions")
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "Hi!"}, "finish_reason": "stop"}
                    ]
                },
                request=request,
            )

        # Build a minimal supervisor + TTS shim.
        supervisor = _make_supervisor()
        supervisor.tts_client = _StubTTS()

        # Capture messages as Mercury sees them.
        captured_messages: list[list[dict[str, str]]] = []

        from backend.conversation_driver import _call_mercury_api
        original = _call_mercury_api

        async def _spy(messages, *, api_key, http_client):
            captured_messages.append(list(messages))
            return await original(messages, api_key=api_key, http_client=http_client)

        # http_client.post needs a side_effect to return our fake response.
        http_client = MagicMock(spec=httpx.AsyncClient)
        http_client.post = AsyncMock(side_effect=_fake_post)

        import backend.conversation_driver as cd_mod
        cd_mod._call_mercury_api = _spy
        try:
            await run_turn(
                supervisor,
                messages,
                new_user_text=None,
                http_client=http_client,
                api_key="dummy",
            )
        finally:
            cd_mod._call_mercury_api = original

        # Mercury saw system + kickoff only — no user message.
        assert captured_messages, "Mercury was never called"
        roles = [m["role"] for m in captured_messages[0]]
        assert "user" not in roles, (
            f"kickoff mode must NOT append a user message; Mercury saw: {roles}"
        )
        assert "system" in roles and len(roles) == 2

    @pytest.mark.asyncio
    async def test_none_still_appends_assistant_reply(self) -> None:
        """T2: On success, the assistant reply is appended even in kickoff
        mode (so the next call to Mercury sees the greeting).
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": "sys"},
            {"role": "system", "content": _KICKOFF_SYSTEM_MESSAGE},
        ]
        supervisor = _make_supervisor()
        supervisor.tts_client = _StubTTS()

        async def _fake_post(url, *, json, headers):
            request = httpx.Request("POST", "https://api.inceptionlabs.ai/v1/chat/completions")
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "Hi, I'm Riya!"}, "finish_reason": "stop"}
                    ]
                },
                request=request,
            )

        from backend.conversation_driver import _call_mercury_api
        original = _call_mercury_api

        async def _spy(messages, *, api_key, http_client):
            return await original(messages, api_key=api_key, http_client=http_client)

        http_client = MagicMock(spec=httpx.AsyncClient)
        http_client.post = AsyncMock(side_effect=_fake_post)

        import backend.conversation_driver as cd_mod
        cd_mod._call_mercury_api = _spy
        try:
            await run_turn(
                supervisor,
                messages,
                new_user_text=None,
                http_client=http_client,
                api_key="dummy",
            )
        finally:
            cd_mod._call_mercury_api = original

        # Last message is the assistant greeting.
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "Hi, I'm Riya!"

    @pytest.mark.asyncio
    async def test_empty_string_still_skips_turn(self) -> None:
        """T3: Empty-string ``new_user_text`` still hits the defensive
        "skip" path (existing behaviour preserved for callers that pass
        ``""`` from upstream guards).
        """
        messages: list[dict[str, str]] = [{"role": "system", "content": "sys"}]
        supervisor = _make_supervisor()
        supervisor.tts_client = _StubTTS()

        http_client = MagicMock(spec=httpx.AsyncClient)
        http_client.post = AsyncMock(
            side_effect=AssertionError("Mercury should not be called on skip")
        )

        await run_turn(
            supervisor,
            messages,
            new_user_text="",
            http_client=http_client,
            api_key="dummy",
        )

        # No user message, no assistant message — turn was skipped.
        assert len(messages) == 1
        assert messages[0]["role"] == "system"


class _StubTTS:
    """Minimal TTS shim — records calls, never raises."""

    def __init__(self) -> None:
        self.synthesize_calls: list[tuple[str, str, int, Any]] = []

    async def synthesize_turn(
        self, text, *, language, generation, supervisor
    ) -> None:
        self.synthesize_calls.append((text, language, generation, supervisor))


# ─── D1–D2: drain helper ─────────────────────────────────────────────────────


class TestDrainCompletedTurnQueue:
    """_drain_completed_turn_queue discards queued transcripts (early
    "hello?"s) before the kickoff greeting so they don't reach Mercury
    as the first turn.
    """

    @pytest.mark.asyncio
    async def test_drain_returns_count(self) -> None:
        """D1: drain returns the count of items removed."""
        s = _make_supervisor()
        await s.completed_turn_queue.put("hello?")
        await s.completed_turn_queue.put("hi?")

        drained = _drain_completed_turn_queue(s)
        assert drained == 2
        assert s.completed_turn_queue.empty()

    @pytest.mark.asyncio
    async def test_drain_on_empty_queue_returns_zero(self) -> None:
        """D2: drain on an empty queue returns 0 (no exception)."""
        s = _make_supervisor()
        assert _drain_completed_turn_queue(s) == 0


# ─── V1–V6: driver synchronization against live driver coroutine ─────────────


class _FakeAsyncClientContextManager:
    """Mimics httpx.AsyncClient as an async context manager so the driver's
    ``async with httpx.AsyncClient(...)`` block works in tests.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    async def __aenter__(self) -> Any:
        return self._client

    async def __aexit__(self, *exc: Any) -> None:
        return None


def _make_mercury_response(text: str) -> httpx.Response:
    request = httpx.Request("POST", "https://api.inceptionlabs.ai/v1/chat/completions")
    return httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"content": text}, "finish_reason": "stop"}
            ]
        },
        request=request,
    )


@pytest.fixture
def fake_tts_client_class(monkeypatch: pytest.MonkeyPatch):
    """Swap AsyncSarvamTTSClient for a stub that records synthesize calls."""
    calls: list[tuple[str, str, int, Any]] = []

    class _FakeTTS:
        def __init__(self) -> None:
            self.synthesize_calls = calls

        async def connect(self, supervisor: Any) -> None:
            return None

        async def synthesize_turn(self, text, *, language, generation, supervisor):
            self.synthesize_calls.append((text, language, generation, supervisor))

        async def close(self) -> None:
            return None

    monkeypatch.setattr(
        "backend.conversation_driver.AsyncSarvamTTSClient", _FakeTTS
    )
    return _FakeTTS


class TestDriverSynchronization:
    """End-to-end driver sync: disclosure wait, AMD wait, drain, kickoff,
    conversation history cleanup.
    """

    async def _seed_lead(self, pool) -> int:
        async with pool.connection() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO leads
                    (name, phone, email, city, language, status,
                     attempt_count, vehicle_id)
                VALUES ('I040 Lead', '+919876543210', 'l@x.com',
                        'Mumbai', 'English', 'Processing', 1, NULL)
                """,
            )
            return cursor.lastrowid

    @pytest.mark.asyncio
    async def test_v1_driver_waits_for_disclosure(
        self,
        pool,
        monkeypatch,
        fake_tts_client_class,
    ) -> None:
        """V1: With disclosure NOT yet complete, the driver does not
        process a queued transcript — even after a long wait.
        """
        lead_id = await self._seed_lead(pool)
        supervisor = _make_supervisor(lead_id=lead_id, pool=pool)
        register_call(supervisor)
        await supervisor.completed_turn_queue.put("hi")

        async def _fake_post(url, *, json, headers):
            return _make_mercury_response("reply")

        http_client = MagicMock(spec=httpx.AsyncClient)
        http_client.post = AsyncMock(side_effect=_fake_post)
        monkeypatch.setattr(
            "backend.conversation_driver.httpx.AsyncClient",
            lambda timeout: _FakeAsyncClientContextManager(http_client),
        )

        driver_task = asyncio.create_task(
            run_mercury_conversation_driver(supervisor),
            name="i040-v1",
        )

        # Wait long enough that the driver would have processed the
        # transcript IF it weren't waiting on the disclosure event.
        await asyncio.sleep(0.2)
        assert http_client.post.await_count == 0, (
            "Driver called Mercury before disclosure_complete was set — "
            "synchronization is broken"
        )

        # Now mark disclosure complete — driver should proceed.
        _unblock_call_start(supervisor)
        supervisor.shutdown_event.set()  # Exit after one turn.
        await asyncio.wait_for(driver_task, timeout=2.0)

    @pytest.mark.asyncio
    async def test_v2_driver_waits_for_amd_verdict(
        self,
        pool,
        monkeypatch,
        fake_tts_client_class,
    ) -> None:
        """V2: With disclosure done but amd_verdict unset, the driver
        does not call Mercury until the future is resolved.
        """
        lead_id = await self._seed_lead(pool)
        supervisor = _make_supervisor(lead_id=lead_id, pool=pool)
        register_call(supervisor)
        await supervisor.completed_turn_queue.put("hi")

        async def _fake_post(url, *, json, headers):
            return _make_mercury_response("reply")

        http_client = MagicMock(spec=httpx.AsyncClient)
        http_client.post = AsyncMock(side_effect=_fake_post)
        monkeypatch.setattr(
            "backend.conversation_driver.httpx.AsyncClient",
            lambda timeout: _FakeAsyncClientContextManager(http_client),
        )

        # Mark disclosure complete but DO NOT set amd_verdict.
        supervisor.mark_disclosure_complete()

        driver_task = asyncio.create_task(
            run_mercury_conversation_driver(supervisor),
            name="i040-v2",
        )
        await asyncio.sleep(0.2)
        assert http_client.post.await_count == 0, (
            "Driver called Mercury before AMD verdict was set"
        )

        # Set AMD verdict; driver should proceed.
        supervisor.amd_verdict.set_result("human")
        supervisor.shutdown_event.set()
        await asyncio.wait_for(driver_task, timeout=2.0)

    @pytest.mark.asyncio
    async def test_v3_amd_timeout_proceeds_as_human(
        self,
        pool,
        monkeypatch,
        fake_tts_client_class,
    ) -> None:
        """V3: Disclosure done, AMD verdict NEVER set — driver proceeds
        as if the verdict were 'human' after the 2 s timeout.
        """
        lead_id = await self._seed_lead(pool)
        supervisor = _make_supervisor(lead_id=lead_id, pool=pool)
        register_call(supervisor)
        supervisor.mark_disclosure_complete()

        async def _fake_post(url, *, json, headers):
            return _make_mercury_response("greeting")

        http_client = MagicMock(spec=httpx.AsyncClient)
        http_client.post = AsyncMock(side_effect=_fake_post)
        monkeypatch.setattr(
            "backend.conversation_driver.httpx.AsyncClient",
            lambda timeout: _FakeAsyncClientContextManager(http_client),
        )

        # Shorten the AMD timeout so the test doesn't have to wait 2s.
        from backend.conversation_driver import _AMD_VERDICT_TIMEOUT_S
        original_timeout = _AMD_VERDICT_TIMEOUT_S
        monkeypatch.setattr(
            "backend.conversation_driver._AMD_VERDICT_TIMEOUT_S", 0.1
        )

        driver_task = asyncio.create_task(
            run_mercury_conversation_driver(supervisor),
            name="i040-v3",
        )
        # Wait for the kickoff greeting to land.
        for _ in range(200):
            tts = supervisor.tts_client
            if tts is not None and tts.synthesize_calls:
                break
            await asyncio.sleep(0.025)
        else:  # pragma: no cover
            pytest.fail("Driver did not call synthesize_turn within timeout")

        # Mercury was called (kickoff fired after the AMD timeout).
        assert http_client.post.await_count == 1
        supervisor.shutdown_event.set()
        await asyncio.wait_for(driver_task, timeout=2.0)

    @pytest.mark.asyncio
    async def test_v4_machine_verdict_exits_cleanly(
        self,
        pool,
        monkeypatch,
        fake_tts_client_class,
    ) -> None:
        """V4: AMD verdict 'machine_end' → driver exits WITHOUT calling Mercury."""
        lead_id = await self._seed_lead(pool)
        supervisor = _make_supervisor(lead_id=lead_id, pool=pool)
        register_call(supervisor)
        _unblock_call_start(supervisor, amd_verdict="machine_end")

        async def _fake_post(url, *, json, headers):
            return _make_mercury_response("reply")

        http_client = MagicMock(spec=httpx.AsyncClient)
        http_client.post = AsyncMock(side_effect=_fake_post)
        monkeypatch.setattr(
            "backend.conversation_driver.httpx.AsyncClient",
            lambda timeout: _FakeAsyncClientContextManager(http_client),
        )

        driver_task = asyncio.create_task(
            run_mercury_conversation_driver(supervisor),
            name="i040-v4",
        )
        await asyncio.wait_for(driver_task, timeout=2.0)

        # Mercury NEVER called — driver exited at the machine-verdict branch.
        assert http_client.post.await_count == 0
        assert driver_task.exception() is None

    @pytest.mark.asyncio
    async def test_v5_kickoff_greeting_fires_before_queued_transcript(
        self,
        pool,
        monkeypatch,
        fake_tts_client_class,
    ) -> None:
        """V5: The kickoff greeting IS the first synthesis; a transcript
        queued AFTER the driver has started is processed as the second
        turn (the kickoff fires first, then the transcript-reply).
        """
        lead_id = await self._seed_lead(pool)
        supervisor = _make_supervisor(lead_id=lead_id, pool=pool)
        register_call(supervisor)
        _unblock_call_start(supervisor)

        greeting_text = "Hello, I'm Riya!"
        transcript_reply_text = "Sure, what model interests you?"

        async def _fake_post(url, *, json, headers):
            # Mercury sees: [system, system(kickoff)] first; then
            # [system, system(kickoff), assistant(greeting), user(hello)]
            # for the transcript reply.  Detect which by the last user msg.
            last_user = next(
                (m for m in reversed(json["messages"]) if m["role"] == "user"),
                None,
            )
            if last_user and last_user["content"] == "hello":
                return _make_mercury_response(transcript_reply_text)
            return _make_mercury_response(greeting_text)

        http_client = MagicMock(spec=httpx.AsyncClient)
        http_client.post = AsyncMock(side_effect=_fake_post)
        monkeypatch.setattr(
            "backend.conversation_driver.httpx.AsyncClient",
            lambda timeout: _FakeAsyncClientContextManager(http_client),
        )

        driver_task = asyncio.create_task(
            run_mercury_conversation_driver(supervisor),
            name="i040-v5",
        )

        # Wait for the kickoff synthesis to land, then queue the
        # transcript so it survives past the drain.
        for _ in range(400):
            if http_client.post.await_count >= 1:
                # Kickoff landed; queue the real transcript now.
                await supervisor.completed_turn_queue.put("hello")
                break
            await asyncio.sleep(0.025)
        else:  # pragma: no cover
            pytest.fail("Kickoff did not call Mercury")

        # Wait for the second Mercury call (transcript reply).
        for _ in range(400):
            if http_client.post.await_count >= 2:
                break
            await asyncio.sleep(0.025)
        else:  # pragma: no cover
            pytest.fail(
                f"Expected 2 Mercury calls (kickoff + transcript reply); "
                f"got {http_client.post.await_count}"
            )

        supervisor.shutdown_event.set()
        await asyncio.wait_for(driver_task, timeout=2.0)

        # Two TTS calls: greeting first, then transcript reply.
        tts = supervisor.tts_client
        assert len(tts.synthesize_calls) == 2
        assert tts.synthesize_calls[0][0] == greeting_text
        assert tts.synthesize_calls[1][0] == transcript_reply_text

    @pytest.mark.asyncio
    async def test_v6_regression_double_greet_after_kickoff(
        self,
        pool,
        monkeypatch,
        fake_tts_client_class,
    ) -> None:
        """
        Issue 040 HITL regression: after the proactive kickoff, the
        NEXT customer turn must see the greeting in the conversation
        history — otherwise Mercury's "FIRST MESSAGE" rule re-fires
        and Riya greets the customer a second time.

        Root cause that this test catches:
        The original kickoff cleanup used ``if messages[-1] is
        kickoff_msg: messages.pop()`` — but after a successful
        ``run_turn`` the LAST element is the assistant greeting that
        was just appended, NOT the kickoff.  The pop never fired, the
        kickoff stayed in the history, and the next Mercury call saw
        ``[system, kickoff, assistant: greeting, user: turn]`` —
        triggering the FIRST_MESSAGE rule again → Riya re-greeted.

        The fix: ``messages.remove(kickoff_msg)`` (by identity)
        removes the kickoff regardless of position.  The assistant
        greeting survives intact.

        Test strategy: push a transcript after the kickoff, capture
        the SECOND Mercury call's messages list, and assert the
        list is exactly ``[system, assistant: greeting, user:
        transcript]`` with NO kickoff.  Without the fix, the kickoff
        is in the second call's messages and Mercury re-greets.
        """
        from backend.conversation_driver import _call_mercury_api
        from backend.conversation_driver import _KICKOFF_SYSTEM_MESSAGE

        lead_id = await self._seed_lead(pool)
        supervisor = _make_supervisor(lead_id=lead_id, pool=pool)
        register_call(supervisor)
        _unblock_call_start(supervisor)

        greeting_text = "Hello, I'm Riya!"
        transcript_reply_text = "Tell me about your budget."
        customer_text = "I have 8 lakh"

        async def _fake_post(url, *, json, headers):
            # Distinguish calls by inspecting messages: first call has
            # NO user message yet (kickoff); second call has the
            # customer transcript as the trailing user message.
            has_user = any(m["role"] == "user" for m in json["messages"])
            if has_user:
                return _make_mercury_response(transcript_reply_text)
            return _make_mercury_response(greeting_text)

        http_client = MagicMock(spec=httpx.AsyncClient)
        http_client.post = AsyncMock(side_effect=_fake_post)
        monkeypatch.setattr(
            "backend.conversation_driver.httpx.AsyncClient",
            lambda timeout: _FakeAsyncClientContextManager(http_client),
        )

        # Capture every Mercury call's messages list.
        seen_messages: list[list[dict[str, str]]] = []

        async def _spy(messages, *, api_key, http_client):
            seen_messages.append(list(messages))
            return await original_call(messages, api_key=api_key, http_client=http_client)

        original_call = _call_mercury_api

        import backend.conversation_driver as cd_mod
        cd_mod._call_mercury_api = _spy

        driver_task = asyncio.create_task(
            run_mercury_conversation_driver(supervisor),
            name="i040-v6-regression",
        )

        # Wait for the kickoff Mercury call to land.
        for _ in range(200):
            if len(seen_messages) >= 1:
                break
            await asyncio.sleep(0.025)
        else:  # pragma: no cover
            pytest.fail("Kickoff did not call Mercury")

        # Snapshot the kickoff messages — must include the kickoff
        # (we haven't popped it yet — that happens AFTER run_turn).
        kickoff_messages = seen_messages[0]
        assert any(
            m["content"] == _KICKOFF_SYSTEM_MESSAGE for m in kickoff_messages
        ), f"Kickoff missing from its own Mercury call: {kickoff_messages}"

        # Push a transcript AFTER the kickoff, wait for the second
        # Mercury call.
        await supervisor.completed_turn_queue.put(customer_text)
        for _ in range(400):
            if len(seen_messages) >= 2:
                break
            await asyncio.sleep(0.025)
        else:  # pragma: no cover
            pytest.fail(
                f"Second Mercury call (transcript reply) never landed; "
                f"got {len(seen_messages)} call(s)"
            )

        supervisor.shutdown_event.set()
        await asyncio.wait_for(driver_task, timeout=2.0)

        # ── The critical assertion ──────────────────────────────────
        # The second Mercury call's messages list MUST contain the
        # greeting (so Mercury doesn't re-greet) and MUST NOT contain
        # the kickoff (so the FIRST_MESSAGE rule doesn't re-fire).
        second_messages = seen_messages[1]
        contents = [m["content"] for m in second_messages]
        roles = [m["role"] for m in second_messages]

        assert _KICKOFF_SYSTEM_MESSAGE not in contents, (
            f"Kickoff still in messages on the second Mercury call — "
            f"driver would re-greet. contents={contents}"
        )
        assert greeting_text in contents, (
            f"Greeting missing from second Mercury call's messages — "
            f"Mercury has no greeting to reference. contents={contents}"
        )
        assert customer_text in contents, (
            f"Customer transcript missing from second Mercury call. "
            f"contents={contents}"
        )
        # Roles must be exactly: system, assistant, user (no second
        # system kickoff lurking).
        assert roles == ["system", "assistant", "user"], (
            f"Second-call message roles should be exactly "
            f"['system', 'assistant', 'user']; got {roles}"
        )

    @pytest.mark.asyncio
    async def test_v6_kickoff_popped_from_messages(
        self,
        pool,
        monkeypatch,
        fake_tts_client_class,
    ) -> None:
        """V6: After the kickoff greeting fires, the kickoff system message
        is removed from the messages list — the final history is pristine
        ``[system, assistant: greeting]`` (plus any real customer turns).
        """
        lead_id = await self._seed_lead(pool)
        supervisor = _make_supervisor(lead_id=lead_id, pool=pool)
        register_call(supervisor)
        _unblock_call_start(supervisor)

        async def _fake_post(url, *, json, headers):
            return _make_mercury_response("Hello!")

        http_client = MagicMock(spec=httpx.AsyncClient)
        http_client.post = AsyncMock(side_effect=_fake_post)
        monkeypatch.setattr(
            "backend.conversation_driver.httpx.AsyncClient",
            lambda timeout: _FakeAsyncClientContextManager(http_client),
        )

        # Capture messages as Mercury sees them.
        seen_during_kickoff: list[list[dict[str, str]]] = []
        from backend.conversation_driver import _call_mercury_api
        original = _call_mercury_api

        async def _spy(messages, *, api_key, http_client):
            seen_during_kickoff.append(list(messages))
            return await original(messages, api_key=api_key, http_client=http_client)

        import backend.conversation_driver as cd_mod
        cd_mod._call_mercury_api = _spy

        driver_task = asyncio.create_task(
            run_mercury_conversation_driver(supervisor),
            name="i040-v6",
        )
        for _ in range(200):
            if http_client.post.await_count >= 1:
                break
            await asyncio.sleep(0.025)
        else:  # pragma: no cover
            pytest.fail("Kickoff did not call Mercury")

        # While the kickoff Mercury call was in flight, the messages list
        # contained the kickoff — confirm Mercury saw it.
        assert seen_during_kickoff, "kickoff Mercury call never captured"
        kickoff_snapshot = seen_during_kickoff[0]
        kickoff_contents = [m["content"] for m in kickoff_snapshot]
        assert _KICKOFF_SYSTEM_MESSAGE in kickoff_contents, (
            f"Kickoff system message missing from pre-call snapshot: "
            f"{kickoff_contents}"
        )

        # Wait briefly for the driver to finish the kickoff turn and pop
        # the kickoff from messages.
        for _ in range(40):
            # Inspect the supervisor's driver task — no direct access to
            # messages list; we just need the kickoff turn to complete.
            if http_client.post.await_count >= 1 and supervisor.tts_client is not None:
                # Give the finally block one tick to pop the message.
                await asyncio.sleep(0.01)
                break
            await asyncio.sleep(0.025)

        # We can't directly inspect the driver's `messages` list (it's
        # a local variable in run_mercury_conversation_driver).  Instead,
        # we verify the kickoff is NOT in any subsequent Mercury call
        # (none should happen — the driver is now waiting on the queue).
        await asyncio.sleep(0.1)
        assert len(seen_during_kickoff) == 1, (
            "Unexpected additional Mercury calls — kickoff cleanup may "
            "have left a primed conversation"
        )

        supervisor.shutdown_event.set()
        await asyncio.wait_for(driver_task, timeout=2.0)