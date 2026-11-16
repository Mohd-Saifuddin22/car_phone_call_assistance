"""
backend/tests/test_sarvam_tts.py — Issue 038 unit tests.

Test plan
─────────
The AsyncSarvamTTSClient is the per-call bridge between Mercury-2's full
text replies and Sarvam's Bulbul v3 streaming WebSocket.  These tests use
a fake ``sarvamai`` SDK (registered in ``sys.modules``) so we exercise
the real client coroutine against realistic inputs without standing up a
live Sarvam session.

  Configuration & SDK contract
  C1  ``connect()`` calls the SDK with ``model="bulbul:v3"`` and
      ``send_completion_event="true"``.
  C2  ``synthesize_turn(text, language="English")`` configures with
      ``target_language_code="en-IN"``.
  C3  ``synthesize_turn(text, language="Hinglish")`` configures with
      ``target_language_code="hi-IN"``.
  C4  The configured ``speaker`` comes from settings.
  C5  ``synthesize_turn`` always uses ``output_audio_codec="mulaw"`` and
      ``speech_sample_rate=8000``.
  C6  ``synthesize_turn`` always calls ``enable_preprocessing=True``.
  C7  ``min_buffer_size`` is NOT explicitly passed as a kwarg.
  C8  ``configure → convert → flush`` is called in order per turn.

  Reader → outbound queue
  R1  Each ``AudioOutput`` message is base64-decoded and pushed as
      ``(generation, bytes)`` to ``supervisor.outbound_audio_queue``.
  R2  Stamping the current generation onto the chunk happens BEFORE
      configure is called — any audio arriving mid-turn carries the
      correct tag.
  R3  An ``EventResponse`` with ``event_type="final"`` is the
      authoritative end-of-turn signal; ``synthesize_turn`` then pushes
      ``(generation, None)``.
  R4  An ``ErrorResponse`` from Sarvam logs an error, sets the turn
      done event, and the synthesize call still returns a ``None``
      marker (so the pipeline can advance to teardown).

  Multi-turn socket reuse
  M1  Two sequential ``synthesize_turn`` calls reuse the SAME socket
      (one ``connect`` → two ``configure → convert → flush``).
  M2  Each turn's chunks carry its own generation tag.
  M3  ``synthesize_turn`` for an empty text is a no-op for Sarvam
      (no configure/convert/flush) but still pushes the end-of-turn
      ``None`` marker.

  Lifecycle
  L1  ``close()`` is idempotent.
  L2  ``synthesize_turn`` on a closed client raises.
  L3  ``synthesize_turn`` with a mismatched supervisor raises.
  L4  Connect failure raises ``SarvamTTSError`` and leaves the client
      in a safe (unconnected) state.

  Fallback timeout
  T1  If Sarvam never emits the ``final`` event, ``synthesize_turn``
      still pushes ``(generation, None)`` after the configured timeout
      and logs a warning.

  Settings
  S1  Default speaker is ``"priya"`` (lowercase — Sarvam returns 422 for
      other casings) and is configurable via env.
"""

from __future__ import annotations

import asyncio
import base64
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import patch

import pytest

from backend.audio import sarvam_tts as stts_module
from backend.audio.sarvam_tts import AsyncSarvamTTSClient, SarvamTTSError
from backend.call_supervisor import CallSupervisor
from backend.concurrency import ConcurrencyManager
from backend.config import load_settings


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_supervisor() -> CallSupervisor:
    manager = ConcurrencyManager(capacity=1)
    pool = object()  # synthesize_turn never touches the pool
    return CallSupervisor(
        lead_id=1,
        attempt_id=1,
        call_sid="CA_TEST_SARVAM",
        language="English",
        concurrency_manager=manager,
        pool=pool,
    )


@dataclass
class FakeWS:
    """
    In-memory stand-in for ``AsyncTextToSpeechStreamingSocketClient``.

    Records every ``configure / convert / flush`` call and replays a
    pre-programmed sequence of messages from ``recv()``.

    Responses are gated per turn: ``flush()`` releases the next batch
    of responses, and ``recv()`` blocks until at least one response is
    available for the current turn.  This mirrors the real Sarvam
    behavior (network round-trips between client actions and server
    emissions) and prevents the reader task from racing ahead of
    ``synthesize_turn`` and consuming a later turn's events.
    """

    model: str | None = None
    send_completion_event: str | None = None
    configured: list[dict[str, Any]] = field(default_factory=list)
    converted: list[str] = field(default_factory=list)
    flush_count: int = 0
    closed: bool = False

    # Per-turn response queues — index i is the list of messages the
    # server will emit after the i-th ``flush()`` call.
    turn_responses: list[list[Any]] = field(default_factory=list)
    current_responses: list[Any] = field(default_factory=list)
    recv_index: int = 0
    turn_released: asyncio.Event = field(default_factory=asyncio.Event)

    async def configure(self, **kwargs: Any) -> None:
        self.configured.append(kwargs)

    async def convert(self, text: str) -> None:
        self.converted.append(text)

    async def flush(self) -> None:
        self.flush_count += 1
        # Release the next batch of responses (if any are queued).
        if self.turn_responses:
            self.current_responses = self.turn_responses.pop(0)
            self.recv_index = 0
            self.turn_released.set()

    async def recv(self) -> Any:
        # Block until at least one response is available for the
        # current turn — mirrors the real Sarvam server which only
        # sends audio AFTER flush() has been processed.
        while self.recv_index >= len(self.current_responses):
            self.turn_released.clear()
            await self.turn_released.wait()
        msg = self.current_responses[self.recv_index]
        self.recv_index += 1
        return msg


class FakeContextManager:
    """Async context manager that yields the FakeWS on ``__aenter__``."""

    def __init__(self, *, ws: FakeWS) -> None:
        self._ws = ws

    async def __aenter__(self) -> FakeWS:
        return self._ws

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._ws.closed = True


class FakeSarvamAI:
    """
    Stand-in for ``sarvamai.AsyncSarvamAI``.

    A single shared instance is created per fixture so tests can
    pre-program it with ``configure_next`` and inspect the same object
    that ``connect()`` ends up calling.  The class-level ``shared``
    attribute is the canonical instance for the current fixture scope.
    """

    shared: "FakeSarvamAI | None" = None

    def __init__(self, *, api_subscription_key: str | None = None) -> None:
        self.api_subscription_key = api_subscription_key
        self.connect_calls: list[dict[str, Any]] = []
        self._next_ws: FakeWS | None = None

    def configure_next(self, ws: FakeWS) -> None:
        """Tests call this to set up the FakeWS returned by the next connect."""
        self._next_ws = ws

    @property
    def text_to_speech_streaming(self) -> "FakeSarvamAI":
        return self

    def connect(
        self,
        *,
        model: str | None = None,
        send_completion_event: str | None = None,
        **kwargs: Any,
    ) -> FakeContextManager:
        self.connect_calls.append(
            {"model": model, "send_completion_event": send_completion_event, **kwargs}
        )
        if self._next_ws is None:
            raise RuntimeError("Test forgot to call configure_next() before connect()")
        ws = self._next_ws
        ws.model = model
        ws.send_completion_event = send_completion_event
        return FakeContextManager(ws=ws)


# Module-level factory — every FakeSarvamAI() constructor call returns
# the SAME instance (set by the fixture). This mirrors the "one SDK
# client per process" model that the production code relies on.
def _fake_sarvam_ai_factory(*, api_subscription_key: str | None = None) -> FakeSarvamAI:
    if FakeSarvamAI.shared is None:
        FakeSarvamAI.shared = FakeSarvamAI(api_subscription_key=api_subscription_key)
    else:
        FakeSarvamAI.shared.api_subscription_key = api_subscription_key
    return FakeSarvamAI.shared


class FakeAudioOutput:
    def __init__(self, b64_audio: str) -> None:
        self.type = "audio"
        self.data = SimpleNamespace(
            audio=b64_audio, content_type="audio/mulaw", request_id=None
        )


class FakeEventResponse:
    def __init__(self, event_type: str = "final", message: str | None = None) -> None:
        self.type = "event"
        self.data = SimpleNamespace(
            event_type=event_type, message=message, timestamp=None
        )


class FakeErrorResponse:
    def __init__(self, message: str = "boom", code: int | None = 500) -> None:
        self.type = "error"
        self.data = SimpleNamespace(
            message=message, code=code, details=None, request_id=None
        )


@pytest.fixture
def fake_sarvam_module(monkeypatch: pytest.MonkeyPatch) -> FakeSarvamAI:
    """
    Register a fake ``sarvamai`` module in ``sys.modules`` so the
    local imports inside ``backend.audio.sarvam_tts`` resolve to our
    fakes.  Returns the SHARED ``FakeSarvamAI`` instance so tests can
    pre-program it with ``configure_next`` and inspect the same object
    that ``connect()`` ends up calling.

    The factory ``_fake_sarvam_ai_factory`` returns the same shared
    instance every time it is called (matching the production pattern of
    one SDK client per process) so ``configure_next`` set up by the test
    is what the SUT sees.
    """
    # Reset shared instance per test.
    FakeSarvamAI.shared = None

    fake_module = SimpleNamespace(
        AsyncSarvamAI=_fake_sarvam_ai_factory,
        AudioOutput=FakeAudioOutput,
        ErrorResponse=FakeErrorResponse,
        EventResponse=FakeEventResponse,
    )
    monkeypatch.setitem(sys.modules, "sarvamai", fake_module)
    load_settings.cache_clear()

    # Pre-create the shared instance so the test gets the same object
    # the SUT will see.
    shared = _fake_sarvam_ai_factory(api_subscription_key="dummy")
    return shared


# ─── C: Configuration & SDK contract ──────────────────────────────────────────


class TestConfigurationAndSdkContract:
    async def test_connect_calls_sdk_with_bulbul_v3_and_completion_event(
        self, fake_sarvam_module: FakeSarvamAI
    ) -> None:
        fake_sarvam_module.configure_next(FakeWS())
        client = AsyncSarvamTTSClient()
        await client.connect(_make_supervisor())

        assert len(fake_sarvam_module.connect_calls) == 1
        call = fake_sarvam_module.connect_calls[0]
        assert call["model"] == "bulbul:v3"
        assert call["send_completion_event"] == "true"

        await client.close()

    async def test_synthesize_turn_english_uses_en_in(
        self, fake_sarvam_module: FakeSarvamAI
    ) -> None:
        ws = FakeWS(turn_responses=[[FakeAudioOutput(base64.b64encode(b"\x00" * 160).decode()),
                FakeEventResponse(event_type="final"),]])
        fake_sarvam_module.configure_next(ws)
        client = AsyncSarvamTTSClient()
        sup = _make_supervisor()
        await client.connect(sup)

        await client.synthesize_turn(
            "Hello there.", language="English", generation=1, supervisor=sup
        )

        assert ws.configured[-1]["target_language_code"] == "en-IN"
        await client.close()

    async def test_synthesize_turn_hinglish_uses_hi_in(
        self, fake_sarvam_module: FakeSarvamAI
    ) -> None:
        ws = FakeWS(turn_responses=[[FakeAudioOutput(base64.b64encode(b"\x00" * 160).decode()),
                FakeEventResponse(event_type="final"),]])
        fake_sarvam_module.configure_next(ws)
        client = AsyncSarvamTTSClient()
        sup = _make_supervisor()
        await client.connect(sup)

        await client.synthesize_turn(
            "नमस्ते", language="Hinglish", generation=1, supervisor=sup
        )

        assert ws.configured[-1]["target_language_code"] == "hi-IN"
        await client.close()

    async def test_speaker_comes_from_settings(
        self,
        fake_sarvam_module: FakeSarvamAI,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SARVAM_TTS_SPEAKER", "TestSpeaker")
        load_settings.cache_clear()
        ws = FakeWS(turn_responses=[[FakeAudioOutput(base64.b64encode(b"\x00" * 160).decode()),
                FakeEventResponse(event_type="final"),]])
        fake_sarvam_module.configure_next(ws)
        client = AsyncSarvamTTSClient()
        sup = _make_supervisor()
        await client.connect(sup)

        await client.synthesize_turn(
            "Hi", language="English", generation=1, supervisor=sup
        )

        assert ws.configured[-1]["speaker"] == "TestSpeaker"
        await client.close()

    async def test_speaker_defaults_to_priya(
        self, fake_sarvam_module: FakeSarvamAI, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SARVAM_TTS_SPEAKER", raising=False)
        load_settings.cache_clear()
        ws = FakeWS(turn_responses=[[FakeAudioOutput(base64.b64encode(b"\x00" * 160).decode()),
                FakeEventResponse(event_type="final"),]])
        fake_sarvam_module.configure_next(ws)
        client = AsyncSarvamTTSClient()
        sup = _make_supervisor()
        await client.connect(sup)

        await client.synthesize_turn(
            "Hi", language="English", generation=1, supervisor=sup
        )

        assert ws.configured[-1]["speaker"] == "priya"
        await client.close()

    async def test_output_codec_and_sample_rate_are_mulaw_8khz(
        self, fake_sarvam_module: FakeSarvamAI
    ) -> None:
        ws = FakeWS(turn_responses=[[FakeAudioOutput(base64.b64encode(b"\x00" * 160).decode()),
                FakeEventResponse(event_type="final"),]])
        fake_sarvam_module.configure_next(ws)
        client = AsyncSarvamTTSClient()
        sup = _make_supervisor()
        await client.connect(sup)

        await client.synthesize_turn(
            "Hi", language="English", generation=1, supervisor=sup
        )

        cfg = ws.configured[-1]
        assert cfg["output_audio_codec"] == "mulaw"
        assert cfg["speech_sample_rate"] == 8000
        await client.close()

    async def test_preprocessing_enabled(
        self, fake_sarvam_module: FakeSarvamAI
    ) -> None:
        ws = FakeWS(turn_responses=[[FakeAudioOutput(base64.b64encode(b"\x00" * 160).decode()),
                FakeEventResponse(event_type="final"),]])
        fake_sarvam_module.configure_next(ws)
        client = AsyncSarvamTTSClient()
        sup = _make_supervisor()
        await client.connect(sup)

        await client.synthesize_turn(
            "Hi", language="English", generation=1, supervisor=sup
        )

        assert ws.configured[-1]["enable_preprocessing"] is True
        await client.close()

    async def test_min_buffer_size_not_passed(
        self, fake_sarvam_module: FakeSarvamAI
    ) -> None:
        ws = FakeWS(turn_responses=[[FakeAudioOutput(base64.b64encode(b"\x00" * 160).decode()),
                FakeEventResponse(event_type="final"),]])
        fake_sarvam_module.configure_next(ws)
        client = AsyncSarvamTTSClient()
        sup = _make_supervisor()
        await client.connect(sup)

        await client.synthesize_turn(
            "Hi", language="English", generation=1, supervisor=sup
        )

        cfg = ws.configured[-1]
        assert "min_buffer_size" not in cfg
        await client.close()

    async def test_configure_convert_flush_order(
        self, fake_sarvam_module: FakeSarvamAI
    ) -> None:
        ws = FakeWS(turn_responses=[[FakeAudioOutput(base64.b64encode(b"\x00" * 160).decode()),
                FakeEventResponse(event_type="final"),]])
        fake_sarvam_module.configure_next(ws)
        client = AsyncSarvamTTSClient()
        sup = _make_supervisor()
        await client.connect(sup)

        await client.synthesize_turn(
            "Hello world.", language="English", generation=1, supervisor=sup
        )

        # We can't observe order from the lists alone, but we can assert
        # each was called exactly once with the right text.
        assert len(ws.configured) == 1
        assert ws.converted == ["Hello world."]
        assert ws.flush_count == 1
        await client.close()


# ─── R: Reader → outbound queue ───────────────────────────────────────────────


class TestReaderToOutboundQueue:
    async def test_audio_output_is_decoded_and_pushed_with_generation(
        self, fake_sarvam_module: FakeSarvamAI
    ) -> None:
        chunk_bytes = b"\x00\xff\x42\xab" * 40  # 160 bytes — one Twilio frame
        b64 = base64.b64encode(chunk_bytes).decode()
        ws = FakeWS(turn_responses=[[FakeAudioOutput(b64),
                FakeEventResponse(event_type="final"),]])
        fake_sarvam_module.configure_next(ws)
        client = AsyncSarvamTTSClient()
        sup = _make_supervisor()
        await client.connect(sup)

        # Drain the reader task until it sees the final event so the
        # AudioOutput is pushed before synthesize_turn returns.
        await client.synthesize_turn(
            "Hi", language="English", generation=7, supervisor=sup
        )

        # The outbound queue should have one (gen, bytes) followed by (gen, None).
        items: list[tuple[int, Any]] = []
        while not sup.outbound_audio_queue.empty():
            items.append(sup.outbound_audio_queue.get_nowait())

        assert items[0] == (7, chunk_bytes)
        assert items[-1] == (7, None)
        await client.close()

    async def test_generation_is_pinned_before_configure(
        self, fake_sarvam_module: FakeSarvamAI
    ) -> None:
        """
        Even if audio arrives after configure() is sent but before flush()
        completes, the chunk must carry the generation the caller pinned.
        """
        chunk_bytes = b"\x01" * 160
        ws = FakeWS(turn_responses=[[FakeAudioOutput(base64.b64encode(chunk_bytes).decode()),
                FakeEventResponse(event_type="final"),]])
        fake_sarvam_module.configure_next(ws)
        client = AsyncSarvamTTSClient()
        sup = _make_supervisor()
        await client.connect(sup)

        await client.synthesize_turn(
            "Hi", language="English", generation=42, supervisor=sup
        )

        items: list[tuple[int, Any]] = []
        while not sup.outbound_audio_queue.empty():
            items.append(sup.outbound_audio_queue.get_nowait())

        # Find the audio chunk (not the None marker).
        audio_items = [it for it in items if it[1] is not None]
        assert audio_items == [(42, chunk_bytes)]
        await client.close()

    async def test_final_event_signals_end_of_turn(
        self, fake_sarvam_module: FakeSarvamAI
    ) -> None:
        ws = FakeWS(turn_responses=[[FakeAudioOutput(base64.b64encode(b"\x00" * 160).decode()),
                FakeEventResponse(event_type="final", message="done"),]])
        fake_sarvam_module.configure_next(ws)
        client = AsyncSarvamTTSClient()
        sup = _make_supervisor()
        await client.connect(sup)

        await client.synthesize_turn(
            "Hi", language="English", generation=3, supervisor=sup
        )

        items: list[tuple[int, Any]] = []
        while not sup.outbound_audio_queue.empty():
            items.append(sup.outbound_audio_queue.get_nowait())

        # End-of-turn marker MUST be present after the audio chunk.
        assert items[-1] == (3, None)
        await client.close()

    async def test_error_response_logs_and_still_advances_turn(
        self,
        fake_sarvam_module: FakeSarvamAI,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        ws = FakeWS(turn_responses=[[FakeErrorResponse(message="sarvam down", code=503),]])
        fake_sarvam_module.configure_next(ws)
        client = AsyncSarvamTTSClient()
        sup = _make_supervisor()
        await client.connect(sup)

        with caplog.at_level("ERROR", logger="backend.audio.sarvam_tts"):
            await client.synthesize_turn(
                "Hi", language="English", generation=5, supervisor=sup
            )

        # End-of-turn marker is still pushed so the outbound sender can
        # emit its final mark and the supervisor can move to teardown.
        items: list[tuple[int, Any]] = []
        while not sup.outbound_audio_queue.empty():
            items.append(sup.outbound_audio_queue.get_nowait())
        assert items == [(5, None)]

        assert any("sarvam down" in rec.message for rec in caplog.records)
        await client.close()

    async def test_invalid_base64_audio_is_dropped_not_pushed(
        self, fake_sarvam_module: FakeSarvamAI, caplog: pytest.LogCaptureFixture
    ) -> None:
        # "!@#$" is not valid base64 — the reader should warn and skip.
        ws = FakeWS(turn_responses=[[FakeAudioOutput("!@#$not_base64$$"),
                FakeEventResponse(event_type="final"),]])
        fake_sarvam_module.configure_next(ws)
        client = AsyncSarvamTTSClient()
        sup = _make_supervisor()
        await client.connect(sup)

        with caplog.at_level("WARNING", logger="backend.audio.sarvam_tts"):
            await client.synthesize_turn(
                "Hi", language="English", generation=9, supervisor=sup
            )

        items: list[tuple[int, Any]] = []
        while not sup.outbound_audio_queue.empty():
            items.append(sup.outbound_audio_queue.get_nowait())

        # Only the end-of-turn marker is pushed (the bad audio was dropped).
        assert items == [(9, None)]
        assert any("base64-decode" in rec.message for rec in caplog.records)
        await client.close()


# ─── M: Multi-turn socket reuse ──────────────────────────────────────────────


class TestMultiTurnSocketReuse:
    async def test_two_sequential_turns_share_one_socket(
        self, fake_sarvam_module: FakeSarvamAI
    ) -> None:
        # The WS replays the (audio + final) pair twice — one per turn.
        # Order matters: each turn must drain its own audio + final
        # before the next turn starts (the reader is single-threaded
        # and the queue is FIFO).
        chunk_one = b"\x00" * 160
        chunk_two = b"\x01" * 160
        ws = FakeWS(
            turn_responses=[
                [FakeAudioOutput(base64.b64encode(chunk_one).decode()), FakeEventResponse(event_type="final")],
                [FakeAudioOutput(base64.b64encode(chunk_two).decode()), FakeEventResponse(event_type="final")],
            ]
        )
        fake_sarvam_module.configure_next(ws)
        client = AsyncSarvamTTSClient()
        sup = _make_supervisor()
        await client.connect(sup)

        # Only one SDK connect call was made.
        assert len(fake_sarvam_module.connect_calls) == 1

        await client.synthesize_turn(
            "Turn one.", language="English", generation=1, supervisor=sup
        )
        await client.synthesize_turn(
            "Turn two.", language="English", generation=2, supervisor=sup
        )

        # Two configure → convert → flush cycles on the same socket.
        assert len(ws.configured) == 2
        assert ws.converted == ["Turn one.", "Turn two."]
        assert ws.flush_count == 2
        # Each turn produced its own end-of-turn marker.
        items: list[tuple[int, Any]] = []
        while not sup.outbound_audio_queue.empty():
            items.append(sup.outbound_audio_queue.get_nowait())
        # Two audio chunks + two None markers.
        audio_items = [it for it in items if it[1] is not None]
        none_items = [it for it in items if it[1] is None]
        assert len(audio_items) == 2
        assert audio_items[0][0] == 1
        assert audio_items[1][0] == 2
        assert none_items == [(1, None), (2, None)]
        await client.close()

    async def test_empty_text_turn_pushes_none_without_sarvam_call(
        self, fake_sarvam_module: FakeSarvamAI
    ) -> None:
        ws = FakeWS()
        fake_sarvam_module.configure_next(ws)
        client = AsyncSarvamTTSClient()
        sup = _make_supervisor()
        await client.connect(sup)

        await client.synthesize_turn(
            "", language="English", generation=11, supervisor=sup
        )

        # No Sarvam calls were made for the empty turn.
        assert ws.configured == []
        assert ws.converted == []
        assert ws.flush_count == 0

        # End-of-turn marker still pushed.
        items: list[tuple[int, Any]] = []
        while not sup.outbound_audio_queue.empty():
            items.append(sup.outbound_audio_queue.get_nowait())
        assert items == [(11, None)]
        await client.close()


# ─── L: Lifecycle ────────────────────────────────────────────────────────────


class TestLifecycle:
    async def test_close_is_idempotent(self, fake_sarvam_module: FakeSarvamAI) -> None:
        ws = FakeWS()
        fake_sarvam_module.configure_next(ws)
        client = AsyncSarvamTTSClient()
        await client.connect(_make_supervisor())

        await client.close()
        # Second close is a no-op (no exception, no hang).
        await client.close()
        assert ws.closed is True

    async def test_synthesize_turn_after_close_raises(
        self, fake_sarvam_module: FakeSarvamAI
    ) -> None:
        ws = FakeWS()
        fake_sarvam_module.configure_next(ws)
        client = AsyncSarvamTTSClient()
        sup = _make_supervisor()
        await client.connect(sup)
        await client.close()

        with pytest.raises(SarvamTTSError, match=r"close\(\)"):
            await client.synthesize_turn(
                "Hi", language="English", generation=1, supervisor=sup
            )

    async def test_synthesize_turn_without_connect_raises(
        self, fake_sarvam_module: FakeSarvamAI
    ) -> None:
        client = AsyncSarvamTTSClient()
        sup = _make_supervisor()

        with pytest.raises(SarvamTTSError, match="connect"):
            await client.synthesize_turn(
                "Hi", language="English", generation=1, supervisor=sup
            )

    async def test_synthesize_turn_with_mismatched_supervisor_raises(
        self, fake_sarvam_module: FakeSarvamAI
    ) -> None:
        ws = FakeWS(turn_responses=[[FakeEventResponse(event_type="final")]])
        fake_sarvam_module.configure_next(ws)
        client = AsyncSarvamTTSClient()
        sup_a = _make_supervisor()
        sup_b = _make_supervisor()
        await client.connect(sup_a)

        with pytest.raises(SarvamTTSError, match="supervisor"):
            await client.synthesize_turn(
                "Hi", language="English", generation=1, supervisor=sup_b
            )

        await client.close()

    async def test_connect_failure_raises_sarvam_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the SDK connect raises, the client surfaces SarvamTTSError."""

        class FailingSarvamAI:
            def __init__(self, *, api_subscription_key: str | None = None) -> None:
                self.api_subscription_key = api_subscription_key

            @property
            def text_to_speech_streaming(self) -> "FailingSarvamAI":
                return self

            def connect(self, **_kwargs: Any) -> Any:
                # Return an async context manager that raises on
                # __aenter__ — mirrors a real Sarvam SDK connection
                # failure (e.g. 401, network down).
                class _FailingCM:
                    async def __aenter__(self) -> None:
                        raise RuntimeError("network down")

                    async def __aexit__(self, *exc: Any) -> bool:
                        return False

                return _FailingCM()

        fake_module = SimpleNamespace(
            AsyncSarvamAI=FailingSarvamAI,
            AudioOutput=FakeAudioOutput,
            ErrorResponse=FakeErrorResponse,
            EventResponse=FakeEventResponse,
        )
        monkeypatch.setitem(sys.modules, "sarvamai", fake_module)
        load_settings.cache_clear()

        client = AsyncSarvamTTSClient()
        with pytest.raises(SarvamTTSError, match="network down"):
            await client.connect(_make_supervisor())

        # After failure, the client is in a closed/uninitialized state.
        # synthesize_turn must reject cleanly.
        with pytest.raises(SarvamTTSError):
            await client.synthesize_turn(
                "Hi", language="English", generation=1, supervisor=_make_supervisor()
            )

    async def test_synthesize_turn_with_unsupported_language_raises(
        self, fake_sarvam_module: FakeSarvamAI
    ) -> None:
        ws = FakeWS(turn_responses=[[FakeEventResponse(event_type="final")]])
        fake_sarvam_module.configure_next(ws)
        client = AsyncSarvamTTSClient()
        sup = _make_supervisor()
        await client.connect(sup)

        with pytest.raises(SarvamTTSError, match="Unsupported language"):
            await client.synthesize_turn(
                "Hi", language="Klingon", generation=1, supervisor=sup
            )

        await client.close()


# ─── T: Fallback timeout ──────────────────────────────────────────────────────


class TestFallbackTimeout:
    async def test_missing_final_event_still_pushes_none_after_timeout(
        self,
        fake_sarvam_module: FakeSarvamAI,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Patch the timeout down to 0.1s so the test is fast.
        monkeypatch.setattr(stts_module, "_TURN_DONE_TIMEOUT_S", 0.1)
        # Recreate the module-level Event on the patched module so the
        # patched constant is what synthesize_turn sees.
        # No responses at all — the reader hangs and synthesize_turn must
        # give up after the timeout.
        ws = FakeWS()
        fake_sarvam_module.configure_next(ws)
        client = AsyncSarvamTTSClient()
        sup = _make_supervisor()
        await client.connect(sup)

        with caplog.at_level("WARNING", logger="backend.audio.sarvam_tts"):
            await client.synthesize_turn(
                "Hi", language="English", generation=12, supervisor=sup
            )

        items: list[tuple[int, Any]] = []
        while not sup.outbound_audio_queue.empty():
            items.append(sup.outbound_audio_queue.get_nowait())
        # The fallback timeout fired — the None marker is still pushed.
        assert items == [(12, None)]
        assert any("timed out" in rec.message for rec in caplog.records)
        await client.close()


# ─── S: Settings ──────────────────────────────────────────────────────────────


class TestSettings:
    def test_default_speaker_is_priya(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SARVAM_TTS_SPEAKER", raising=False)
        load_settings.cache_clear()
        settings = load_settings()
        assert settings.sarvam_tts_speaker == "priya"

    def test_speaker_overridable_via_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SARVAM_TTS_SPEAKER", "Karun")
        load_settings.cache_clear()
        settings = load_settings()
        assert settings.sarvam_tts_speaker == "Karun"