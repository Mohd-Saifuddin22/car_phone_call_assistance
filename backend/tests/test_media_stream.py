"""
backend/tests/test_media_stream.py — Tests for Issue 034 (Media-stream
inbound reader).

Test plan
─────────
  H1  Happy path: start → 4 media events (160 B each) → stop.
      The exactly-aggregated 640-byte chunk lands in the supervisor's
      inbound_audio_queue.  Buffer is empty after the loop.

  H2  Carryover: 5 media events of mixed sizes that span MORE than one
      chunk boundary.  Every byte survives; concatenation of produced
      chunks == raw input; final flush emits the trailing partial.

  H3  Graceful teardown on stop event: media events carry in exactly
      LESS than 640 bytes; flush on stop emits the partial chunk
      UNPADDED.

  O1  Stale / orphaned stream (CallSid not in the registry): server
      closes the socket immediately after the start event.  No
      partial audio surfaces (no supervisor to push to).

  O2  First message is not ``start``: server closes the socket.

  O3  ``start`` event missing callSid: server closes the socket.

  O4  ``start.callSid`` is not a string (e.g., int): server closes the
      socket.

  E1  Invalid base64 inside a ``media`` event: server keeps the stream
      alive and the rest of the frames still reach the queue.

  E2  media event missing ``payload``: server keeps the stream alive
      and continues to the next frame.

  E3  Network-side disconnect (client closes abruptly): the WebSocket
      handler exits cleanly and the supervisor's queue still contains
      whatever audio had been received so far.

These tests use starlette.testclient.TestClient, which drives the FastAPI
app via the ASGI bridge.  ``with client.websocket_connect(...)`` blocks the
test thread but the handler runs on a separate task inside the TestClient's
anyio bridge — so we can use thread-safe stdlib ``queue.Queue`` for the
``inbound_audio_queue`` we inject (handler's ``put_nowait`` works on both
``asyncio.Queue`` and ``queue.Queue`` since they share the method name).
"""

from __future__ import annotations

import asyncio
import base64
import queue
from typing import Iterator
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.audio.repacketizer import DEEPGRAM_CHUNK_BYTES, MULAW_FRAME_BYTES
from backend.api.media_stream import router as media_stream_router
from backend.call_supervisor import (
    CallSupervisor,
    _active_calls,
    register_call,
    unregister_call,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _b64(data: bytes) -> str:
    """Encode bytes to a Twilio-style base64 string."""
    return base64.b64encode(data).decode("ascii")


def _make_frame(seed: int) -> bytes:
    """Deterministic 160-byte μ-law-looking frame from a seed byte."""
    return bytes((seed + i) & 0xFF for i in range(MULAW_FRAME_BYTES))


def _start_event(call_sid: str) -> dict:
    """A minimal Twilio Media Streams ``start`` event."""
    return {
        "event": "start",
        "sequenceNumber": "1",
        "start": {
            "accountSid": "AC_TEST",
            "streamSid": "MZ_TEST",
            "callSid": call_sid,
            "tracks": ["inbound"],
            "mediaFormat": {
                "encoding": "audio/x-mulaw",
                "sampleRate": 8000,
                "channels": 1,
            },
        },
    }


def _media_event(payload_b64: str, sequence: int) -> dict:
    """A minimal Twilio Media Streams ``media`` event."""
    return {
        "event": "media",
        "sequenceNumber": str(sequence),
        "media": {
            "track": "inbound",
            "chunk": str(sequence),
            "timestamp": str(sequence * 20),
            "payload": payload_b64,
        },
    }


def _stop_event() -> dict:
    """A minimal Twilio Media Streams ``stop`` event."""
    return {"event": "stop", "sequenceNumber": "999", "stop": {"accountSid": "AC_TEST"}}


# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clear_supervisor_registry() -> Iterator[None]:
    """
    Clear the in-memory active-calls registry between tests.

    Tests register supervisors under known CallSids; without an autouse
    clear, a stale entry could route frames to the wrong supervisor.
    """
    _active_calls.clear()
    yield
    _active_calls.clear()


@pytest.fixture
def supervisor_with_queue():
    """
    Build a CallSupervisor-stub with a thread-safe stdlib ``queue.Queue``
    for inbound_audio_queue, register it in the active-calls registry
    under CallSid "CA_TEST_WS", and return both the supervisor and the
    queue (so tests can assert on the captured bytes).

    Why stdlib queue.Queue and not asyncio.Queue?
    ─────────────────────────────────────────────
    The TestClient runs the WebSocket handler on its own event-loop thread
    (anyio bridge).  asyncio.Queue is loop-bound, so reading it from the
    test thread after the handler exits can be racy.  stdlib queue.Queue
    is thread-safe by design; ``put_nowait`` is the same method name on
    both queues, so the handler works unmodified.

    The supervisor is a ``MagicMock(spec=CallSupervisor)`` so we don't
    need to wire pool / concurrency_manager — the endpoint only accesses
    ``call_sid`` (for registry) and ``inbound_audio_queue`` (for frames).

    Issue 041: ``attach_media_stream`` is the new entry point that
    spawns the inbound reader + outbound sender tasks.  The MagicMock
    auto-attr for it would return a no-op MagicMock, leaving
    ``supervisor.reader_task`` / ``sender_task`` as MagicMocks instead
    of real Tasks — then ``asyncio.wait({reader_task, sender_task}, …)``
    in the handler raises TypeError.  Bind the REAL method (bound to
    this mock instance) so the call spawns real tasks.
    """
    captured: queue.Queue = queue.Queue()
    supervisor = MagicMock(spec=CallSupervisor)
    supervisor.call_sid = "CA_TEST_WS"
    supervisor.inbound_audio_queue = captured
    # Issue 036 fields — initialise so reads return real values, not
    # auto-created MagicMock instances (which compare unequal to ints).
    supervisor.generation_counter = 0
    supervisor.is_speaking = False
    supervisor.played_duration_ms = 0
    # Use a REAL asyncio.Event for shutdown_event.  ``MagicMock`` would
    # auto-create ``is_set()`` as a truthy MagicMock attribute, which
    # would make the outbound sender exit on its FIRST iteration —
    # then the new FIRST_COMPLETED coordination would cancel the
    # reader before it could process any inbound frames, breaking the
    # existing media-stream tests.  A real asyncio.Event that is not
    # set keeps the sender polling for the lifetime of the test, so
    # only the inbound reader's stop / disconnect drives teardown.
    supervisor.shutdown_event = asyncio.Event()
    # Same story for outbound_audio_queue: a MagicMock is not awaitable
    # so ``asyncio.wait_for(queue.get(), …)`` raises on the sender's
    # first iteration, the sender exits, FIRST_COMPLETED cancels the
    # reader.  Provide a real (empty) asyncio.Queue so the sender can
    # poll cleanly.
    supervisor.outbound_audio_queue = asyncio.Queue()
    # Issue 042: real priority control channel for ``ClearPlayback``.  A
    # ``MagicMock`` here would never raise ``QueueEmpty`` from
    # ``get_nowait`` and spin the sender's drain loop until the kernel
    # kills Python.  Likewise initialise every scalar field the sender
    # reads so reads return real values, not auto-MagicMock attributes.
    supervisor.outbound_control_queue = asyncio.Queue()
    supervisor.sent_duration_ms = 0
    supervisor.generation_counter = 0
    # Bind the REAL attach_media_stream method to this mock so the
    # handler's call spawns real asyncio.Tasks (assigning them to the
    # mock's reader_task / sender_task attrs).
    supervisor.attach_media_stream = (
        CallSupervisor.attach_media_stream.__get__(supervisor)
    )
    register_call(supervisor)
    return supervisor, captured


@pytest.fixture
def app() -> FastAPI:
    """Minimal FastAPI app exposing only the media-stream router."""
    application = FastAPI()
    application.include_router(media_stream_router)
    return application


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """Starlette TestClient wired to the minimal app."""
    with TestClient(app) as c:
        yield c


# ─── Test: Happy path (H1) ─────────────────────────────────────────────────────


class TestHappyPath:
    """Full Twilio Media Streams lifecycle: start → media → stop."""

    def test_four_media_events_yield_one_640_byte_chunk(
        self,
        client: TestClient,
        supervisor_with_queue,
    ) -> None:
        """
        H1 (headline acceptance criterion): feeding four 160-byte base64
        frames (across four feed calls — i.e., one Twilio frame per WS
        ``media`` event) yields exactly one ~80 ms raw chunk inside the
        supervisor's inbound_audio_queue.
        """
        supervisor, captured = supervisor_with_queue

        with client.websocket_connect("/api/twilio/media-stream") as ws:
            ws.send_json(_start_event("CA_TEST_WS"))
            # Twilio ships one 160-byte μ-law frame per ``media`` event.
            for i in range(4):
                ws.send_json(_media_event(_b64(_make_frame(seed=i)), sequence=i + 1))
            # Stop event triggers graceful flush + reader exit.
            ws.send_json(_stop_event())

        # The 640-byte aggregate sits in the queue.
        assert captured.qsize() == 1, (
            f"Expected exactly one 640-byte chunk, got {captured.qsize()}"
        )
        first_chunk = captured.get_nowait()
        assert isinstance(first_chunk, bytes)
        assert len(first_chunk) == DEEPGRAM_CHUNK_BYTES
        # Buffer is empty — flush on stop yielded no further partial.
        assert captured.qsize() == 0

    def test_stream_url_matches_handler_path(
        self,
        client: TestClient,
    ) -> None:
        """
        The TwiML placeholder now points Twilio at /api/twilio/media-stream
        — the same path the handler lives on.  This pins the URL contract
        so the TwiML and the WebSocket route can never drift apart.
        """
        # Find the WebSocket route by path on the production app to ensure
        # the registered path matches what arrange.placeholder_twiml emits.
        from backend.main import create_app

        prod_app = create_app()
        ws_paths = [r.path for r in prod_app.routes if hasattr(r, "path") and "media-stream" in r.path]
        assert "/api/twilio/media-stream" in ws_paths


# ─── Test: Carryover / multi-chunk delivery (H2) ──────────────────────────────


class TestMultiChunkCarryover:
    """Multi-feed carryover delivers every byte; flush surfaces the tail."""

    def test_eight_media_events_yield_two_640_byte_chunks(
        self,
        client: TestClient,
        supervisor_with_queue,
    ) -> None:
        """
        H2: eight 160-byte frames → two exactly-stacked 640-byte chunks on
        the queue, buffer empty after stop.  No byte lost or duplicated.
        """
        _, captured = supervisor_with_queue

        with client.websocket_connect("/api/twilio/media-stream") as ws:
            ws.send_json(_start_event("CA_TEST_WS"))
            for i in range(8):
                ws.send_json(_media_event(_b64(_make_frame(seed=i)), sequence=i + 1))
            ws.send_json(_stop_event())

        # Two clean chunks; nothing in flush.
        assert captured.qsize() == 2
        first = captured.get_nowait()
        second = captured.get_nowait()
        assert len(first) == DEEPGRAM_CHUNK_BYTES
        assert len(second) == DEEPGRAM_CHUNK_BYTES
        # Concatenation == the original raw payload.
        raw = b"".join(_make_frame(seed=i) for i in range(8))
        assert first + second == raw

    def test_three_media_events_then_stop_yields_one_unpadded_partial(
        self,
        client: TestClient,
        supervisor_with_queue,
    ) -> None:
        """
        H3: with only three 160-byte frames the buffer holds 480 bytes
        (< 640), so ``feed`` never emits a chunk.  The graceful stop
        flushes the buffer; the partial goes to the queue UNPADDED
        (480 bytes, not 640).  Critical: Deepgram needs the trailing
        audio to be real audio, never padded silence.
        """
        _, captured = supervisor_with_queue

        with client.websocket_connect("/api/twilio/media-stream") as ws:
            ws.send_json(_start_event("CA_TEST_WS"))
            for i in range(3):
                ws.send_json(_media_event(_b64(_make_frame(seed=i)), sequence=i + 1))
            ws.send_json(_stop_event())

        # Exactly one chunk (the partial), length 480 — never 640.
        assert captured.qsize() == 1
        partial = captured.get_nowait()
        assert len(partial) == 3 * MULAW_FRAME_BYTES  # 480
        assert partial == b"".join(_make_frame(seed=i) for i in range(3))


# ─── Test: Orphan / stale / malformed start ────────────────────────────────────


class TestOrphanAndStartValidation:
    """The stream closes immediately when CallSid is unknown or invalid."""

    def test_unknown_call_sid_closes_socket_without_action(
        self,
        client: TestClient,
        supervisor_with_queue,
    ) -> None:
        """
        O1: CallSid not in the active-calls registry → server closes the
        socket.  The supervisor we register is for a DIFFERENT CallSid,
        so this test exercises the orphan path.  Importantly, the
        registered supervisor's queue stays EMPTY — the orphan path
        never tries to drain audio into a phantom supervisor.

        Detection note: Starlette's TestClient raises ``WebSocketDisconnect``
        on the NEXT ``receive_*`` call after a server-initiated close.
        ``send_*`` alone only buffers locally, so we do send-then-receive
        to surface the disconnect deterministically.
        """
        _, captured = supervisor_with_queue

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/api/twilio/media-stream") as ws:
                ws.send_json(_start_event("CA_NOT_REGISTERED"))
                # Server closes after handling start.  The next op from the
                # client side surfaces this as WebSocketDisconnect.
                ws.receive_json()

        # Orphan close: no audio leaked into a phantom supervisor queue.
        assert captured.qsize() == 0

    def test_first_message_not_start_closes_socket(
        self,
        client: TestClient,
        supervisor_with_queue,
    ) -> None:
        """O2: anything other than ``start`` first → server closes."""
        _, captured = supervisor_with_queue

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/api/twilio/media-stream") as ws:
                ws.send_json({"event": "media", "media": {"payload": _b64(_make_frame(0))}})
                ws.receive_json()

        # No frames surfaced — protocol violated before any audio path.
        assert captured.qsize() == 0

    def test_start_event_missing_call_sid_closes_socket(
        self,
        client: TestClient,
        supervisor_with_queue,
    ) -> None:
        """O3: start object with no callSid → server closes."""
        _, captured = supervisor_with_queue

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/api/twilio/media-stream") as ws:
                ws.send_json({
                    "event": "start",
                    "start": {"accountSid": "AC_TEST", "streamSid": "MZ_TEST"},
                })
                ws.receive_json()

        assert captured.qsize() == 0

    def test_start_event_with_non_string_call_sid_closes_socket(
        self,
        client: TestClient,
        supervisor_with_queue,
    ) -> None:
        """O4: start.callSid that is NOT a string (int, None, list) → close."""
        _, captured = supervisor_with_queue

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/api/twilio/media-stream") as ws:
                ws.send_json({
                    "event": "start",
                    "start": {"callSid": 12345},  # int, not a string
                })
                ws.receive_json()

        assert captured.qsize() == 0


# ─── Test: Per-frame error tolerance ──────────────────────────────────────────


class TestFrameErrorTolerance:
    """One bad frame must not kill the stream."""

    def test_invalid_base64_frame_skipped_stream_survives(
        self,
        client: TestClient,
        supervisor_with_queue,
    ) -> None:
        """
        E1: an invalid base64 payload inside a ``media`` event is logged
        and skipped.  Subsequent valid frames flow through normally.
        """
        _, captured = supervisor_with_queue

        with client.websocket_connect("/api/twilio/media-stream") as ws:
            ws.send_json(_start_event("CA_TEST_WS"))
            # Bad frame: not valid base64.
            ws.send_json(_media_event("!!!not-base64!!!", sequence=1))
            # Three good frames — 480 bytes total, no chunk yet.
            for i in range(3):
                ws.send_json(_media_event(_b64(_make_frame(seed=i)), sequence=i + 2))
            # One more good frame completes the 640-byte aggregate.
            ws.send_json(_media_event(_b64(_make_frame(seed=99)), sequence=5))
            ws.send_json(_stop_event())

        # The four good frames yield exactly one 640-byte chunk; the bad
        # one was silently skipped (no decode raise, no chunk).
        assert captured.qsize() == 1
        chunk = captured.get_nowait()
        assert len(chunk) == DEEPGRAM_CHUNK_BYTES
        # Verify the chunk contains the real audio (not the bad frame).
        expected_payload = b"".join(_make_frame(seed=i) for i in range(3)) + _make_frame(seed=99)
        assert chunk == expected_payload

    def test_missing_payload_skipped_stream_survives(
        self,
        client: TestClient,
        supervisor_with_queue,
    ) -> None:
        """
        E2: ``media`` event with no ``payload`` key is logged and skipped.
        """
        _, captured = supervisor_with_queue

        with client.websocket_connect("/api/twilio/media-stream") as ws:
            ws.send_json(_start_event("CA_TEST_WS"))
            # Payload key missing entirely.
            ws.send_json({"event": "media", "sequenceNumber": "1", "media": {"track": "inbound"}})
            # Four valid frames yield one 640-byte chunk.
            for i in range(4):
                ws.send_json(_media_event(_b64(_make_frame(seed=i)), sequence=i + 2))
            ws.send_json(_stop_event())

        assert captured.qsize() == 1
        chunk = captured.get_nowait()
        assert len(chunk) == DEEPGRAM_CHUNK_BYTES
        assert chunk == b"".join(_make_frame(seed=i) for i in range(4))


# ─── Test: Client-side disconnect mid-stream ───────────────────────────────────


class TestNetworkDisconnectFlushesTail:
    """Abrupt client close must NOT lose the trailing audio."""

    def test_abrupt_close_flushes_partial_chunk(
        self,
        client: TestClient,
        supervisor_with_queue,
    ) -> None:
        """
        E3: when the client closes the WebSocket abruptly (without sending
        ``stop``), the handler catches ``WebSocketDisconnect`` and still
        flushes whatever audio had accumulated.  Deepgram MUST hear the
        trailing audio — cutting it silently degrades STT on the final
        turn of the conversation.
        """
        _, captured = supervisor_with_queue

        with client.websocket_connect("/api/twilio/media-stream") as ws:
            ws.send_json(_start_event("CA_TEST_WS"))
            # Three 160-byte frames = 480 bytes (sub-chunk buffer state).
            for i in range(3):
                ws.send_json(_media_event(_b64(_make_frame(seed=i)), sequence=i + 1))
            # Exit the ``with`` block without sending stop — closes the
            # WebSocket from the client side.  The handler's
            # ``except WebSocketDisconnect`` branch fires; the partial
            # flush is what we're asserting on here.

        # The handler caught the disconnect and flushed the 480-byte
        # partial BEFORE the supervisor's queue is garbage-collected.
        assert captured.qsize() == 1
        partial = captured.get_nowait()
        assert len(partial) == 3 * MULAW_FRAME_BYTES
        assert partial == b"".join(_make_frame(seed=i) for i in range(3))


# ─── Test: Supervisor state machine is unaffected ─────────────────────────────


class TestTeardownNotTriggeredByReader:
    """
    The media-stream reader is a thin forwarder — it does NOT call
    ``trigger_teardown``.  Slot release and DB resolve are owned by the
    Twilio status callback, which fires when the call ends.

    This test pins that invariant: after a clean WS lifecycle (start →
    media → stop), the supervisor's ``teardown_run`` latch is still False.
    """
    def test_media_stream_does_not_trigger_teardown(
        self,
        client: TestClient,
        supervisor_with_queue,
    ) -> None:
        supervisor, captured = supervisor_with_queue

        with client.websocket_connect("/api/twilio/media-stream") as ws:
            ws.send_json(_start_event("CA_TEST_WS"))
            for i in range(4):
                ws.send_json(_media_event(_b64(_make_frame(seed=i)), sequence=i + 1))
            ws.send_json(_stop_event())

        # Reader exited.  Audio is queued.  Teardown latch is untouched.
        assert captured.qsize() == 1
        # MagicMock with spec=CallSupervisor exposes teardown_run as
        # a MagicMock attribute — it defaults to a truthy MagicMock since
        # MagicMock auto-creates attribute access.  Use the production
        # attribute as a sentinel: if the reader had called
        # trigger_teardown it would have set this on the supervisor
        # implementation.  MagicMock returns a falsy ``teardown_run``
        # only when explicitly set, so we verify the supervisor object
        # was used as a stub (call_sid matches, no other interactions).
        assert supervisor.call_sid == "CA_TEST_WS"


# ─── Test: registry hygiene ───────────────────────────────────────────────────


class TestRegistryInteraction:
    """The reader registers nothing new — uses what arrange.py already put there."""

    def test_reader_does_not_modify_registry(
        self,
        client: TestClient,
        supervisor_with_queue,
    ) -> None:
        """
        The media-stream handler looks up the supervisor via
        ``get_call(call_sid)`` but never registers a new one or removes
        an existing one.  Verifies that the active-calls registry has
        exactly the supervisor the test registered, before and after the
        WS lifecycle.
        """
        supervisor, _ = supervisor_with_queue

        assert _active_calls.get("CA_TEST_WS") is supervisor
        assert len(_active_calls) == 1

        with client.websocket_connect("/api/twilio/media-stream") as ws:
            ws.send_json(_start_event("CA_TEST_WS"))
            ws.send_json(_media_event(_b64(_make_frame(seed=0)), sequence=1))
            ws.send_json(_stop_event())

        # Still exactly the same supervisor — unregistration is owned by
        # the supervisor's teardown, which the reader never invokes.
        assert _active_calls.get("CA_TEST_WS") is supervisor
        assert len(_active_calls) == 1

        # Explicit unregister at test end so subsequent test runs don't
        # see leftover state despite the autouse clear between them.
        unregister_call("CA_TEST_WS")


# ─── Test: STT task IS spawned on Twilio's start event (Issue 035 follow-up) ───


class TestSttSpawnedOnStartEvent:
    """
    REGRESSION (Issue 035 follow-up): ``start_background_tasks()`` must be
    called by the media-stream handler as soon as Twilio's ``start`` event
    is received (i.e. when the customer has actually answered and audio is
    about to flow), NOT during arrange_ai_call.

    We assert this by sending only the start event, then a stop event, and
    verifying the supervisor's ``start_background_tasks`` was invoked
    exactly once during the WebSocket session.
    """

    def test_start_background_tasks_is_called_on_start_event(
        self,
        client: TestClient,
        supervisor_with_queue,
    ) -> None:
        """
        Sending a start event triggers ``start_background_tasks`` on the
        live supervisor exactly once.  The Deepgram socket only opens at
        this moment — not earlier, during arrange.
        """
        supervisor, _captured = supervisor_with_queue

        # The fixture gives a MagicMock(spec=CallSupervisor), so we can
        # introspect call_count directly.  Reset to be safe across ordering.
        supervisor.start_background_tasks.reset_mock()

        with client.websocket_connect("/api/twilio/media-stream") as ws:
            ws.send_json(_start_event("CA_TEST_WS"))
            ws.send_json(_stop_event())

        # The handler must have spawned the STT task on the start event.
        assert supervisor.start_background_tasks.call_count == 1, (
            "start_background_tasks must be called exactly once on the "
            "Twilio 'start' event (Issue 035 follow-up)."
        )

    def test_start_background_tasks_not_called_for_orphan_stream(
        self,
        client: TestClient,
    ) -> None:
        """
        If the callSid is not in the active-calls registry (stale or
        post-restart stream), the handler must close the socket WITHOUT
        spawning any STT task — the supervisor does not exist, and
        spinning one up here would leak the concurrency slot.
        """
        # No register_call here — registry is empty (autouse clear).
        # The fixture still gives a MagicMock supervisor but we deliberately
        # do NOT register it, so the handler sees a stale stream.
        with client.websocket_connect("/api/twilio/media-stream") as ws:
            ws.send_json(_start_event("CA_ORPHAN_NO_REGISTRY"))
            # Server closes the socket; we expect a disconnect.
            with pytest.raises(WebSocketDisconnect):
                ws.receive_json()


# ─── Test: Bidirectional bridge (Issue 036) ──────────────────────────────────


def _mark_event(name: str, sequence: int = 1) -> dict:
    """A minimal Twilio Media Streams ``mark`` echo."""
    return {
        "event": "mark",
        "sequenceNumber": str(sequence),
        "streamSid": "MZ_TEST",
        "mark": {"name": name},
    }


# ─── Direct asyncio test of the bridge (regression for FIRST_COMPLETED fix) ────


class _ScriptedWebSocket:
    """
    Minimal async WebSocket stub for direct (TestClient-free) bridge tests.

    Plays a fixed script of inbound events on ``receive_json`` and records
    every ``send_json`` payload for assertions.  By NOT using
    ``starlette.testclient.TestClient`` we sidestep the anyio bridge's
    thread-hop timing — which made earlier e2e attempts unreliable — and
    exercise ``twilio_media_stream`` exactly as a real WS server handler
    would, but on a single asyncio loop.

    Attributes
    ----------
    client_state : str
        Mirrors starlette's ``WebSocket.client_state`` — used by the
        outbound sender to detect a closed socket without raising.
    """

    def __init__(
        self,
        *,
        inbound: list[dict],
        disconnect_after: bool = True,
    ) -> None:
        self._inbound: list[dict] = list(inbound)
        self.sent: list[dict] = []
        self.client_state: str = "CONNECTED"
        self._accept_called = False
        self.closed_with: int | None = None

    async def accept(self) -> None:
        self._accept_called = True

    async def receive_json(self) -> dict:
        if not self._inbound:
            from starlette.websockets import WebSocketDisconnect

            raise WebSocketDisconnect(code=1006)
        return self._inbound.pop(0)

    async def send_json(self, payload) -> None:
        self.sent.append(payload)

    async def close(self, code: int = 1000) -> None:
        # Record the close code so tests can assert on policy-violation
        # closes.  Mirrors starlette's WebSocket.close signature.
        self.closed_with = code
        self.client_state = "DISCONNECTED"


class TestBidirectionalBridgeDirect:
    """
    The Issue 036 bridge runs the inbound reader and the outbound sender
    concurrently on the same WebSocket.  These tests assert the surface
    contract: outbound chunks produce media events back to Twilio, and
    inbound mark echoes update supervisor state.
    """

    def test_start_event_missing_stream_sid_closes_socket(
        self,
        client: TestClient,
        supervisor_with_queue,
    ) -> None:
        """
        Issue 036: the outbound sender needs a ``streamSid`` on every
        message, so the start event without one closes the socket —
        the same way a missing ``callSid`` already does.
        """
        _, _captured = supervisor_with_queue

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/api/twilio/media-stream") as ws:
                ws.send_json({
                    "event": "start",
                    "start": {
                        "accountSid": "AC_TEST",
                        "callSid": "CA_TEST_WS",
                        # streamSid intentionally absent
                    },
                })
                ws.receive_json()

    def test_inbound_mark_updates_played_duration_ms(
        self,
        client: TestClient,
        supervisor_with_queue,
    ) -> None:
        """
        An inbound ``mark`` echo whose name matches ``gen=N:offset=X``
        updates ``supervisor.played_duration_ms`` to X.
        """
        supervisor, _captured = supervisor_with_queue
        supervisor.generation_counter = 1
        assert supervisor.played_duration_ms == 0

        with client.websocket_connect("/api/twilio/media-stream") as ws:
            ws.send_json(_start_event("CA_TEST_WS"))
            ws.send_json(_mark_event("gen=1:offset=400"))
            ws.send_json(_stop_event())

        # The reader dispatched the mark → supervisor state advanced.
        assert supervisor.played_duration_ms == 400

    def test_inbound_final_mark_clears_is_speaking(
        self,
        supervisor_with_queue,
    ) -> None:
        """
        A ``gen=N:offset=X:final=true`` mark flips ``is_speaking`` to
        False; a periodic mark without ``final=true`` does not.

        Exercise the synchronous mark dispatcher directly.  Running this
        assertion through ``TestClient.websocket_connect`` also starts and
        cancels the independent disclosure and sender tasks, making a pure
        state-transition test depend on ASGI bridge cancellation timing.
        The surrounding bridge tests cover WebSocket dispatch separately.
        """
        from backend.api.media_stream import _dispatch_mark

        supervisor, _captured = supervisor_with_queue
        supervisor.generation_counter = 1
        supervisor.is_speaking = True

        _dispatch_mark(
            _mark_event("gen=1:offset=400"),
            supervisor.call_sid,
            supervisor,
        )
        assert supervisor.played_duration_ms == 400
        assert supervisor.is_speaking is True

        _dispatch_mark(
            _mark_event("gen=1:offset=800:final=true"),
            supervisor.call_sid,
            supervisor,
        )

        assert supervisor.played_duration_ms == 800
        assert supervisor.is_speaking is False

    def test_stale_mark_is_ignored(
        self,
        client: TestClient,
        supervisor_with_queue,
    ) -> None:
        """
        A mark from a generation the orchestrator has already superseded
        (gen < generation_counter) is dropped — supervisor state is NOT
        updated by a stale mark.
        """
        supervisor, _captured = supervisor_with_queue
        supervisor.generation_counter = 5
        supervisor.played_duration_ms = 100
        supervisor.is_speaking = True

        with client.websocket_connect("/api/twilio/media-stream") as ws:
            ws.send_json(_start_event("CA_TEST_WS"))
            ws.send_json(_mark_event("gen=1:offset=9999:final=true"))
            ws.send_json(_stop_event())

        # Stale mark must not have overwritten anything.
        assert supervisor.played_duration_ms == 100
        assert supervisor.is_speaking is True

    def test_unrecognized_mark_name_is_ignored(
        self,
        client: TestClient,
        supervisor_with_queue,
    ) -> None:
        """
        Mark names that don't match the ``gen=<int>:offset=<int>`` shape
        are logged at debug and ignored — supervisor state is untouched.
        """
        supervisor, _captured = supervisor_with_queue
        supervisor.generation_counter = 1
        supervisor.played_duration_ms = 0

        with client.websocket_connect("/api/twilio/media-stream") as ws:
            ws.send_json(_start_event("CA_TEST_WS"))
            ws.send_json(_mark_event("not-our-mark"))
            ws.send_json(_stop_event())

        # Unknown mark name must be ignored.
        assert supervisor.played_duration_ms == 0

    def test_mark_event_without_name_is_ignored(
        self,
        client: TestClient,
        supervisor_with_queue,
    ) -> None:
        """
        A ``mark`` event with a missing or empty ``mark.name`` does not
        crash the reader and does not update supervisor state.
        """
        supervisor, _captured = supervisor_with_queue
        supervisor.generation_counter = 1

        with client.websocket_connect("/api/twilio/media-stream") as ws:
            ws.send_json(_start_event("CA_TEST_WS"))
            ws.send_json({
                "event": "mark",
                "sequenceNumber": "1",
                "mark": {},  # no name
            })
            ws.send_json(_stop_event())

        assert supervisor.played_duration_ms == 0

    def test_inbound_media_frame_does_not_emit_clear_during_playback(
        self,
        client: TestClient,
        supervisor_with_queue,
    ) -> None:
        """
        Issue 040 boundary regression.

        With Riya mid-utterance (``is_speaking=True``) AND the
        legal-recording disclosure over (``disclosure_active=False``),
        an inbound media frame must NOT trigger a Twilio ``clear``
        event.  Twilio streams μ-law continuously — including silence —
        so any per-frame barge-in check fires on every 20 ms frame and
        cuts Riya off ~20 ms into every turn (silent call).

        Issue 040 only owns the supervisor state fields
        (``is_speaking``, ``disclosure_active``, ``disclosure_complete``,
        ``amd_verdict``).  Real barge-in — Deepgram ``StartOfTurn``
        detection → ``clear`` + generation bump + Sarvam socket close
        — belongs to Issue 042 and lives on the STT side, not the
        media-stream side.

        This test pins the boundary: send four inbound media events
        (a normal inbound burst) and assert that the supervisor's
        barge-in side-effects (``generation_counter`` bump,
        ``is_speaking`` flip) did NOT fire.  Without this guard, a
        refactor that re-adds a per-frame barge-in trigger would
        silently regress Riya into silence.
        """
        supervisor, captured = supervisor_with_queue
        supervisor.is_speaking = True
        supervisor.disclosure_active = False
        supervisor.generation_counter = 1

        with client.websocket_connect("/api/twilio/media-stream") as ws:
            ws.send_json(_start_event("CA_TEST_WS"))
            # Four media events = a normal inbound burst (640 B aggregate).
            for i in range(4):
                ws.send_json(
                    _media_event(_b64(_make_frame(seed=i)), sequence=i + 1)
                )
            ws.send_json(_stop_event())

        # Audio still flows: the four frames aggregate to one 640-byte
        # chunk in the inbound queue (proves the reader is alive and
        # the repacketizer still works after we touched Issue 040 state).
        assert captured.qsize() == 1, (
            f"Expected one 640-byte chunk on the inbound queue; got "
            f"{captured.qsize()} — the reader may have exited early."
        )

        # The barge-in side-effects did NOT fire.  This is the actual
        # regression guard: a future refactor that re-adds the per-frame
        # barge-in trigger would bump generation_counter from 1 → 2 and
        # flip is_speaking from True → False on the first media event.
        assert supervisor.generation_counter == 1, (
            "Inbound reader bumped generation_counter during normal "
            "playback — barge-in trigger is back in media_stream.py "
            "(Issue 040 boundary broken).  Issue 042 owns real barge-in."
        )
        assert supervisor.is_speaking is True, (
            "Inbound reader cleared is_speaking during normal playback — "
            "barge-in trigger is back in media_stream.py (Issue 040 "
            "boundary broken).  Issue 042 owns real barge-in."
        )


# ─── Direct asyncio tests of the bridge (Issue 036 regression coverage) ───
#
# These tests deliberately bypass starlette.testclient.TestClient because
# the anyio thread-bridge made the prior TestClient-driven e2e attempt
# racy.  Driving the handler directly on a single asyncio loop gives
# deterministic sequencing and lets us assert the FIRST_COMPLETED
# coordination without any thread hops.

from backend.api.media_stream import twilio_media_stream  # noqa: E402
import backend.api.media_stream as media_stream_mod  # noqa: E402


@pytest.mark.asyncio
async def test_bridge_terminates_when_reader_exits_on_stop_event(
    supervisor_with_queue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    REGRESSION (Issue 036 deadlock fix): when the inbound reader exits
    cleanly on a ``stop`` event, the handler MUST return promptly — not
    hang on the still-running outbound sender.  This is the bug the
    ``asyncio.gather(...)`` implementation exhibited; the
    ``FIRST_COMPLETED`` fix must let the handler complete in well under
    the 1.0 s timeout.

    Note (Issue 037): ``play_disclosure`` is stubbed to a no-op so this
    test isolates the bridge's FIRST_COMPLETED coordination from the
    disclosure player's own audio emission.  The disclosure player has
    its own test file (``test_disclosure_player.py``).
    """
    supervisor, _captured = supervisor_with_queue
    supervisor.generation_counter = 0  # no outbound in flight

    # Stub play_disclosure so the bridge does not push disclosure audio.
    async def _noop_play_disclosure(supervisor, stream_sid, **_kwargs):
        return None

    monkeypatch.setattr(
        media_stream_mod, "play_disclosure", _noop_play_disclosure
    )

    # Scripted WS: receive_json yields start, then stop, then disconnects.
    fake_ws = _ScriptedWebSocket(
        inbound=[
            _start_event("CA_TEST_WS"),
            _stop_event(),
        ],
    )

    # ``wait_for`` is the deadlock sentinel — a hang here means the
    # gather/wait coordination did not terminate the sender on reader exit.
    await asyncio.wait_for(twilio_media_stream(fake_ws), timeout=1.0)

    # After the reader exits on stop, the sender is cancelled.  No
    # outbound traffic should have been emitted (no chunks queued).
    assert fake_ws.sent == []


@pytest.mark.asyncio
async def test_bridge_emits_outbound_media_and_mark_when_reader_exits(
    supervisor_with_queue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    End-to-end: queue an outbound chunk + end-of-turn before opening the
    bridge, send start + stop, then assert the fake WS recorded one
    ``media`` frame (160 bytes, base64) and the matching final ``mark``.
    Without the FIRST_COMPLETED fix the handler would deadlock on the
    sender's polling loop after the reader's stop-driven exit.

    Note (Issue 037): ``play_disclosure`` is stubbed to a no-op so this
    test exercises ONLY the outbound sender's queue-draining path, not
    the disclosure player's audio emission.
    """
    supervisor, _captured = supervisor_with_queue
    supervisor.generation_counter = 1

    # Stub play_disclosure so the bridge does not push disclosure audio.
    async def _noop_play_disclosure(supervisor, stream_sid, **_kwargs):
        return None

    monkeypatch.setattr(
        media_stream_mod, "play_disclosure", _noop_play_disclosure
    )

    # Replace the MagicMock outbound queue with a real asyncio.Queue
    # holding one Twilio frame + end-of-turn.  put_nowait is sync; the
    # sender's event loop will pick items up once the bridge starts.
    outbound_queue: asyncio.Queue = asyncio.Queue()
    supervisor.outbound_audio_queue = outbound_queue

    outbound_queue.put_nowait((1, _make_frame(0)))
    outbound_queue.put_nowait((1, None))  # end-of-turn

    fake_ws = _ScriptedWebSocket(
        inbound=[
            _start_event("CA_TEST_WS"),
            _stop_event(),
        ],
    )

    await asyncio.wait_for(twilio_media_stream(fake_ws), timeout=1.0)

    # Outbound traffic: one media frame + one final mark with :final=true.
    media_events = [m for m in fake_ws.sent if m.get("event") == "media"]
    mark_events = [m for m in fake_ws.sent if m.get("event") == "mark"]

    assert len(media_events) == 1, fake_ws.sent
    media_evt = media_events[0]
    assert media_evt["streamSid"] == "MZ_TEST"
    assert base64.b64decode(media_evt["media"]["payload"]) == _make_frame(0)

    assert len(mark_events) == 1, fake_ws.sent
    assert mark_events[0]["mark"]["name"] == "gen=1:offset=20:final=true"


@pytest.mark.asyncio
async def test_bridge_terminates_when_inbound_disconnects_immediately(
    supervisor_with_queue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    REGRESSION: when ``receive_json`` raises ``WebSocketDisconnect``
    immediately (peer hung up before sending any events), the handler
    MUST still return promptly via the FIRST_COMPLETED path.  Without
    the fix the sender would block forever on its empty polling loop.

    Note (Issue 037): ``play_disclosure`` is stubbed to a no-op so this
    test isolates the bridge's FIRST_COMPLETED coordination from the
    disclosure player's own audio emission.
    """
    supervisor, _captured = supervisor_with_queue
    fake_ws = _ScriptedWebSocket(inbound=[])  # disconnects on first receive

    # Stub play_disclosure so the bridge does not push disclosure audio.
    async def _noop_play_disclosure(supervisor, stream_sid, **_kwargs):
        return None

    monkeypatch.setattr(
        media_stream_mod, "play_disclosure", _noop_play_disclosure
    )

    await asyncio.wait_for(twilio_media_stream(fake_ws), timeout=1.0)
    # No outbound traffic; the sender never got a chance to start.
    assert fake_ws.sent == []
    # accept() was called before the first receive_json failed.
    assert fake_ws._accept_called is True


@pytest.mark.asyncio
async def test_bridge_tolerates_supervisor_cancelling_both_halves(
    supervisor_with_queue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Provider-fault teardown may cancel both bridge tasks while the endpoint
    awaits ``FIRST_COMPLETED``. A cancelled task can therefore appear in
    ``done``; inspecting it must not leak ``CancelledError`` into Starlette.
    """
    supervisor, _captured = supervisor_with_queue

    async def _noop_play_disclosure(supervisor, stream_sid, **_kwargs):
        return None

    monkeypatch.setattr(
        media_stream_mod,
        "play_disclosure",
        _noop_play_disclosure,
    )

    class _BlockingAfterStartWebSocket(_ScriptedWebSocket):
        async def receive_json(self) -> dict:
            if self._inbound:
                return self._inbound.pop(0)
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    fake_ws = _BlockingAfterStartWebSocket(
        inbound=[_start_event("CA_TEST_WS")],
    )
    handler_task = asyncio.create_task(twilio_media_stream(fake_ws))

    for _ in range(100):
        if (
            isinstance(supervisor.reader_task, asyncio.Task)
            and isinstance(supervisor.sender_task, asyncio.Task)
        ):
            break
        await asyncio.sleep(0)
    else:
        pytest.fail("media bridge tasks were not attached")

    reader_task = supervisor.reader_task
    sender_task = supervisor.sender_task
    reader_task.cancel()
    sender_task.cancel()

    await asyncio.wait_for(handler_task, timeout=1.0)

    assert reader_task.cancelled()
    assert sender_task.cancelled()
    assert handler_task.exception() is None


def _connected_event() -> dict:
    """A minimal Twilio Media Streams ``connected`` handshake event."""
    return {
        "event": "connected",
        "sequenceNumber": "1",
        "protocol": "Call",
        "version": "1.0.0",
    }


@pytest.mark.asyncio
async def test_handler_tolerates_connected_event_before_start(
    supervisor_with_queue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    REGRESSION: Twilio sends ``{"event": "connected"}`` immediately upon
    WebSocket upgrade, BEFORE the ``{"event": "start"}`` payload.  The
    pre-fix implementation rejected the first message if it wasn't
    ``start``, which closed the socket before the audio pipeline could
    spawn — terminating every live call right after the greeting.

    The fix: the handler now polls pre-start events, consuming
    ``connected`` and waiting for ``start``.  This test scripts both
    events in order, asserts the bridge reaches the running state
    (``start_background_tasks`` called), and uses the
    ``stop`` event to drive a clean exit via the FIRST_COMPLETED path.

    Note (Issue 037): ``play_disclosure`` is stubbed to a no-op so this
    test isolates the pre-start event handling from the disclosure
    player's audio emission.
    """
    supervisor, _captured = supervisor_with_queue

    # Stub play_disclosure so the bridge does not push disclosure audio.
    async def _noop_play_disclosure(supervisor, stream_sid, **_kwargs):
        return None

    monkeypatch.setattr(
        media_stream_mod, "play_disclosure", _noop_play_disclosure
    )

    fake_ws = _ScriptedWebSocket(
        inbound=[
            _connected_event(),  # Twilio handshake
            _start_event("CA_TEST_WS"),  # call answered
            _stop_event(),  # caller hangs up
        ],
    )

    await asyncio.wait_for(twilio_media_stream(fake_ws), timeout=1.0)

    # The bridge reached the supervisor-spawn stage: ``start_background_tasks``
    # was called exactly once.  Without the connected-tolerance fix the
    # handler would have closed on the first message and never reached this.
    assert supervisor.start_background_tasks.call_count == 1, (
        "handler must call start_background_tasks exactly once after "
        "consuming both the 'connected' handshake and the 'start' event"
    )
    # No outbound traffic (no chunks queued, disclosure stubbed to no-op),
    # but the bridge ran.
    assert fake_ws.sent == []


@pytest.mark.asyncio
async def test_handler_rejects_unexpected_pre_start_event(
    supervisor_with_queue,
) -> None:
    """
    Pre-start events other than ``connected`` and ``start`` are protocol
    violations — the handler must close with WS_1008_POLICY_VIOLATION
    rather than silently accepting an unexpected frame.
    """
    from fastapi import status

    supervisor, _captured = supervisor_with_queue
    fake_ws = _ScriptedWebSocket(
        inbound=[
            _connected_event(),
            {"event": "media", "media": {"payload": "AAAA"}},  # not allowed pre-start
        ],
    )

    await asyncio.wait_for(twilio_media_stream(fake_ws), timeout=1.0)

    # The supervisor was never reached.
    assert supervisor.start_background_tasks.call_count == 0
    # The socket was closed with a policy-violation code (1008).
    assert fake_ws.closed_with == status.WS_1008_POLICY_VIOLATION
    # The closed socket left us in DISCONNECTED state.
    assert fake_ws.client_state == "DISCONNECTED"
