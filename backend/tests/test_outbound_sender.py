"""
backend/tests/test_outbound_sender.py — Issue 036 unit tests.

Test plan
─────────
The outbound sender is the sole writer to the Twilio WebSocket on the
server side.  These tests use a fake WebSocket (a tiny record-and-fail
stub) so we exercise the real sender coroutine against realistic inputs
without standing up a real Twilio session.

  Generation + is_speaking
  G1  First frame of a new generation flips ``is_speaking`` to True.
  G2  Items below the current generation are dropped (stale check).
  G3  Bumping the generation mid-stream resets the repacketizer so a
      partial from the previous generation does not bleed into the new
      turn's first chunk.

  Mark cadence + end-of-turn
  M1  Periodic marks fire every ``_MARK_EVERY_N_FRAMES`` frames.
  M2  The end-of-turn ``None`` chunk emits a final mark whose name
      encodes ``:final=true``.
  M3  Mark names round-trip through ``parse_mark_name``.

  Lifecycle
  L1  ``shutdown_event`` set before any item exits the loop cleanly.
  L2  WebSocket send failure exits the loop without raising.
  L3  Sender never touches the inbound queue (sole-writer guarantee).
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any
from unittest.mock import MagicMock

import pytest

from backend.audio.outbound_sender import (
    _MARK_EVERY_N_FRAMES,
    parse_mark_name,
    run_outbound_sender,
)
from backend.audio.repacketizer import MULAW_FRAME_BYTES
from backend.call_supervisor import CallSupervisor
from backend.concurrency import ConcurrencyManager


# ─── Test helpers ─────────────────────────────────────────────────────────────


class FakeWebSocket:
    """
    Minimal async-WS stub that records every ``send_json`` payload and
    can be programmed to raise on the next call.

    Mirrors the subset of the starlette WebSocket interface the sender
    actually touches (``send_json`` and the ``client_state`` attribute).
    """

    def __init__(self, *, fail_after: int | None = None) -> None:
        self.sent: list[dict[str, Any]] = []
        self._fail_after = fail_after
        self.client_state = "CONNECTED"

    async def send_json(self, payload: Any) -> None:
        if self._fail_after is not None and len(self.sent) >= self._fail_after:
            self.client_state = "DISCONNECTED"
            from starlette.websockets import WebSocketDisconnect

            raise WebSocketDisconnect(code=1006)
        self.sent.append(payload)

    def frames(self) -> list[str]:
        """Convenience: just the payloads, in order, for media events."""
        return [
            evt["media"]["payload"]
            for evt in self.sent
            if evt.get("event") == "media"
        ]

    def marks(self) -> list[dict[str, Any]]:
        """Convenience: just the mark events, in order."""
        return [evt for evt in self.sent if evt.get("event") == "mark"]


def _make_supervisor() -> CallSupervisor:
    """
    Build a real CallSupervisor with stub pool + concurrency manager.

    Tests don't exercise the DB / hangup paths so we hand the supervisor
    MagicMocks for those fields.
    """
    manager = ConcurrencyManager(capacity=1)
    pool = MagicMock()
    return CallSupervisor(
        lead_id=1,
        attempt_id=1,
        call_sid="CA_TEST_OUT",
        language="English",
        concurrency_manager=manager,
        pool=pool,
    )


def _make_frame(seed: int) -> bytes:
    """Deterministic 160-byte μ-law-looking frame."""
    return bytes((seed + i) & 0xFF for i in range(MULAW_FRAME_BYTES))


async def _run_until_idle(
    supervisor: CallSupervisor,
    websocket: FakeWebSocket,
    *,
    stream_sid: str = "MZ_TEST_OUT",
) -> None:
    """
    Run the sender until it exits.

    The sender's main loop polls the queue with a bounded timeout (to
    react to shutdown_event).  Tests must therefore set shutdown_event
    after queueing all items — otherwise the sender polls forever.

    This helper exists to centralize that lifecycle so every test stays
    symmetric: push items, start task, signal shutdown, await task.
    """
    task = asyncio.create_task(
        run_outbound_sender(supervisor, websocket, stream_sid),
    )
    # Let the task run for one poll-interval so it processes the
    # already-queued items before we signal shutdown.
    await asyncio.sleep(0.2)
    supervisor.shutdown_event.set()
    await asyncio.wait_for(task, timeout=2.0)


# ─── G1 — first frame of new generation flips is_speaking ────────────────────


class TestGenerationTransitions:
    """Generation transitions + is_speaking bookkeeping."""

    @pytest.mark.asyncio
    async def test_first_frame_of_new_generation_flips_is_speaking(self) -> None:
        """Pushing (1, chunk) on an empty queue sends frames and sets is_speaking."""
        supervisor = _make_supervisor()
        ws = FakeWebSocket()

        # Set generation to 1 (orchestrator bumped from 0 → 1).
        supervisor.generation_counter = 1
        assert supervisor.is_speaking is False

        # Feed exactly one Twilio frame's worth of bytes (160 B → 1 frame),
        # then end-of-turn so the sender drains.
        await supervisor.outbound_audio_queue.put((1, _make_frame(0)))
        await supervisor.outbound_audio_queue.put((1, None))

        await _run_until_idle(supervisor, ws)

        assert supervisor.is_speaking is True
        assert len(ws.frames()) == 1

    @pytest.mark.asyncio
    async def test_stale_items_are_dropped_silently(self) -> None:
        """
        Items with generation < supervisor.generation_counter are stale
        (orchestrator bumped past them on barge-in) — drop without sending
        any frames.
        """
        supervisor = _make_supervisor()
        ws = FakeWebSocket()

        # Orchestrator bumped to gen=5; the queue still holds a stale gen=1 chunk.
        supervisor.generation_counter = 5
        await supervisor.outbound_audio_queue.put((1, _make_frame(0)))
        await supervisor.outbound_audio_queue.put((1, _make_frame(1)))
        # End-of-turn so the sender exits.
        await supervisor.outbound_audio_queue.put((5, None))

        await _run_until_idle(supervisor, ws)

        # Only the gen=5 end-of-turn mark was emitted — the gen=1 chunks
        # were stale and dropped.  ``is_speaking`` is True here because
        # the gen=5 turn's first (and only) mark was emitted; only the
        # inbound reader's mark-echo dispatch flips it back to False.
        assert len(ws.frames()) == 0
        assert len(ws.marks()) == 1
        assert ws.marks()[0]["mark"]["name"] == "gen=5:offset=0:final=true"
        assert supervisor.is_speaking is True

    @pytest.mark.asyncio
    async def test_new_generation_resets_repacketizer_partial(self) -> None:
        """
        When a new generation arrives mid-stream, leftover partial bytes
        from the previous generation's repacketizer are discarded so
        they don't bleed into the new turn's first chunk.
        """
        supervisor = _make_supervisor()
        ws = FakeWebSocket()

        # Generation 1: feed a partial that does NOT yield a full 160-byte frame
        # (e.g. 80 bytes) — repacketizer retains them.
        supervisor.generation_counter = 1
        await supervisor.outbound_audio_queue.put((1, bytes(range(80))))

        # Orchestrator bumps to gen=2 mid-stream (barge-in).
        supervisor.generation_counter = 2
        # Now feed one full frame in gen=2 + end-of-turn so the sender exits.
        await supervisor.outbound_audio_queue.put((2, _make_frame(0)))
        await supervisor.outbound_audio_queue.put((2, None))

        await _run_until_idle(supervisor, ws)

        # Exactly one frame sent: the gen=2 frame.  The 80 partial bytes
        # from gen=1 must have been discarded, not prepended to the new
        # frame (which would have produced 240 bytes — out of spec).
        assert len(ws.frames()) == 1
        decoded = base64.b64decode(ws.frames()[0])
        assert len(decoded) == MULAW_FRAME_BYTES
        assert decoded == _make_frame(0)


# ─── M1 + M2 — mark cadence + end-of-turn ──────────────────────────────────


class TestMarks:
    """Mark emission cadence + end-of-turn final mark."""

    @pytest.mark.asyncio
    async def test_periodic_mark_every_n_frames(self) -> None:
        """
        After every ``_MARK_EVERY_N_FRAMES`` frames, the sender emits a
        periodic mark whose name encodes the current offset in ms.
        """
        supervisor = _make_supervisor()
        ws = FakeWebSocket()
        supervisor.generation_counter = 1

        # Feed enough bytes for ``_MARK_EVERY_N_FRAMES`` + 1 frames in a
        # single chunk (zero-copy via the repacketizer's feed).
        total_bytes = (_MARK_EVERY_N_FRAMES + 1) * MULAW_FRAME_BYTES
        await supervisor.outbound_audio_queue.put((1, bytes(total_bytes)))

        # End-of-turn so the sender exits cleanly.
        await supervisor.outbound_audio_queue.put((1, None))

        await _run_until_idle(supervisor, ws)

        frames = ws.frames()
        marks = ws.marks()
        assert len(frames) == _MARK_EVERY_N_FRAMES + 1

        # One periodic mark (after N frames) + one final mark.
        assert len(marks) == 2

        periodic = marks[0]["mark"]["name"]
        assert periodic == f"gen=1:offset={_MARK_EVERY_N_FRAMES * 20}"

        final = marks[1]["mark"]["name"]
        assert final == f"gen=1:offset={(_MARK_EVERY_N_FRAMES + 1) * 20}:final=true"

    @pytest.mark.asyncio
    async def test_short_turn_emits_only_final_mark(self) -> None:
        """
        A turn shorter than the periodic-mark threshold (N frames)
        emits no periodic marks — only the end-of-turn final mark.
        """
        supervisor = _make_supervisor()
        ws = FakeWebSocket()
        supervisor.generation_counter = 1

        # Feed 5 frames (< N=20), then end-of-turn.
        await supervisor.outbound_audio_queue.put((1, bytes(5 * MULAW_FRAME_BYTES)))
        await supervisor.outbound_audio_queue.put((1, None))

        await _run_until_idle(supervisor, ws)

        assert len(ws.frames()) == 5
        marks = ws.marks()
        assert len(marks) == 1
        assert marks[0]["mark"]["name"] == "gen=1:offset=100:final=true"


# ─── M3 — parse_mark_name round-trip ──────────────────────────────────────


class TestParseMarkName:
    """encode_mark_name is private; parse_mark_name is the public surface."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("gen=1:offset=400", (1, 400, False)),
            ("gen=7:offset=12345:final=true", (7, 12345, True)),
            ("gen=0:offset=0", (0, 0, False)),
            ("gen=42:offset=9999:final=false", (42, 9999, False)),
        ],
    )
    def test_parse_valid_names(
        self, name: str, expected: tuple[int, int, bool]
    ) -> None:
        assert parse_mark_name(name) == expected

    @pytest.mark.parametrize(
        "name",
        [
            "",
            "not-a-mark",
            "gen=abc:offset=400",
            "gen=1",
            "offset=400",
            "gen=:offset=400",
            "gen=1:offset=",
            "gen=1:offset=400:final=yes",
        ],
    )
    def test_parse_invalid_names_returns_none(self, name: str) -> None:
        assert parse_mark_name(name) is None


# ─── L1 + L2 — lifecycle ────────────────────────────────────────────────────


class TestLifecycle:
    """Shutdown + WS disconnect paths."""

    @pytest.mark.asyncio
    async def test_shutdown_event_exits_cleanly(self) -> None:
        """
        Setting ``supervisor.shutdown_event`` makes the loop exit
        promptly (within the poll interval), even with items still
        queued behind it.
        """
        supervisor = _make_supervisor()
        ws = FakeWebSocket()
        supervisor.generation_counter = 1

        # Queue one item but signal shutdown BEFORE the task starts.
        # The first iteration of the loop's shutdown check fires before
        # any item is popped, so the queue item is left untouched.
        supervisor.outbound_audio_queue.put_nowait((1, _make_frame(0)))
        supervisor.shutdown_event.set()

        task = asyncio.create_task(
            run_outbound_sender(supervisor, ws, "MZ_TEST_OUT"),
        )
        await asyncio.wait_for(task, timeout=2.0)

        # Shutdown won the race: the queued item was NOT emitted.
        assert ws.sent == []
        # …and it is still in the queue for inspection by teardown.
        assert supervisor.outbound_audio_queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_ws_disconnect_exits_silently(self) -> None:
        """
        A WS-side failure on send_json causes the sender to log + return
        without raising.  The exception does NOT propagate out of
        ``run_outbound_sender`` — callers (media_stream.gather) depend
        on this.
        """
        supervisor = _make_supervisor()
        ws = FakeWebSocket(fail_after=1)  # raise after one send
        supervisor.generation_counter = 1

        await supervisor.outbound_audio_queue.put((1, bytes(2 * MULAW_FRAME_BYTES)))

        task = asyncio.create_task(
            run_outbound_sender(supervisor, ws, "MZ_TEST_OUT"),
        )
        # Must NOT raise — sender swallows the disconnect.
        await asyncio.wait_for(task, timeout=2.0)
        assert task.exception() is None

    @pytest.mark.asyncio
    async def test_sender_does_not_touch_inbound_queue(self) -> None:
        """
        Sole-writer guarantee: the outbound sender MUST never call
        ``inbound_audio_queue.put_nowait`` or similar.  The inbound
        queue is the reader's exclusive hand-off to the STT task.
        """
        supervisor = _make_supervisor()
        ws = FakeWebSocket()
        supervisor.generation_counter = 1

        # Sentinel attribute — if the sender ever touches inbound_audio_queue,
        # this attribute gets read and the test will detect it.
        inbound_touch_count = 0
        original_queue = supervisor.inbound_audio_queue
        sentinel_queue = MagicMock(wraps=original_queue)

        def _track_put_nowait(item: Any) -> None:
            nonlocal inbound_touch_count
            inbound_touch_count += 1
            original_queue.put_nowait(item)

        sentinel_queue.put_nowait.side_effect = _track_put_nowait
        supervisor.inbound_audio_queue = sentinel_queue

        await supervisor.outbound_audio_queue.put((1, _make_frame(0)))
        await supervisor.outbound_audio_queue.put((1, None))

        await _run_until_idle(supervisor, ws)

        # Outbound sender is sole-writer to WS; it never writes inbound.
        assert inbound_touch_count == 0


# ─── End-to-end feed-through sanity ────────────────────────────────────────


class TestEndToEnd:
    """A full feed-through produces the expected WS message shape."""

    @pytest.mark.asyncio
    async def test_media_event_shape_matches_twilio_contract(self) -> None:
        """
        Each media event sent on the WS has the exact Twilio contract
        shape: ``event``, ``streamSid``, ``media.payload`` (base64).
        """
        supervisor = _make_supervisor()
        ws = FakeWebSocket()
        supervisor.generation_counter = 1

        await supervisor.outbound_audio_queue.put((1, _make_frame(0)))
        await supervisor.outbound_audio_queue.put((1, None))

        await _run_until_idle(supervisor, ws)

        # Find the media event.
        media_events = [e for e in ws.sent if e.get("event") == "media"]
        assert len(media_events) == 1
        evt = media_events[0]
        assert evt["event"] == "media"
        assert evt["streamSid"] == "MZ_TEST_OUT"
        assert "media" in evt
        assert "payload" in evt["media"]
        # Payload must round-trip as the original 160 bytes.
        assert base64.b64decode(evt["media"]["payload"]) == _make_frame(0)