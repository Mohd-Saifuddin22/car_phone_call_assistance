"""
backend/audio/outbound_sender.py — Twilio Media Streams outbound sender (Issue 036).

Bridges the conversation driver's TTS chunks (typically streamed from Sarvam
over its WebSocket) onto the Twilio Media Streams WebSocket as uniform
20 ms μ-law frames. Runs concurrently with the inbound reader as the other
half of the bidirectional media-stream bridge.

Sole-writer guarantee
─────────────────────
This coroutine is the ONLY writer to the Twilio WebSocket on the server
side.  The inbound reader (in ``backend.api.media_stream``) never sends
anything back — it only reads media and mark events from Twilio.  All
outbound messages are produced here.

Lifecycle
─────────
* Started by the media-stream handler after the ``start`` event arrives,
  via ``asyncio.gather(reader_task, run_outbound_sender(...))``.
* Exits when ANY of these is true:
    - ``supervisor.shutdown_event`` is set (graceful teardown).
    - The WebSocket's send raises (peer disconnected, e.g. on ``stop``).
    - The coroutine returns naturally (rare — no shutdown_event and
      the WS stays open forever).

Generation-counter contract
───────────────────────────
Each item on ``supervisor.outbound_audio_queue`` is ``(generation, chunk)``.
* If ``generation < supervisor.generation_counter``, the item is stale
  (queued before the orchestrator bumped the counter on barge-in) — drop it
  and reset the repacketizer to discard any leftover partial bytes.
* If ``generation > last_seen_generation``, this is a new generation in
  flight.  Reset the repacketizer (so partial bytes from the previous
  generation do not bleed into the new turn), flip ``is_speaking = True``
  on the supervisor, and reset the per-generation frame counter.
* ``None`` chunk is the end-of-turn signal: flush the repacketizer
  (yielding the silence-padded trailing frame) and emit a final mark.

Mark cadence
────────────
A ``mark`` event is sent after every ``_MARK_EVERY_N_FRAMES`` frames
(~400 ms at 20 ms / frame) AND once at the end of every turn (when the
end-of-turn ``None`` is received).  The mark name encodes ``gen`` and
``offset_ms`` (and ``final=true`` for the end-of-turn one) so the inbound
reader can recover played duration and turn-completion state.

Pacing
──────
We send frames as fast as they come off the repacketizer — Twilio owns the
playback buffer and absorbs the cadence.  Adding artificial pacing here
would only add latency without making playback more correct.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from starlette.websockets import WebSocketDisconnect, WebSocketState

from backend.audio.repacketizer import MULAW_FRAME_MS, OutboundRepacketizer

if TYPE_CHECKING:
    from backend.call_supervisor import CallSupervisor

__all__ = ["run_outbound_sender", "parse_mark_name", "ClearPlayback"]

log = logging.getLogger(__name__)


# ─── Outbound control commands ───────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ClearPlayback:
    """
    Priority control command telling the sole outbound WebSocket writer to
    flush Twilio's buffered playback (barge-in, Issue 042).

    Delivered on ``supervisor.outbound_control_queue`` — a **dedicated
    priority channel** the sender drains before pulling audio and after every
    send, so the clear is never stuck behind queued (now-stale) audio.  It is
    also tolerated on ``outbound_audio_queue`` (union-typed) as a defensive
    fallback.  The sender emits ``{"event": "clear", "streamSid": <active
    sid>}`` using its own stored stream SID — the barge-in stop sequence and
    the Deepgram listener never touch the socket directly (single-writer
    invariant).

    ``generation`` is the post-bump counter value (log correlation only).
    ``ack`` is an optional future the sender resolves once the clear has
    actually been written — ``True`` on success, ``False`` on send failure —
    so the barge-in stop sequence can confirm the clear went out (with a
    bounded wait) instead of assuming a queued command was sent.
    """

    generation: int
    ack: "asyncio.Future[bool] | None" = field(default=None, compare=False)


# ─── Tunables ────────────────────────────────────────────────────────────────

#: Emit a periodic mark every N frames.  At 20 ms / frame this is ~400 ms.
#: The cadence is short enough that a stale mark is bounded in how
#: misleading it can be, and long enough that the mark stream is not
#: dominated by overhead.
_MARK_EVERY_N_FRAMES: int = 20


# ─── Mark-name encoding ─────────────────────────────────────────────────────

#: Regex used to parse ``gen=<int>:offset=<int>`` out of an inbound mark
#: echo.  ``final=true`` is optional — if present, the reader sets
#: ``is_speaking = False``.  Trailing fields after offset (e.g. ``:final=true``)
#: are tolerated and ignored by the reader.
_MARK_NAME_RE = re.compile(
    r"^gen=(?P<gen>\d+):offset=(?P<offset>\d+)(?::final=(?P<final>true|false))?$"
)


def _encode_mark_name(generation: int, offset_ms: int, *, final: bool) -> str:
    """
    Build the outbound ``mark.name`` string.

    Format: ``gen=<int>:offset=<int>[:final=true]``.  Round-trippable
    through ``parse_mark_name`` so the inbound reader can extract gen
    and offset to update ``played_duration_ms`` and ``is_speaking``.
    """
    if final:
        return f"gen={generation}:offset={offset_ms}:final=true"
    return f"gen={generation}:offset={offset_ms}"


def parse_mark_name(name: str) -> tuple[int, int, bool] | None:
    """
    Parse an inbound mark echo back into ``(generation, offset_ms, is_final)``.

    Returns ``None`` if the name doesn't match the expected format.  A
    non-matching mark is logged and ignored; the reader never raises on
    an unexpected shape.

    Exposed publicly so ``backend.api.media_stream`` (and its tests) can
    share one implementation.
    """
    match = _MARK_NAME_RE.match(name)
    if match is None:
        return None
    return (
        int(match.group("gen")),
        int(match.group("offset")),
        match.group("final") == "true",
    )


# ─── Send helpers ───────────────────────────────────────────────────────────


async def _send_media(websocket, stream_sid: str, frame_b64: str) -> None:
    """
    Send one outbound Twilio media frame.

    Raises on a closed socket — callers must handle the disconnect
    (the sender loop converts these to a graceful exit).
    """
    await websocket.send_json(
        {
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": frame_b64},
        }
    )


async def _send_mark(
    websocket,
    stream_sid: str,
    generation: int,
    offset_ms: int,
    *,
    is_final: bool,
) -> None:
    """Emit a single ``mark`` event with the encoded name."""
    await websocket.send_json(
        {
            "event": "mark",
            "streamSid": stream_sid,
            "mark": {"name": _encode_mark_name(generation, offset_ms, final=is_final)},
        }
    )


async def _send_clear(websocket, stream_sid: str) -> None:
    """
    Emit the Twilio ``clear`` event to flush the buffered outbound playback.

    Shape is exactly ``{"event": "clear", "streamSid": <active sid>}`` per the
    Twilio Media Streams contract.  After a clear, Twilio drops every media
    frame it has buffered but not yet played, and emits any pending ``mark``
    echoes — those carry the pre-barge-in generation, so the inbound reader's
    stale-generation guard ignores them.

    Raises on a closed socket — the sender loop converts that to a graceful
    exit like any other send failure.
    """
    await websocket.send_json({"event": "clear", "streamSid": stream_sid})


# ─── Main entry point ──────────────────────────────────────────────────────


async def run_outbound_sender(
    supervisor: "CallSupervisor",
    websocket,
    stream_sid: str,
) -> None:
    """
    Long-lived per-call task that writes Riya's TTS audio to the Twilio WS.

    Parameters
    ----------
    supervisor : CallSupervisor
        The live supervisor for this call.  Provides ``outbound_audio_queue``,
        ``shutdown_event``, ``generation_counter``, and ``is_speaking``.
    websocket : starlette.websockets.WebSocket
        The same WebSocket the inbound reader is consuming.  This coroutine
        is its sole writer.
    stream_sid : str
        The ``streamSid`` from the ``start`` event.  Required on every
        outbound media/mark message per the Twilio Media Streams contract.

    Exits when
    ----------
    * ``supervisor.shutdown_event`` becomes set (graceful teardown).
    * Any ``send_json`` call raises (peer disconnected — Twilio sent
      ``stop`` or the call ended; we treat both identically).
    """
    repacketizer = OutboundRepacketizer()
    last_seen_generation: int = 0
    frames_emitted_in_gen: int = 0

    # Dedicated priority control channel (Issue 042).  ``getattr`` keeps the
    # sender usable against minimal stub supervisors in unit tests that never
    # exercise barge-in.
    #
    # CRITICAL: only an ``asyncio.Queue`` (or None) is safe.  ``MagicMock``
    # has a non-raising ``get_nowait`` that would spin forever, allocating
    # objects until the kernel kills Python.  Any other type is treated as
    # absent — the sender continues polling audio but never enters a drain
    # loop on it.  We also ensure ``supervisor.outbound_audio_queue`` is a real
    # queue for the same reason (the await on a non-queue would raise).
    raw_control = getattr(supervisor, "outbound_control_queue", None)
    if raw_control is None or isinstance(raw_control, asyncio.Queue):
        control_queue = raw_control
    else:
        log.error(
            "outbound_sender: CallSid=%s outbound_control_queue is %s, "
            "expected asyncio.Queue — barge-in clear path disabled for "
            "this run.",
            supervisor.call_sid,
            type(raw_control).__name__,
        )
        control_queue = None
    raw_audio = getattr(supervisor, "outbound_audio_queue", None)
    if raw_audio is not None and not isinstance(raw_audio, asyncio.Queue):
        log.error(
            "outbound_sender: CallSid=%s outbound_audio_queue is %s, "
            "expected asyncio.Queue — sender cannot proceed.",
            supervisor.call_sid,
            type(raw_audio).__name__,
        )
        return

    # Missing audio queue (no upstream pipeline ever connected).  There
    # is nothing for the sender to forward and no busy-loop to spin in —
    # exit cleanly so the bridge can wind down via the supervisor's
    # teardown path.  The control channel is still honoured if it ever
    # receives a ClearPlayback (e.g. an out-of-band teardown request);
    # we await it with the same concurrent waiter race below and never
    # loop on ``asyncio.sleep(0)``.
    audio_available = raw_audio is not None

    log.info(
        "outbound_sender: starting CallSid=%s streamSid=%s audio=%s",
        supervisor.call_sid,
        stream_sid,
        audio_available,
    )

    def _try_get_nowait(queue: Any) -> Any:
        """
        Non-blocking ``Queue.get`` that returns ``None`` on empty / missing.

        Used for the fast-path top-of-loop check so a ``ClearPlayback`` or
        an audio frame is consumed synchronously in the same event-loop
        iteration as the source's enqueue — critical for the media-stream
        test (pre-loaded events on the audio queue are processed before
        the bridge can cancel the sender) and for the concurrent waiter
        race test (a ``ClearPlayback`` enqueued on the control channel is
        observed on the very next iteration without polling).
        """
        if queue is None:
            return None
        try:
            return queue.get_nowait()
        except (asyncio.QueueEmpty, AttributeError, TypeError):
            return None

    # ── Clear send + stale-guard helpers ─────────────────────────────────
    async def _send_clear_cmd(cmd: ClearPlayback) -> None:
        """
        Write the Twilio ``clear`` for ``cmd`` and resolve its ack future.

        On success the ack resolves ``True``; on any send failure it resolves
        ``False`` and the exception is re-raised so the sender's outer handler
        can wind the bridge down.  Either way the barge-in stop sequence's
        bounded ack-wait unblocks — it never hangs on an unsent clear.
        """
        try:
            await _send_clear(websocket, stream_sid)
        except BaseException:
            if cmd.ack is not None and not cmd.ack.done():
                cmd.ack.set_result(False)
            raise
        if cmd.ack is not None and not cmd.ack.done():
            cmd.ack.set_result(True)
        log.info(
            "outbound_sender: cleared Twilio playback buffer CallSid=%s gen=%d.",
            supervisor.call_sid,
            cmd.generation,
        )

    async def _send_one_clear_now(cmd: ClearPlayback) -> None:
        """
        Process a single ``ClearPlayback`` already dequeued from the control
        queue: send it (which resolves its ack) and reset the per-generation
        repacketizer + counters.  Bounded single-step: never loops, never
        recurses.  Malformed commands are logged + discarded.
        """
        nonlocal repacketizer, frames_emitted_in_gen
        if not isinstance(cmd, ClearPlayback):
            log.warning(
                "outbound_sender: CallSid=%s control queue got malformed "
                "command %s — discarding (bounded single-step).",
                supervisor.call_sid,
                type(cmd).__name__,
            )
            return
        # Defensive stale-clear guard: if the supervisor's generation has
        # already moved past this command's generation, drop the clear
        # entirely (mark its ack False if requested) so an old clear can
        # never reach the wire and reset a newer-generation playback
        # buffer.  The supervisor's own media_path_broken guard further
        # below adds an unconditional drop on top of this.
        if cmd.generation < supervisor.generation_counter or getattr(
            supervisor, "media_path_broken", False
        ):
            log.warning(
                "outbound_sender: CallSid=%s discarding stale/stale-clear "
                "gen=%d (current=%d, media_path_broken=%s).",
                supervisor.call_sid,
                cmd.generation,
                supervisor.generation_counter,
                getattr(supervisor, "media_path_broken", False),
            )
            if cmd.ack is not None and not cmd.ack.done():
                cmd.ack.set_result(False)
            return
        await _send_clear_cmd(cmd)
        repacketizer = OutboundRepacketizer()
        frames_emitted_in_gen = 0
        supervisor.sent_duration_ms = 0

    try:
        while True:
            # ── Shutdown check FIRST (top of loop) ───────────────────────
            # Done BEFORE any queue interaction so a pre-set shutdown event
            # never touches the audio/control queues (which would lose the
            # queued audio item — ``test_shutdown_event_exits_cleanly``
            # asserts ``qsize() == 1`` after a pre-set shutdown exit).
            if supervisor.shutdown_event.is_set():
                log.info(
                    "outbound_sender: shutdown_event set, exiting CallSid=%s.",
                    supervisor.call_sid,
                )
                return

            # ── Missing audio queue: no busy loop, no waiting ──────────────
            # Honour any already-queued control commands once (so a
            # teardown-side ClearPlayback still flushes), then exit cleanly.
            # We never enter the concurrent waiter race below because
            # ``raw_audio`` is None — there is nothing to wait for and
            # looping on ``asyncio.sleep(0)`` would spin.
            if raw_audio is None:
                control_cmd = _try_get_nowait(control_queue)
                if control_cmd is not None:
                    await _send_one_clear_now(control_cmd)
                log.info(
                    "outbound_sender: no audio queue CallSid=%s — exiting.",
                    supervisor.call_sid,
                )
                return

            # ── FAST PATH: try non-blocking gets in priority order ─────────
            # Control wins over audio (per Issue 042 barge-in priority).
            # Audio is consumed here (not just peeked) so the media-stream
            # test's pre-loaded audio is processed synchronously in the
            # same event-loop iteration BEFORE the bridge can cancel us.
            control_cmd = _try_get_nowait(control_queue)
            audio_item = _try_get_nowait(raw_audio) if raw_audio is not None else None

            if control_cmd is not None:
                await _send_one_clear_now(control_cmd)
                # If audio was ALSO ready (both-ready case), defer it to the
                # next iteration so it passes through the generation
                # staleness check — never put it back at the queue tail,
                # which could reorder audio.
                if audio_item is not None:
                    item = audio_item
                    # fall through to the audio processing below
                else:
                    continue
            elif audio_item is not None:
                item = audio_item
            else:
                # ── Both queues empty: CONCURRENT WAITER RACE ──────────────
                # We create real ``asyncio.Task`` waiters for control, audio
                # and shutdown, then race them with ``asyncio.wait(
                # FIRST_COMPLETED)``.  ``asyncio.wait`` yields to the event
                # loop and the waiter tasks resolve naturally — no polling,
                # no separate coroutine bodies, no nested timeouts.
                #
                # ``asyncio.create_task`` (not ``ensure_future``) is used so
                # the task names are traceable.  The waiters are stored in a
                # dict keyed by tag so the winner can be identified.
                waiters: dict[str, "asyncio.Task[Any]"] = {}
                if control_queue is not None:
                    waiters["control"] = asyncio.create_task(
                        control_queue.get(),
                        name=f"sender-wait-control-{supervisor.call_sid}",
                    )
                if raw_audio is not None:
                    waiters["audio"] = asyncio.create_task(
                        raw_audio.get(),
                        name=f"sender-wait-audio-{supervisor.call_sid}",
                    )
                waiters["shutdown"] = asyncio.create_task(
                    supervisor.shutdown_event.wait(),
                    name=f"sender-wait-shutdown-{supervisor.call_sid}",
                )

                # ``asyncio.wait(FIRST_COMPLETED)`` returns a snapshot of
                # the ``done`` set at the moment it wakes — but other
                # waiters may complete between the snapshot being taken and
                # our cleanup running.  Critically, those completing waiters
                # have ALREADY dequeued their queue item, so cancelling
                # them post-hoc would silently drop the dequeued item (audio
                # or clear).  The correct pattern: after the wait returns,
                # cancel only tasks that are NOT already done, then collect
                # results from ALL waiters via ``asyncio.gather`` —
                # ``return_exceptions=True`` surfaces ``CancelledError`` for
                # cancelled waiters and the actual dequeued value for done
                # ones.  No ``task.exception()`` on a just-cancelled task
                # (which is invalid).
                try:
                    await asyncio.wait(
                        list(waiters.values()),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                except BaseException:
                    # Bridge cancelled us (or any other BaseException).
                    # Apply the same cancel-and-gather cleanup below, then
                    # re-raise.  We do NOT call ``task.exception()`` here —
                    # we always ``await`` the task so its cancellation is
                    # observed properly.
                    for t in waiters.values():
                        if not t.done():
                            t.cancel()
                    try:
                        await asyncio.gather(
                            *waiters.values(),
                            return_exceptions=True,
                        )
                    except BaseException:
                        pass
                    raise

                # Cancel only the still-pending waiters.  Tasks that
                # completed after the ``asyncio.wait`` snapshot are
                # already done (``t.done()`` is True) — cancelling them
                # now would silently drop the dequeued item.  Tasks that
                # never completed are cancelled so the queue getter is
                # released (the item, if any) is put back by the queue.
                for t in waiters.values():
                    if not t.done():
                        t.cancel()

                # ``asyncio.gather(return_exceptions=True)`` awaits every
                # waiter — done ones return their dequeued value (or
                # raise), cancelled ones return ``CancelledError``.  This
                # is the single source of truth for the race results;
                # we never inspect ``t in done`` (the stale snapshot).
                gathered = await asyncio.gather(
                    *waiters.values(),
                    return_exceptions=True,
                )

                # Build tag -> consumed-item map.  ``CancelledError`` means
                # the waiter did NOT consume an item; an ordinary value
                # means the item was dequeued and MUST be processed (it is
                # NEVER re-queued at the tail — see point 6 of the spec).
                # Any other exception is logged and treated as "no item".
                done_results: dict[str, Any] = {}
                for tag, result in zip(waiters.keys(), gathered):
                    if isinstance(result, asyncio.CancelledError):
                        # Waiter was cancelled before it could consume an
                        # item; the item (if any) is back in the queue
                        # via asyncio.Queue's getter-revocation semantics.
                        continue
                    if isinstance(result, BaseException):
                        log.debug(
                            "outbound_sender: waiter %s raised %s — "
                            "ignored.",
                            tag,
                            result,
                        )
                        continue
                    done_results[tag] = result

                # ── Resolve the winner(s) ───────────────────────────────────
                # Multiple winners can fire at the same iteration — resolve
                # in priority order: shutdown → control → audio.  If both
                # control AND audio completed in this iteration, the
                # audio result is preserved in the local ``item`` variable
                # and processed on the NEXT iteration (where the
                # generation-staleness check filters stale items).  The
                # audio is never silently dropped or re-queued at the tail.
                if "shutdown" in done_results:
                    log.info(
                        "outbound_sender: shutdown won race CallSid=%s.",
                        supervisor.call_sid,
                    )
                    return
                if "control" in done_results:
                    cmd = done_results["control"]
                    await _send_one_clear_now(cmd)
                    if "audio" in done_results:
                        item = done_results["audio"]
                        # fall through to audio processing below
                    else:
                        continue
                elif "audio" in done_results:
                    item = done_results["audio"]
                else:
                    # All waiters were cancelled or raised (e.g. bridge
                    # cancelled us mid-race).  Re-loop and try again — the
                    # shutdown check at the top will catch the bridge exit.
                    continue

            if isinstance(item, ClearPlayback):
                await _send_one_clear_now(item)
                continue

            # ── Unpack ─────────────────────────────────────────────────────
            if not isinstance(item, tuple) or len(item) != 2:
                log.error(
                    "outbound_sender: CallSid=%s got malformed item %r — "
                    "dropping and continuing.",
                    supervisor.call_sid,
                    type(item).__name__,
                )
                continue
            generation, chunk = item
            if not isinstance(generation, int):
                log.error(
                    "outbound_sender: CallSid=%s got non-int generation %r "
                    "— dropping.",
                    supervisor.call_sid,
                    generation,
                )
                continue
            if chunk is not None and not isinstance(chunk, (bytes, bytearray)):
                log.error(
                    "outbound_sender: CallSid=%s got non-bytes chunk %r "
                    "— dropping.",
                    supervisor.call_sid,
                    type(chunk).__name__,
                )
                continue

            # ── Stale check: orchestrator bumped past this generation ─────
            if generation < supervisor.generation_counter:
                if chunk is not None:
                    repacketizer = OutboundRepacketizer()
                log.debug(
                    "outbound_sender: CallSid=%s dropped stale item "
                    "gen=%d (current=%d).",
                    supervisor.call_sid,
                    generation,
                    supervisor.generation_counter,
                )
                continue

            # ── Generation transition ─────────────────────────────────────
            if generation != last_seen_generation:
                repacketizer = OutboundRepacketizer()
                frames_emitted_in_gen = 0
                last_seen_generation = generation
                supervisor.sent_duration_ms = 0
                if not supervisor.is_speaking:
                    supervisor.is_speaking = True

            # ── Feed chunk or flush on end-of-turn ────────────────────────
            if chunk is not None:
                frames = repacketizer.feed(bytes(chunk))
            else:
                frames = repacketizer.flush()

            abandoned = False
            for frame_b64 in frames:
                if generation < supervisor.generation_counter:
                    repacketizer = OutboundRepacketizer()
                    abandoned = True
                    break
                await _send_media(websocket, stream_sid, frame_b64)
                frames_emitted_in_gen += 1
                supervisor.sent_duration_ms = frames_emitted_in_gen * MULAW_FRAME_MS
                # A clear queued during that send must be the NEXT event.
                # We check the control queue non-blocking on the same coroutine
                # so a mid-frame barge-in preempts the remaining frames of
                # the in-progress chunk before any further Twilio traffic.
                if control_queue is not None:
                    try:
                        mid_cmd = control_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        mid_cmd = None
                    if mid_cmd is not None:
                        await _send_one_clear_now(mid_cmd)
                        abandoned = True
                        break
                if (
                    frames_emitted_in_gen % _MARK_EVERY_N_FRAMES == 0
                    and chunk is not None
                ):
                    if generation < supervisor.generation_counter:
                        repacketizer = OutboundRepacketizer()
                        abandoned = True
                        break
                    offset_ms = frames_emitted_in_gen * MULAW_FRAME_MS
                    await _send_mark(
                        websocket, stream_sid, generation, offset_ms,
                        is_final=False,
                    )
                    if control_queue is not None:
                        try:
                            mid_cmd = control_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            mid_cmd = None
                        if mid_cmd is not None:
                            await _send_one_clear_now(mid_cmd)
                            abandoned = True
                            break

            if (
                chunk is None
                and not abandoned
                and generation >= supervisor.generation_counter
            ):
                offset_ms = frames_emitted_in_gen * MULAW_FRAME_MS
                await _send_mark(
                    websocket, stream_sid, generation, offset_ms,
                    is_final=True,
                )
                if control_queue is not None:
                    try:
                        mid_cmd = control_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        mid_cmd = None
                    if mid_cmd is not None:
                        await _send_one_clear_now(mid_cmd)

    except Exception as exc:
        # The reader treats WebSocketDisconnect as a graceful exit (the WS
        # is already closed).  Everything else is an unexpected failure
        # — log loudly so the operator can investigate, then exit; the
        # supervisor's teardown path owns the slot/DB cleanup regardless.
        # We do NOT raise: the gather() in media_stream.py uses
        # return_exceptions=True, and a raised exception here would only
        # cancel the inbound reader.
        ws_state = getattr(websocket, "client_state", None)
        is_disconnect = (
            isinstance(exc, WebSocketDisconnect)
            or ws_state == WebSocketState.DISCONNECTED
        )
        if is_disconnect:
            log.info(
                "outbound_sender: WebSocket disconnect CallSid=%s "
                "(frames_sent_in_last_gen=%d) — exiting cleanly.",
                supervisor.call_sid,
                frames_emitted_in_gen,
            )
        else:
            log.exception(
                "outbound_sender: CallSid=%s unexpected exception — exiting.",
                supervisor.call_sid,
            )
        return