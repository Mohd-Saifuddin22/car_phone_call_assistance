"""
backend/api/media_stream.py — Twilio Media Streams bridge (Issues 034 + 036).

This module is the server side of the Twilio Media Streams WebSocket
established by ``<Connect><Stream>`` TwiML returned from
``/api/twilio/placeholder-twiml``.

Bidirectional bridge (Issue 036)
────────────────────────────────
Once the call is answered the bridge runs two concurrent coroutines on the
same WebSocket:

* **Inbound reader** — aggregates 4 × 20 ms Twilio frames into ~80 ms raw
  chunks via ``InboundRepacketizer`` and pushes them onto
  ``supervisor.inbound_audio_queue`` for the Deepgram STT task.  Also
  intercepts ``mark`` events Twilio echoes back, parses them, and updates
  ``supervisor.played_duration_ms`` / ``is_speaking``.
* **Outbound sender** — ``backend.audio.outbound_sender.run_outbound_sender``
  drains ``supervisor.outbound_audio_queue`` (TTS chunks from the
  conversation driver), repacketizes them into Twilio's 20 ms frames, and
  sends ``media`` + ``mark`` events.  It is the SOLE writer to this WebSocket.

Twilio's Media Streams contract
───────────────────────────────
Inbound events arrive in this order:

    1. ``{"event": "connected", ...}``      ← sent just after WS upgrade (optional)
    2. ``{"event": "start", "start": {...}}`` ← exactly one start event
    3. ``{"event": "media", ...}`` and / or ``{"event": "mark", ...}`` ← interleaved
    4. ``{"event": "stop", ...}``            ← exactly one stop event on graceful end

Each inbound ``media.payload`` is a base64-encoded μ-law 8 kHz mono **20 ms
frame of exactly 160 bytes**.  Deepgram Flux prefers **~80 ms (640-byte)
raw chunks** — exactly four Twilio frames — and we do that aggregation here.

Outbound events we send back to Twilio:

    * ``{"event": "media", "streamSid": "...", "media": {"payload": "..."}}``
      — one 20 ms μ-law frame per message.
    * ``{"event": "mark", "streamSid": "...", "mark": {"name": "..."}}``
      — played-position checkpoint.  Twilio echoes the same event back
      when the corresponding audio has actually been played (or when the
      buffer is cleared).  Our mark names encode ``gen=<int>:offset=<ms>``
      (and optionally ``:final=true``) so the inbound reader can recover
      generation + offset from the echo.

Why this lives in its own endpoint and not inside arrange.py
───────────────────────────────────────────────────────────
The WebSocket bridge is a long-lived, I/O-heavy piece.  Mixing it into
``arrange.py`` (which already owns the webhook fleet and DB transactions)
would force that module to import Twilio-specific asyncio plumbing and pull
in the audio repacketizer.  Keeping it standalone keeps arrange.py focused
on the call-control skeleton and lets the bridge evolve independently as
the downstream pipeline (Issue 035 Deepgram + Issue 040 greeting + Issue 041
orchestrator) lands.

Non-blocking guarantees
──────────────────────
* The inbound reader pushes every complete 640-byte chunk onto
  ``supervisor.inbound_audio_queue`` via ``put_nowait`` (no await, no
  yield) — never blocks on downstream generation.
* The outbound sender pulls from ``supervisor.outbound_audio_queue`` via
  bounded ``asyncio.wait_for`` polls so it can react to
  ``supervisor.shutdown_event`` within ~100 ms.

Lifecycle / teardown ownership
───────────────────────────────
This endpoint is a **thin forwarder**.  It does NOT own the slot release or
final teardown of the call.  When the WebSocket closes (Twilio stop event or
network drop), the bridge logs and exits; the actual slot release + DB
resolve is owned exclusively by the ``twilio_status_callback`` endpoint
(which fires when Twilio marks the call terminal).  This split is critical:
two distinct closing events (Media-Stream stop vs. Call-status completed)
must converge through exactly one teardown path to avoid double-release.

Protocol errors / orphan streams
─────────────────────────────────
* First message is not a ``start`` event → log + close.  Twilio retries
  on non-2xx close, so a malformed peer won't loop us.
* ``start.callSid`` is missing or empty → log + close.  We MUST have a
  CallSid to route audio to the right supervisor.
* ``start.streamSid`` is missing or empty → log + close.  We MUST have a
  streamSid to address outbound media + mark messages per Twilio's contract.
* ``start.callSid`` is unknown to the active-calls registry → log + close
  immediately.  The call ended before we connected (e.g., the supervisor
  was already torn down via the status callback), or the server restarted
  and the in-memory registry is empty.  Trust the startup sweep (Issue
  029) to resolve the orphaned lead via the database — we MUST NOT spin
  up a fresh supervisor here, since the slot has already been released.
* Bad base64 payload inside a ``media`` event → ``InboundRepacketizer.feed``
  raises ``ValueError``.  We catch and continue (next frame might be fine).
  Twilio's TCP layer guarantees ordering within a stream so we don't need
  to fail the whole stream over one bad frame.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from backend.audio.outbound_sender import parse_mark_name
from backend.audio.repacketizer import InboundRepacketizer
from backend.call_supervisor import get_call

if TYPE_CHECKING:
    from backend.call_supervisor import CallSupervisor

log = logging.getLogger(__name__)

__all__ = ["router", "play_disclosure"]

router = APIRouter(prefix="/api/twilio", tags=["twilio"])


# ─── Disclosure player (Issue 037) ───────────────────────────────────────────

#: Default location of the pre-rendered disclosure μ-law files (relative to
#: this module).  ``backend/audio/assets/disclosure_{en|hi}.ulaw``.  These
#: files are FROZEN — the generator lives in ``backend/scripts/
#: generate_disclosures.py`` and only the operator runs it (the assets dir
#: is not auto-overwritten at startup or in tests).
DEFAULT_DISCLOSURE_ASSETS_DIR: Path = (
    Path(__file__).resolve().parent.parent / "audio" / "assets"
)

#: How long ``play_disclosure`` waits for the final-mark echo before
#: giving up and marking the disclosure complete anyway.  Sized as the
#: file's natural playback duration + a small slack for network jitter
#: + the trailing repacketizer silence frame.  The fallback exists so a
#: lost mark echo cannot hang the call indefinitely.
_DISCLOSURE_TIMEOUT_BUFFER_S: float = 0.5

#: Poll interval for the final-mark wait.  50 ms is fine-grained enough
#: that the driver's AMD wait (2 s timeout) sees ``disclosure_complete``
#: set within ~50 ms of playback end; coarse enough that the loop is
#: invisible CPU overhead.
_DISCLOSURE_WAIT_POLL_INTERVAL_S: float = 0.05


def _disclosure_asset_path(
    supervisor: "CallSupervisor",
    *,
    assets_dir: Path | None = None,
) -> Path:
    """
    Return the absolute path to the pre-rendered disclosure file for the
    given supervisor's language.

    The two assets are frozen — generated by
    ``backend/scripts/generate_disclosures.py`` via Sarvam REST and
    stripped of the RIFF/WAVE container (see ``test_generate_disclosures.py``
    for the format-guard contract).  English → ``disclosure_en.ulaw``,
    Hinglish → ``disclosure_hi.ulaw``.  Anything else falls back to
    ``disclosure_en.ulaw`` so a misconfigured lead does not crash the
    player — the English disclosure is the safe default.
    """
    base = assets_dir if assets_dir is not None else DEFAULT_DISCLOSURE_ASSETS_DIR
    # English is the safe default for any non-Hinglish language so a
    # misconfigured lead (e.g. a future language value, a typo, or a
    # corrupted DB row) never crashes the player with a missing-asset
    # path — it picks the file we know exists.
    suffix = "hi" if supervisor.language == "Hinglish" else "en"
    return base / f"disclosure_{suffix}.ulaw"


async def play_disclosure(
    supervisor: "CallSupervisor",
    stream_sid: str,
    *,
    assets_dir: Path | None = None,
    timeout_buffer_s: float = _DISCLOSURE_TIMEOUT_BUFFER_S,
    poll_interval_s: float = _DISCLOSURE_WAIT_POLL_INTERVAL_S,
) -> None:
    """
    Play the legal-recording disclosure audio through the live Twilio
    Media Streams WebSocket (Issue 037).

    Pipeline
    ────────
    1. Load the raw headerless μ-law bytes for the supervisor's language
       from ``backend/audio/assets/disclosure_{en|hi}.ulaw``.
    2. Push the bytes as ``(generation, chunk)`` onto the supervisor's
       outbound audio queue, then push ``(generation, None)`` as the
       end-of-turn marker.  The outbound sender slices the chunk into
       20 ms μ-law frames + emits the final mark on the ``None``.
    3. Wait for the final-mark echo to come back through the inbound
       reader — ``supervisor.is_speaking`` flips from ``True`` back to
       ``False`` when the mark is observed (see ``_dispatch_mark`` in
       this module).  The wait is bounded by ``len(bytes) / 8000 +
       timeout_buffer_s`` so a lost mark cannot hang the call.
    4. Call ``supervisor.mark_disclosure_complete()`` — flips
       ``disclosure_active = False`` and sets the ``disclosure_complete``
       event so the driver can proceed past its disclosure wait.

    Generation-counter policy
    ─────────────────────────
    This coroutine does NOT bump ``supervisor.generation_counter`` —
    it uses the value as it stands.  Barge-in (Issue 042) is the
    authority on generation bumps; the disclosure is non-interruptible
    in Issue 040's design (any per-frame barge-in check fires on every
    20 ms frame and cuts Riya off mid-disclosure).

    Cancellation safety
    ───────────────────
    The wait loop races against ``supervisor.shutdown_event``.  If
    teardown fires mid-disclosure (AMD machine verdict, status-callback
    hangup), the supervisor sets ``shutdown_event`` and the player
    exits cleanly WITHOUT calling ``mark_disclosure_complete()`` — the
    call is ending, the driver is being cancelled, and a half-played
    disclosure should not be marked complete.

    ``asyncio.CancelledError`` from the handler returning is NOT
    caught: the player is a fire-and-forget task that the handler
    cancels on exit, and propagating ``CancelledError`` is the standard
    pattern for cleanly terminating such tasks.  Other exceptions are
    logged but do NOT propagate (the player is spawned as a background
    task and an unhandled exception would only show up as an "unretrieved
    exception" warning from asyncio).

    Returns
    -------
    None — this coroutine always returns cleanly (or via
    ``CancelledError`` if the handler cancels it).
    """
    asset_path = _disclosure_asset_path(supervisor, assets_dir=assets_dir)

    try:
        disclosure_bytes = asset_path.read_bytes()
    except FileNotFoundError:
        # Asset missing is a deployment bug — the frozen .ulaw files
        # should always exist in production.  Log loudly and mark the
        # disclosure complete so the driver does not hang on its wait
        # for the rest of the call.  Better to greet without disclosure
        # than to freeze the call at connect.
        log.error(
            "play_disclosure: asset not found at %s for CallSid=%s — "
            "marking disclosure complete anyway to avoid stuck call",
            asset_path,
            supervisor.call_sid,
        )
        supervisor.mark_disclosure_complete()
        return
    except OSError as exc:
        log.error(
            "play_disclosure: failed to read asset %s for CallSid=%s: %s "
            "— marking disclosure complete anyway",
            asset_path,
            supervisor.call_sid,
            exc,
        )
        supervisor.mark_disclosure_complete()
        return

    if not disclosure_bytes:
        log.error(
            "play_disclosure: asset %s is empty for CallSid=%s — "
            "marking complete immediately",
            asset_path,
            supervisor.call_sid,
        )
        supervisor.mark_disclosure_complete()
        return

    duration_s = len(disclosure_bytes) / 8000.0  # μ-law is 1 byte/sample
    timeout_s = duration_s + timeout_buffer_s
    generation = supervisor.generation_counter

    log.info(
        "play_disclosure: starting CallSid=%s language=%s "
        "size=%d bytes (~%.2fs) generation=%d",
        supervisor.call_sid,
        supervisor.language,
        len(disclosure_bytes),
        duration_s,
        generation,
    )

    # Enqueue the chunk + end-of-turn marker.  ``put_nowait`` is
    # synchronous (no await, no yield) so any enqueue failure raises
    # immediately rather than blocking the caller.
    try:
        supervisor.outbound_audio_queue.put_nowait((generation, disclosure_bytes))
        supervisor.outbound_audio_queue.put_nowait((generation, None))
    except Exception as exc:  # asyncio.QueueFull, AttributeError, etc.
        log.error(
            "play_disclosure: enqueue failed for CallSid=%s: %s — "
            "marking complete anyway",
            supervisor.call_sid,
            exc,
        )
        supervisor.mark_disclosure_complete()
        return

    # ── Manual priming of is_speaking ─────────────────────────────────────
    # The outbound sender only flips ``is_speaking = True`` when it sees
    # a NEW generation (``generation > last_seen_generation``).  The
    # disclosure is the FIRST audio on the call, so both values are 0
    # at enqueue time — the comparison is False and the sender does NOT
    # flip the flag.  Without intervention, ``is_speaking`` stays False
    # for the entire disclosure playback, so when the inbound reader
    # processes the final-mark echo (``is_final=True``) it sets
    # ``is_speaking = False`` — a no-op.  The player's wait loop never
    # observes the True → False edge and falls through to the
    # duration-timeout backstop every call.
    #
    # Priming it here lets the inbound reader's mark echo do its job:
    # once the final mark arrives, the flag flips True → False, the
    # wait loop sees the edge, and ``mark_disclosure_complete`` fires
    # naturally — well before the timeout backstop would.
    supervisor.is_speaking = True
    log.debug(
        "play_disclosure: primed is_speaking=True for CallSid=%s "
        "generation=%d (manual priming — see HITL feedback)",
        supervisor.call_sid,
        generation,
    )

    # Wait for the final-mark echo.  The outbound sender consumes our
    # chunk + None, emits frames + a final mark, the inbound reader
    # receives the mark echo and flips ``is_speaking`` from True → False.
    #
    # We must observe is_speaking going True at least once BEFORE
    # treating False as the end of playback — otherwise we'd race the
    # sender startup and incorrectly mark complete immediately if the
    # sender hasn't yet set is_speaking=True.
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    observed_speaking = False

    try:
        while True:
            if supervisor.shutdown_event.is_set():
                # Teardown fired mid-disclosure.  Exit without marking
                # complete — the call is ending, the driver will be
                # cancelled, and a half-played disclosure is not
                # legally sufficient.
                log.info(
                    "play_disclosure: shutdown_event set mid-playback "
                    "CallSid=%s — exiting without mark_disclosure_complete",
                    supervisor.call_sid,
                )
                return

            if supervisor.is_speaking:
                observed_speaking = True

            if observed_speaking and not supervisor.is_speaking:
                # Mark echo arrived (or sender finished while we
                # weren't polling — covers both).
                log.info(
                    "play_disclosure: final-mark echo received "
                    "CallSid=%s — marking disclosure complete",
                    supervisor.call_sid,
                )
                supervisor.mark_disclosure_complete()
                return

            if loop.time() >= deadline:
                # Duration timeout backstop: the mark echo was lost
                # (Twilio glitch, network jitter).  Mark complete anyway
                # so the driver can proceed — better to greet without
                # the last 500 ms of disclosure than to hang.
                log.warning(
                    "play_disclosure: timeout after %.2fs "
                    "(expected ~%.2fs) for CallSid=%s — "
                    "marking complete anyway",
                    timeout_s,
                    duration_s,
                    supervisor.call_sid,
                )
                supervisor.mark_disclosure_complete()
                return

            await asyncio.sleep(poll_interval_s)
    except asyncio.CancelledError:
        # The media-stream handler cancelled us on exit.  Do not mark
        # complete (the call is ending, the driver is being cancelled,
        # a half-played disclosure should not be marked complete).
        log.info(
            "play_disclosure: cancelled for CallSid=%s",
            supervisor.call_sid,
        )
        raise
    except Exception as exc:  # pragma: no cover — defensive
        # Last-resort safety net.  Without this an unhandled exception
        # in the player would surface as "task exception was never
        # retrieved" from asyncio — the player is a fire-and-forget
        # background task, so nobody awaits it to retrieve exceptions.
        log.exception(
            "play_disclosure: unexpected error for CallSid=%s: %s — "
            "marking complete anyway",
            supervisor.call_sid,
            exc,
        )
        try:
            supervisor.mark_disclosure_complete()
        except Exception:  # pragma: no cover — belt-and-braces
            pass


# ─── Mark-event dispatch ─────────────────────────────────────────────────────


def _dispatch_mark(message: dict, call_sid: str, supervisor) -> None:
    """
    Handle an inbound ``mark`` event from Twilio.

    Twilio echoes back each ``mark`` we send when the corresponding media
    has been played (or cleared).  Our mark names encode
    ``gen=<int>:offset=<int>[:final=true]`` so we can recover:

    * whether the mark is from a still-current generation (drop stale ones),
    * the played offset in ms (write to ``supervisor.played_duration_ms``),
    * whether the mark signals turn completion (flip ``is_speaking``).

    A mark whose name does not match the expected format is logged at
    debug and ignored — Twilio's contract permits arbitrary mark names
    (e.g., test clients), and we MUST NOT raise on them.
    """
    mark = message.get("mark")
    if not isinstance(mark, dict):
        log.debug(
            "media-stream: CallSid=%s mark event missing body — ignoring.",
            call_sid,
        )
        return
    name = mark.get("name")
    if not isinstance(name, str) or not name:
        log.debug(
            "media-stream: CallSid=%s mark event missing name — ignoring.",
            call_sid,
        )
        return

    parsed = parse_mark_name(name)
    if parsed is None:
        # Not one of ours — Twilio permits arbitrary mark names; log and
        # leave supervisor state alone.
        log.debug(
            "media-stream: CallSid=%s unrecognized mark name %r — ignoring.",
            call_sid,
            name,
        )
        return

    generation, offset_ms, is_final = parsed

    # Stale check: mark echoes for a generation we've already superseded
    # (e.g. barge-in bumped the counter).  Drop the update entirely so
    # played_duration_ms and is_speaking stay on the current generation.
    if generation < supervisor.generation_counter:
        log.debug(
            "media-stream: CallSid=%s stale mark gen=%d < current=%d — "
            "ignoring.",
            call_sid,
            generation,
            supervisor.generation_counter,
        )
        return

    # Monotonic guard: only advance played_duration_ms forward. Defensive
    # belt-and-braces — in normal operation offsets grow monotonically
    # within a generation, but if a late periodic mark arrives interleaved
    # with the final mark we don't want a smaller value to clobber the
    # newer one.
    if offset_ms >= supervisor.played_duration_ms:
        supervisor.played_duration_ms = offset_ms

    if is_final:
        supervisor.is_speaking = False


# ─── Inbound reader (extracted) ─────────────────────────────────────────────


async def _inbound_reader_loop(
    websocket: WebSocket,
    call_sid: str,
    supervisor,
    stream_sid: str,
) -> None:
    """
    Run the inbound media + mark reader for the lifetime of the WS.

    Exits cleanly when:
      * a ``stop`` event is received (Twilio's graceful goodbye);
      * the WebSocket disconnects (``WebSocketDisconnect``);
      * an unexpected exception escapes per-frame handling — logged
        but does not propagate (the bridge owner decides what to do).

    Per Issue 034's contract, the repacketizer's partial buffer is flushed
    on exit so Deepgram hears the trailing audio.

    NOTE (Issue 040 boundary): this loop does NOT implement barge-in.
    Twilio streams μ-law frames continuously — including silence — so
    any per-frame barge-in check fires on every 20 ms frame and cuts
    Riya off ~20 ms into every turn.  Real barge-in (gated on Deepgram
    ``StartOfTurn``, with ``clear`` + generation bump + Sarvam socket
    close) belongs to Issue 042.  Issue 040 only owns the supervisor
    state (``is_speaking``, ``disclosure_active``, ``disclosure_complete``,
    ``amd_verdict``) — not the barge-in trigger itself.
    """
    repacketizer = InboundRepacketizer()
    log.info(
        "media-stream: CallSid=%s reader loop started.",
        call_sid,
    )

    try:
        while True:
            message = await websocket.receive_json()

            if not isinstance(message, dict):
                log.debug(
                    "media-stream: CallSid=%s got non-dict message %r — skipping.",
                    call_sid,
                    type(message).__name__,
                )
                continue

            event = message.get("event")

            if event == "media":
                media = message.get("media")
                if not isinstance(media, dict):
                    log.debug(
                        "media-stream: CallSid=%s 'media' event missing body — skipping.",
                        call_sid,
                    )
                    continue
                payload_b64 = media.get("payload")
                if not isinstance(payload_b64, str):
                    log.debug(
                        "media-stream: CallSid=%s 'media.payload' not a str — skipping.",
                        call_sid,
                    )
                    continue

                try:
                    chunks = repacketizer.feed(payload_b64)
                except (ValueError, TypeError) as exc:
                    log.warning(
                        "media-stream: CallSid=%s invalid media payload "
                        "(%s: %s) — skipping frame.",
                        call_sid,
                        type(exc).__name__,
                        exc,
                    )
                    continue

                for chunk in chunks:
                    supervisor.inbound_audio_queue.put_nowait(chunk)

                # NOTE (Issue 040 boundary): no barge-in here.  Real
                # barge-in is Issue 042 and lives on the Deepgram
                # ``StartOfTurn`` path.  This loop only feeds audio to
                # the STT pipeline.

            elif event == "mark":
                _dispatch_mark(message, call_sid, supervisor)

            elif event == "stop":
                # Flush trailing partial so Deepgram hears the last syllable.
                tail = repacketizer.flush()
                for chunk in tail:
                    supervisor.inbound_audio_queue.put_nowait(chunk)
                log.info(
                    "media-stream: CallSid=%s sent 'stop' — reader exiting "
                    "(flushed %d trailing byte(s)).",
                    call_sid,
                    sum(len(c) for c in tail),
                )
                return

            elif event == "connected":
                # Optional event Twilio sends right after WS upgrade, before
                # 'start'.  Already accepted the WS above; this is observability.
                log.debug(
                    "media-stream: CallSid=%s received 'connected' event.",
                    call_sid,
                )

            else:
                log.debug(
                    "media-stream: CallSid=%s unknown event %r — ignoring.",
                    call_sid,
                    event,
                )

    except WebSocketDisconnect as exc:
        # Network drop or Twilio-side hangup.  Flush whatever audio
        # arrived so Deepgram gets the trailing audio.
        tail = repacketizer.flush()
        for chunk in tail:
            try:
                supervisor.inbound_audio_queue.put_nowait(chunk)
            except asyncio.QueueFull:  # pragma: no cover — unbounded by default
                log.warning(
                    "media-stream: CallSid=%s queue full on disconnect flush.",
                    call_sid,
                )
        log.info(
            "media-stream: CallSid=%s disconnected (code=%s, flushed %d "
            "trailing byte(s)). Reader exiting.",
            call_sid,
            exc.code,
            sum(len(c) for c in tail),
        )

    except Exception as exc:
        log.exception(
            "media-stream: CallSid=%s reader raised %s — exiting cleanly. "
            "Investigate: the inbound reader is supposed to swallow per-frame "
            "errors and keep the stream alive.",
            call_sid,
            type(exc).__name__,
        )


# ─── WebSocket endpoint ────────────────────────────────────────────────────────


@router.websocket("/media-stream")
async def twilio_media_stream(websocket: WebSocket) -> None:
    """
    Twilio Media Streams WebSocket — bidirectional bridge.

    Runs two concurrent coroutines on the same WebSocket until either
    exits:

    * ``_inbound_reader_loop`` — reads media + mark events from Twilio and
      forwards the audio to the Deepgram STT pipeline via
      ``supervisor.inbound_audio_queue``.
    * ``run_outbound_sender`` — writes TTS audio from
      ``supervisor.outbound_audio_queue`` to Twilio as 20 ms μ-law
      ``media`` frames, plus periodic and end-of-turn ``mark`` events.

    The two coroutines are independent.  ``asyncio.gather(..., return_exceptions=True)``
    is used so that one side's failure does not cancel the other; any
    exception that escaped per-frame error handling is logged and the
    bridge winds down together.

    Closes the socket (without teardown) on:
      * First message not being a ``start`` event with a known ``callSid``
        and ``streamSid``.
      * Either side of the bridge exits (Twilio's ``stop`` event, network
        drop, supervisor shutdown).

    No authentication — Twilio posts to this endpoint directly.  CallSid
    registration in the active-calls registry (set up by ``arrange_ai_call``
    after Twilio call placement) is the only access control.

    Twilio's Media Streams contract:
      https://www.twilio.com/docs/voice/media-streams/websocket-messages
    """
    await websocket.accept()

    # ── Step 1: Read pre-start events until we get ``start`` ──────────────
    # Twilio's Media Streams contract sends ``{"event": "connected"}``
    # immediately on WebSocket upgrade, BEFORE the ``{"event": "start"}``
    # event.  Rejecting the connected handshake (which the previous
    # implementation did) closed the socket before the audio pipeline
    # could spawn, terminating live calls right after the greeting.
    # Loop here tolerates connected; anything else is a protocol violation.
    start_message: dict | None = None
    while start_message is None:
        try:
            msg = await websocket.receive_json()
        except WebSocketDisconnect as exc:
            # Twilio (or a malicious peer) disconnected before sending start.
            # No state to clean up — the reader never started.
            log.info(
                "media-stream: client disconnected before start (code=%s). "
                "Reader exiting.",
                exc.code,
            )
            return
        except (json.JSONDecodeError, ValueError) as exc:
            # Invalid JSON in a pre-start message — protocol violation.
            log.warning(
                "media-stream: pre-start message not valid JSON (%s). Closing.",
                exc,
            )
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        if not isinstance(msg, dict):
            # Silently skip — Starlette can yield odd shapes; Twilio's
            # contract guarantees a JSON object per WS frame, but we
            # don't want to be brittle about it pre-start.
            continue

        event = msg.get("event")
        if event == "connected":
            # Expected pre-start handshake — Twilio confirms the WS
            # upgrade is complete and the media plane is wired up.  No
            # state change; keep reading until ``start`` arrives.
            log.debug("media-stream: received pre-start 'connected' event.")
            continue
        if event == "start":
            start_message = msg
            break

        # Any other event before ``start`` is unexpected — close with a
        # policy-violation code.  Per Twilio's docs, only ``connected``
        # and ``start`` are valid pre-start messages.
        log.warning(
            "media-stream: unexpected pre-start event %r — closing.",
            event,
        )
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    assert start_message is not None  # invariant: loop exits only via break
    first_message = start_message

    start_payload = first_message.get("start")
    if not isinstance(start_payload, dict):
        log.warning(
            "media-stream: 'start' event missing 'start' object. Closing."
        )
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    call_sid = start_payload.get("callSid")
    if not isinstance(call_sid, str) or not call_sid:
        log.warning(
            "media-stream: 'start' event has no callSid. Closing."
        )
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # Issue 036: streamSid is required by every outbound media + mark
    # message per Twilio's contract.  Validate it eagerly so we don't
    # accept a stream we can't address outbound.
    stream_sid = start_payload.get("streamSid")
    if not isinstance(stream_sid, str) or not stream_sid:
        log.warning(
            "media-stream: 'start' event has no streamSid. Closing."
        )
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # ── Step 2: Look up the live supervisor ────────────────────────────────
    # If no supervisor is registered for this CallSid, the call has either
    # already ended (status callback fired and un-registered) or the server
    # restarted (in-memory registry is empty across restarts).  In both
    # cases the startup sweep (Issue 029) owns the database resolution —
    # we MUST NOT spin up a supervisor here, because the concurrency slot
    # has already been released and we'd just leak again.
    supervisor = get_call(call_sid)
    if supervisor is None:
        log.warning(
            "media-stream: no supervisor for CallSid=%s (stale stream or "
            "post-restart). Closing without action.",
            call_sid,
        )
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # ── Step 3: Spawn STT + run the bidirectional bridge ────────────────────
    # Issue 035 follow-up: spawn the Deepgram STT reader *now*, not during
    # ``arrange_ai_call``.  Twilio's ``start`` event fires when the call is
    # actually answered and the media plane is wired up — opening the STT
    # socket any earlier would let it idle during the ringing phase and
    # burn through our fast-reconnect budget before any audio arrives.
    #
    # The supervisor owns the task and awaits its clean shutdown in
    # ``trigger_teardown``; ``start_background_tasks`` is non-blocking
    # (``create_task``) and a failure to spawn is logged but never raised.
    supervisor.start_background_tasks()

    log.info(
        "media-stream: CallSid=%s streamSid=%s stream attached. "
        "Bidirectional bridge running.",
        call_sid,
        stream_sid,
    )

    # First-one-wins coordination (Issue 036 blocker fix).
    #
    # ``asyncio.gather(reader, sender, return_exceptions=True)`` was wrong
    # here: the inbound reader exits naturally on a Twilio ``stop`` event
    # or a ``WebSocketDisconnect``, but the outbound sender is an
    # infinite polling loop that only exits on ``shutdown_event`` or a
    # WS send failure.  When the reader finishes cleanly, ``gather``
    # blocks indefinitely waiting for the sender.  ``asyncio.wait`` with
    # ``FIRST_COMPLETED`` lets us react to either half finishing, then
    # explicitly cancel the other so the handler can return promptly.
    # Issue 041: the supervisor owns the reader + sender tasks (it
    # cancels them in ``trigger_teardown``).  We still need them as
    # local references to drive the FIRST_COMPLETED + cancel pattern
    # below — Starlette requires the endpoint coroutine to stay alive
    # for the connection.  The two paths converge cleanly: cancel() on
    # a done task is idempotent, so whichever fires first wins and the
    # other is a no-op.
    supervisor.attach_media_stream(websocket, stream_sid)
    reader_task = supervisor.reader_task
    sender_task = supervisor.sender_task

    # Issue 037: spawn the legal-recording disclosure player as a
    # fire-and-forget background task once the reader + sender are up.
    # The player pushes the pre-rendered μ-law bytes, waits for the
    # final-mark echo (is_speaking flips False), and calls
    # ``mark_disclosure_complete()`` so the driver proceeds.  It is
    # NOT in the FIRST_COMPLETED wait set below — its lifecycle is
    # independent of the bridge halves, and the call needs to keep
    # running while the disclosure plays (~10 s for English, ~11 s for
    # Hinglish) without returning the handler.
    disclosure_task = asyncio.create_task(
        play_disclosure(supervisor, stream_sid),
        name=f"disclosure-player-{call_sid}",
    )

    done, pending = await asyncio.wait(
        {reader_task, sender_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    # Cancel the survivor so its half unwinds cleanly.  ``gather`` with
    # ``return_exceptions=True`` so a ``CancelledError`` raised by cancel
    # does not escape — we only care about whether the task finished.
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

    # Cancel the disclosure player too.  It is NOT in the wait set above
    # (it runs independently of the bridge), so we must clean it up
    # here.  Its ``play_disclosure`` coroutine is cancellation-safe:
    # on cancel it logs and re-raises CancelledError without marking
    # the disclosure complete (the call is ending, half-played
    # disclosures should not be marked legally complete).
    if not disclosure_task.done():
        disclosure_task.cancel()
        try:
            await disclosure_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.warning(
                "media-stream: CallSid=%s disclosure player raised on "
                "cancel: %s",
                call_sid,
                exc,
            )

    # Surface any exception that escaped per-frame error handling in the
    # half that finished first.  The pending half's CancelledError is
    # already absorbed by the gather above.  Teardown may also cancel a
    # task just before ``asyncio.wait`` returns it in ``done``; calling
    # ``Task.exception()`` on that task raises ``CancelledError`` rather
    # than returning an exception object, so treat that expected state as
    # a clean bridge shutdown.
    for t in done:
        try:
            exc = t.exception()
        except asyncio.CancelledError:
            continue
        if exc is not None:
            log.exception(
                "media-stream: CallSid=%s bridge half raised — %s.",
                call_sid,
                type(exc).__name__,
                exc_info=exc,
            )

    # Slot release + DB resolve is owned by twilio_status_callback,
    # NOT by this endpoint.  The WS is left for Starlette to clean up
    # when this handler returns.
