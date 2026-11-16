"""
backend/audio/deepgram_stt.py — Deepgram Flux STT reader (Issue 035).

The "brain's ear" — converts the raw aggregated μ-law audio chunks arriving on
``supervisor.inbound_audio_queue`` into confirmed customer turns, which it
pushes onto ``supervisor.completed_turn_queue`` for the conversation driver
(Issue 039) to consume.

Why this module owns the whole socket
──────────────────────────────────────
Deepgram Flux is a **continuous, persistent** WebSocket — one socket carries
the whole call's audio, not per-turn.  That makes it fundamentally different
from per-turn AI providers (Mercury-2, Sarvam, Gemini) and means the module
has to handle three things those don't:

  1. **Reconnect with backoff.**  A single transient network blip should not
     kill the call.  We reconnect up to ``_MAX_RECONNECT_ATTEMPTS`` times,
     draining the inbound queue first (stale audio would scramble Flux's
     real-time turn detection).  Exhaustion triggers supervisor teardown
     with ``provider_fault=True``.
  2. **Graceful shutdown.**  When the supervisor's ``shutdown_event`` fires,
     the reader sends ``{"type": "CloseStream"}`` and drains any final
     transcript Flux flushes — losing the trailing turn would be a silent
     data loss.  The whole shutdown dance runs without triggering the
     reconnect loop.
  3. **Mid-stream fault isolation.**  Provider blips must not propagate to
     the call pipeline.  Every external interaction is wrapped — socket
     errors raise, the outer loop catches, the supervisor only sees a
     clean teardown when retries are exhausted.

Connection params (URL query string)
────────────────────────────────────
* ``model`` — ``flux-general-en`` for English-only leads, ``flux-general-multi``
  for Hinglish leads (see ``build_connection_params``).  Selection is
  driven by ``CallSupervisor.language`` so the same code path serves both.
* ``encoding=mulaw`` / ``sample_rate=8000`` — Twilio's native codec.
* ``eot_threshold`` / ``eot_timeout_ms`` — from settings.
* ``language_hint=hi&language_hint=en`` — Hinglish only, repeated parameter,
  urlencode with ``doseq=True``.  English calls send no language hints.
* ``keyterm=...`` (one per entry in ``DEEPGRAM_KEYTERMS``) — same shape.

Turn lifecycle events (Flux docs: ``state.md``)
───────────────────────────────────────────────
v1 ships with eager-eot OFF, so only ``StartOfTurn`` / ``Update`` / ``EndOfTurn``
arrive in practice; we still log ``EagerEndOfTurn`` / ``TurnResumed`` at debug
when they fire (v2 speculation toggle).  Only ``EndOfTurn`` produces a
transcript — everything else is observability.

Lifecycle of ``run_deepgram_reader``
────────────────────────────────────
::

    while not (shutdown_event set):
        try:
            await _run_one_session(supervisor)
            return                                # clean exit
        except CancelledError:
            if shutdown_event: return             # graceful
            raise
        except Exception:
            reconnect += 1
            if reconnect >= max: → teardown
            drain inbound_audio_queue
            await asyncio.sleep(backoff)

The task ends on:
  * Graceful shutdown (``shutdown_event`` set + cancel + CloseStream sent).
  * Reconnect exhaustion (→ supervisor ``trigger_teardown(provider_fault=True)``).
  * Task cancellation mid-flight (propagated unless accompanied by shutdown_event).

Module non-goals
────────────────
* No audio buffering or repacketization — ``InboundRepacketizer`` already does
  that upstream in ``backend/api/media_stream.py``.
* No turn-boundary decisions of its own — it only forwards what Flux reports.
* No retry/backoff on individual sends — the outer reconnect loop owns that.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from typing import TYPE_CHECKING
from urllib.parse import urlencode

# Imported at module level (not lazily inside the function) so unit tests can
# patch ``backend.audio.deepgram_stt.websockets.connect`` cleanly.  The import
# is side-effect-free: ``websockets`` is a pure-Python asyncio library with
# no I/O at import time.
import websockets
from websockets.exceptions import ConnectionClosed

from backend.audio.lexicon import DEEPGRAM_KEYTERMS
from backend.config import load_settings

if TYPE_CHECKING:
    from backend.call_supervisor import CallSupervisor

__all__ = [
    "run_deepgram_reader",
    "build_connection_params",
    "dispatch_event",
    "drain_inbound_queue",
]

log = logging.getLogger(__name__)


# ─── Tunables ─────────────────────────────────────────────────────────────────

#: Deepgram Flux endpoint. ``/v2/listen`` is required — ``/v1/listen`` does not
#: speak Flux.
_DEEPGRAM_WS_BASE: str = "wss://api.deepgram.com/v2/listen"

#: Hard cap on reconnect attempts before giving up and triggering teardown.
_MAX_RECONNECT_ATTEMPTS: int = 3

#: Initial backoff between reconnect attempts (seconds). Doubles per attempt.
_INITIAL_RECONNECT_BACKOFF_S: float = 1.0
_BACKOFF_GROWTH: float = 0.5  # next backoff += this many seconds

#: Soft cap on how long we wait for a graceful close before giving up.
_CLOSE_STREAM_TIMEOUT_S: float = 3.0

#: Reader-loop poll interval (seconds).  Between events the reader sleeps
#: on ``ws.recv()`` with this timeout to wake up and re-check the shutdown
#: event.  100 ms is the sweet spot — fast enough to feel snappy on a
#: teardown, slow enough that the overhead is invisible on a normal call.
_READER_POLL_INTERVAL_S: float = 0.1

#: Deepgram Flux model identifiers.  Selected per-call based on the lead's
#: preferred language (see ``build_connection_params``):
#: - English-only leads use ``flux-general-en`` for higher accuracy on
#:   monolingual English audio.
#: - Hinglish leads use ``flux-general-multi`` with the language hints below
#:   so Flux can render a per-word language tag for mixed-script speech.
_DEEPGRAM_MODEL_ENGLISH: str = "flux-general-en"
_DEEPGRAM_MODEL_MULTILINGUAL: str = "flux-general-multi"

#: Language hints sent to the multilingual model for Hinglish calls (Hindi
#: + English).  We ask Flux to weight both so a mixed-script sentence gets
#: a fair per-word tag.  English-only calls send NO ``language_hint`` query
#: params — telling Flux to bias toward languages the customer is not
#: speaking actively degrades transcription accuracy.
_LANGUAGE_HINTS: tuple[str, ...] = ("hi", "en")


# ─── Public API ───────────────────────────────────────────────────────────────


async def run_deepgram_reader(supervisor: "CallSupervisor") -> None:
    """
    Long-lived per-call task that owns the Deepgram Flux WebSocket.

    Parameters
    ----------
    supervisor : CallSupervisor
        The supervisor created in ``arrange_ai_call`` after Twilio call
        placement succeeds.  Provides:

        * ``inbound_audio_queue`` — μ-law audio chunks ready for Flux.
        * ``completed_turn_queue`` — output target for finished transcripts.
        * ``shutdown_event`` — set when teardown starts.
        * ``trigger_teardown`` — called with ``provider_fault=True`` if the
          reconnect budget is exhausted.
        * ``call_sid`` — used only for log correlation.

    Lifecycle
    ─────────
    This coroutine returns when:

    * **Graceful shutdown** — supervisor ``shutdown_event`` is set, the
      reader sends ``CloseStream`` and drains the final transcript.  This
      path runs inside ``_run_one_session`` — the outer loop sees a clean
      return and exits without reconnecting.
    * **Reconnect exhaustion** — ``_MAX_RECONNECT_ATTEMPTS`` consecutive
      session failures → ``supervisor.trigger_teardown(provider_fault=True)``.
    * **Cancellation** — the supervisor cancels this task during teardown.
      If ``shutdown_event`` is also set, the cancellation is treated as
      graceful; otherwise it propagates.

    Failure semantics
    ─────────────────
    Any uncaught exception in the session body (WebSocket failure, malformed
    URL, etc.) is treated as a transient failure and triggers a reconnect —
    except for ``CancelledError``, which respects ``shutdown_event``.
    """
    reconnect_attempts: int = 0

    while True:
        try:
            await _run_one_session(supervisor)
            # Clean exit — graceful shutdown completed.
            log.info(
                "run_deepgram_reader: session ended cleanly for CallSid=%s.",
                supervisor.call_sid,
            )
            return

        except asyncio.CancelledError:
            # Task being cancelled — likely from trigger_teardown.
            # If the supervisor flagged a graceful shutdown, treat it as
            # such and exit; otherwise propagate (outer test harness / loop
            # cancellation).
            if supervisor.shutdown_event.is_set():
                log.info(
                    "run_deepgram_reader: CancelledError after shutdown_event "
                    "for CallSid=%s — exiting cleanly without reconnect.",
                    supervisor.call_sid,
                )
                return
            log.warning(
                "run_deepgram_reader: CancelledError WITHOUT shutdown_event "
                "for CallSid=%s — propagating.",
                supervisor.call_sid,
            )
            raise

        except Exception as exc:
            reconnect_attempts += 1
            if reconnect_attempts >= _MAX_RECONNECT_ATTEMPTS:
                log.error(
                    "run_deepgram_reader: %d consecutive session failures "
                    "for CallSid=%s; last error: %s. Triggering teardown "
                    "with provider_fault=True.",
                    reconnect_attempts,
                    supervisor.call_sid,
                    exc,
                )
                await supervisor.trigger_teardown(
                    outcome_status="Failed",
                    failure_reason="error",
                    provider_fault=True,
                )
                return

            backoff = (
                _INITIAL_RECONNECT_BACKOFF_S
                + _BACKOFF_GROWTH * (reconnect_attempts - 1)
            )
            log.warning(
                "run_deepgram_reader: session failed (attempt %d/%d) "
                "CallSid=%s: %s. Draining stale audio and retrying in %.1fs.",
                reconnect_attempts,
                _MAX_RECONNECT_ATTEMPTS,
                supervisor.call_sid,
                exc,
                backoff,
            )

            # Drop any audio sitting in the inbound queue — replaying it on
            # a fresh session would scramble Flux's real-time turn detection
            # (it would look like the customer just spoke a burst of words
            # right at session start, potentially firing a phantom EOT).
            drain_inbound_queue(supervisor)

            await asyncio.sleep(backoff)


def build_connection_params(
    api_key: str,
    *,
    eot_threshold: float,
    eot_timeout_ms: int,
    language: str,
) -> tuple[str, list[tuple[str, str]]]:
    """
    Build the WebSocket URL and Authorization header for Deepgram Flux.

    Exposed publicly so unit tests can assert the exact wire format without
    needing a real session.

    Parameters
    ----------
    api_key
        The Deepgram API key (raw, pre-formatted).  The Authorization header
        value is ``Token <key>`` per Deepgram's WebSocket contract.
    eot_threshold
        The politeness dial — how certain Flux must be that the customer is
        done before firing ``EndOfTurn``.  Caller passes this from settings.
    eot_timeout_ms
        The silence backstop — maximum milliseconds before forcing
        ``EndOfTurn`` regardless of confidence.  Caller passes from settings.
    language
        The lead's preferred language — ``"English"`` or ``"Hinglish"``.
        Selects the Flux model and (for Hinglish) the language-hint list:

        * English → ``model=flux-general-en`` and **no** ``language_hint``
          params.  Hinting Hindi on a monolingual English call degrades
          transcription.
        * Hinglish → ``model=flux-general-multi`` and ``language_hint``
          repeated for Hindi + English so Flux can render a per-word tag
          for mixed-script speech.

    Returns
    -------
    (url, headers)
        ``url`` is the full ``wss://.../v2/listen?...`` query string.
        ``headers`` is a list of ``(name, value)`` pairs to pass to
        ``websockets.connect(additional_headers=...)``.

    Raises
    ------
    ValueError
        If ``language`` is not one of the two supported values.

    URL serialization
    ────────────────
    We deliberately use ``urllib.parse.urlencode(..., doseq=True)`` so that
    ``keyterm`` (a list of ~75 entries) is serialized as **repeated query
    parameters** rather than comma-joined.  For Hinglish the same trick
    repeats ``language_hint=hi`` and ``language_hint=en``; English calls
    emit no language hints at all.

    Deepgram's WebSocket contract expects repeated ``keyterm=`` and
    ``language_hint=`` params; comma-joined values would be a single
    malformed token.  Building this manually is error-prone — let the stdlib
    do it.
    """
    if language == "English":
        model = _DEEPGRAM_MODEL_ENGLISH
        language_hints: tuple[str, ...] = ()
    elif language == "Hinglish":
        model = _DEEPGRAM_MODEL_MULTILINGUAL
        language_hints = _LANGUAGE_HINTS
    else:
        raise ValueError(
            f"build_connection_params: language must be 'English' or "
            f"'Hinglish', got {language!r}"
        )

    params: list[tuple[str, str]] = [
        ("model", model),
        ("encoding", "mulaw"),
        ("sample_rate", "8000"),
        ("eot_threshold", str(eot_threshold)),
        ("eot_timeout_ms", str(eot_timeout_ms)),
    ]
    for hint in language_hints:
        params.append(("language_hint", hint))
    for term in DEEPGRAM_KEYTERMS:
        params.append(("keyterm", term))

    query = urlencode(params, doseq=True)
    url = f"{_DEEPGRAM_WS_BASE}?{query}"
    headers: list[tuple[str, str]] = [("Authorization", f"Token {api_key}")]
    return url, headers


def _connect_with_headers(
    url: str,
    headers: list[tuple[str, str]],
):
    """
    Open a WebSocket using the header keyword supported by the installed API.

    The top-level ``websockets.connect`` may resolve to the legacy API with
    ``extra_headers`` (as it does in the host's 13.1 user-site package) or the
    modern API with ``additional_headers`` (as it does in the isolated 16.1
    environment). Both shapes are inside the project's declared dependency
    range, so select the keyword from the callable's real signature instead of
    forwarding an unsupported name through ``**kwargs`` to the event loop.
    """
    parameters = inspect.signature(websockets.connect).parameters
    if "additional_headers" in parameters:
        return websockets.connect(url, additional_headers=headers)
    if "extra_headers" in parameters:
        return websockets.connect(url, extra_headers=headers)
    raise RuntimeError(
        "Unsupported websockets.connect API: expected additional_headers "
        "or extra_headers"
    )


def dispatch_event(event: dict, supervisor: "CallSupervisor") -> None:
    """
    Route a parsed Deepgram Flux event to the right log line / queue push.

    Pure with respect to ``event``; mutates supervisor state only via the
    well-defined hand-off (pushing onto ``completed_turn_queue``).

    Event semantics (Flux ``state.md``)
    ───────────────────────────────────
    * ``StartOfTurn``     — customer started speaking (non-empty transcript
                            guaranteed). Logged at debug; no action.
    * ``EagerEndOfTurn``  — Flux is "probably" done.  Only fired if
                            ``eager_eot_threshold`` is configured (v2). v1
                            won't see this; logging is defensive.
    * ``TurnResumed``     — customer kept talking after EagerEndOfTurn.
                            Only fired in v2. Logged at debug.
    * ``EndOfTurn``       — confirmed end of customer turn.  Pushes the
                            non-empty transcript onto
                            ``supervisor.completed_turn_queue``.
    * ``Update`` / unknown — partial transcript updates mid-turn; logged
                            at debug.  Never pushed to the consumer — Flux
                            guarantees the EndOfTurn transcript will be the
                            final/canonical one.
    """
    event_type = event.get("event")
    turn_index = event.get("turn_index")

    if event_type == "StartOfTurn":
        log.debug(
            "Deepgram STT: StartOfTurn CallSid=%s turn_index=%s transcript=%r",
            supervisor.call_sid,
            turn_index,
            (event.get("transcript") or "").strip(),
        )
        # Issue 042: customer speech onset is the barge-in trigger.  Flux
        # guarantees a non-empty transcript on StartOfTurn, so this is the
        # recommended interruption signal.  The supervisor decides whether to
        # actually fire (Riya speaking, disclosure not playing, not already
        # interrupting) — we just deliver the signal.  Resolved via getattr so
        # the stub supervisors used in dispatch unit tests (which lack the
        # method) are unaffected — mirrors the matcher fan-out guard below.
        start_bargein = getattr(supervisor, "maybe_start_bargein", None)
        if start_bargein is not None:
            try:
                start_bargein()
            except Exception as exc:  # pragma: no cover — defensive
                log.warning(
                    "Deepgram STT: barge-in trigger raised CallSid=%s "
                    "turn_index=%s: %s — ignoring.",
                    supervisor.call_sid,
                    turn_index,
                    exc,
                )
        return

    if event_type == "EagerEndOfTurn":
        # v1 ships with eager-eot OFF, so this is rarely seen.  Logged at
        # debug so v2 can flip the speculation switch without changing the
        # reader's behavior.
        log.debug(
            "Deepgram STT: EagerEndOfTurn CallSid=%s turn_index=%s transcript=%r",
            supervisor.call_sid,
            turn_index,
            (event.get("transcript") or "").strip(),
        )
        return

    if event_type == "TurnResumed":
        log.debug(
            "Deepgram STT: TurnResumed CallSid=%s turn_index=%s",
            supervisor.call_sid,
            turn_index,
        )
        return

    if event_type == "EndOfTurn":
        transcript = (event.get("transcript") or "").strip()
        if not transcript:
            # Empty EndOfTurn (a brief "hmm" that Flux still gated on).  Skip
            # rather than push empty strings downstream — the conversation
            # driver treats empty transcripts as filler (no Mercury call).
            log.debug(
                "Deepgram STT: EndOfTurn with empty transcript CallSid=%s "
                "turn_index=%s — skipping.",
                supervisor.call_sid,
                turn_index,
            )
            return
        # put_nowait — never blocks.  Unbounded queue means a slow consumer
        # can theoretically accumulate, but the orchestrator will see the
        # backlog and adjust.
        try:
            supervisor.completed_turn_queue.put_nowait(transcript)
        except asyncio.QueueFull:  # pragma: no cover — queue is unbounded
            log.warning(
                "Deepgram STT: completed_turn_queue full CallSid=%s — "
                "dropping transcript turn_index=%s.",
                supervisor.call_sid,
                turn_index,
            )
            return

        # Issue 041: fan out to the matcher consumer in parallel with the
        # conversation driver.  The matcher ranks vehicles off the driver's
        # critical path so the driver doesn't wait for extraction/scoring
        # before responding to the customer.  Best-effort: if the matcher
        # isn't spawned (test fixture, missing catalogue), skip silently.
        matcher = getattr(supervisor, "matcher_task", None)
        if matcher is not None:
            try:
                matcher.enqueue_turn(transcript)
            except Exception as exc:  # pragma: no cover — defensive
                log.warning(
                    "Deepgram STT: matcher enqueue failed CallSid=%s "
                    "turn_index=%s: %s — driver still sees the transcript.",
                    supervisor.call_sid,
                    turn_index,
                    exc,
                )

        log.info(
            "Deepgram STT: EndOfTurn CallSid=%s turn_index=%s transcript=%r",
            supervisor.call_sid,
            turn_index,
            transcript,
        )
        return

    # Anything else (Update, future-proofing, unknown): debug log only.
    log.debug(
        "Deepgram STT: unhandled event type=%r CallSid=%s turn_index=%s",
        event_type,
        supervisor.call_sid,
        turn_index,
    )


def drain_inbound_queue(supervisor: "CallSupervisor") -> int:
    """
    Drop every audio chunk currently buffered on ``supervisor.inbound_audio_queue``.

    Called immediately before a reconnect so the fresh session doesn't see
    stale audio that would scramble Flux's real-time turn detection.

    Returns
    -------
    int
        Number of chunks drained.  Useful for logging / metrics.
    """
    drained = 0
    while True:
        try:
            supervisor.inbound_audio_queue.get_nowait()
            drained += 1
        except asyncio.QueueEmpty:
            break
    return drained


# ─── Internals ────────────────────────────────────────────────────────────────


async def _run_one_session(supervisor: "CallSupervisor") -> None:
    """
    One connect → run → graceful close cycle.

    Raises any socket/IO error to the outer ``run_deepgram_reader`` reconnect
    loop.  Returns cleanly only when ``shutdown_event`` fires and the reader
    finishes its ``CloseStream`` handshake.
    """
    settings = load_settings()
    api_key = settings.deepgram_api_key.get_secret_value()

    url, headers = build_connection_params(
        api_key,
        eot_threshold=settings.deepgram_eot_threshold,
        eot_timeout_ms=settings.deepgram_eot_timeout_ms,
        language=supervisor.language,
    )

    # Reflect the model actually used in the URL for log correlation.  We
    # re-derive it from the supervisor's language so the log line never
    # disagrees with what build_connection_params actually emitted.
    if supervisor.language == "English":
        model_for_log = _DEEPGRAM_MODEL_ENGLISH
        hints_for_log: tuple[str, ...] = ()
    else:
        model_for_log = _DEEPGRAM_MODEL_MULTILINGUAL
        hints_for_log = _LANGUAGE_HINTS

    log.info(
        "Deepgram STT: connecting CallSid=%s model=%s "
        "encoding=mulaw sample_rate=8000 eot_threshold=%s eot_timeout_ms=%d "
        "keyterms=%d language_hints=%s",
        supervisor.call_sid,
        model_for_log,
        settings.deepgram_eot_threshold,
        settings.deepgram_eot_timeout_ms,
        len(DEEPGRAM_KEYTERMS),
        list(hints_for_log),
    )

    async with _connect_with_headers(url, headers) as ws:
        log.info(
            "Deepgram STT: connected CallSid=%s — starting writer + reader.",
            supervisor.call_sid,
        )

        writer_task = asyncio.create_task(
            _writer_loop(ws, supervisor),
            name=f"deepgram-writer-{supervisor.call_sid}",
        )
        reader_task = asyncio.create_task(
            _reader_loop(ws, supervisor),
            name=f"deepgram-reader-{supervisor.call_sid}",
        )

        try:
            # The reader owns the lifecycle — it exits on shutdown_event
            # (graceful) or on ConnectionClosed (failure).  Awaiting it is
            # therefore waiting for "session ended".
            await reader_task
        finally:
            # Reader is done for whatever reason — make sure the writer
            # doesn't leak.  cancel() wakes the await on inbound_audio_queue;
            # we swallow the CancelledError because the writer's job is done.
            if not writer_task.done():
                writer_task.cancel()
                try:
                    await writer_task
                except asyncio.CancelledError:
                    pass
                except ConnectionClosed:
                    # Race: reader's ConnectionClosed propagated into the
                    # writer's send() before cancel landed.  That's fine —
                    # both sides observed the disconnect.
                    pass


async def _writer_loop(ws, supervisor: "CallSupervisor") -> None:
    """
    Drain ``supervisor.inbound_audio_queue`` and forward each chunk to Deepgram.

    Pure forwarder: no buffering, no batching, no interpretation.  The
    repacketizer upstream already produced 80 ms Deepgram-ready chunks; we
    just ship them.

    Exits via:
      * ``CancelledError`` — supervisor shut us down.
      * ``ConnectionClosed`` — Deepgram dropped; bubbles to outer loop.
      * ``ws.send`` raising — any socket-level error bubbles.
    """
    while True:
        chunk = await supervisor.inbound_audio_queue.get()
        try:
            await ws.send(chunk)
        except ConnectionClosed:
            # Reader will see this too; the outer reconnect loop owns the
            # recovery decision.  Just exit so the session can wind down.
            raise
        except Exception:
            # Any other ws.send failure (e.g. socket closed mid-send).  Re-raise
            # so the outer loop can treat it as a session failure.
            raise


async def _reader_loop(ws, supervisor: "CallSupervisor") -> None:
    """
    Consume Deepgram Flux events until the session ends.

    Lifecycle:
      * Normal shutdown — supervisor.shutdown_event is set.  Send
        ``CloseStream``, drain any final transcript, return.
      * Network drop — ``ws.recv`` raises ``ConnectionClosed`` → bubble up.
      * Malformed JSON — silently skip; Flux promises well-formed messages.

    Polling
    ───────
    ``ws.recv()`` blocks indefinitely while waiting for the next event, so
    we wrap each call in ``asyncio.wait_for(...)`` with a short timeout
    (``_READER_POLL_INTERVAL_S``).  On timeout we re-check the shutdown event
    and try again.  This bounds shutdown-detection latency to the poll
    interval — typically ≤100 ms — while keeping the normal-event path
    unchanged.
    """
    while True:
        # Check shutdown before the recv — no point receiving a message we're
        # about to close on top of.
        if supervisor.shutdown_event.is_set():
            await _send_close_stream_and_drain(ws, supervisor)
            return

        try:
            raw_message = await asyncio.wait_for(
                ws.recv(),
                timeout=_READER_POLL_INTERVAL_S,
            )
        except asyncio.TimeoutError:
            # No new event in the poll window — loop back to re-check shutdown.
            continue
        except ConnectionClosed:
            log.info(
                "Deepgram STT: ConnectionClosed CallSid=%s.",
                supervisor.call_sid,
            )
            raise

        try:
            event = json.loads(raw_message)
        except (json.JSONDecodeError, TypeError):
            log.warning(
                "Deepgram STT: malformed JSON from Flux CallSid=%s — "
                "skipping (raw=%r).",
                supervisor.call_sid,
                raw_message[:200] if isinstance(raw_message, (str, bytes))
                    else type(raw_message).__name__,
            )
            continue

        if not isinstance(event, dict):
            log.debug(
                "Deepgram STT: non-dict message CallSid=%s — skipping.",
                supervisor.call_sid,
            )
            continue

        dispatch_event(event, supervisor)


async def _send_close_stream_and_drain(ws, supervisor: "CallSupervisor") -> None:
    """
    Send ``{"type": "CloseStream"}`` and flush any final transcript Flux emits.

    Per Flux docs (``close-stream.md``), the server processes any cached
    audio, sends the final ``TurnInfo`` event, then terminates the WebSocket.
    Any final ``EndOfTurn`` carries the trailing transcript — we must catch it
    and push it onto ``completed_turn_queue`` or lose it.
    """
    log.info(
        "Deepgram STT: shutdown_event set — sending CloseStream CallSid=%s.",
        supervisor.call_sid,
    )
    try:
        await asyncio.wait_for(
            ws.send(json.dumps({"type": "CloseStream"})),
            timeout=_CLOSE_STREAM_TIMEOUT_S,
        )
    except (ConnectionClosed, asyncio.TimeoutError, Exception) as exc:
        log.warning(
            "Deepgram STT: CloseStream send failed CallSid=%s (%s) — "
            "exiting without draining final transcript.",
            supervisor.call_sid,
            exc,
        )
        return

    try:
        async for raw_message in ws:
            try:
                event = json.loads(raw_message)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(event, dict):
                continue

            event_type = event.get("event")
            if event_type == "EndOfTurn":
                transcript = (event.get("transcript") or "").strip()
                if transcript:
                    try:
                        supervisor.completed_turn_queue.put_nowait(transcript)
                        log.info(
                            "Deepgram STT: final transcript from CloseStream "
                            "CallSid=%s transcript=%r",
                            supervisor.call_sid,
                            transcript,
                        )
                    except asyncio.QueueFull:  # pragma: no cover
                        log.warning(
                            "Deepgram STT: completed_turn_queue full during "
                            "CloseStream drain CallSid=%s — dropping final.",
                            supervisor.call_sid,
                        )
            # Other events during the close handshake (Update, StartOfTurn
            # for a turn Flux already decided was done, etc.) are ignored —
            # we only care about the final EndOfTurn.
    except ConnectionClosed:
        # Expected — Flux closes the socket after the final TurnInfo.
        pass
    except asyncio.TimeoutError:
        log.warning(
            "Deepgram STT: timed out waiting for final transcript after "
            "CloseStream CallSid=%s.",
            supervisor.call_sid,
        )
