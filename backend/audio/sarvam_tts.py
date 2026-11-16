"""
backend/audio/sarvam_tts.py — Sarvam Bulbul v3 Streaming TTS Client (Issue 038).

Wraps the ``sarvamai.AsyncSarvamAI.text_to_speech_streaming.connect()`` WebSocket
behind a small per-call orchestrator that:

  * Owns one persistent WebSocket for the duration of the call (no per-turn
    handshake — saves the connect-time latency on every Riya utterance).
  * Translates the Sarvam streaming protocol into the existing per-call
    ``outbound_audio_queue`` contract the outbound sender already consumes
    (items are ``(generation, chunk_or_None)`` tuples; ``None`` marks end of
    turn).
  * Honors ``supervisor.language`` → BCP-47 mapping (English → ``en-IN``,
    Hinglish → ``hi-IN``).
  * Pulls the speaker name from settings (``Settings.sarvam_tts_speaker``) so
    operators can A/B voices without a redeploy.
  * Spots Sarvam's completion event (``EventResponse`` with
    ``data.event_type == "final"``) and uses it as the authoritative
    end-of-turn signal — that is when we push the ``None`` chunk so the
    outbound sender emits its final ``mark`` and the inbound reader flips
    ``is_speaking`` back to ``False``.

Why a single persistent socket
───────────────────────────────
Spike 026 confirmed:

  * μ-law 8 kHz is a valid ``output_audio_codec`` (the SDK docstring claims
    "MP3 only" but that is stale — mulaw is accepted).
  * TTFB on the very first turn is ~267 ms; reopening the socket per turn
    would add a fresh handshake every utterance, doubling that floor on a
    conversation that already lives at the edge of tolerable latency.
  * ``configure()`` re-applies cleanly between turns (any text buffered
    auto-flushes per the SDK docstring).
  * There is **no** cancel/clear control message — the only way to abandon
    in-flight synthesis is to close the socket. Barge-in (issue 042) will
    therefore call ``close()`` and lazily reopen on the next turn.

Contract — items pushed to ``supervisor.outbound_audio_queue``
──────────────────────────────────────────────────────────────
* For every ``AudioOutput`` message: ``(generation, raw_ulaw_bytes)``
  where ``generation`` is the value ``synthesize_turn`` pinned at the start
  of the turn (``None`` is never used here — those bytes are real audio).
* Exactly one ``(generation, None)`` per successful turn, pushed by
  ``synthesize_turn`` after the reader task observes Sarvam's final event
  (or a fallback timeout fires — see ``_TURN_DONE_TIMEOUT_S``).

Caller responsibilities
───────────────────────
The orchestrator (issue 039) must serialize calls to ``synthesize_turn``
(one at a time). Concurrent calls are not supported — the SDK's
``configure → convert → flush`` ordering assumes a single in-flight turn.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import TYPE_CHECKING

from backend.config import load_settings

if TYPE_CHECKING:
    from backend.call_supervisor import CallSupervisor

__all__ = ["AsyncSarvamTTSClient"]

log = logging.getLogger(__name__)


# ─── Tunables ─────────────────────────────────────────────────────────────────

#: Sarvam TTS model. ``bulbul:v3`` is the only model that handles the mixed
#: Devanagari + Latin script our consultation script depends on (Rung 1 spike).
_TTS_MODEL: str = "bulbul:v3"

#: BCP-47 target language code for the Hinglish (Devanagari + Latin) call path.
_HINGLISH_LANG_CODE: str = "hi-IN"

#: BCP-47 target language code for the English-only call path.
_ENGLISH_LANG_CODE: str = "en-IN"

#: TTS output codec — μ-law 8 kHz mono matches the upstream Twilio Media
#: Streams contract end-to-end. No resampling or transcoding required.
_TTS_OUTPUT_CODEC: str = "mulaw"

#: TTS sample rate — must match Twilio's native 8 kHz.
_TTS_SAMPLE_RATE: int = 8000

#: Whether Sarvam should preprocess mixed Hindi/English words + numeric
#: entities (lakh, EMI, dates) before synthesis. ``True`` is required for
#: natural Hinglish; English-only calls still benefit (dates + numerals).
_TTS_PREPROCESSING: bool = True

#: Tell Sarvam to emit an explicit ``EventResponse(event_type="final")``
#: once the last ``AudioOutput`` for the turn has been generated. Without
#: this, the only "end of turn" signal is silence — which we cannot
#: distinguish from a slow network.
_TTS_SEND_COMPLETION_EVENT: str = "true"

#: How long ``synthesize_turn`` will wait for the reader task to observe
#: the final event before falling back to pushing the ``None`` marker
#: anyway. Sized generously above Sarvam's worst-case synthesis latency
#: (multi-second long utterances, network jitter) so a healthy turn never
#: trips it. The fallback is only there to keep the pipeline moving if
#: Sarvam ever changes its protocol or drops the completion event.
_TURN_DONE_TIMEOUT_S: float = 30.0


# ─── Custom exceptions ────────────────────────────────────────────────────────


class SarvamTTSError(RuntimeError):
    """Raised when the Sarvam streaming TTS client cannot complete a turn."""

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.__cause__ = cause


# ─── Client ───────────────────────────────────────────────────────────────────


class AsyncSarvamTTSClient:
    """
    Per-call Sarvam Bulbul v3 streaming TTS client.

    One instance is created at the start of a call, ``connect()``-ed once,
    and ``synthesize_turn()``-ed repeatedly for the call's duration.
    ``close()`` releases the socket and reader task (called from
    ``CallSupervisor.trigger_teardown``).

    The class is **not** thread-safe — designed to be driven from the single
    asyncio event loop carrying the voice pipeline. Do not share an instance
    across coroutines that run concurrently.
    """

    __slots__ = (
        # ── SDK handles (populated by connect()) ─────────────────────────
        "_client",
        "_ws_context_manager",
        "_ws",
        # ── Per-call state ────────────────────────────────────────────────
        "_supervisor",
        "_reader_task",
        "_current_generation",
        "_turn_done_event",
        # ── Lifecycle flags ───────────────────────────────────────────────
        "_connected",
        "_closed",
    )

    def __init__(self) -> None:
        # SDK client + WebSocket — populated by connect().
        self._client = None  # type: ignore[assignment]
        self._ws_context_manager = None  # type: ignore[assignment]
        self._ws = None  # type: ignore[assignment]

        # The supervisor whose outbound_audio_queue receives audio chunks.
        # Wired in connect() and cleared in close().
        self._supervisor: "CallSupervisor | None" = None  # type: ignore[assignment]

        # Background task that drains the WebSocket and pushes audio chunks
        # to the supervisor. Spawned in connect(), awaited in close().
        self._reader_task: asyncio.Task[None] | None = None

        # Stamped onto every AudioOutput chunk the reader pushes, so the
        # outbound sender can drop stale frames after barge-in (issue 042).
        # ``synthesize_turn`` pins this BEFORE calling configure() so any
        # audio arriving mid-turn carries the correct tag.
        self._current_generation: int = 0

        # Set by the reader when it observes Sarvam's ``event_type="final"``
        # completion event. ``synthesize_turn`` waits on it (with a timeout)
        # before pushing the end-of-turn ``None`` marker.
        self._turn_done_event: asyncio.Event = asyncio.Event()

        # Lifecycle guards — used to make close() idempotent and to refuse
        # synthesize_turn on a half-initialized client.
        self._connected: bool = False
        self._closed: bool = False

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    async def connect(self, supervisor: "CallSupervisor") -> None:
        """
        Open the persistent Sarvam TTS WebSocket and spawn the reader task.

        Parameters
        ----------
        supervisor : CallSupervisor
            The supervisor for the current call. Its ``outbound_audio_queue``
            receives audio chunks and the ``None`` end-of-turn marker.

        Raises
        ------
        SarvamTTSError
            If the WebSocket cannot be opened (bad credentials, network
            failure, SDK API error). The caller should treat this as a
            pre-call failure — typically by triggering the supervisor's
            teardown with ``provider_fault=True``.
        """
        if self._connected:
            raise SarvamTTSError(
                "AsyncSarvamTTSClient.connect called twice without close()"
            )
        if self._closed:
            raise SarvamTTSError(
                "AsyncSarvamTTSClient.connect called on a closed client"
            )

        settings = load_settings()
        api_key = settings.sarvam_api_key.get_secret_value()

        # Import here (not at module scope) so the module is importable in
        # environments that lack the SDK (e.g. lightweight unit tests that
        # stub the entire sarvamai package).
        from sarvamai import AsyncSarvamAI

        try:
            self._client = AsyncSarvamAI(api_subscription_key=api_key)
            self._ws_context_manager = self._client.text_to_speech_streaming.connect(
                model=_TTS_MODEL,
                send_completion_event=_TTS_SEND_COMPLETION_EVENT,
            )
            self._ws = await self._ws_context_manager.__aenter__()
        except Exception as exc:
            log.error(
                "sarvam_tts.connect: failed to open WebSocket: %s",
                exc,
                exc_info=True,
            )
            # Best-effort cleanup of partially-initialized state.
            await self._safe_release_context_manager()
            raise SarvamTTSError(
                f"Failed to open Sarvam TTS WebSocket: {exc}", cause=exc
            ) from exc

        self._supervisor = supervisor
        self._connected = True

        # Spawn the background reader task. We deliberately do NOT make
        # this a long-lived per-call supervisor task (issue 041 will
        # consolidate task ownership); for now, the client owns the
        # reader and close() awaits it.
        self._reader_task = asyncio.create_task(
            self._reader_loop(),
            name="sarvam-tts-reader",
        )
        log.info(
            "sarvam_tts.connect: WebSocket open model=%s speaker=%s CallSid=%s",
            _TTS_MODEL,
            settings.sarvam_tts_speaker,
            supervisor.call_sid,
        )

    async def close(self) -> None:
        """
        Tear down the client: cancel the reader task, close the WebSocket.

        Idempotent — safe to call multiple times. Always returns cleanly;
        exceptions during socket close are logged but not raised, since
        teardown must never block the supervisor's other cleanup steps.
        """
        if self._closed:
            return
        self._closed = True

        # ── Cancel the reader first so it stops consuming the socket ─────
        if self._reader_task is not None and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                log.warning(
                    "sarvam_tts.close: reader task raised during shutdown: %s",
                    exc,
                )
            self._reader_task = None

        # ── Exit the SDK's async context manager (closes the socket) ────
        await self._safe_release_context_manager()

        # ── Drop references ──────────────────────────────────────────────
        self._supervisor = None  # type: ignore[assignment]
        self._ws = None
        self._client = None
        self._connected = False
        log.info("sarvam_tts.close: client released")

    # ─── Turn API ─────────────────────────────────────────────────────────────

    async def synthesize_turn(
        self,
        text: str,
        *,
        language: str,
        generation: int,
        supervisor: "CallSupervisor",
    ) -> None:
        """
        Synthesize one utterance of Riya's reply.

        Sends ``configure → convert → flush`` over the persistent socket,
        waits for Sarvam's ``final`` event (or a fallback timeout), then
        pushes ``(generation, None)`` to the supervisor's outbound queue
        so the sender emits its final ``mark``.

        Parameters
        ----------
        text : str
            The complete utterance Riya should speak. **Not** streamed —
            we send the full text in one ``convert()`` call per the
            LLM→TTS coupling decision (buffer Mercury's full reply, then
            stream to Sarvam — see backend/call_supervisor.py docstring).
        language : str
            ``"English"`` or ``"Hinglish"``. Mapped to the BCP-47 code
            Sarvam expects (``en-IN`` / ``hi-IN``).
        generation : int
            The supervisor's current generation counter. Stamped onto
            every audio chunk so the outbound sender can drop stale
            frames after barge-in. The caller must pass the value
            BEFORE calling this method so a barge-in that races with
            the synthesize call has a consistent tag.
        supervisor : CallSupervisor
            The supervisor for the current call. Its ``outbound_audio_queue``
            receives audio chunks and the end-of-turn ``None`` marker.
            Must match the supervisor passed to ``connect()``.

        Raises
        ------
        SarvamTTSError
            If the client is not connected, or Sarvam returns an error
            response mid-turn, or the WebSocket raises mid-turn.
        """
        if not self._connected or self._ws is None:
            raise SarvamTTSError(
                "synthesize_turn called before connect() or after close()"
            )
        if self._closed:
            raise SarvamTTSError("synthesize_turn called on a closed client")
        if supervisor is not self._supervisor:
            raise SarvamTTSError(
                "synthesize_turn called with a supervisor that does not "
                "match the one passed to connect()"
            )
        if not text:
            # Treat empty text as a no-op turn — push the end-of-turn marker
            # so the outbound sender still emits its final mark, but skip
            # the Sarvam round-trip entirely.
            log.debug(
                "sarvam_tts.synthesize_turn: empty text for generation=%d — "
                "pushing end-of-turn marker without a Sarvam call",
                generation,
            )
            supervisor.outbound_audio_queue.put_nowait((generation, None))
            return

        # Pin the generation BEFORE the first Sarvam message so any audio
        # that arrives mid-call carries the correct tag. Bumping
        # ``_turn_done_event`` clears any stale signal from a prior turn.
        self._current_generation = generation
        self._turn_done_event.clear()

        settings = load_settings()
        target_language_code = self._language_to_bcp47(language)

        try:
            await self._ws.configure(
                target_language_code=target_language_code,
                speaker=settings.sarvam_tts_speaker,
                output_audio_codec=_TTS_OUTPUT_CODEC,
                speech_sample_rate=_TTS_SAMPLE_RATE,
                enable_preprocessing=_TTS_PREPROCESSING,
            )
            await self._ws.convert(text)
            await self._ws.flush()
        except Exception as exc:
            log.error(
                "sarvam_tts.synthesize_turn: Sarvam call failed for "
                "generation=%d: %s",
                generation,
                exc,
                exc_info=True,
            )
            raise SarvamTTSError(
                f"Sarvam turn failed (generation={generation}): {exc}",
                cause=exc,
            ) from exc

        # Wait for the reader task to observe Sarvam's ``final`` event
        # (the authoritative end-of-turn signal). The fallback timeout
        # only fires if Sarvam stops sending the event entirely — in
        # which case we still push ``None`` so the outbound sender can
        # emit its final mark and the pipeline can advance.
        try:
            await asyncio.wait_for(
                self._turn_done_event.wait(),
                timeout=_TURN_DONE_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            log.warning(
                "sarvam_tts.synthesize_turn: timed out after %.1fs waiting "
                "for Sarvam final event (generation=%d) — pushing "
                "end-of-turn marker anyway",
                _TURN_DONE_TIMEOUT_S,
                generation,
            )

        # Authoritative end-of-turn signal for the outbound sender.
        # ``put_nowait`` is non-blocking and synchronous on the queue's
        # internal lock, so this cannot interleave with reader pushes.
        supervisor.outbound_audio_queue.put_nowait((generation, None))

    # ─── Background reader ────────────────────────────────────────────────────

    async def _reader_loop(self) -> None:
        """
        Drain the WebSocket and push decoded audio chunks to the supervisor.

        Runs until the WebSocket raises, the supervisor's shutdown_event is
        set, or the task is cancelled (in close()). Three message shapes
        are handled:

        * ``AudioOutput`` — base64 decode ``data.audio`` and push
          ``(self._current_generation, raw_bytes)`` to the supervisor's
          outbound audio queue.
        * ``EventResponse`` with ``data.event_type == "final"`` — Sarvam's
          authoritative end-of-turn signal. Set ``_turn_done_event`` so
          the in-flight ``synthesize_turn`` can return. Any
          ``data.message`` is logged at debug for observability.
        * ``ErrorResponse`` — logged at error and the WebSocket is
          considered broken; loop exits so the supervisor's teardown can
          engage.

        Anything else (defensive) is logged at debug and ignored.
        """
        # Imports are local so the module is importable without the SDK
        # present (some tests stub the entire sarvamai package).
        from sarvamai import AudioOutput, ErrorResponse, EventResponse

        ws = self._ws
        supervisor = self._supervisor
        if ws is None or supervisor is None:
            log.error(
                "sarvam_tts._reader_loop: started without a WebSocket or "
                "supervisor — exiting"
            )
            return

        log.debug("sarvam_tts._reader_loop: starting CallSid=%s", supervisor.call_sid)
        try:
            while True:
                # Honour supervisor teardown so we don't block a clean
                # close() — the WebSocket itself will close on its own
                # once the supervisor tears down.
                if supervisor.shutdown_event.is_set():
                    log.debug(
                        "sarvam_tts._reader_loop: supervisor shutdown_event "
                        "set, exiting"
                    )
                    return

                try:
                    msg = await ws.recv()
                except Exception as exc:
                    # ConnectionClosed / CancelledError / network blip —
                    # any reader-side failure exits the loop. The
                    # supervisor's own watchdog (issue 043) handles the
                    # upstream "no audio for too long" path; we just need
                    # to stop reading and let the teardown sequence run.
                    log.warning(
                        "sarvam_tts._reader_loop: ws.recv() raised, exiting "
                        "(CallSid=%s): %s",
                        supervisor.call_sid,
                        exc,
                    )
                    return

                if isinstance(msg, AudioOutput):
                    self._handle_audio_output(msg)
                elif isinstance(msg, EventResponse):
                    self._handle_event_response(msg)
                elif isinstance(msg, ErrorResponse):
                    self._handle_error_response(msg)
                else:
                    log.debug(
                        "sarvam_tts._reader_loop: unhandled message type "
                        "%s (CallSid=%s)",
                        type(msg).__name__,
                        supervisor.call_sid,
                    )
        except asyncio.CancelledError:
            # close() cancelled us — propagate silently.
            raise
        except Exception as exc:  # pragma: no cover — defensive
            log.exception(
                "sarvam_tts._reader_loop: unexpected exception "
                "(CallSid=%s): %s",
                supervisor.call_sid if supervisor else "?",
                exc,
            )

    # ─── Reader handlers (one responsibility each) ──────────────────────────

    def _handle_audio_output(self, msg: object) -> None:
        """Decode a Sarvam AudioOutput and push it to the outbound queue."""
        # ``msg.data.audio`` is documented as a base64-encoded string. The
        # model is frozen + ``extra="allow"`` so we read defensively via
        # ``getattr`` in case Sarvam extends the schema.
        data = getattr(msg, "data", None)
        audio_b64 = getattr(data, "audio", None)
        if not isinstance(audio_b64, str) or not audio_b64:
            log.warning(
                "sarvam_tts: AudioOutput with missing/empty audio payload "
                "(generation=%d) — skipping",
                self._current_generation,
            )
            return

        try:
            chunk_bytes = base64.b64decode(audio_b64, validate=True)
        except (ValueError, TypeError) as exc:
            # ``validate=True`` raises on non-alphabet characters. Anything
            # else is a programming error worth surfacing.
            log.warning(
                "sarvam_tts: failed to base64-decode AudioOutput audio "
                "(generation=%d, len=%d): %s",
                self._current_generation,
                len(audio_b64),
                exc,
            )
            return

        supervisor = self._supervisor
        if supervisor is None:
            return

        # Issue 042: accumulate decoded μ-law bytes per generation so a
        # barge-in can compute the interrupted turn's ``total_ms`` from audio
        # SYNTHESIZED (some of which may still be queued or in flight to
        # Twilio), not merely from what the outbound sender has put on the
        # wire.  ``isinstance`` guards against minimal stub supervisors in
        # unit tests that do not carry the counter dict.
        counters = getattr(supervisor, "synthesized_bytes_by_generation", None)
        if isinstance(counters, dict):
            gen = self._current_generation
            counters[gen] = counters.get(gen, 0) + len(chunk_bytes)

        supervisor.outbound_audio_queue.put_nowait(
            (self._current_generation, chunk_bytes)
        )

    def _handle_event_response(self, msg: object) -> None:
        """React to Sarvam's EventResponse messages — currently only ``final``."""
        data = getattr(msg, "data", None)
        event_type = getattr(data, "event_type", None)
        message = getattr(data, "message", None)

        if event_type == "final":
            log.debug(
                "sarvam_tts: received final event for generation=%d (%s)",
                self._current_generation,
                message,
            )
            self._turn_done_event.set()
            return

        # Defensive — log unknown event types at debug so a future Sarvam
        # protocol extension does not crash us.
        log.debug(
            "sarvam_tts: unhandled EventResponse event_type=%r message=%r",
            event_type,
            message,
        )

    def _handle_error_response(self, msg: object) -> None:
        """Log Sarvam errors loudly. The supervisor's watchdog handles recovery."""
        data = getattr(msg, "data", None)
        message = getattr(data, "message", None)
        code = getattr(data, "code", None)
        log.error(
            "sarvam_tts: Sarvam ErrorResponse code=%s message=%s "
            "(generation=%d) — exiting reader loop",
            code,
            message,
            self._current_generation,
        )
        # Setting the event lets any in-flight synthesize_turn return
        # immediately rather than waiting on the timeout — the turn is
        # broken anyway, and the supervisor's watchdog will pick up the
        # silence.
        self._turn_done_event.set()

    # ─── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _language_to_bcp47(language: str) -> str:
        """Map the supervisor's ``"English"`` / ``"Hinglish"`` enum to BCP-47."""
        if language == "Hinglish":
            return _HINGLISH_LANG_CODE
        if language == "English":
            return _ENGLISH_LANG_CODE
        raise SarvamTTSError(
            f"Unsupported language {language!r}; expected 'English' or 'Hinglish'"
        )

    async def _safe_release_context_manager(self) -> None:
        """Exit the SDK's async context manager, swallowing errors."""
        cm = self._ws_context_manager
        if cm is None:
            return
        try:
            await cm.__aexit__(None, None, None)
        except Exception as exc:
            log.warning(
                "sarvam_tts: SDK context manager exit raised: %s", exc
            )
        finally:
            self._ws_context_manager = None