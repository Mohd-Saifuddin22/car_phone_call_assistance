"""
backend/call_supervisor.py — Call Supervisor & Idempotent Teardown (Issue 032).

The CallSupervisor is the single owner of the per-call teardown lifecycle for
a live outbound AI call.  Every ending trigger — webhooks (status-callback,
amd-callback), the orphan-window handler in arrange_ai_call, future pipeline
faults — funnels through ``trigger_teardown()``, which is guarded by a one-shot
latch so the teardown sequence runs exactly once regardless of how many
triggers fire concurrently.

This module also owns the in-memory **active-calls registry** (``_active_calls``)
keyed by Twilio CallSid.  Webhooks look up the live supervisor via
``get_call(call_sid)`` and signal it; if none exists (stale callback,
post-restart) they return 200 and trust the startup sweep (Issue 029) as the
authoritative resolver for orphaned leads via the database.

The teardown sequence (fixed, in order)
───────────────────────────────────────
1. End the Twilio call (status="completed") — skipped if the outcome implies
   the call is already gone.
2. # TODO(034): stop pipeline tasks (STT/TTS readers, driver, sender).
3. # TODO(038): finalize Matcher (compute recommendations + blurb).
4. Persist outcome to DB (``call_attempts`` + ``leads``).
5. # TODO(046): close provider sockets (STT, TTS).
6. Release the concurrency slot — single guaranteed release.
7. Unregister from the active-calls registry.

Module exports
──────────────
Registry (in-memory, module-level):
    register_call(supervisor)  — add to registry by CallSid.
    get_call(call_sid)         — look up live supervisor; None if absent.
    unregister_call(call_sid)   — remove from registry (no-op if absent).

Helpers (moved from backend.api.arrange):
    _twilio_hangup_call(call_sid, account_sid, auth_token) -> bool
    _resolve_failed_attempt(pool, attempt_id, lead_id, failure_reason) -> None
    _resolve_completed_attempt(pool, attempt_id, lead_id, completed_timeout) -> None

Class:
    CallSupervisor  — single owner of teardown for one call.

No module calls ``load_settings()`` at import time.
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.audio.outbound_sender import ClearPlayback
    from backend.concurrency import ConcurrencyManager
    from backend.database._pool import ConnectionPool
    from backend.interrupted_turn import InterruptedTurn

__all__ = [
    "CallSupervisor",
    "register_call",
    "get_call",
    "unregister_call",
]

log = logging.getLogger(__name__)


# ─── Tunables ─────────────────────────────────────────────────────────────────

#: How long ``trigger_teardown`` waits for the Deepgram STT reader to
#: self-close gracefully (i.e. observe ``shutdown_event`` between events,
#: send ``{"type": "CloseStream"}``, and drain the final transcript) before
#: falling back to ``cancel()``.  Kept short so a hung reader cannot stall
#: the rest of the teardown sequence (DB resolve, slot release, unregister).
_STT_GRACEFUL_SHUTDOWN_TIMEOUT_S: float = 1.5


#: Bounded wait for the outbound sender to acknowledge a barge-in
#: ``ClearPlayback`` (Issue 042).  The stop sequence must confirm the sole WS
#: writer actually sent the clear before it cancels the turn and closes
#: Sarvam — but it must never hang if the sender is gone (no media stream, or
#: already exited).  Sized generously above a single ``send_json`` round-trip;
#: on timeout the sequence proceeds (the generation bump already invalidated
#: the stale audio on the sender side).
_CLEAR_ACK_TIMEOUT_S: float = 2.0


class BargeinClearResult(Enum):
    """
    Outcome of the barge-in ``ClearPlayback`` request to the sole WS writer.

    Distinguishes "we confirmed the clear went out" from "we could not
    confirm the clear went out" — the latter mandates safe teardown because
    Twilio's playback buffer may not have been flushed and stale audio
    could keep playing past the barge-in.
    """

    SUCCESS = "success"
    """Sender reported ``ack=True`` — the clear frame was sent."""

    SEND_FAILURE = "send_failure"
    """Sender reported ``ack=False`` — ``send_json`` raised."""

    ACK_TIMEOUT = "ack_timeout"
    """Sender did not ack within ``_CLEAR_ACK_TIMEOUT_S`` — clear may or may
    not have been sent; we cannot confirm."""

    SHUTDOWN_DURING_ACK = "shutdown_during_ack"
    """Teardown was already in progress when the clear was enqueued —
    the sender is exiting and will not process further commands."""

    ENQUEUE_FAILED = "enqueue_failed"
    """The control queue refused the put (missing/invalid channel) —
    the clear was never delivered."""


class BargeinClearFailureError(Exception):
    """
    Raised by the barge-in stop sequence when the ``ClearPlayback`` could
    not be confirmed as delivered to Twilio.

    The stop sequence always completes its normal cleanup (cancel the
    current turn, close/detach Sarvam, finalise interrupted metadata) and
    THEN escalates to ``trigger_teardown(provider_fault=True)`` — so the
    caller does NOT need to catch this exception, but it is exposed for
    tests and observability.
    """

    def __init__(self, call_sid: str, result: "BargeinClearResult") -> None:
        super().__init__(
            f"barge-in clear could not be confirmed delivered for "
            f"CallSid={call_sid} (result={result.value})"
        )
        self.call_sid = call_sid
        self.result = result


# ─── Active-calls registry (module-level, in-memory) ──────────────────────────

_active_calls: dict[str, "CallSupervisor"] = {}
"""
In-memory registry of live call supervisors, keyed by Twilio CallSid.

Lifecycle
─────────
- ``register_call(supervisor)`` adds to the registry after Twilio call
  creation succeeds (in arrange_ai_call).
- ``get_call(call_sid)`` returns the supervisor if present (used by webhooks).
- ``unregister_call(call_sid)`` removes from the registry at the end of
  teardown.

Cross-restart safety
────────────────────
The registry is in-memory only.  After a backend restart, no supervisors exist.
The startup sweep (Issue 029) handles orphaned Processing leads authoritatively
via the database — webhooks hitting "no supervisor found" return 200 and rely
on that sweep, not on in-memory state.
"""


def register_call(supervisor: "CallSupervisor") -> None:
    """Register a supervisor as the live owner of its CallSid."""
    _active_calls[supervisor.call_sid] = supervisor
    log.debug(
        "CallSupervisor registered: CallSid=%s lead=%d attempt=%d",
        supervisor.call_sid,
        supervisor.lead_id,
        supervisor.attempt_id,
    )


def get_call(call_sid: str) -> "CallSupervisor | None":
    """Look up the live supervisor for the given CallSid, or None if none."""
    return _active_calls.get(call_sid)


def unregister_call(call_sid: str) -> None:
    """Remove the supervisor for the given CallSid from the registry (no-op if absent)."""
    _active_calls.pop(call_sid, None)
    log.debug("CallSupervisor unregistered: CallSid=%s", call_sid)


# ─── Twilio helpers (moved from backend.api.arrange) ──────────────────────────


def _twilio_hangup_call(
    call_sid: str,
    twilio_account_sid: str,
    twilio_auth_token: str,
) -> bool:
    """
    Synchronously hang up a live Twilio call — MUST be called via run_in_executor.

    Sets call status to "completed" which causes Twilio to end the call.
    Returns True on success, False on any failure (missing credentials, missing
    CallSid, Twilio REST exception).  The caller treats both outcomes
    identically — teardown always completes; the hangup is best-effort
    credit-saving, not the source of truth.

    Moved from backend.api.arrange (Issue 032) so the supervisor can import it
    without creating a circular dependency.
    """
    from twilio.rest import Client as TwilioClient
    from twilio.base.exceptions import TwilioRestException

    if not twilio_account_sid or not twilio_auth_token:
        log.warning(
            "Twilio hangup skipped: credentials not configured (CallSid=%s).",
            call_sid,
        )
        return False

    if not call_sid:
        log.warning("Twilio hangup skipped: empty CallSid.")
        return False

    client = TwilioClient(twilio_account_sid, twilio_auth_token)

    try:
        # Setting status="completed" terminates the call cleanly.  This is
        # the documented way to end an in-progress Twilio call from the
        # server side (as opposed to letting the customer hang up).
        client.calls(call_sid).update(status="completed")
        log.info("Twilio call %s hung up via status=completed.", call_sid)
        return True
    except TwilioRestException as exc:
        log.error(
            "Twilio hangup failed for CallSid=%s: %s",
            call_sid,
            exc,
        )
        return False


# ─── Recording helpers (Issue 037) ────────────────────────────────────────────


def _twilio_delete_recordings(
    call_sid: str,
    twilio_account_sid: str,
    twilio_auth_token: str,
) -> int:
    """
    Synchronously delete every recording attached to the given call — MUST
    be called via run_in_executor.

    Issue 037 (discard-on-voicemail): Twilio's ``record=True`` flag
    captures every outbound call.  When the call resolves as
    ``no_answer`` (machine/fax), we MUST NOT retain the captured audio —
    it is a recording of a non-consenting party (the voicemail box) and
    keeping it would breach the consent disclosure we just played.
    This helper enumerates the call's recordings and issues a delete for
    each, then returns the count actually deleted.

    Defensive against every realistic failure mode:
      * Missing credentials / CallSid → returns 0 without raising.
      * No recordings on the call (yet, or never) → returns 0.
      * Twilio REST error during list or delete → logged, returns the
        count deleted so far (or 0).  Caller treats partial failure as
        best-effort cleanup; teardown proceeds regardless.

    Lives in ``call_supervisor.py`` (next to ``_twilio_hangup_call``) so
    the supervisor's teardown sequence can call it directly without an
    import cycle through ``backend.api.arrange``.
    """
    from twilio.rest import Client as TwilioClient
    from twilio.base.exceptions import TwilioRestException

    if not twilio_account_sid or not twilio_auth_token:
        log.warning(
            "Twilio recording delete skipped: credentials not configured "
            "(CallSid=%s).",
            call_sid,
        )
        return 0

    if not call_sid:
        log.warning("Twilio recording delete skipped: empty CallSid.")
        return 0

    client = TwilioClient(twilio_account_sid, twilio_auth_token)
    deleted = 0

    try:
        recordings = client.calls(call_sid).recordings.list()
    except TwilioRestException as exc:
        log.warning(
            "Twilio recording list failed for CallSid=%s: %s "
            "— skipping deletion (recording may already be gone).",
            call_sid,
            exc,
        )
        return 0
    except Exception as exc:  # pragma: no cover — defensive
        log.warning(
            "Twilio recording list raised unexpectedly for CallSid=%s: %s",
            call_sid,
            exc,
        )
        return 0

    for rec in recordings:
        try:
            client.recordings(rec.sid).delete()
            deleted += 1
            log.info(
                "Twilio recording %s deleted for CallSid=%s.",
                rec.sid,
                call_sid,
            )
        except TwilioRestException as exc:
            log.warning(
                "Twilio recording delete failed for %s (CallSid=%s): %s",
                rec.sid,
                call_sid,
                exc,
            )
        except Exception as exc:  # pragma: no cover — defensive
            log.warning(
                "Twilio recording delete raised unexpectedly for %s "
                "(CallSid=%s): %s",
                rec.sid,
                call_sid,
                exc,
            )

    return deleted


async def _delete_call_recordings_async(call_sid: str) -> int:
    """
    Async wrapper around :func:`_twilio_delete_recordings` — reads
    Twilio credentials from settings, runs the blocking SDK call in a
    thread-pool executor, and returns the deleted-recording count.

    Safe to call with no live event loop (the wrapper is a coroutine;
    it only blocks inside ``run_in_executor``).  Safe to call from any
    teardown step — the caller awaits it but does not propagate
    exceptions (the underlying sync helper swallows them and logs).
    """
    from backend.config import load_settings

    settings = load_settings()
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(
            None,
            _twilio_delete_recordings,
            call_sid,
            settings.twilio_account_sid,
            settings.twilio_auth_token.get_secret_value(),
        )
    except Exception as exc:  # pragma: no cover — belt-and-braces
        log.warning(
            "delete_call_recordings_async: unexpected failure for "
            "CallSid=%s: %s — continuing teardown.",
            call_sid,
            exc,
        )
        return 0


# ─── Attempt resolution helpers (moved from backend.api.arrange) ───────────────


async def _resolve_failed_attempt(
    pool,
    attempt_id: int,
    lead_id: int,
    failure_reason: str,
) -> None:
    """
    Resolve a failed call_attempt and transition the lead to Failed.

    Single transaction:
      1. UPDATE call_attempts SET outcome=failure_reason, ended_at=now.
      2. UPDATE leads SET status='Failed', failure_reason, updated_at=now.

    Moved from backend.api.arrange (Issue 032).  Used directly by
    arrange_ai_call's exception paths (where no supervisor exists) and by
    CallSupervisor._persist_outcome.
    """
    from backend.database._txn import transaction

    async with pool.connection() as conn:
        async with transaction(conn):
            await conn.execute(
                """
                UPDATE call_attempts
                SET outcome = ?, ended_at = datetime('now')
                WHERE id = ?;
                """,
                (failure_reason, attempt_id),
            )
            await conn.execute(
                """
                UPDATE leads
                SET status          = 'Failed',
                    failure_reason   = ?,
                    completed_timeout = 0,
                    updated_at        = datetime('now')
                WHERE id = ?;
                """,
                (failure_reason, lead_id),
            )


async def _resolve_completed_attempt(
    pool,
    attempt_id: int,
    lead_id: int,
    completed_timeout: bool,
) -> None:
    """
    Resolve a completed call_attempt and transition the lead to Completed.

    Single transaction:
      1. UPDATE call_attempts SET outcome='completed', ended_at=now.
      2. UPDATE leads SET status='Completed', completed_timeout, updated_at=now.

    Moved from backend.api.arrange (Issue 032).  Used by CallSupervisor._persist_outcome.
    """
    from backend.database._txn import transaction

    async with pool.connection() as conn:
        async with transaction(conn):
            await conn.execute(
                """
                UPDATE call_attempts
                SET outcome = 'completed', ended_at = datetime('now')
                WHERE id = ?;
                """,
                (attempt_id,),
            )
            await conn.execute(
                """
                UPDATE leads
                SET status           = 'Completed',
                    failure_reason    = NULL,
                    completed_timeout = ?,
                    updated_at        = datetime('now')
                WHERE id = ?;
                """,
                (1 if completed_timeout else 0, lead_id),
            )


# ─── CallSupervisor ───────────────────────────────────────────────────────────


class CallSupervisor:
    """
    Single owner of the per-call teardown lifecycle.

    Created by arrange_ai_call after Twilio call placement succeeds and the
    call has been registered in the active-calls registry.  Signalled by every
    ending trigger (webhooks, orphan-window handler, future pipeline faults).
    The one-shot latch in ``trigger_teardown()`` guarantees the teardown
    sequence runs exactly once, regardless of how many triggers fire.

    Parameters
    ----------
    lead_id : int
        The lead row id this call belongs to.
    attempt_id : int
        The call_attempts row id.
    call_sid : str
        The Twilio CallSid.
    concurrency_manager : ConcurrencyManager
        The manager whose slot will be released by this supervisor (single
        owner of the slot lifecycle for this call).
    pool : ConnectionPool
        The DB connection pool — used to persist the outcome.
    """

    # __slots__ pins the instance shape: fewer attribute lookups, fixed
    # memory footprint per live call, and any typo on a private attribute
    # surfaces as AttributeError at construction time instead of as a
    # silent bug.  asyncio.Lock is fine in slots (it owns its own __dict__
    # internally); the lock and queue are stored as references, not copied.
    __slots__ = (
        # ── Identity ────────────────────────────────────────────────────────
        "lead_id",
        "attempt_id",
        "call_sid",
        "language",
        # ── Owned resources ────────────────────────────────────────────────
        "_concurrency_manager",
        "_pool",
        "inbound_audio_queue",
        # ── Pipeline task topology (Issue 035) ────────────────────────────
        # ``completed_turn_queue`` is the writer→reader hand-off: the STT
        # task pushes confirmed EndOfTurn transcripts here for the
        # conversation driver (Issue 039) to consume.
        # ``shutdown_event`` is the one-bit teardown signal the STT task
        # polls between events; setting it tells the reader to send
        # ``CloseStream`` and exit cleanly without triggering a reconnect.
        # ``stt_task`` owns the Deepgram Flux WebSocket lifecycle.
        "completed_turn_queue",
        "shutdown_event",
        "stt_task",
        # ── Outbound audio (Issue 036) ──────────────────────────────────────
        # ``outbound_audio_queue`` carries ``(generation, chunk_or_None)``
        # tuples from the conversation driver (Issue 039) to the outbound
        # sender. ``None`` chunk signals "flush the repacketizer and emit
        # the final mark".  Unbounded — same trade-off as inbound.
        # ``generation_counter`` is the authoritative current generation,
        # bumped by the orchestrator on barge-in (Issue 042). The sender
        # drops any item whose generation is below this value.
        # ``is_speaking`` flips True when the sender starts a new
        # generation's first frame and back to False when the inbound
        # mark echo with ``final=true`` arrives.
        # ``played_duration_ms`` is updated by the inbound reader when a
        # mark echo arrives — the orchestrator uses it to decide how
        # much of a turn actually reached the customer (Issue 040/042).
        "outbound_audio_queue",
        "generation_counter",
        "is_speaking",
        "played_duration_ms",
        # ── Issue 042 — barge-in state ─────────────────────────────────────
        # ``sent_duration_ms`` mirrors ``played_duration_ms`` but tracks how
        # much audio the outbound sender has actually put on the wire for the
        # current generation (frames * 20 ms).  Barge-in snapshots it as the
        # interrupted turn's ``total_ms`` — real frame accounting, never a
        # text-proportional estimate.
        # ``current_turn_task`` is the in-flight ``run_turn`` coroutine wrapped
        # in its own task.  Barge-in cancels ONLY this — never ``mercury_task``
        # (the long-lived driver that owns the conversation history).
        # ``current_turn_text`` is the full text Riya is synthesizing this
        # turn, captured so barge-in can store it as interrupted-turn metadata.
        # ``bargein_task`` owns the asynchronous stop sequence spawned from the
        # Deepgram listener; the supervisor holds it so teardown can cancel it
        # and so its exceptions are observed via a done-callback.
        # ``_bargein_in_progress`` is a synchronous re-entrancy latch set the
        # instant a StartOfTurn is accepted, so a second StartOfTurn arriving
        # before the stop sequence disarms ``is_speaking`` cannot spawn a
        # duplicate stop sequence.
        # ``interrupted_turns`` is the in-memory seam Issue 048 persists.
        "sent_duration_ms",
        "current_turn_task",
        "current_turn_text",
        "bargein_task",
        "_bargein_in_progress",
        "interrupted_turns",
        # ── Issue 042 hardening ────────────────────────────────────────────
        # ``outbound_control_queue`` is the dedicated priority channel the
        # barge-in stop sequence uses to deliver a ``ClearPlayback`` to the
        # sole WS writer ahead of any queued (now-stale) audio.
        # ``bargein_complete`` is the completion barrier the Mercury driver
        # waits on after an interrupted turn — it stays set except while a
        # stop sequence is running, so the next turn never races the still-
        # closing Sarvam socket.
        # ``synthesized_bytes_by_generation`` accumulates decoded μ-law bytes
        # Sarvam produced per generation, so an interrupted turn's ``total_ms``
        # reflects audio SYNTHESIZED (some may still be queued), not merely
        # audio sent to Twilio.
        "outbound_control_queue",
        "bargein_complete",
        "synthesized_bytes_by_generation",
        # ── Issue 042 follow-on: broken-media-path flag ───────────────────
        # Set when the barge-in ``ClearPlayback`` could not be confirmed
        # delivered to Twilio (sender reported ``ack=False``, ack timed
        # out, the control queue refused the put, or the sender is gone).
        # Once set, EVERY subsequent ``ClearPlayback`` enqueued by a
        # later barge-in is silently discarded by the sender (with its
        # ack resolved ``False``) so an old clear can never reach the wire
        # and reset a newer-generation playback buffer.  The supervisor
        # also escalates the current call to safe teardown.
        "media_path_broken",
        # ── Conversation driver + TTS (Issue 039) ─────────────────────────
        # ``tts_client`` is owned by the Mercury driver task — it opens the
        # Sarvam WebSocket on first turn and closes it on teardown. The
        # supervisor only owns the reference so teardown can guarantee the
        # socket is closed in Step 5.
        # ``mercury_task`` is the long-lived driver coroutine created by
        # ``start_background_tasks()``. The supervisor awaits it in Step 2
        # (alongside the STT task) so any in-flight Mercury call returns
        # cleanly before the slot is released.
        "tts_client",
        "mercury_task",
        # ── Issue 041 — WebSocket bridge + matcher ownership ────────────────
        # ``reader_task`` is the per-call Twilio Media Streams inbound
        # reader coroutine (aggregates 20 ms frames → 80 ms chunks, then
        # pushes onto inbound_audio_queue).  ``sender_task`` is the
        # per-call outbound sender coroutine (slices outbound chunks →
        # 20 ms μ-law frames + marks).  Both are spawned by
        # ``attach_media_stream`` (called from the media-stream WS
        # handler) and owned exclusively by the supervisor for teardown.
        # ``matcher_task`` is the matcher consumer coroutine that
        # processes each completed customer turn through the
        # preference extractor + scorer (Issue 022).  Spawned by
        # ``start_background_tasks`` alongside the STT and driver.
        "reader_task",
        "sender_task",
        "matcher_task",
        # Direct handle on the matcher's internal asyncio.Task so
        # ``trigger_teardown`` can cancel it aggressively when
        # ``matcher.close()`` hangs.  Captured in ``start_background_tasks``.
        "_matcher_internal_task",
        # ── Issue 040 — Call-start synchronization + AMD verdict future ────
        # ``disclosure_complete`` is set once the pre-rendered legal
        # recording disclosure has finished playing.  The conversation
        # driver blocks on this event before entering the transcript-
        # polling loop so Riya does not greet over the disclosure.
        # ``disclosure_active`` mirrors the same condition as a cheap bool
        # (used in the hot path of every inbound media chunk to gate
        # barge-in — never ``await`` on the event in that path).  Both
        # flip together via ``mark_disclosure_complete``.
        # ``amd_verdict`` is a Future carrying the Twilio async-AMD
        # verdict string (``"human"`` / ``"machine_end"`` / ``"fax"`` /
        # ``"unknown"`` …).  Populated by the AMD webhook (which now
        # calls ``set_result`` for EVERY verdict — human/unknown
        # included — BEFORE any teardown).  The driver awaits this
        # future with a bounded timeout so a dropped webhook cannot
        # stall the greeting indefinitely.
        "disclosure_complete",
        "disclosure_active",
        "amd_verdict",
        # ── Teardown latch + coordination ──────────────────────────────────
        "_teardown_run",
        "_teardown_lock",
        # ── Diagnostic outcome capture ─────────────────────────────────────
        "_last_outcome_status",
        "_last_failure_reason",
        "_last_completed_timeout",
    )

    def __init__(
        self,
        *,
        lead_id: int,
        attempt_id: int,
        call_sid: str,
        language: str,
        concurrency_manager: "ConcurrencyManager",
        pool: "ConnectionPool",
    ) -> None:
        # ``language`` is the lead's preferred language (English | Hinglish).
        # It is required — it drives STT model + language-hint selection in
        # ``backend.audio.deepgram_stt`` (and will drive TTS voice and
        # Mercury variant strings when those lands).  No default: silently
        # picking a language would corrupt transcription.
        if language not in ("English", "Hinglish"):
            raise ValueError(
                f"CallSupervisor.language must be 'English' or 'Hinglish', "
                f"got {language!r}"
            )
        self.lead_id = lead_id
        self.attempt_id = attempt_id
        self.call_sid = call_sid
        self.language = language
        self._concurrency_manager = concurrency_manager
        self._pool = pool

        # Issue 034: the media-stream inbound reader (backend/api/media_stream.py)
        # pushes ~80 ms Deepgram-ready 640-byte raw chunks onto this queue via
        # put_nowait so the reader never blocks on downstream generation.  The
        # downstream pipeline (STT reader) consumes from it; the supervisor itself
        # only owns the lifecycle (no consumer logic here).  Default size is
        # unbounded — backpressure is the responsibility of the orchestrator
        # (Issue 041), not this queue.
        self.inbound_audio_queue: "asyncio.Queue[bytes]" = asyncio.Queue()

        # Issue 035: pipeline task topology.
        #
        # ``completed_turn_queue`` carries finalized customer transcripts from
        # the Deepgram STT reader to the conversation driver (Issue 039).  The
        # STT task pushes here via ``put_nowait`` on every confirmed
        # ``EndOfTurn``; never blocks.  Unbounded by default — backpressure
        # is the orchestrator's problem.
        self.completed_turn_queue: "asyncio.Queue[str]" = asyncio.Queue()
        # ``shutdown_event`` is the one-bit teardown signal the STT task polls
        # between events.  Set before cancelling the task so the reader
        # knows to send ``CloseStream`` (graceful) instead of treating the
        # cancel as a fault.
        self.shutdown_event: asyncio.Event = asyncio.Event()
        # ``stt_task`` is the handle for the Deepgram Flux reader coroutine
        # created by ``start_background_tasks()``.  The supervisor owns its
        # lifecycle and awaits it during teardown so its socket is closed
        # cleanly before the slot is released.
        self.stt_task: "asyncio.Task[None] | None" = None

        # ── Issue 036: outbound audio plumbing ─────────────────────────────
        # ``outbound_audio_queue`` carries ``(generation, audio_chunk_or_None)``
        # items from the conversation driver. ``None`` chunk signals "end of
        # turn: flush the repacketizer and emit a final mark".  Unbounded
        # by default — same trade-off as inbound; backpressure belongs to
        # the orchestrator (Issue 041), not this queue.
        self.outbound_audio_queue: "asyncio.Queue[tuple[int, bytes | None] | ClearPlayback]" = asyncio.Queue()
        # ``generation_counter`` is the authoritative current generation;
        # the orchestrator bumps it on barge-in.  Starts at 0 (no
        # generation in flight) so the first utterance uses gen=1.
        self.generation_counter: int = 0
        # ``is_speaking`` is True while Riya's audio is on the wire.
        # Flipped to True by the sender when it starts a new generation's
        # first frame; flipped back to False by the inbound reader when
        # the corresponding final mark echo arrives.
        self.is_speaking: bool = False
        # ``played_duration_ms`` reports the offset (in milliseconds) of
        # the most recent mark echo for the current generation. Updated
        # by the inbound reader when a mark event arrives. The orchestrator
        # uses this to reason about how much of a turn actually reached
        # the customer (e.g. for barge-in accounting and final-turn blurb
        # generation in Issue 046).
        self.played_duration_ms: int = 0

        # ── Issue 042: barge-in state ──────────────────────────────────────
        # ``sent_duration_ms`` tracks how much audio the outbound sender has
        # actually put on the Twilio wire for the current generation
        # (frames * 20 ms).  It is NOT the interrupted turn's ``total_ms`` —
        # that comes from bytes SYNTHESIZED by Sarvam (see
        # ``synthesized_bytes_by_generation`` below); kept separately as it is
        # still useful context (played <= sent <= synthesized).
        self.sent_duration_ms: int = 0
        # The in-flight ``run_turn`` task for the current utterance.  Set by
        # the conversation driver's cancellable-turn wrapper via try/finally;
        # barge-in cancels ONLY this task, never ``mercury_task``.
        self.current_turn_task: "asyncio.Task[None] | None" = None
        # The full text Riya is currently synthesizing (set by ``run_turn``
        # just before it hands the reply to TTS).  Barge-in snapshots it as
        # the interrupted turn's full intended text.
        self.current_turn_text: str | None = None
        # The asynchronous barge-in stop sequence task (spawned from the
        # Deepgram listener via ``maybe_start_bargein``).  Owned here so
        # teardown can cancel it and its exceptions are observed.
        self.bargein_task: "asyncio.Task[None] | None" = None
        # Synchronous re-entrancy latch — set the moment a StartOfTurn is
        # accepted so a rapid second StartOfTurn cannot spawn a duplicate
        # stop sequence before ``is_speaking`` is disarmed.
        self._bargein_in_progress: bool = False
        # In-memory record of interrupted Riya turns (Issue 042); Issue 048
        # persists these to ``transcript_turns``.  ``InterruptedTurn`` is
        # imported lazily where records are appended (barge-in stop sequence)
        # to keep this module import-light; the annotation is a string.
        self.interrupted_turns: "list[InterruptedTurn]" = []

        # ── Issue 042 hardening: priority clear channel + barrier + bytes ──
        # Dedicated priority channel for the barge-in ``ClearPlayback`` so it
        # is never stuck behind queued (now-stale) audio on the audio queue.
        self.outbound_control_queue: "asyncio.Queue[ClearPlayback]" = asyncio.Queue()
        # Completion barrier: SET whenever no barge-in stop sequence is
        # running.  ``maybe_start_bargein`` clears it synchronously on accept;
        # the stop sequence sets it in its ``finally``.  The Mercury driver
        # waits on it after an interrupted turn so the next turn cannot start
        # (and reconnect Sarvam) while the old socket is still being closed.
        self.bargein_complete: asyncio.Event = asyncio.Event()
        self.bargein_complete.set()
        # Cumulative decoded μ-law bytes Sarvam produced, keyed by generation.
        # Updated by the Sarvam reader; read (and retired) by the barge-in
        # stop sequence to compute ``total_ms`` from synthesized audio.
        self.synthesized_bytes_by_generation: dict[int, int] = {}
        # Broken-media-path flag: set when a barge-in ``ClearPlayback`` could
        # not be confirmed delivered to Twilio.  Once set, all subsequent
        # ``ClearPlayback`` commands on this call are silently discarded by
        # the sender so an old clear can never reset a newer-generation
        # playback buffer; the supervisor also escalates to safe teardown.
        self.media_path_broken: bool = False

        # ── Issue 039: conversation driver + TTS client ─────────────────
        # Both are populated by ``start_background_tasks()`` after the STT
        # task is spawned.  ``tts_client`` is owned by the Mercury driver
        # task — it opens the Sarvam WebSocket on its first turn and the
        # driver is responsible for closing it on teardown.  The
        # supervisor only stores the reference so teardown's Step 5 can
        # close the socket as a guaranteed-release safety net.
        self.tts_client: "object | None" = None  # AsyncSarvamTTSClient at runtime
        self.mercury_task: "asyncio.Task[None] | None" = None

        # ── Issue 041: WebSocket bridge tasks (Issue 036 + matcher) ────────
        # These are NOT spawned in __init__ — they are populated later:
        #
        #   * reader_task / sender_task — by ``attach_media_stream``
        #     (called from the WebSocket handler after the stream's
        #     ``start`` event arrives and the supervisor is registered).
        #     Until then they stay None and the WebSocket handler does
        #     NOT yet have a Twilio stream to bridge.
        #
        #   * matcher_task — by ``start_background_tasks`` after the
        #     STT task spawns.  The matcher is a long-lived consumer
        #     that reads transcripts and ranks vehicles off the driver's
        #     critical path, so it can start as soon as STT is up.
        self.reader_task: "asyncio.Task[None] | None" = None
        self.sender_task: "asyncio.Task[None] | None" = None
        self.matcher_task: "object | None" = None  # MatcherConsumer at runtime
        # Matcher wraps an internal asyncio.Task; we keep a direct ref
        # so teardown can cancel it aggressively (see trigger_teardown).
        self._matcher_internal_task: "asyncio.Task[None] | None" = None

        # One-shot latch: once True, trigger_teardown() is a no-op.
        self._teardown_run = False
        # asyncio.Lock guards against simultaneous triggers racing into the
        # teardown body before the latch flips.  Cheap, event-loop-only.
        self._teardown_lock = asyncio.Lock()

        # Last-request-wins outcome capture (diagnostic only — the latch
        # ensures only the FIRST request's args drive the actual teardown).
        self._last_outcome_status: str | None = None
        self._last_failure_reason: str | None = None
        self._last_completed_timeout: bool = False

        # ── Issue 040: call-start synchronization + AMD verdict future ────
        # ``disclosure_complete`` is set when the legal-recording
        # disclosure finishes playing.  The driver blocks on this event
        # so Riya never greets over the disclosure.  Created without a
        # running loop binding (``asyncio.Event()`` is loop-agnostic);
        # the supervisor is itself constructed inside an async handler so
        # there IS a running loop at ``__init__`` time, but no future
        # call ever requires the loop reference directly.
        self.disclosure_complete: asyncio.Event = asyncio.Event()
        # ``disclosure_active`` mirrors the disclosure as a bool for the
        # hot path: barge-in checks this on every inbound media chunk,
        # which would be needlessly expensive against an ``Event``.
        # ``True`` means "the disclosure is still playing — barge-in is
        # DISABLED".  Flipped to ``False`` by ``mark_disclosure_complete``
        # at the same moment ``disclosure_complete`` is set.
        self.disclosure_active: bool = True
        # ``amd_verdict`` is populated by the Twilio AMD webhook with the
        # raw ``AnsweredBy`` string (``"human"`` / ``"machine_end"`` /
        # ``"fax"`` / ``"unknown"`` …).  The driver awaits this future
        # with a bounded timeout (see ``run_mercury_conversation_driver``)
        # so a dropped webhook cannot stall the greeting forever — on
        # timeout the driver proceeds as if the verdict were ``human``.
        # ``asyncio.Future()`` is bound to the running loop at
        # construction (the supervisor is always built inside an async
        # handler), so no deprecation warning fires.
        self.amd_verdict: "asyncio.Future[str]" = asyncio.Future()

    @property
    def teardown_run(self) -> bool:
        """Whether the teardown sequence has already started (test observability)."""
        return self._teardown_run

    # ── Issue 040: disclosure complete setter ─────────────────────────────────

    def mark_disclosure_complete(self) -> None:
        """
        Atomically flip ``disclosure_active`` to ``False`` and set the
        ``disclosure_complete`` event.

        Called by the playback path that emits the legal-recording
        disclosure audio (currently the outbound sender / future
        disclosure playback coroutine) once the last byte has been
        delivered to Twilio.  Flipping the bool and setting the event in
        the same call removes the race window where the driver could
        observe ``disclosure_complete.is_set() == True`` while barge-in
        still saw ``disclosure_active == True`` (or vice versa) and made
        the wrong decision.

        Idempotent — both fields are monotonic and the ``Event.set()``
        and the bool assignment are themselves idempotent.  Safe to call
        from a finally clause or after a supervisor restart.
        """
        # Bool first, then event — order matters here only insofar as the
        # bool is the hot-path read on every inbound media chunk and the
        # event is the one-shot signal the driver awaits.  Setting the
        # bool first ensures any consumer waking from the event wait
        # already sees ``disclosure_active == False``.
        self.disclosure_active = False
        self.disclosure_complete.set()
        log.info(
            "CallSupervisor: disclosure marked complete CallSid=%s",
            self.call_sid,
        )

    # ── Issue 042: barge-in ───────────────────────────────────────────────────

    def begin_new_turn_generation(self) -> int:
        """
        Advance to a fresh generation for the next Riya utterance and reset
        the per-generation mark/playback accounting.

        Called once at the start of every Riya turn — the kickoff greeting
        included — by the conversation driver's cancellable-turn wrapper.  A
        distinct generation per utterance is what lets the outbound sender
        re-arm ``is_speaking`` (its generation-transition branch) on every
        turn, not just the first; without it barge-in would only ever fire on
        the greeting.  Playback/sent counters reset to 0 so a barge-in mid-turn
        snapshots durations for THIS turn, and stale mark echoes from the
        previous generation are rejected by the inbound reader's guards.

        Returns
        -------
        int
            The new generation counter value.
        """
        self.generation_counter += 1
        self.played_duration_ms = 0
        self.sent_duration_ms = 0
        self.current_turn_text = None
        return self.generation_counter

    def maybe_start_bargein(self) -> bool:
        """
        Fire the barge-in stop sequence on customer speech onset — at most
        once per Riya turn.

        Called synchronously from the Deepgram Flux listener on a
        ``StartOfTurn`` event (guaranteed non-empty transcript).  All guards
        are synchronous and run before any ``await`` so a second StartOfTurn
        arriving on the reader's next iteration — before the stop sequence has
        disarmed ``is_speaking`` — cannot spawn a duplicate sequence.

        Barge-in fires only when Riya is actually speaking, the (legally
        non-interruptible) disclosure is not playing, no stop sequence is
        already running, and teardown has not begun.

        Returns
        -------
        bool
            True iff a stop-sequence task was spawned.
        """
        if not self.is_speaking:
            return False
        if self.disclosure_active:
            return False
        if self._bargein_in_progress:
            return False
        if self.shutdown_event.is_set():
            return False

        # Latch synchronously BEFORE spawning so the next StartOfTurn is a
        # no-op until the stop sequence completes and the done-callback clears
        # the latch.  Clear the completion barrier in the same synchronous
        # window so a driver that observes the inner-turn cancellation can
        # never see the barrier still set from a prior (finished) barge-in.
        self._bargein_in_progress = True
        self.bargein_complete.clear()
        try:
            self.bargein_task = asyncio.create_task(
                self._run_bargein_stop_sequence(),
                name=f"bargein-{self.call_sid}",
            )
        except RuntimeError as exc:
            # No running loop — should never happen (the listener always runs
            # inside the event loop) but must not wedge the latch or barrier.
            self._bargein_in_progress = False
            self.bargein_complete.set()
            log.exception(
                "CallSupervisor.maybe_start_bargein: failed to spawn stop "
                "sequence CallSid=%s (%s).",
                self.call_sid,
                exc,
            )
            return False
        self.bargein_task.add_done_callback(self._on_bargein_done)
        return True

    def _on_bargein_done(self, task: "asyncio.Task[None]") -> None:
        """
        Clear the re-entrancy latch and surface any stop-sequence exception.

        Registered as the ``bargein_task`` done-callback so a crash in the
        stop sequence is logged (not silently swallowed as an unretrieved
        task exception) and the latch is always released for the next turn.
        """
        self._bargein_in_progress = False
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            log.error(
                "CallSupervisor: barge-in stop sequence raised CallSid=%s: %s",
                self.call_sid,
                exc,
                exc_info=exc,
            )

    async def _run_bargein_stop_sequence(self) -> None:
        """
        The full barge-in stop sequence (Issue 042), run in fixed order:

        A. Snapshot the interrupted generation id, intended text, and played
           ms BEFORE mutating anything.  (``total_ms`` is finalized in F, once
           Sarvam can no longer add audio.)
        B. Bump ``generation_counter`` and disarm ``is_speaking`` — every
           queued/in-flight audio item for the old generation is now stale and
           the outbound sender drops it on sight.
        C. Enqueue a ``ClearPlayback`` on the dedicated priority control queue
           and WAIT (bounded, shutdown-aware) for the sole WS writer to
           acknowledge it actually sent the Twilio ``clear`` — we do not
           proceed to turn cancellation / Sarvam shutdown merely assuming the
           clear was sent.
        D. Cancel + await ONLY ``current_turn_task`` (the in-flight
           ``run_turn``).  ``mercury_task`` — the long-lived driver that owns
           the conversation history and waits on the completion barrier — is
           never awaited here (that would be a circular wait).
        E. Close and detach the Sarvam TTS client (the only way to abandon
           in-flight synthesis); the next turn lazily reconnects.
        F. Finalize ``total_ms`` from the interrupted generation's SYNTHESIZED
           byte counter (now that the Sarvam socket is closed and no more
           audio can arrive), record the interrupted-turn metadata, and retire
           the counter.

        The completion barrier (``bargein_complete``) is released in
        ``finally`` on every path — success, exception, cancellation, teardown
        — so the Mercury driver can never deadlock waiting on it.

        The outer phone call, WS reader/sender, Deepgram connection, Mercury
        driver and conversation history all stay alive.
        """
        from backend.interrupted_turn import InterruptedTurn

        try:
            # ── A. Snapshot (before any mutation) ──────────────────────────
            interrupted_generation = self.generation_counter
            played_ms = self.played_duration_ms
            intended_text = self.current_turn_text or ""

            log.info(
                "CallSupervisor: barge-in on customer speech CallSid=%s "
                "interrupted_gen=%d played_ms=%d",
                self.call_sid,
                interrupted_generation,
                played_ms,
            )

            # ── B. Bump generation + disarm speaking/re-entry state ────────
            self.generation_counter += 1
            self.is_speaking = False

            # ── C. Flush Twilio playback via the priority channel + ack ────
            clear_result = await self._request_and_confirm_clear(
                self.generation_counter
            )
            if clear_result is not BargeinClearResult.SUCCESS:
                # The clear could NOT be confirmed delivered to Twilio.
                # Twilio's playback buffer may not have been flushed — we
                # cannot trust that Riya has gone silent.  Mark the media
                # path broken so every subsequent ClearPlayback is silently
                # discarded by the sender (and never reaches the wire to
                # reset a newer-generation playback buffer).  The rest of
                # the normal stop-sequence cleanup (D/E/F) still runs
                # below — the Sarvam socket is still closed, the current
                # turn is still cancelled, interrupted metadata is still
                # recorded.  After that we escalate to safe teardown so
                # the call ends rather than the driver proceeding to a new
                # turn against a half-broken media path.
                self.media_path_broken = True
                log.error(
                    "CallSupervisor: barge-in clear failed "
                    "(result=%s) CallSid=%s — marking media path broken; "
                    "will escalate to safe teardown after stop sequence.",
                    clear_result.value,
                    self.call_sid,
                )

            # ── D. Cancel + await ONLY the in-flight turn task ─────────────
            current = asyncio.current_task()
            turn_task = self.current_turn_task
            if (
                turn_task is not None
                and turn_task is not current
                and not turn_task.done()
            ):
                turn_task.cancel()
                try:
                    await turn_task
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    log.warning(
                        "CallSupervisor: in-flight turn raised during barge-in "
                        "cancel CallSid=%s: %s — continuing stop sequence.",
                        self.call_sid,
                        exc,
                    )

            # ── E. Close + detach the Sarvam client (next turn reconnects) ─
            # Null the handle FIRST so a concurrent teardown reading it sees
            # None and skips its own close — ``close()`` is idempotent anyway.
            tts_client = self.tts_client
            self.tts_client = None
            if tts_client is not None:
                try:
                    await tts_client.close()
                except Exception as exc:
                    log.warning(
                        "CallSupervisor: Sarvam close raised during barge-in "
                        "CallSid=%s: %s — continuing stop sequence.",
                        self.call_sid,
                        exc,
                    )

            # ── F. Finalize total_ms from SYNTHESIZED bytes (post-close) ───
            # Only now that the Sarvam socket is closed can we be sure no more
            # audio will land for the interrupted generation.  μ-law 8 kHz is
            # 1 byte/sample, so duration_ms = bytes * 1000 / 8000.  Accumulate
            # first, divide once (never round per chunk).  ``pop`` retires the
            # counter so it cannot affect a later turn.
            synthesized_bytes = self.synthesized_bytes_by_generation.pop(
                interrupted_generation, 0
            )
            total_ms = synthesized_bytes * 1000 // 8000

            self.interrupted_turns.append(
                InterruptedTurn.from_snapshot(
                    generation=interrupted_generation,
                    full_text=intended_text,
                    played_ms=played_ms,
                    total_ms=total_ms,
                )
            )
            log.info(
                "CallSupervisor: barge-in stop sequence complete CallSid=%s "
                "new_gen=%d played_ms=%d total_ms=%d interrupted_turns=%d",
                self.call_sid,
                self.generation_counter,
                played_ms,
                total_ms,
                len(self.interrupted_turns),
            )

            # ── G. Clear failure → escalate to safe teardown ───────────────
            # If the ClearPlayback could not be confirmed delivered (set
            # in step C above), the media path is unreliable.  After ALL
            # normal cleanup completes, escalate to safe teardown so the
            # Mercury driver does NOT start another turn against a half-
            # broken state.  Triggered AFTER the barrier is set in the
            # finally below — but the await is safe to call here because
            # trigger_teardown's own latch is one-shot.
            #
            # ``is not None`` guards against the self-trigger case where
            # the stop sequence is itself running under teardown's
            # already-acquired lock.
            if self.media_path_broken and not self._teardown_run:
                # ``await`` here is the canonical escape hatch; the call
                # below acquires the teardown lock, runs the full 7-step
                # sequence, and returns.  Step 1 (Twilio hangup) will run
                # cleanly because the media path is broken anyway.
                await self.trigger_teardown(
                    outcome_status="Failed",
                    failure_reason="error",
                    provider_fault=True,
                )
        finally:
            # Release the completion barrier on EVERY path so the Mercury
            # driver's post-interruption wait can never deadlock.
            self.bargein_complete.set()

    async def _request_and_confirm_clear(
        self, generation: int
    ) -> "BargeinClearResult":
        """
        Enqueue a ``ClearPlayback`` on the priority control queue and wait
        (bounded, shutdown-aware) for the sole WS writer to confirm it sent
        the Twilio ``clear``.

        Returns a ``BargeinClearResult`` — NEVER raises.  The caller MUST
        inspect the result and escalate to safe teardown on any non-
        ``SUCCESS`` outcome because Twilio's playback buffer may not have
        been flushed, which means Riya has not gone silent and stale audio
        could keep playing past the barge-in.  The stop sequence's media-
        path-broken guard ensures no subsequent ClearPlayback (from a
        later barge-in) can reach the wire and reset a newer-generation
        playback buffer.

        Bounded by ``_CLEAR_ACK_TIMEOUT_S`` so the stop sequence can never
        hang waiting for a sender that has already exited (no media
        stream, or teardown in progress).
        """
        from backend.audio.outbound_sender import ClearPlayback

        if self.shutdown_event.is_set():
            return BargeinClearResult.SHUTDOWN_DURING_ACK

        control_queue = getattr(self, "outbound_control_queue", None)
        if not isinstance(control_queue, asyncio.Queue):
            log.warning(
                "CallSupervisor: outbound_control_queue is missing or "
                "invalid (CallSid=%s) — cannot enqueue ClearPlayback.",
                self.call_sid,
            )
            return BargeinClearResult.ENQUEUE_FAILED

        loop = asyncio.get_running_loop()
        ack: "asyncio.Future[bool]" = loop.create_future()
        try:
            control_queue.put_nowait(
                ClearPlayback(generation=generation, ack=ack)
            )
        except Exception as exc:  # asyncio.QueueFull / AttributeError
            log.warning(
                "CallSupervisor: failed to enqueue ClearPlayback CallSid=%s: "
                "%s — cannot confirm clear.",
                self.call_sid,
                exc,
            )
            return BargeinClearResult.ENQUEUE_FAILED

        # Race the ack against the shutdown event so a sender that exits
        # before acking does NOT hang this stop sequence.
        shutdown_wait = asyncio.create_task(self.shutdown_event.wait())
        try:
            done, _pending = await asyncio.wait(
                {ack, shutdown_wait},
                timeout=_CLEAR_ACK_TIMEOUT_S,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            if not shutdown_wait.done():
                shutdown_wait.cancel()
                try:
                    await shutdown_wait
                except (asyncio.CancelledError, Exception):
                    pass

        if shutdown_wait in done:
            log.warning(
                "CallSupervisor: shutdown_event fired while awaiting "
                "ClearPlayback ack (CallSid=%s).",
                self.call_sid,
            )
            return BargeinClearResult.SHUTDOWN_DURING_ACK

        if ack not in done:
            log.warning(
                "CallSupervisor: Twilio clear ack timed out (%.1fs) "
                "CallSid=%s — cannot confirm clear was delivered.",
                _CLEAR_ACK_TIMEOUT_S,
                self.call_sid,
            )
            return BargeinClearResult.ACK_TIMEOUT

        try:
            result_value = ack.result()
        except asyncio.CancelledError:
            return BargeinClearResult.ACK_TIMEOUT
        except Exception as exc:  # pragma: no cover — defensive
            log.warning(
                "CallSupervisor: Twilio clear ack raised CallSid=%s: %s.",
                self.call_sid,
                exc,
            )
            return BargeinClearResult.ACK_TIMEOUT

        if result_value is False:
            log.warning(
                "CallSupervisor: Twilio clear reported send failure "
                "CallSid=%s.",
                self.call_sid,
            )
            return BargeinClearResult.SEND_FAILURE

        return BargeinClearResult.SUCCESS

    # ── Background task lifecycle (Issue 035) ─────────────────────────────────

    def start_background_tasks(self) -> None:
        """
        Spawn the long-lived per-call pipeline tasks (Issue 035+).

        Must be called **after** ``register_call(supervisor)`` (so the supervisor
        is reachable from webhooks) and **before** Twilio's media stream starts
        sending audio — the STT reader needs to be alive and consuming the
        ``inbound_audio_queue`` from the moment the customer hears the
        greeting, otherwise the first ~80 ms of customer audio is lost.

        Idempotency
        ───────────
        If ``stt_task`` is already set and still running, this is a no-op.
        Defensive against accidental double-call from future code paths.

        Failure to spawn the task is logged but never raised — the call can
        still proceed (Twilio will dial, the call will run, we just won't get
        transcripts).  The supervisor's teardown logic tolerates a never-started
        task gracefully (the ``is not None and not task.done()`` guards).
        """
        from backend.audio.deepgram_stt import run_deepgram_reader

        if self.stt_task is not None and not self.stt_task.done():
            log.debug(
                "CallSupervisor.start_background_tasks: STT task already "
                "running for CallSid=%s — no-op.",
                self.call_sid,
            )
            return

        log.info(
            "CallSupervisor.start_background_tasks: spawning STT task "
            "CallSid=%s lead=%d attempt=%d.",
            self.call_sid,
            self.lead_id,
            self.attempt_id,
        )
        try:
            self.stt_task = asyncio.create_task(
                run_deepgram_reader(self),
                name=f"deepgram-stt-{self.call_sid}",
            )
        except RuntimeError as exc:
            # No running event loop — this should never happen in production
            # (the supervisor is always created inside an async handler) but
            # is logged loudly so the operator notices.
            log.exception(
                "CallSupervisor.start_background_tasks: failed to spawn STT "
                "task for CallSid=%s (%s). The call will run without STT.",
                self.call_sid,
                exc,
            )

        # ── Issue 039: spawn the Mercury-2 conversation driver ───────────
        # The driver owns the TTS client lifecycle (open on first turn,
        # close on its own clean exit OR via the supervisor's Step 2
        # cancellation). It is spawned AFTER the STT task because both
        # tasks need to be alive before audio can flow end-to-end.
        from backend.conversation_driver import run_mercury_conversation_driver

        try:
            self.mercury_task = asyncio.create_task(
                run_mercury_conversation_driver(self),
                name=f"mercury-driver-{self.call_sid}",
            )
        except RuntimeError as exc:
            log.exception(
                "CallSupervisor.start_background_tasks: failed to spawn "
                "Mercury driver task for CallSid=%s (%s). The call will "
                "run without Mercury / TTS.",
                self.call_sid,
                exc,
            )

        # ── Issue 041: instantiate + start the matcher consumer ──────────
        # The matcher is a serial consumer that extracts preferences from
        # each completed customer turn and ranks vehicles off the
        # driver's critical path.  It needs the in-memory catalogue
        # (loaded once at startup in backend.main.lifespan) — we read it
        # lazily so a test that never spawns the matcher doesn't need the
        # catalogue.
        from backend.catalogue import vehicles as _vehicles
        from backend.matcher_consumer import MatcherConsumer

        try:
            self.matcher_task = MatcherConsumer(vehicles=_vehicles)
            self.matcher_task.start()
            # ``start()`` reassigns ``self.matcher_task._task``; capture it
            # so the supervisor can cancel it aggressively in teardown
            # (instead of relying on the graceful ``close()`` sentinel
            # path, which would wait for the queue to drain first).
            self._matcher_internal_task = self.matcher_task._task  # type: ignore[attr-defined]
            log.info(
                "CallSupervisor.start_background_tasks: matcher consumer "
                "started for CallSid=%s.",
                self.call_sid,
            )
        except RuntimeError as exc:
            log.exception(
                "CallSupervisor.start_background_tasks: failed to start "
                "matcher for CallSid=%s (%s). The call will run without "
                "matcher (recommendations won't be ranked).",
                self.call_sid,
                exc,
            )

    # ── Issue 041: WebSocket bridge ownership ────────────────────────────────

    def attach_media_stream(self, websocket, stream_sid: str) -> None:
        """
        Spawn the inbound_reader_loop and run_outbound_sender tasks and
        store their references on the supervisor.

        Called by the WebSocket handler (``twilio_media_stream``) right
        after the supervisor is registered and the ``start`` event
        arrives.  The handler ALSO awaits the bridge locally (Starlette
        requires the endpoint coroutine to stay alive for the connection),
        but the supervisor owns the task references so the teardown
        sequence can cancel them deterministically without racing the
        endpoint's own FIRST_COMPLETED + cancel() pattern.  The cancel
        is idempotent: whichever path cancels first wins, the other is
        a no-op.

        Idempotency: if either task is already set and not done, that
        spawn is skipped.  Defensive against double-call.

        The matcher is NOT spawned here — it's started by
        ``start_background_tasks`` so it can begin processing transcripts
        as soon as STT is up, before the WebSocket even opens.
        """
        from backend.api.media_stream import _inbound_reader_loop
        from backend.audio.outbound_sender import run_outbound_sender

        if self.reader_task is None or self.reader_task.done():
            log.info(
                "CallSupervisor.attach_media_stream: spawning reader_task "
                "CallSid=%s streamSid=%s.",
                self.call_sid,
                stream_sid,
            )
            try:
                self.reader_task = asyncio.create_task(
                    _inbound_reader_loop(websocket, self.call_sid, self, stream_sid),
                    name=f"media-stream-reader-{self.call_sid}",
                )
            except RuntimeError as exc:
                log.exception(
                    "CallSupervisor.attach_media_stream: failed to spawn "
                    "reader_task CallSid=%s (%s). The bridge will not "
                    "forward inbound media.",
                    self.call_sid,
                    exc,
                )

        if self.sender_task is None or self.sender_task.done():
            log.info(
                "CallSupervisor.attach_media_stream: spawning sender_task "
                "CallSid=%s streamSid=%s.",
                self.call_sid,
                stream_sid,
            )
            try:
                self.sender_task = asyncio.create_task(
                    run_outbound_sender(self, websocket, stream_sid),
                    name=f"media-stream-sender-{self.call_sid}",
                )
            except RuntimeError as exc:
                log.exception(
                    "CallSupervisor.attach_media_stream: failed to spawn "
                    "sender_task CallSid=%s (%s). The bridge will not "
                    "send outbound media.",
                    self.call_sid,
                    exc,
                )

    async def trigger_teardown(
        self,
        outcome_status: str,
        failure_reason: str | None = None,
        provider_fault: bool = False,
        completed_timeout: bool = False,
        call_already_terminal: bool = False,
    ) -> None:
        """
        Idempotent teardown — runs the fixed sequence exactly once.

        The one-shot latch (``_teardown_run``) plus the asyncio.Lock guards
        against both concurrent invocations and sequential re-entry.  Multiple
        triggers firing on the same call (e.g., status-callback + amd-callback
        arriving in either order) all see the same outcome: the sequence runs
        once with the FIRST trigger's arguments, and subsequent triggers
        return immediately.

        Sequence (fixed, in order):
            1. End the Twilio call (skipped iff ``call_already_terminal=True``).
            2. Stop pipeline tasks (Issue 035 + Issue 041):
               a. STT (Deepgram reader) — graceful: set ``shutdown_event``,
                  then ``asyncio.wait_for(stt_task, timeout=1.5)`` to let the
                  reader self-close via CloseStream; ``cancel()`` is a
                  backstop if it does not return in time.
               b. Mercury driver, inbound reader, outbound sender —
                  aggressive ``cancel()`` on each (with self-cancellation
                  guard so the calling task is never cancelled+awaited
                  on itself).
               c. Matcher consumer — graceful ``close()`` (places
                  shutdown sentinel and awaits the internal task); the
                  internal task is cancelled as a backstop if close() hangs.
            3. TODO(038): finalize Matcher (write ranked recommendations to DB).
            4. Persist outcome to DB.
            5. TODO(046): close provider sockets (TTS socket already closed
               above as part of Step 2).
            6. Release the concurrency slot.
            7. Unregister from the active-calls registry.

        Parameters
        ----------
        outcome_status : str
            "Completed" or "Failed".
        failure_reason : str | None
            Required when outcome_status is "Failed"; one of
            "no_answer" | "dropped" | "error".  Ignored for "Completed".
        provider_fault : bool
            Reserved for Issue 043 (mid-call watchdog).  Currently logged but
            not used to branch behavior.
        completed_timeout : bool
            Set when outcome_status is "Completed" via the max-duration
            backstop (Issue 047).  Stored on the leads row.
        call_already_terminal : bool
            Set to True ONLY by triggers that fire AFTER Twilio has already
            ended the call — currently the Twilio status-callback endpoint,
            which only ever fires with terminal CallStatus values
            (completed / busy / no-answer / canceled / failed).  In that
            case the line is already down; calling client.calls(sid).update
            would just bounce off Twilio.

            Set to False (default) for any trigger where the Twilio line
            may still be live — AMD verdicts (voicemail / fax), the
            orphan-window handler in arrange_ai_call, and future pipeline
            triggers (Issue 046 clean close).  We must always actively hang
            up in those cases so the customer isn't left stranded on a silent
            line and Twilio credits aren't wasted.

            Note: this flag replaced an earlier heuristic that inferred
            "already gone" from the ``outcome_status == "Completed"``
            branch.  That heuristic was inverted: clean closes via the
            consultation pipeline have outcome="Completed" but the Twilio
            line is still live (we're the ones who must end it).  Only the
            Twilio status callback actually arrives AFTER Twilio has ended
            the call, so we made that explicit.
        """
        # ── Latch — second and later triggers return immediately ──────────
        if self._teardown_run:
            log.debug(
                "CallSupervisor.trigger_teardown: CallSid=%s already ran; "
                "ignoring duplicate trigger (outcome=%s reason=%s).",
                self.call_sid,
                outcome_status,
                failure_reason,
            )
            return

        async with self._teardown_lock:
            # Re-check inside the lock — first to acquire wins.
            if self._teardown_run:
                return
            self._teardown_run = True

            # Record the winning outcome.
            self._last_outcome_status = outcome_status
            self._last_failure_reason = failure_reason
            self._last_completed_timeout = completed_timeout

            log.info(
                "CallSupervisor teardown starting: CallSid=%s lead=%d "
                "attempt=%d outcome=%s reason=%s completed_timeout=%s "
                "provider_fault=%s call_already_terminal=%s",
                self.call_sid,
                self.lead_id,
                self.attempt_id,
                outcome_status,
                failure_reason,
                completed_timeout,
                provider_fault,
                call_already_terminal,
            )

            # ── Step 1: End the Twilio call ─────────────────────────────────
            # Always attempt the hangup unless the trigger explicitly signals
            # that Twilio has already ended the call (only the status
            # callback knows this for sure — it only fires with terminal
            # CallStatus values from Twilio).  _twilio_hangup_call is
            # tolerant: it returns False (rather than raising) on
            # already-terminal calls, so even if our knowledge is slightly
            # stale, the hangup attempt is safe.
            if call_already_terminal:
                log.debug(
                    "CallSupervisor: skipping Twilio hangup for CallSid=%s "
                    "(trigger signalled call_already_terminal=True).",
                    self.call_sid,
                )
            else:
                await self._hangup_call()

            # ── Step 2: stop pipeline tasks (Issue 035) ────────────────────
            # Signal the STT task to wind down gracefully: set the shutdown
            # event first so the reader loop, on its next iteration, sees it
            # and sends ``{"type": "CloseStream"}`` to flush the final
            # transcript.  We then give the task a bounded wait to self-close
            # on its own; ``cancel()`` is a backstop, not the primary path,
            # because cancelling an ``asyncio.wait`` on ``ws.recv()`` throws
            # ``CancelledError`` instantly and bypasses the top-of-loop
            # shutdown check entirely — losing the final transcript flush.
            #
            # The order matters: set shutdown_event *first*, then wait_for
            # — the reader checks shutdown_event between events and would
            # otherwise never see the signal.
            #
            # SELF-CANCELLATION GUARD: ``trigger_teardown`` is frequently
            # invoked FROM one of the per-call pipeline tasks (the Mercury
            # driver on a provider fault, the STT reader on reconnect
            # exhaustion).  Naively cancelling + awaiting that task from
            # inside itself raises ``RuntimeError: Task cannot await on
            # itself`` and a swallowed ``CancelledError`` can mask the
            # true cause.  We capture the currently-running task here and
            # SKIP cancelling/awaiting it below — when the calling task
            # returns from this coroutine naturally, it exits cleanly on
            # its own.
            current_task = asyncio.current_task()
            self.shutdown_event.set()
            if (
                self.stt_task is not None
                and self.stt_task is not current_task
                and not self.stt_task.done()
            ):
                try:
                    # Bounded grace window for the reader to observe
                    # shutdown_event, send CloseStream, drain the final
                    # transcript, and return cleanly.  Any non-Cancelled
                    # exception the task raised during self-close is
                    # logged below but does NOT block teardown.
                    await asyncio.wait_for(
                        self.stt_task,
                        timeout=_STT_GRACEFUL_SHUTDOWN_TIMEOUT_S,
                    )
                    log.debug(
                        "CallSupervisor: STT task self-closed via CloseStream "
                        "CallSid=%s.",
                        self.call_sid,
                    )
                except asyncio.TimeoutError:
                    # Reader didn't self-close in time — fall back to
                    # cancel() so a hung STT socket cannot stall the rest
                    # of teardown (DB resolve, slot release, unregister).
                    log.warning(
                        "CallSupervisor: STT task did not self-close within "
                        "%.2fs CallSid=%s — forcing cancel.",
                        _STT_GRACEFUL_SHUTDOWN_TIMEOUT_S,
                        self.call_sid,
                    )
                    self.stt_task.cancel()
                    try:
                        await self.stt_task
                    except asyncio.CancelledError:
                        # Expected — the task was cancelled in the cancel() above.
                        pass
                    except Exception as exc:
                        # The task raised during forced shutdown — log but
                        # don't block the rest of teardown.  The DB resolve
                        # + slot release below must run regardless.
                        log.warning(
                            "CallSupervisor: STT task raised during forced "
                            "cancel CallSid=%s: %s — continuing teardown.",
                            self.call_sid,
                            exc,
                        )
                except Exception as exc:
                    # Self-close path raised an exception (e.g. a WebSocket
                    # error during the CloseStream handshake).  Log and
                    # continue — the rest of teardown must still run.
                    log.warning(
                        "CallSupervisor: STT task raised during graceful "
                        "shutdown CallSid=%s: %s — continuing teardown.",
                        self.call_sid,
                        exc,
                    )
            # Clear the handle regardless of self-close vs forced-cancel
            # vs already-done — guarantees the supervisor releases the
            # reference so the next call to trigger_teardown is a no-op
            # via the latch and no dangling task survives.
            self.stt_task = None

            # ── Step 2b: stop Mercury driver + bridge tasks + matcher ────────
            # Issue 041: all five long-lived per-call tasks are owned by
            # the supervisor and cancelled here.  Strategy:
            #
            #   * STT (deepgram reader) — graceful above (Step 2).  STT's
            #     reconnect-on-failure loop means a hard cancel can leak
            #     a live Deepgram WebSocket; graceful via shutdown_event
            #     is the only safe path.
            #
            #   * Mercury driver — aggressive cancel.  The driver may be
            #     mid Mercury-API-call when teardown fires; cancelling it
            #     cleanly exits the run loop.  The outer driver loop
            #     survives the inner-task cancellation (Issue 041 wraps
            #     ``run_turn`` in an inner task precisely so this cancel
            #     doesn't kill the driver task itself).
            #
            #   * Inbound reader (Twilio WS reader) — aggressive cancel.
            #     When shutdown_event is set the inbound reader exits its
            #     ``ws.recv()`` loop on the next iteration, so this is a
            #     backstop for stuck frames.
            #
            #   * Outbound sender (Twilio WS writer) — aggressive cancel.
            #     The sender's polling loop watches shutdown_event too,
            #     but cancel() guarantees we don't block on a slow send.
            #
            #   * Matcher consumer — graceful via close() (places shutdown
            #     sentinel, awaits task).  The matcher drains its own
            #     queue before exiting so partial preference state isn't
            #     lost on clean calls.  For aggressive cancellation, the
            #     internal task is cancelled separately as a backstop.
            #
            # SELF-CANCELLATION GUARD: ``trigger_teardown`` is frequently
            # invoked FROM one of these tasks (Mercury on a provider
            # fault, STT on reconnect exhaustion).  Catching the current
            # task in ``current_task`` and skipping cancel/await on it
            # avoids ``RuntimeError: Task cannot await on itself``.
            current_task = asyncio.current_task()

            # ── Issue 042 — stop the in-flight turn + barge-in sequence ─────
            # ``current_turn_task`` is the cancellable ``run_turn`` wrapper and
            # ``bargein_task`` is any running stop sequence.  Cancel both here
            # so teardown never leaks them.  shutdown_event is already set
            # above, so cancelling ``current_turn_task`` lets the driver's
            # wrapper observe the cancel, fall through to its shutdown check,
            # and exit cleanly before the Mercury driver cancel below.  The
            # self-cancellation guard skips whichever task is running this
            # teardown (barge-in never calls teardown, but the guard is cheap
            # and defensive).
            for _bargein_attr in ("current_turn_task", "bargein_task"):
                _task = getattr(self, _bargein_attr)
                if (
                    _task is not None
                    and _task is not current_task
                    and not _task.done()
                ):
                    _task.cancel()
                    try:
                        await _task
                    except asyncio.CancelledError:
                        pass
                    except Exception as exc:
                        log.warning(
                            "CallSupervisor: %s raised during cancel "
                            "CallSid=%s: %s — continuing teardown.",
                            _bargein_attr,
                            self.call_sid,
                            exc,
                        )
                setattr(self, _bargein_attr, None)

            # Release the barge-in completion barrier so a Mercury driver
            # blocked on it (post-interruption wait) unblocks and observes
            # shutdown_event.  The stop sequence's own ``finally`` also sets
            # this, but a teardown that cancelled the stop task mid-flight
            # relies on this belt-and-braces set (plus the driver's wait is
            # itself shutdown-aware).
            self.bargein_complete.set()

            # ── Mercury driver (aggressive cancel) ─────────────────────────
            if (
                self.mercury_task is not None
                and self.mercury_task is not current_task
                and not self.mercury_task.done()
            ):
                self.mercury_task.cancel()
                try:
                    await self.mercury_task
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    log.warning(
                        "CallSupervisor: Mercury driver raised during "
                        "cancel CallSid=%s: %s — continuing teardown.",
                        self.call_sid,
                        exc,
                    )
            self.mercury_task = None

            # ── Inbound reader (aggressive cancel) ─────────────────────────
            if (
                self.reader_task is not None
                and self.reader_task is not current_task
                and not self.reader_task.done()
            ):
                self.reader_task.cancel()
                try:
                    await self.reader_task
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    log.warning(
                        "CallSupervisor: inbound reader raised during "
                        "cancel CallSid=%s: %s — continuing teardown.",
                        self.call_sid,
                        exc,
                    )
            self.reader_task = None

            # ── Outbound sender (aggressive cancel) ─────────────────────────
            if (
                self.sender_task is not None
                and self.sender_task is not current_task
                and not self.sender_task.done()
            ):
                self.sender_task.cancel()
                try:
                    await self.sender_task
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    log.warning(
                        "CallSupervisor: outbound sender raised during "
                        "cancel CallSid=%s: %s — continuing teardown.",
                        self.call_sid,
                        exc,
                    )
            self.sender_task = None

            # ── Matcher consumer (graceful close + cancel backstop) ────────
            # Prefer graceful close so any in-flight turn completes and
            # its ranking state is available for the supervisor's
            # _persist_outcome → recommendations write.  If close()
            # hangs, cancel the internal task directly as a backstop.
            if self.matcher_task is not None:
                try:
                    await self.matcher_task.close()
                except Exception as exc:
                    log.warning(
                        "CallSupervisor: matcher close raised CallSid=%s: %s "
                        "— falling back to aggressive cancel.",
                        self.call_sid,
                        exc,
                    )
                    internal = self._matcher_internal_task
                    if (
                        internal is not None
                        and internal is not current_task
                        and not internal.done()
                    ):
                        internal.cancel()
                        try:
                            await internal
                        except asyncio.CancelledError:
                            pass
                        except Exception as exc2:
                            log.warning(
                                "CallSupervisor: matcher internal task "
                                "raised during cancel CallSid=%s: %s.",
                                self.call_sid,
                                exc2,
                            )
            self.matcher_task = None
            self._matcher_internal_task = None

            # ── TTS client close (Sarvam socket) ───────────────────────────
            if self.tts_client is not None:
                try:
                    await self.tts_client.close()
                except Exception as exc:
                    log.warning(
                        "CallSupervisor: TTS client close raised "
                        "CallSid=%s: %s — continuing teardown.",
                        self.call_sid,
                        exc,
                    )
                self.tts_client = None

            # ── Step 3: finalize Matcher (write recommendations) ──────────
            # Issue 041 wires up the matcher consumer; Issue 038 will
            # write the final ranked recommendations to the DB here.
            # TODO(038): finalize Matcher

            # ── Step 4: Persist outcome to DB ─────────────────────────────
            # Persistence failures must NOT block slot release or registry
            # cleanup — the supervisor must always reach its terminal state.
            # Log loudly so the inconsistency is diagnosable.
            try:
                await self._persist_outcome(
                    outcome_status=outcome_status,
                    failure_reason=failure_reason,
                    completed_timeout=completed_timeout,
                )
            except Exception as exc:
                log.exception(
                    "CallSupervisor: failed to persist outcome for "
                    "CallSid=%s lead=%d: %s",
                    self.call_sid,
                    self.lead_id,
                    exc,
                )

            # ── Step 4b: Discard recordings on no_answer (Issue 037) ─────
            # The ``record=True`` flag we set on calls.create (Issue 037)
            # captures every outbound call, including voicemails.  When
            # the call resolves as ``no_answer`` (machine / fax), the
            # captured audio is a recording of a non-consenting party
            # (the voicemail box owner never consented to recording) and
            # MUST NOT be retained.
            #
            # Belt-and-braces: this same delete is ALSO issued from the
            # ``/api/twilio/recording-callback`` endpoint when Twilio's
            # recording-completion notification fires.  Either path
            # wins — whichever fires first deletes the recording; the
            # second path finds an empty list and no-ops.  Together
            # they cover every timing race between the AMD/terminal
            # callbacks and Twilio's recording finalisation.
            #
            # Exceptions from the async wrapper are swallowed inside
            # the helper — teardown continues regardless.  The async
            # call IS awaited (not fire-and-forget) so the count is
            # available for logging before we release the slot.
            if failure_reason == "no_answer":
                deleted = await _delete_call_recordings_async(self.call_sid)
                log.info(
                    "CallSupervisor: discarded %d recording(s) for "
                    "no_answer CallSid=%s.",
                    deleted,
                    self.call_sid,
                )

            # ── Step 5: close provider sockets ────────────────────────────
            # TODO(046): close provider sockets — STT socket is already closed
            # by the awaited stt_task above; TTS socket lands here when the
            # outbound audio sender is wired in (Issue 036/046).

            # ── Step 6: Release the concurrency slot — single owner ──────
            self._release_slot()

            # ── Step 7: Unregister from active-calls registry ─────────────
            unregister_call(self.call_sid)

            log.info(
                "CallSupervisor teardown complete: CallSid=%s "
                "outcome=%s reason=%s",
                self.call_sid,
                outcome_status,
                failure_reason,
            )

    # ── Internals ─────────────────────────────────────────────────────────────

    async def _hangup_call(self) -> None:
        """Issue the Twilio hangup via run_in_executor.  Mirrors _twilio_create_call."""
        from backend.config import load_settings

        settings = load_settings()
        loop = asyncio.get_running_loop()
        try:
            ok = await loop.run_in_executor(
                None,
                _twilio_hangup_call,
                self.call_sid,
                settings.twilio_account_sid,
                settings.twilio_auth_token.get_secret_value(),
            )
            log.info(
                "CallSupervisor: Twilio hangup %s for CallSid=%s.",
                "succeeded" if ok else "failed-or-already-terminal",
                self.call_sid,
            )
        except Exception as exc:
            # Hangup must never block the teardown sequence.
            log.error(
                "CallSupervisor: hangup raised for CallSid=%s: %s",
                self.call_sid,
                exc,
            )

    def _release_slot(self) -> None:
        """Release the concurrency slot — single owner, idempotent guard."""
        try:
            self._concurrency_manager.release()
            log.debug(
                "CallSupervisor: concurrency slot released for CallSid=%s "
                "lead=%d attempt=%d.",
                self.call_sid,
                self.lead_id,
                self.attempt_id,
            )
        except RuntimeError:
            # Double-release guard — log but don't crash.  The supervisor
            # is the single owner of the slot for this call, but the
            # startup sweep / manager.reset() can put the count in
            # unexpected states; we'd rather log and move on than block
            # teardown.
            log.warning(
                "CallSupervisor: double-release guard fired for CallSid=%s "
                "lead=%d attempt=%d — investigate (release called twice?).",
                self.call_sid,
                self.lead_id,
                self.attempt_id,
            )

    async def _persist_outcome(
        self,
        *,
        outcome_status: str,
        failure_reason: str | None,
        completed_timeout: bool,
    ) -> None:
        """Update call_attempts + leads atomically."""
        if outcome_status == "Failed":
            await _resolve_failed_attempt(
                pool=self._pool,
                attempt_id=self.attempt_id,
                lead_id=self.lead_id,
                failure_reason=failure_reason or "error",
            )
        elif outcome_status == "Completed":
            await _resolve_completed_attempt(
                pool=self._pool,
                attempt_id=self.attempt_id,
                lead_id=self.lead_id,
                completed_timeout=completed_timeout,
            )
        else:
            raise ValueError(
                f"CallSupervisor: unknown outcome_status {outcome_status!r}"
            )