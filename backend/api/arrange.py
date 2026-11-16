"""
backend/api/arrange.py — Arrange AI Call handler (Issues 030 + 031 + 032).

Provides:
  1. POST /api/admin/leads/{lead_id}/arrange — admin-triggered call-control skeleton.
     - CSRF-protected.
     - Atomic idempotency: conditional UPDATE inside transaction prevents races.
     - Non-blocking acquire() — rejects at-capacity immediately with HTTP 409.
     - Single DB transaction: inserts call_attempts row AND flips lead to Processing,
       bumping attempt_count and updated_at.
     - Places outbound Twilio call with status callbacks and async AMD enabled
       (machine_detection="Enable", async_amd_status_callback → endpoint #2).
     - Issue 032: after Twilio call placement succeeds, instantiates a
       CallSupervisor and registers it in the active-calls registry,
       transferring slot ownership to it.  The orphan window (Twilio live but
       SID-store DB write failed) triggers supervisor teardown synchronously.
     - Sync error handling: on Twilio creation failure (returns None), no
       supervisor is created — resolves attempt + releases slot directly
       in the exception handler.
  2. POST /api/twilio/amd-callback — Twilio async AMD verdict webhook (Issue 031).
     - On machine_* / fax verdict: looks up the live CallSupervisor and signals
       it via trigger_teardown(outcome_status="Failed", failure_reason="no_answer").
       The supervisor owns hangup, DB resolve, slot release, and unregister.
     - On human / unknown verdict: no-op — call is allowed to proceed
       (Spike 025 trial-account convention treats unknown as human).
     - Issue 032: no direct DB writes, no direct slot release — all teardown
       work is owned by the supervisor.  If the supervisor is missing from the
       registry (stale / post-restart), return 200 and trust the startup sweep
       (Issue 029) to resolve the orphaned lead authoritatively.
     - TODO(037): discard partial recording on machine verdict.
     - TODO(037/040): push human/unknown verdict into per-call session state
       so the greeting pipeline can gate on it.
  3. POST /api/twilio/status-callback — Twilio webhook for call status updates.
     - Issue 032: thin signaler — looks up the live CallSupervisor and signals
       it.  All teardown work (hangup if needed, DB resolve, slot release,
       unregister) is owned by the supervisor.  REMOVE-IN-032 stopgap is gone.
     - Updates attempt and lead state driven by the outcome classifier.
     - Issue 031: AMD verdicts are no longer consumed here — they arrive at
       /api/twilio/amd-callback (endpoint #2) and are resolved there.
  4. GET /api/twilio/placeholder-twiml — TwiML that opens the Media Stream.
     Issue 034: returns <Say> + <Connect><Stream url="..."/></Connect>.  The
     <Connect> verb is TERMINAL — it keeps the call live and dedicated to
     the WebSocket until the customer hangs up (or our side does).  This
     is the entire entry point for the voice pipeline: Twilio opens the
     WebSocket when the call is answered and audio flows both ways through
     it until teardown.

Design
──────
ConcurrencyManager is a module-level global (set during FastAPI lifespan startup).
The status callback and AMD callback handlers are unauthenticated (Twilio posts
to them).

P1: Real Twilio placement — phone fetched in handler, passed to _place_twilio_call_impl.
P2: Sync error handling — slot released and attempt resolved inside handler on
    Twilio-creation failure (no supervisor exists yet at that point).
P3: Atomic idempotency — conditional UPDATE WHERE status<>'Processing' inside transaction.
P4: REMOVED (Issue 032).  Webhook-driven slot release is gone — the
    CallSupervisor owns the slot lifecycle for its call, period.
P5: Blocking Twilio SDK call runs via run_in_executor.
P6: AMD unconditional; placeholder-twiml endpoint added; matcher_usable noted as placeholder.
P7: Issue 031 — dedicated /api/twilio/amd-callback endpoint; resolve-before-hangup
    ordering to prevent race with trailing status callback; carry-forward TODO
    for human/unknown verdict gating in the greeting pipeline.
P8: Issue 032 — CallSupervisor owns the teardown lifecycle (one-shot latch);
    webhooks are thin signalers; orphan window handled in arrange_ai_call;
    REMOVE-IN-032 stopgap deleted.

No module calls load_settings() at import time.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from backend.api.auth import get_current_admin, get_pool
from backend.call_supervisor import (
    CallSupervisor,
    _resolve_completed_attempt,
    _resolve_failed_attempt,
    _twilio_hangup_call,
    get_call,
    register_call,
)
from backend.concurrency import ConcurrencyManager
from backend.database._sessions import Session
from backend.database._txn import transaction
from backend.outcome_classifier import CallOutcome, classify

log = logging.getLogger(__name__)


# ─── Internal exception ───────────────────────────────────────────────────────


class TwilioPlacementError(Exception):
    """
    Raised when Twilio call placement fails (API error or returns no CallSid).

    Caught by the arrange_ai_call exception handler, which resolves the
    attempt as 'error', flips the lead to Failed, and releases the slot.
    """

    def __init__(self, lead_id: int, attempt_id: int, reason: str = "error"):
        self.lead_id = lead_id
        self.attempt_id = attempt_id
        self.reason = reason
        super().__init__(f"Twilio placement failed for lead {lead_id} (attempt {attempt_id}): {reason}")

# ─── Module-level concurrency manager (set during lifespan startup) ───────────────

_concurrency_manager: ConcurrencyManager | None = None


def set_concurrency_manager(manager: ConcurrencyManager) -> None:
    """Wire the concurrency manager into the arrange module (called from main.py lifespan)."""
    global _concurrency_manager
    _concurrency_manager = manager


def get_concurrency_manager() -> ConcurrencyManager:
    """Return the wired concurrency manager. Raises RuntimeError if not initialised."""
    if _concurrency_manager is None:
        raise RuntimeError("ConcurrencyManager not initialised. Call set_concurrency_manager() first.")
    return _concurrency_manager


# ─── Request / Response models ───────────────────────────────────────────────────


class ArrangeSuccessResponse(BaseModel):
    """Response body on successful arrange."""

    message: str = Field(default="Call initiated.")
    attempt_id: int = Field(..., description="call_attempts row ID")
    call_sid: str | None = Field(
        None,
        description="Twilio CallSid (None if create-call API failed — attempt is resolved as error)",
    )


class IdempotentSkipResponse(BaseModel):
    """Response body when the lead is already Processing (atomic idempotent no-op)."""

    message: str = Field(default="Lead is already being processed.")
    attempt_id: int = Field(..., description="Existing call_attempts row ID")


class TwilioStatusCallbackResponse(BaseModel):
    """TwiML response to acknowledge the webhook (empty TwiML)."""

    pass


# ─── Router ───────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/api", tags=["arrange"])
_status_callback_router = APIRouter(prefix="/api/twilio", tags=["twilio"])


# ─── Arrange AI Call endpoint ───────────────────────────────────────────────────


@router.post(
    "/admin/leads/{lead_id}/arrange",
    response_model=ArrangeSuccessResponse,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Lead not found"},
        status.HTTP_409_CONFLICT: {
            "description": "Concurrency slot unavailable or lead already Processing (idempotent)"
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Lead has reached max attempts"},
    },
    summary="Arrange AI Call",
    description=(
        "Admin-triggered call-control skeleton for Issue 030. "
        "CSRF-protected. Atomically idempotent (conditional UPDATE inside tx). "
        "Acquires a concurrency slot (non-blocking). "
        "In a single transaction: inserts call_attempts row, flips lead to Processing, "
        "bumps attempt_count and updated_at. "
        "Places outbound Twilio call with status callbacks and async AMD enabled. "
        "On Twilio failure: synchronously resolves attempt + releases slot in handler."
    ),
)
async def arrange_ai_call(
    lead_id: int,
    _: Session = Depends(get_current_admin),
    pool: Annotated[object, Depends(get_pool)] = ...,  # type: ignore[assignment]
) -> ArrangeSuccessResponse:
    """
    Arrange an outbound AI call for the given lead.

    Flow
    ────
    1. Auth / CSRF — enforced by get_current_admin.
    2. Load the lead from the database (status, attempt_count, phone).
       - 404 if not found.
       - 422 if attempt_count >= max_attempts.
    3. Acquire a concurrency slot (non-blocking).
       - If acquire() returns False → HTTP 409.
    4. Single DB transaction:
       a. Conditional UPDATE: flip lead to Processing only if not already Processing.
          - rowcount == 0 → another request won the race → release slot, return idempotent.
          - rowcount == 1 → we won → insert call_attempts row.
       b. Capture the new attempt_id.
    5. Place outbound Twilio call:
       - Async AMD enabled (unconditional).
       - Status callback pointing to POST /api/twilio/status-callback.
       - Blocking SDK call run via run_in_executor to avoid blocking the event loop.
       - Capture the CallSid and store it on the attempt row.
    6. On Twilio failure: synchronously resolve attempt + release slot here.
       (P2 fix — prevents zombie Processing leads and slot leaks.)

    Note on concurrency slot lifecycle (Issue 032):
    ─────────────────────────────────────────────────
    On success: slot is released by the status callback webhook when the call resolves.
    On Twilio failure: slot is released here in the handler (P2 fix).
    """
    from backend.config import load_settings

    manager = get_concurrency_manager()
    settings = load_settings()

    # ── Step 1: Load lead (phone for Twilio + language for STT) ─────────────
    # ``language`` is the lead's preferred language (English | Hinglish); it
    # flows into the CallSupervisor and ultimately picks the Deepgram Flux
    # model + language hints (and, later, the Sarvam TTS speaker and the
    # Mercury variant string).  The schema CHECK constraint guarantees the
    # value is one of those two, so we can pass it through verbatim.
    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            SELECT id, status, attempt_count, phone, language
            FROM leads
            WHERE id = ?;
            """,
            (lead_id,),
        )
        row = await cursor.fetchone()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Lead {lead_id} not found.",
        )

    lead_id_db, current_status, attempt_count, phone, language = row

    # ── Step 2: Max attempts check ───────────────────────────────────────────
    if attempt_count >= settings.max_attempts:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "msg": (
                    f"Lead has reached the maximum number of attempts "
                    f"({settings.max_attempts})."
                ),
                "attempt_count": attempt_count,
            },
        )

    # ── Step 3: Acquire concurrency slot (non-blocking) ──────────────────────
    if not manager.acquire():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "msg": "Concurrency limit reached. No slots available right now.",
                "active_calls": manager.active,
                "capacity": manager.capacity,
            },
        )

    # ── Step 4: Atomic DB transaction — conditional flip + insert ────────────
    #
    # P3 FIX: The idempotency check is moved INSIDE the transaction using a
    # conditional UPDATE.  This eliminates the race: even if two requests arrive
    # simultaneously, only one can win the UPDATE (rowcount=1) because the first
    # to commit flips status to 'Processing'.
    #
    new_attempt_number = attempt_count + 1
    attempt_id: int | None = None
    slot_acquired = True  # tracks whether we still hold the slot
    # Issue 032: supervisor starts as None — it's only assigned after Twilio
    # call placement succeeds (post-flip-insert transaction, post-Twilio-create).
    # Initialising it here lets the generic `except Exception` handler safely
    # reference `supervisor` even when an early failure (e.g., transient DB
    # error during the flip-insert transaction) prevented it from being
    # created.  Without this initialiser, the handler would raise
    # UnboundLocalError, masking the real exception, skipping cleanup,
    # leaving a zombie Processing lead, and leaking the concurrency slot.
    supervisor: CallSupervisor | None = None

    try:
        async with pool.connection() as conn:
            async with transaction(conn):
                # Conditional flip: only succeeds if lead is NOT yet Processing.
                cursor = await conn.execute(
                    """
                    UPDATE leads
                    SET status       = 'Processing',
                        attempt_count = ?,
                        updated_at    = datetime('now')
                    WHERE id         = ?
                      AND status    != 'Processing';
                    """,
                    (new_attempt_number, lead_id),
                )
                rowcount = cursor.rowcount

                if rowcount == 0:
                    # Another request won the race — lead is already Processing.
                    # Release the slot we just acquired.
                    try:
                        manager.release()
                        slot_acquired = False
                    except RuntimeError:
                        log.warning(
                            "arrange_ai_call: race guard fired during idempotency "
                            "resolution for lead %d.",
                            lead_id,
                        )

                    # Look up the existing attempt_id for the idempotent response.
                    cursor2 = await conn.execute(
                        """
                        SELECT id
                        FROM call_attempts
                        WHERE lead_id = ?
                        ORDER BY id DESC
                        LIMIT 1;
                        """,
                        (lead_id,),
                    )
                    existing = await cursor2.fetchone()
                    existing_id = existing[0] if existing else None

                    return IdempotentSkipResponse(
                        message="Lead is already being processed.",
                        attempt_id=existing_id or 0,
                    )

                # We won — insert the call_attempts row.
                cursor = await conn.execute(
                    """
                    INSERT INTO call_attempts (lead_id, attempt_number, started_at)
                    VALUES (?, ?, datetime('now'));
                    """,
                    (lead_id, new_attempt_number),
                )
                attempt_id = cursor.lastrowid

        # ── Step 5: Place outbound Twilio call ─────────────────────────────
        #
        # P5 FIX: client.calls.create() is a blocking I/O call (makes an HTTPS
        # request to Twilio).  Running it directly on the asyncio event loop
        # would block all other coroutines.  We offload it to a thread pool
        # executor so the event loop remains free.
        #
        loop = asyncio.get_running_loop()
        call_sid: str | None = await loop.run_in_executor(
            None,
            _twilio_create_call,
            lead_id,
            attempt_id,
            phone,
            settings.twilio_account_sid,
            settings.twilio_auth_token.get_secret_value(),
            settings.twilio_phone_number,
            settings.public_hostname,
        )

        # P2 FIX: _twilio_create_call returning None is a failure — not an
        # exception, so it falls through without triggering the except block.
        # Raise TwilioPlacementError so the exception handler cleans up:
        # resolves attempt, flips lead to Failed, releases slot.
        if call_sid is None:
            raise TwilioPlacementError(
                lead_id=lead_id,
                attempt_id=attempt_id,
                reason="error",
            )

        # ── Step 5b: Create + register the CallSupervisor (Issue 032) ─────
        # Slot ownership transfers from arrange_ai_call to the supervisor
        # at this point.  The supervisor will release the slot in its
        # teardown sequence when the call ends.  We register BEFORE the
        # SID-store DB write so a trailing webhook can find the supervisor
        # even if the SID-store write fails.
        #
        # The window between supervisor registration and the SID-store DB
        # write is the ORPHAN WINDOW: the call is live at Twilio, the DB
        # doesn't yet know the SID.  If the SID-store write below raises,
        # we trigger supervisor teardown to end the live call + mark
        # Failed/error.  See the except block below.
        supervisor: CallSupervisor | None = CallSupervisor(
            lead_id=lead_id,
            attempt_id=attempt_id,
            call_sid=call_sid,
            language=language,
            concurrency_manager=manager,
            pool=pool,
        )
        register_call(supervisor)
        # From this point, slot_acquired must NOT be released directly by
        # the exception handlers below — the supervisor owns the slot.
        slot_acquired = False

        # NOTE (Issue 035 follow-up): the Deepgram STT reader is intentionally
        # NOT started here.  If we opened the socket now, it would idle while
        # the customer's phone was still ringing — burning through the fast-
        # reconnect budget before any audio ever flowed.  The reader is now
        # spawned by ``/api/twilio/media-stream`` on Twilio's ``start`` event
        # (right after ``get_call(call_sid)`` succeeds) so the socket only
        # exists once media is actually streaming.

        # ── Step 6: Store CallSid on the attempt row ──────────────────────
        try:
            async with pool.connection() as conn:
                await conn.execute(
                    """
                    UPDATE call_attempts
                    SET twilio_call_sid = ?
                    WHERE id = ?;
                    """,
                    (call_sid, attempt_id),
                )
        except Exception as exc:
            # ORPHAN WINDOW — Twilio has the live call, our DB doesn't have
            # the SID yet.  Trigger the supervisor to end the call and
            # persist Failed/error so the dashboard reflects reality.
            # The supervisor's latch prevents double-execution if the
            # exception path is re-entered.
            log.error(
                "arrange_ai_call: orphan window — SID-store write failed "
                "for lead %d (attempt %d, CallSid=%s). Triggering supervisor "
                "teardown to end the live call. Error: %s",
                lead_id,
                attempt_id,
                call_sid,
                exc,
            )
            await supervisor.trigger_teardown(
                outcome_status="Failed",
                failure_reason="error",
            )
            raise

    except TwilioPlacementError:
        # P2 FIX: Twilio placement returned None (Twilio not configured or API
        # error caught inside _twilio_create_call).  The attempt has already been
        # inserted and the lead flipped to Processing, but NO supervisor was
        # created (no live call exists).  We resolve the attempt as 'error'
        # directly and release the slot directly — the supervisor does not
        # exist to own this teardown.
        log.warning(
            "arrange_ai_call: Twilio placement failed for lead %d (attempt %s). "
            "Resolving as error and releasing slot.",
            lead_id,
            attempt_id,
        )
        if attempt_id is not None:
            await _resolve_failed_attempt(
                pool=pool,
                attempt_id=attempt_id,
                lead_id=lead_id,
                failure_reason="error",
            )
        if slot_acquired:
            try:
                manager.release()
                slot_acquired = False
            except RuntimeError:
                log.warning(
                    "arrange_ai_call: double-release guard fired for lead %d (attempt %s).",
                    lead_id,
                    attempt_id,
                )
        return ArrangeSuccessResponse(
            message="Call placement failed — attempt resolved as error.",
            attempt_id=attempt_id,
            call_sid=None,
        )

    except Exception as exc:
        # P2 FIX: Any other failure (unexpected exception, DB write error,
        # orphan-window SID-store failure, etc.) must be handled here.  The
        # cleanup path depends on whether a supervisor was created:
        #
        #   - supervisor is not None: the supervisor owns the slot for this
        #     call — signal its teardown.  The supervisor's latch ensures
        #     the slot is released exactly once even if the orphan-window
        #     except already triggered teardown above.
        #   - supervisor is None: failure happened before Twilio call
        #     placement succeeded — direct cleanup.
        #
        # Either way, the call_attempt row has already been inserted with
        # status=Processing, so we must always resolve the attempt to
        # prevent a zombie Processing lead.
        log.error(
            "arrange_ai_call: unexpected error for lead %d — error: %s",
            lead_id,
            exc,
        )
        if supervisor is not None:
            # Supervisor owns teardown (slot release, unregister, etc.).
            # Its latch protects against double-execution.
            log.info(
                "arrange_ai_call: supervisor exists for lead %d — delegating "
                "teardown to it.",
                lead_id,
            )
            await supervisor.trigger_teardown(
                outcome_status="Failed",
                failure_reason="error",
            )
        else:
            # No supervisor — exception happened before Twilio placement
            # succeeded (between the flip-insert transaction and step 5).
            # Direct cleanup.
            log.warning(
                "arrange_ai_call: no supervisor for lead %d — direct cleanup.",
                lead_id,
            )
            if attempt_id is not None:
                await _resolve_failed_attempt(
                    pool=pool,
                    attempt_id=attempt_id,
                    lead_id=lead_id,
                    failure_reason="error",
                )
            if slot_acquired:
                try:
                    manager.release()
                    slot_acquired = False
                except RuntimeError:
                    log.warning(
                        "arrange_ai_call: double-release guard fired for lead %d (attempt %s).",
                        lead_id,
                        attempt_id,
                    )
        raise

    return ArrangeSuccessResponse(
        message="Call initiated.",
        attempt_id=attempt_id,
        call_sid=call_sid,
    )


# ─── Twilio call placement (runs in thread pool executor) ─────────────────────


def _twilio_create_call(
    lead_id: int,
    attempt_id: int,
    to_phone: str,
    twilio_account_sid: str,
    twilio_auth_token: str,
    twilio_phone_number: str,
    public_hostname: str,
) -> str | None:
    """
    Synchronous Twilio call creation — MUST be called via run_in_executor.

    Parameters
    ----------
    All required config values are passed explicitly so this function is fully
    synchronous (no asyncio, no load_settings import).

    Returns
    -------
    str | None — Twilio CallSid on success, None on failure.

    Notes (P1 / P5 / P6)
    ─────────────────────
    - AMD is unconditional: async_amd=True always (Spike 025 design note).
    - Status callback events: initiated, ringing, answered, completed.
    - TwiML URL points to /api/twilio/placeholder-twiml.
    """
    from twilio.rest import Client as TwilioClient
    from twilio.base.exceptions import TwilioRestException

    if not twilio_account_sid or not twilio_auth_token:
        log.warning(
            "Twilio not configured. Skipping outbound call placement for lead %d.",
            lead_id,
        )
        return None

    client = TwilioClient(twilio_account_sid, twilio_auth_token)

    # Build the status callback URL.
    if public_hostname:
        status_callback_url = (
            f"https://{public_hostname}/api/twilio/status-callback"
        )
        twiml_url = f"https://{public_hostname}/api/twilio/placeholder-twiml"
        async_amd_status_callback_url = (
            f"https://{public_hostname}/api/twilio/amd-callback"
        )
        # Issue 037: Twilio fires this when the recording file is ready
        # (we only subscribe to "completed" — we do NOT need the in-progress
        # callbacks).  Our handler logs receipt and, for machine/fax
        # verdicts, deletes the recording so we never retain voicemail
        # audio.  Actual download / S3 archival is Issue 049.
        recording_status_callback_url = (
            f"https://{public_hostname}/api/twilio/recording-callback"
        )
    else:
        # Dev mode: no public hostname — use a placeholder TwiML string.
        status_callback_url = "https://localhost/api/twilio/status-callback"
        twiml_url = None
        async_amd_status_callback_url = (
            "https://localhost/api/twilio/amd-callback"
        )
        recording_status_callback_url = (
            "https://localhost/api/twilio/recording-callback"
        )

    try:
        call = client.calls.create(
            to=to_phone,
            from_=twilio_phone_number,
            status_callback=status_callback_url,
            status_callback_event=["initiated", "ringing", "answered", "completed"],
            status_callback_method="POST",
            # P6 FIX: AMD is unconditional — always request async AMD.
            # Per Spike 025: trial accounts treat AnsweredBy=unknown as human/proceed.
            async_amd="true",
            # Issue 031: enable machine detection so Twilio emits an AMD verdict
            # for every call.  "Enable" returns the verdict as soon as Twilio
            # classifies the called party (faster than "DetectMessageEnd", which
            # waits for the greeting to end).  Voicemail never receives a full
            # consultation.
            machine_detection="Enable",
            # Issue 031: route the async AMD verdict to a dedicated webhook.
            # This keeps machine verdicts out of the status-callback endpoint
            # (which handles only terminal CallStatus transitions), so the two
            # concerns can never tangle.  POST is the default and matches our
            # other webhooks.
            async_amd_status_callback=async_amd_status_callback_url,
            async_amd_status_callback_method="POST",
            # Issue 037: record BOTH inbound + outbound audio so the
            # consultation transcript can be reconstructed post-hoc if
            # the admin needs it.  ``dual`` channels + ``both`` tracks
            # yields two separate WAV files (one per direction) joined
            # by the same RecordingSid.  Per Twilio docs, dual recording
            # with Media Streams is supported but not guaranteed to be
            # perfectly sample-aligned — the first HITL live-call will
            # validate this in production.
            record=True,
            recording_channels="dual",
            recording_track="both",
            # Fire only when the recording file is finalised.  Our
            # webhook logs receipt and discards the file for machine
            # /fax verdicts (no retention of voicemail audio).
            recording_status_callback=recording_status_callback_url,
            recording_status_callback_method="POST",
            recording_status_callback_event=["completed"],
            # P6 FIX: TwiML URL is always provided when public_hostname is set.
            # When not set (dev mode), we omit the URL and let Twilio use a default.
            url=twiml_url,
        )
        log.info(
            "Twilio call placed for lead %d (attempt %d): CallSid=%s",
            lead_id,
            attempt_id,
            call.sid,
        )
        return call.sid
    except TwilioRestException as exc:
        log.error(
            "Twilio API error placing call for lead %d (attempt %d): %s",
            lead_id,
            attempt_id,
            exc,
        )
        return None


# ─── Twilio async AMD callback endpoint (Issue 031) ─────────────────────────────


@_status_callback_router.post(
    "/amd-callback",
    summary="Twilio async AMD callback webhook",
    description=(
        "Webhook endpoint for Twilio's asynchronous Answering Machine Detection "
        "verdict.  When the verdict is machine_* or fax, signals the live "
        "CallSupervisor (Issue 032) to run its idempotent teardown: hang up "
        "the call, persist Failed/no_answer, release the concurrency slot, "
        "and unregister.  When the verdict is human or unknown, no action is "
        "taken — the call is allowed to proceed. "
        "TODO(037): discard partial recording on machine verdict."
    ),
)
async def twilio_amd_callback(
    request: Request,
) -> PlainTextResponse:
    """
    Handle Twilio's asynchronous AMD verdict.

    Issue 032: this endpoint is now a thin signaler.  It looks up the live
    CallSupervisor via the active-calls registry and signals it; the
    supervisor owns all teardown work (hangup, DB resolve, slot release,
    unregister).  If no supervisor exists for the CallSid (stale callback,
    post-restart), we trust the startup sweep (Issue 029) to handle the
    orphaned lead authoritatively via the database and return 200 without
    action — falling back to direct DB writes would re-introduce the
    double-execution / slot-leak risk that the supervisor exists to prevent.

    Issue 040 (driver synchronization): every AMD verdict — human, unknown,
    machine_*, fax — is published onto ``supervisor.amd_verdict`` BEFORE any
    teardown fires.  This unblocks the conversation driver, which awaits
    the future with a bounded timeout (``asyncio.wait_for(..., timeout=2.0)``
    in ``run_mercury_conversation_driver``) so a dropped webhook cannot
    stall the greeting indefinitely.  For machine/fax verdicts the driver
    observes the verdict string and exits cleanly (the supervisor's
    teardown handles the rest).

    Flow
    ────
    1. Parse CallSid and AnsweredBy from form data.  Ignore (200) if either
       is missing — Twilio retries on non-2xx, and a missing field here means
       a malformed callback that we can't act on.
    2. Look up the live CallSupervisor via ``get_call(call_sid)``.  If
       absent (stale callback / post-restart), log and return 200 — the
       startup sweep (Issue 029) is the authoritative resolver and the
       driver is also gone, so there is no consumer for the verdict.
    3. Populate ``supervisor.amd_verdict`` with the AnsweredBy string
       (idempotent: guarded by ``future.done()`` so a duplicate Twilio
       callback doesn't raise ``InvalidStateError`` from ``set_result``).
    4. If AnsweredBy starts with ``"machine"`` OR equals ``"fax"``:
       trigger supervisor teardown (Failed / no_answer).  The driver wakes
       on the future with a machine verdict and exits cleanly.  Otherwise
       (human / unknown): do nothing further — the call proceeds and the
       driver wakes with a non-machine verdict.
    """
    form_data = await request.form()
    call_sid: str | None = form_data.get("CallSid")
    answered_by: str | None = form_data.get("AnsweredBy")

    if not call_sid or not answered_by:
        log.warning(
            "Twilio AMD callback missing CallSid or AnsweredBy: %s",
            dict(form_data),
        )
        return PlainTextResponse("")

    # ── Look up the live supervisor (always — both branches need it) ──────
    # Even human/unknown verdicts must reach the supervisor so the driver
    # can wake up from its amd_verdict await.  Stale / post-restart
    # callbacks (no supervisor in the registry) return 200 and trust the
    # startup sweep (Issue 029) — the driver is also gone in that case.
    supervisor = get_call(call_sid)
    if supervisor is None:
        log.warning(
            "Twilio AMD callback: CallSid=%s not in active-calls registry. "
            "Stale callback or post-restart — startup sweep (Issue 029) "
            "is the authoritative resolver. No-op.",
            call_sid,
        )
        return PlainTextResponse("")

    # ── Populate the AMD verdict future for EVERY verdict (Issue 040) ─────
    # Guarded with ``future.done()`` so a duplicate Twilio callback (Twilio
    # occasionally re-posts) doesn't raise ``InvalidStateError`` from
    # ``set_result``.  The first verdict wins; subsequent ones are silently
    # dropped — the driver has already woken up by then.
    if not supervisor.amd_verdict.done():
        supervisor.amd_verdict.set_result(answered_by)
        log.debug(
            "Twilio AMD callback: CallSid=%s AnsweredBy=%s — verdict "
            "published onto supervisor.amd_verdict.",
            call_sid,
            answered_by,
        )

    # ── Non-human verdict — signal the supervisor's teardown ──────────────
    # Non-human = any "machine_*" verdict OR "fax".  All other verdicts
    # (human, unknown) are treated as "proceed" and the driver wakes up
    # with the verdict string already in hand.
    is_machine = answered_by.startswith("machine") or answered_by == "fax"

    if not is_machine:
        log.debug(
            "Twilio AMD callback: CallSid=%s AnsweredBy=%s — "
            "human/unknown verdict; call proceeds (driver sees verdict "
            "via supervisor.amd_verdict).",
            call_sid,
            answered_by,
        )
        return PlainTextResponse("")

    # ── ORDER MATTERS (Issue 031 follow-up — race-condition hardening) ───────
    # The supervisor's teardown sequence writes the DB resolution BEFORE the
    # hangup, so the trailing `completed` status callback Twilio fires on
    # hangup safely hits the "already resolved → ignore" idempotent guard in
    # twilio_status_callback.  See CallSupervisor.trigger_teardown docstring.
    #
    # call_already_terminal defaults to False — for an AMD machine verdict,
    # the Twilio line is still live (we caught a voicemail), so we MUST hang
    # up to end the call and save Twilio credit.  The AMD future was already
    # populated above, so the driver wakes immediately and observes the
    # machine verdict before this teardown completes its DB write.
    await supervisor.trigger_teardown(
        outcome_status="Failed",
        failure_reason="no_answer",
    )

    log.info(
        "Twilio AMD callback: CallSid=%s AnsweredBy=%s → supervisor signalled "
        "(trigger_teardown outcome=Failed/no_answer).",
        call_sid,
        answered_by,
    )

    # Twilio requires a 200 response.
    return PlainTextResponse("")


# ─── Twilio status callback endpoint ───────────────────────────────────────────

# Twilio CallStatus values that mean the call has actually ended. Only these
# drive attempt resolution in this endpoint — see HITL FIX below.  Async AMD
# verdicts no longer drive resolution here; they arrive at
# /api/twilio/amd-callback and are resolved there.
_TERMINAL_TWILIO_STATUSES = frozenset({"completed", "busy", "no-answer", "canceled", "failed"})


@_status_callback_router.post(
    "/status-callback",
    summary="Twilio status callback webhook",
    description=(
        "Webhook endpoint for Twilio call status callbacks.  "
        "Issue 032: thin signaler — looks up the live CallSupervisor and "
        "signals it via trigger_teardown.  All teardown work (DB resolve, "
        "slot release, unregister) is owned by the supervisor.  "
        "No authentication required (Twilio posts to this endpoint)."
    ),
)
async def twilio_status_callback(
    request: Request,
) -> PlainTextResponse:
    """
    Handle Twilio status callback webhook.

    Issue 032: this endpoint is now a thin signaler — all teardown work is
    owned by the CallSupervisor.  The endpoint parses the form, gates on
    terminal CallStatus, runs the outcome classifier to determine the
    outcome, and signals the supervisor.  No direct DB writes, no direct
    slot release — the REMOVE-IN-032 stopgap is gone.

    Flow
    ────
    1. Parse CallSid and CallStatus from form data.
    2. Gate on terminal CallStatus (HITL FIX from Issue 030).  Non-terminal
       callbacks (initiated/ringing/in-progress) return 200 immediately.
    3. Look up the live CallSupervisor via ``get_call(call_sid)``.
       - Absent (stale / post-restart): log and return 200 — startup sweep
         (Issue 029) is the authoritative resolver for orphaned leads.
       - Present: signal the supervisor with the outcome from the classifier.
    4. The supervisor runs its fixed teardown sequence (DB resolve, slot
       release, unregister).  Idempotent under concurrent triggers via the
       one-shot latch — if the AMD callback fired first and already
       triggered teardown, this call is a no-op.

    Issue 031: ``AnsweredBy`` is intentionally NOT consumed here.  Async AMD
    verdicts arrive via the dedicated ``/api/twilio/amd-callback`` endpoint
    and are resolved there (active hangup + Failed/no_answer).  This endpoint
    only reacts to terminal ``CallStatus`` transitions.
    """
    # Parse form data from Twilio.
    form_data = await request.form()
    call_sid: str | None = form_data.get("CallSid")
    call_status: str | None = form_data.get("CallStatus")

    if not call_sid or not call_status:
        log.warning(
            "Twilio status callback missing CallSid or CallStatus: %s",
            dict(form_data),
        )
        return PlainTextResponse("")

    # ── HITL FIX (Issue 030): only resolve on a TERMINAL Twilio status ───────
    # Twilio sends non-terminal callbacks (initiated / ringing / in-progress)
    # before the call ends. The outcome classifier maps any non-terminal status
    # to Failed/error via its in-flight fallback, so calling it here on an early
    # callback would wrongly resolve a still-live call as Failed/error on the
    # first ring (caught by HITL).
    #
    # Issue 031: the "or non-human async-AMD verdict" branch was removed —
    # AMD verdicts now arrive at /api/twilio/amd-callback and are resolved
    # there.  This keeps terminal-status and AMD-verdict concerns in separate
    # endpoints that cannot tangle.
    if call_status not in _TERMINAL_TWILIO_STATUSES:
        log.debug(
            "Twilio status callback: CallSid=%s non-terminal status=%s — "
            "call still live, not resolving.",
            call_sid,
            call_status,
        )
        return PlainTextResponse("")

    # Normalise Twilio status to our combined_status format.
    combined_status_map = {
        "queued": "created",
        "ringing": "ringing",
        "in-progress": "initiated",
        "completed": "completed",
        "busy": "busy",
        "no-answer": "no_answer",
        "canceled": "canceled",
        "failed": "failed",
    }
    combined_status = combined_status_map.get(call_status, call_status)

    # ── Step 1: Look up the live CallSupervisor ──────────────────────────────
    supervisor = get_call(call_sid)
    if supervisor is None:
        # Stale callback (e.g., fired before the in-process call started, or
        # fired after the supervisor unregistered) or post-restart (no
        # in-memory supervisor exists).  The startup sweep (Issue 029) is
        # the authoritative resolver for orphaned Processing leads via the
        # database.  Trust it and return 200.
        log.warning(
            "Twilio status callback: CallSid=%s not in active-calls registry. "
            "Stale callback or post-restart — startup sweep (Issue 029) is "
            "the authoritative resolver. No-op.",
            call_sid,
        )
        return PlainTextResponse("")

    # ── Step 2: Run the outcome classifier ───────────────────────────────────
    #
    # P6 FIX: matcher_usable is a placeholder.  In the full implementation,
    # this would be read from the call session state stored alongside the attempt.
    # Defaulting to True means max-duration calls are saved when the Matcher
    # gathered usable preferences (safe default).  Set to False when the Matcher
    # gathered nothing to force a drop on max-duration timeout.
    #
    matcher_usable = True  # TODO(030): read from call session state

    outcome: CallOutcome = classify(
        combined_status=combined_status,
        # Issue 031: AMD verdicts no longer arrive via this endpoint — they
        # are consumed and acted on by /api/twilio/amd-callback.  Pass None
        # so the classifier treats the call as "AMD unknown / not yet reported"
        # and routes by CallStatus alone.
        answered_by=None,
        end_consultation_fired=False,  # Driven by Mercury-2 end_consultation tool call (future)
        matcher_usable=matcher_usable,
        provider_fault=False,  # Driven by orchestrator fault detection (future)
    )

    # ── Step 3: Signal the supervisor ────────────────────────────────────────
    # The supervisor runs the fixed teardown sequence (DB resolve, slot
    # release, unregister).  If the AMD callback already triggered teardown,
    # this call no-ops via the one-shot latch.
    #
    # call_already_terminal=True: this endpoint only ever fires with terminal
    # Twilio CallStatus values (completed / busy / no-answer / canceled /
    # failed), which means Twilio has already ended the call.  Skipping the
    # hangup here avoids a wasted Twilio API call (which would just bounce
    # off Twilio).  The AMD callback and orphan-window handler leave it at
    # the default False because the line may still be live when they fire.
    await supervisor.trigger_teardown(
        outcome_status=outcome.status,
        failure_reason=outcome.failure_reason,
        completed_timeout=outcome.completed_timeout,
        call_already_terminal=True,
    )

    log.info(
        "Twilio status callback: CallSid=%s → %s (failure_reason=%s) — "
        "supervisor signalled.",
        call_sid,
        outcome.status,
        outcome.failure_reason,
    )

    # Twilio requires a 200 response. Plain text is fine.
    return PlainTextResponse("")


# ─── Twilio recording-status callback (Issue 037) ────────────────────────────


@_status_callback_router.post(
    "/recording-callback",
    summary="Twilio recording-status callback webhook (Issue 037)",
    description=(
        "Webhook endpoint for Twilio's recording-completion notification. "
        "Receives RecordingSid + RecordingStatus when Twilio finalises the "
        "audio file captured via ``record=True`` on outbound calls. "
        "Actual download / S3 archival is Issue 049 — for now the handler "
        "ONLY logs receipt and, if the call resolved as ``no_answer`` "
        "(machine/fax), deletes the recording so the audio is never "
        "retained.  Belt-and-braces with the supervisor's no_answer "
        "teardown path: whichever fires first wins, the other is a "
        "harmless no-op."
    ),
)
async def twilio_recording_callback(
    request: Request,
) -> PlainTextResponse:
    """
    Receive Twilio's recording-completion notification.

    Issue 037 (discard-on-voicemail): when a call resolves as no_answer
    (machine/fax verdict), the captured recording is of a non-consenting
    party — the voicemail box owner never consented to recording — and
    must be deleted.  This endpoint deletes the recording when the
    callback fires, IF the lead's outcome is already known to be
    ``Failed/no_answer`` in the database.

    If the lead outcome is NOT yet known (callback fired before the AMD
    callback finished persisting), the supervisor's teardown path will
    issue the same delete later — whichever path fires first wins, the
    other finds no recordings to delete and no-ops.

    NOTE: actual download is NOT implemented here.  Issue 049 will fetch
    the recording URL from the form payload and persist it to S3 for
    non-no_answer outcomes.  For now we just log receipt and discard
    when applicable.
    """
    form_data = await request.form()
    call_sid: str | None = form_data.get("CallSid")
    recording_sid: str | None = form_data.get("RecordingSid")
    recording_status: str | None = form_data.get("RecordingStatus")
    recording_url: str | None = form_data.get("RecordingUrl")

    log.info(
        "Twilio recording callback: CallSid=%s RecordingSid=%s "
        "Status=%s (URL=%s)",
        call_sid,
        recording_sid,
        recording_status,
        # Recording URL is masked here — full URL contains the auth
        # token and is PII-adjacent; we log only its presence.
        "<present>" if recording_url else "<absent>",
    )

    if not call_sid:
        log.warning(
            "Twilio recording callback missing CallSid: %s",
            dict(form_data),
        )
        return PlainTextResponse("")

    # Resolve the DB pool from app.state.  We need it to look up the
    # lead outcome to decide whether to discard.  If the pool is not
    # wired (e.g. mid-startup), skip the lookup and let the supervisor
    # teardown path handle the discard.
    pool = getattr(request.app.state, "pool", None)
    if pool is None:
        log.debug(
            "Twilio recording callback: no DB pool wired; skipping "
            "outcome-based discard (supervisor teardown will handle).",
        )
        return PlainTextResponse("")

    # Look up the lead outcome for this CallSid.  If the lead was
    # resolved as ``Failed/no_answer``, delete the recording.  Any
    # other status (Completed / Processing / New) means we keep it.
    try:
        async with pool.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT l.status, l.failure_reason
                FROM call_attempts ca
                JOIN leads l ON ca.lead_id = l.id
                WHERE ca.twilio_call_sid = ?;
                """,
                (call_sid,),
            )
            row = await cursor.fetchone()
    except Exception as exc:  # pragma: no cover — defensive
        log.warning(
            "Twilio recording callback: DB lookup failed for "
            "CallSid=%s: %s — proceeding without discard.",
            call_sid,
            exc,
        )
        return PlainTextResponse("")

    if row is None:
        # CallSid not in our DB — post-restart, stale, or webhook for
        # a call we never made.  Nothing to do.
        log.debug(
            "Twilio recording callback: CallSid=%s not in call_attempts; "
            "ignoring.",
            call_sid,
        )
        return PlainTextResponse("")

    status, failure_reason = row
    if status == "Failed" and failure_reason == "no_answer":
        # Issue 037: discard the voicemail recording.  The async wrapper
        # swallows exceptions so a Twilio glitch cannot surface here.
        from backend.call_supervisor import _delete_call_recordings_async

        deleted = await _delete_call_recordings_async(call_sid)
        log.info(
            "Twilio recording callback: discarded %d recording(s) for "
            "no_answer CallSid=%s (RecordingSid=%s).",
            deleted,
            call_sid,
            recording_sid,
        )
    else:
        log.debug(
            "Twilio recording callback: CallSid=%s outcome=%s reason=%s "
            "— recording retained (download is Issue 049).",
            call_sid,
            status,
            failure_reason,
        )

    return PlainTextResponse("")


# ─── Placeholder TwiML endpoint ───────────────────────────────────────────────
#
# P6 FIX: /api/twilio/placeholder-twiml was referenced by _twilio_create_call
# but did not exist.  Issue 034: this is no longer a "placeholder" — it is the
# LIVE TwiML that opens the bidirectional Media Streams WebSocket and dedicates
# the call to it.  The <Connect> verb is terminal/holding: Twilio keeps the call
# open, our side streams audio in both directions through the WebSocket, and the
# call only ends when the customer hangs up or our orchestrator triggers teardown.
#


@_status_callback_router.api_route(
    "/placeholder-twiml",
    methods=["GET", "POST"],
    response_class=PlainTextResponse,
    summary="TwiML that opens the Media Streams WebSocket (Issue 034)",
    description=(
        "Returns TwiML with <Connect><Stream url=\".../api/twilio/media-stream\"/></Connect>. "
        "The <Connect> verb is terminal — it dedicates the call to the bidirectional "
        "WebSocket until the customer hangs up.  PUBLIC_HOSTNAME controls the stream URL."
    ),
)
async def placeholder_twiml() -> PlainTextResponse:
    """
    Return TwiML that opens the Media Streams WebSocket.

    Issue 034: this is the entry point for the entire voice pipeline.  When the
    call is answered, Twilio fetches this TwiML and follows it top-to-bottom:

        1. <Connect> — terminal/holding verb.  Once Twilio hits this, the
           call stays open and the audio plane is bidirectionally attached
           to the WebSocket at <Stream url=...>.  Every subsequent customer
           utterance arrives as base64 frames at our /api/twilio/media-stream
           endpoint; every TTS frame we send back is played to the customer.

           The pre-rendered legal-recording disclosure is now played by
           ``play_disclosure`` in ``backend/api/media_stream.py`` (Issue
           037) as the FIRST audio emitted on the WebSocket.  Issue 036's
           ``<Say voice="alice">Connecting to the audio pipeline.</Say>``
           stopgap was removed in Issue 037 because it would race the
           disclosure for the customer's first second of audio.
        2. The call ends only when:
             - the customer hangs up (Twilio fires /api/twilio/status-callback), or
             - our orchestrator (Issue 041 + 043) decides to end it (also
               through Twilio's hangup API which fires the status callback).

    PUBLIC_HOSTNAME controls the WebSocket URL:
        - Set (via env or .env):  wss://{PUBLIC_HOSTNAME}/api/twilio/media-stream
        - Empty (dev mode):        ws://127.0.0.1:8000/api/twilio/media-stream

    We read settings lazily on each invocation: load_settings() is lru_cache'd
    so this is a single Settings() instantiation per process, never per request.
    """
    from backend.config import load_settings

    settings = load_settings()
    public_hostname = (settings.public_hostname or "").strip()
    if public_hostname:
        # Production / staging: tunnel hostname (ngrok, cloudflared) is set;
        # Twilio needs the secure wss:// scheme to talk to us.
        stream_url = f"wss://{public_hostname}/api/twilio/media-stream"
    else:
        # Dev mode: PUBLIC_HOSTNAME is empty (no tunnel).  Twilio cannot
        # reach localhost from the public internet, so this URL is only
        # useful for manually-driven smoke tests on the dev box — explicit
        # localhost so the operator can see the URL and not wonder.
        stream_url = "ws://127.0.0.1:8000/api/twilio/media-stream"

    # Issue 037: <Say> preamble removed — the disclosure audio now occupies
    # the customer's first second (played by play_disclosure in
    # backend/api/media_stream.py).  <Connect> is a terminal verb so no
    # verb after it would execute.
    xml = '<?xml version="1.0" encoding="UTF-8"?>'
    xml += "<Response>"
    xml += "<Connect>"
    xml += f'<Stream url="{stream_url}" />'
    xml += "</Connect>"
    xml += "</Response>"
    return PlainTextResponse(xml, media_type="application/xml")


# ─── Pool resolution helper ─────────────────────────────────────────────────────


def get_pool_from_request(request: Request):
    """Resolve the DB connection pool from FastAPI request app state."""
    return request.app.state.pool
