"""
backend/api/leads.py — Lead submission API (Issues 007 / 008 / 009).

Public, unauthenticated endpoint for customer lead submission.

Design
──────
- Validates the raw payload with Pydantic (email format via EmailStr, phone
  format via field_validator).
- Authoritatively re-validates and canonicalizes phone via
  ``backend.contact.normalize_indian_mobile`` → E.164.
- Authoritatively canonicalizes email via ``backend.contact.normalize_email``.
- Guard pipeline (handled by ``backend.domain.submission``):
    * Block-list check (Issue 009): queries ``block_list`` for the canonical
      phone OR canonical email.  If found → silently drops the submission
      and returns the neutral ack (no lead inserted).
    * Deduplication check (Issue 008): within the submission transaction,
      a SELECT checks for an existing open lead (New or Processing).  If
      found the row is NOT inserted and the fixed neutral acknowledgment is
      returned immediately.
    * If only terminal leads (Completed or Failed) exist for the phone,
      the new row is inserted normally.
    * The partial UNIQUE index ``idx_leads_phone_active`` acts as a race-condition
      backstop.  A caught ``sqlite3.IntegrityError`` is mapped to the same
      neutral acknowledgment without a 500 error.
- The response for blocked, duplicate, or DB-constraint duplicates is
  byte-identical to a successful submission (201 + NEUTRAL_MESSAGE JSON).
  This leak-proof contract prevents a malicious actor from probing phone
  numbers to determine who is blocked or has an open request.

Pool injection
─────────────
The DB pool is injected via FastAPI's ``Depends(get_pool)`` — never imported
directly from ``main.py``.  This pattern avoids circular imports and allows
tests to swap the pool cleanly via ``app.dependency_overrides``.

No module calls ``load_settings()`` at import time.
"""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, Field, field_validator

from backend.api.auth import get_current_admin
from backend.contact import normalize_email, normalize_indian_mobile
from backend.database._leads import fetch_leads
from backend.database._leads_deletion import delete_lead_cascade
from backend.database._sessions import Session
from backend.domain.submission import submit_lead as _submit_lead
from backend.domain.submission import NEUTRAL_MESSAGE
from backend.domain.timestamp import to_iso_z

# Sliding-window duration in seconds (1 hour).
_HOUR_SECS = 3600

# ─── Shared FastAPI dependency ──────────────────────────────────────────────────


def get_pool(request: Request):
    """
    FastAPI dependency that resolves the DB connection pool from app state.

    Injected into route handlers via ``Depends(get_pool)`` — pattern will be
    replicated across 12 endpoints.  Tests can override via
    ``app.dependency_overrides[get_pool] = test_pool``.
    """
    return request.app.state.pool


async def rate_limit_check(request: Request) -> None:
    """
    FastAPI dependency that enforces per-IP rate limiting before any other work.

    Uses an in-memory sliding-window counter (Issue 010).  Because the MVP is
    a single-process, single-event-loop application, a plain dict is sufficient
    and avoids any cross-process locking or Redis dependency.

    The IP is read from ``request.client.host`` — acceptable for the MVP.  When
    deployed behind a reverse proxy or tunnel (e.g. ngrok), the real client IP
    may appear in the ``X-Forwarded-For`` header; that header should be trusted
    in those environments, but ``client.host`` is fine for local dev and MVP.

    Behaviour
    ---------
    - Looks up ``request.app.state.rate_limit_store`` (dict[IP, list[float timestamps]])
      and ``request.app.state.rate_limit_limit`` (int max req/IP/hour).
    - Evicts timestamps older than 1 hour (sliding window).
    - If the current count >= limit → raises HTTP 429 Too Many Requests.
    - Otherwise records the current timestamp and lets the request through.

    Raises
    ------
    HTTPException 429 — client has exceeded the per-IP-per-hour submission limit.
    """
    store: dict[str, list[float]] = request.app.state.rate_limit_store
    limit: int = request.app.state.rate_limit_limit

    # request.client.host is None when the client disconnects before the handler runs;
    # fall back to "unknown" so we don't crash (the flood will still be rejected).
    ip: str = getattr(request.client, "host", None) or "unknown"

    now = time.monotonic()
    window = store.setdefault(ip, [])

    # Sliding window: keep only timestamps within the last hour.
    cutoff = now - _HOUR_SECS
    # Slice-assignment in-place is safe because we're the sole writer for this IP.
    window[:] = [ts for ts in window if ts > cutoff]

    if len(window) >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "msg": "Too many requests. Please try again later.",
                # Intentionally generic — no mention of the actual limit or window.
            },
        )

    window.append(now)


# ─── Request / Response models ───────────────────────────────────────────────────


class LeadSubmissionRequest(BaseModel):
    """
    Incoming lead payload — validated at the API boundary.

    Fields
    ──────
    name       : required free-text name
    phone      : required; any common Indian format accepted here; canonicalized
                 to E.164 before storage.
    email      : required; must be a valid email address shape (EmailStr).
                 canonicalized to lower/trimmed form before storage.
    city       : required free-text city
    language   : required; must be ``"English"`` or ``"Hinglish"``
    vehicle_id : optional; if present, stored as the vehicle FK.
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Customer's full name",
    )
    phone: str = Field(
        ...,
        min_length=1,
        max_length=30,
        description="Customer phone number (any common Indian format)",
    )
    email: EmailStr = Field(
        ...,
        description="Customer email address",
    )
    city: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="City / locality",
    )
    language: str = Field(
        ...,
        description="Preferred communication language",
    )
    vehicle_id: int | None = Field(
        default=None,
        description="Selected vehicle ID (null = 'Not specified')",
    )

    @field_validator("language")
    @classmethod
    def _language_must_be_english_or_hinglish(cls, v: str) -> str:
        if v not in ("English", "Hinglish"):
            raise ValueError(
                "language must be one of: English, Hinglish"
            )
        return v


class NeutralAcknowledgment(BaseModel):
    """
    Uniform neutral acknowledgment returned on every successful submission.

    The message is intentionally generic — no internal state (DB IDs, raw payload,
    status) is ever reflected back.
    """

    message: str = Field(
        default=NEUTRAL_MESSAGE,
        description="Generic acknowledgement — never changes.",
    )


# ─── Router ───────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/api", tags=["leads"])


# ─── Endpoint ─────────────────────────────────────────────────────────────────


@router.post(
    "/leads",
    response_model=NeutralAcknowledgment,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a customer lead",
    description=(
        "Public endpoint for the customer form. "
        "Validates and canonicalizes the phone (E.164) and email (lower/trim), "
        "checks block-list (silently drops if blocked), "
        "stores the lead with status New, and returns a neutral acknowledgment."
    ),
)
async def submit_lead(
    payload: LeadSubmissionRequest,
    request: Request,
    pool: Annotated[object, Depends(get_pool)],
    _: Annotated[None, Depends(rate_limit_check)],
) -> NeutralAcknowledgment:
    """
    Create a new ``New`` lead from the submitted form data.

    Rate limiting (Issue 010)
    -------------------------
    ``rate_limit_check`` runs BEFORE any other work — before Pydantic
    validation fires and certainly before DB queries.  This minimises
    compute cost for flood traffic.  A rejected request returns 429.

    Pool injection
    --------------
    The DB pool is injected via ``Depends(get_pool)`` from this module,
    which reads from ``app.state.pool`` set during the FastAPI lifespan.
    This avoids a circular import and keeps the pool swappable for tests.

    Validation pipeline
    ------------------
    1. Pydantic model validation (types, required fields, EmailStr format,
       language enum).  Returns HTTP 422 on failure.
    2. Phone canonicalization (``normalize_indian_mobile`` from Issue 005).
       Returns HTTP 422 on invalid phone.
    3. Email canonicalization (``normalize_email`` from Issue 005).
       Raises HTTP 422 on empty-after-trim (should not occur given EmailStr).
    4. vehicle_id existence check against the in-memory catalogue.
       Returns HTTP 422 if an unknown vehicle_id is supplied.
    5. Guard pipeline (backend.domain.submission.submit_lead):
       a. Block-list check (Issue 009): queries ``block_list`` for canonical
          phone OR canonical email.  If found → silently drops and returns
          the neutral ack (no lead inserted).
       b. Deduplication check (Issue 008): within the submission transaction,
          SELECT checks for an existing open lead (New or Processing).  If
          found, returns the neutral acknowledgment WITHOUT inserting.
       c. If only terminal leads (Completed / Failed) exist, the new row
          is inserted normally.
       d. The partial UNIQUE index acts as a race-condition backstop:
          ``sqlite3.IntegrityError`` is caught and mapped to the same
          neutral acknowledgment (no 500).
    6. Return the fixed neutral acknowledgment (HTTP 201).

    The response is byte-identical for fresh, blocked, and deduplicated
    submissions, preventing state-leakage via response timing or content.
    """
    # ── 1. Phone canonicalization (authoritative, Issue 005) ──────────────────
    try:
        canonical_phone = normalize_indian_mobile(payload.phone)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"phone": str(exc)},
        ) from exc

    # ── 2. Email canonicalization (Issue 005) ─────────────────────────────────
    try:
        canonical_email = normalize_email(payload.email)
    except ValueError as exc:
        # EmailStr already guarantees a valid shape, but normalize_email
        # raises on empty-after-trim as a defence-in-depth measure.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"email": str(exc)},
        ) from exc

    # ── 3. vehicle_id lookup (optional) ─────────────────────────────────────
    vehicle_id: int | None = None
    if payload.vehicle_id is not None:
        from backend.catalogue import get_vehicle_by_id

        record = get_vehicle_by_id(payload.vehicle_id)
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "vehicle_id": (
                        f"Vehicle with id {payload.vehicle_id} does not exist."
                    )
                },
            )
        vehicle_id = payload.vehicle_id

    # ── 4. Delegate to the handler module (block check → dedupe → insert) ──────
    _ = await _submit_lead(
        pool=pool,
        name=payload.name,
        canonical_phone=canonical_phone,
        canonical_email=canonical_email,
        city=payload.city,
        language=payload.language,
        vehicle_id=vehicle_id,
    )

    # ── 5. Neutral acknowledgment (never leaks internal state) ───────────
    return NeutralAcknowledgment(message=NEUTRAL_MESSAGE)


# ─── Admin dashboard endpoints ────────────────────────────────────────────────


class LeadHistoryEntryResponse(BaseModel):
    """Single historical lead entry in the ``history`` array."""

    id: int = Field(..., description="Lead ID")
    status: str = Field(..., description="Lead status at creation time")
    created_at: str = Field(..., description="Submission time in ISO-8601 UTC (Z-suffixed)")


class LeadResponse(BaseModel):
    """
    Serialised lead row for the admin dashboard list view.

    ``created_at`` is formatted as ISO-8601 UTC with a ``Z`` suffix so the
    frontend receives an unambiguous timestamp regardless of server timezone.
    """

    id: int = Field(..., description="Lead ID")
    name: str = Field(..., description="Customer name")
    phone: str = Field(..., description="Customer E.164 phone number")
    email: str = Field(..., description="Customer email address")
    city: str = Field(..., description="City / locality")
    language: str = Field(..., description="Preferred communication language")
    vehicle_id: int | None = Field(None, description="Selected vehicle ID (null = 'Not specified')")
    status: str = Field(..., description="Current lead status")
    failure_reason: str | None = Field(None, description="Failure reason (only present when status='Failed')")
    attempt_count: int = Field(..., description="Number of dial attempts made")
    created_at: str = Field(..., description="Submission time in ISO-8601 UTC (Z-suffixed)")
    history: list[LeadHistoryEntryResponse] = Field(
        ...,
        description="All leads for the same phone, newest-first.  Includes this lead.",
    )


class LeadsListResponse(BaseModel):
    """Paginated-ish response envelope for GET /api/leads."""

    leads: list[LeadResponse] = Field(..., description="All leads, newest-first")
    total: int = Field(..., description="Total number of leads returned")


@router.get(
    "/leads",
    response_model=LeadsListResponse,
    summary="List all leads (admin)",
    description="Authenticated endpoint returning all leads ordered newest-first, with per-phone history attached.",
)
async def list_leads(
    _: Session = Depends(get_current_admin),
    pool: Annotated[object, Depends(get_pool)] = ...,  # type: ignore[assignment]
) -> LeadsListResponse:
    """
    Serve the admin dashboard leads list.

    Auth
    ────
    ``get_current_admin`` enforces a valid session cookie.  GET is exempt from
    CSRF checks (no state-changing side-effects).

    Timestamp contract
    ──────────────────
    ``created_at`` is the raw SQLite UTC string (``YYYY-MM-DD HH:MM:SS``).
    ``to_iso_z`` normalises it to ISO-8601 with an explicit ``Z`` suffix so the
    browser parses it as UTC, not local time.

    Customer history
    ───────────────
    Each lead carries a ``history`` array listing every lead sharing the same
    phone number, ordered newest-first.  This lets the admin see the full
    per-customer timeline at a glance without extra requests.
    """
    raw_leads = await fetch_leads(pool)

    return LeadsListResponse(
        leads=[
            LeadResponse(
                id=lead.id,
                name=lead.name,
                phone=lead.phone,
                email=lead.email,
                city=lead.city,
                language=lead.language,
                vehicle_id=lead.vehicle_id,
                status=lead.status,
                failure_reason=lead.failure_reason,
                attempt_count=lead.attempt_count,
                created_at=to_iso_z(lead.created_at),
                history=[
                    LeadHistoryEntryResponse(
                        id=h.id,
                        status=h.status,
                        created_at=to_iso_z(h.created_at),
                    )
                    for h in lead.history
                ],
            )
            for lead in raw_leads
        ],
        total=len(raw_leads),
    )


# ─── Delete lead endpoint ───────────────────────────────────────────────────────


@router.delete(
    "/leads/{lead_id:int}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    summary="Delete a lead (admin)",
    description=(
        "Hard-delete a lead and all its associated data (call attempts, transcripts, "
        "recommendations). The per-lead recording folder is also wiped. "
        "Session-gated and CSRF-protected. "
        "Returns 409 Conflict if the lead is in Processing status."
    ),
)
async def delete_lead(
    lead_id: int,
    _: Session = Depends(get_current_admin),
    pool: Annotated[object, Depends(get_pool)] = ...,  # type: ignore[assignment]
):
    """
    Delete a lead by ID.

    Auth
    ────
    ``get_current_admin`` enforces a valid session cookie and CSRF token.
    CSRF is enforced for DELETE because it is a state-changing method.

    Guard: Processing status
    ─────────────────────────
    A lead in ``Processing`` status is actively handled by a call worker.
    Deleting it mid-call would corrupt the call state.  We return HTTP 409
    Conflict to block the operation until the lead reaches a terminal state.

    Cascade
    ──────
    ``delete_lead_cascade`` wipes:
      - the per-lead recording folder (``<recordings_root>/<lead_id>/``)
      - the lead row (SQLite ON DELETE CASCADE takes care of call_attempts,
        transcript_turns, and recommendations)
    The ``block_list`` table is intentionally left untouched.
    """
    from backend.config import load_settings

    async with pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT status FROM leads WHERE id = ?",
            (lead_id,),
        )
        row = await cursor.fetchone()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Lead {lead_id} not found.",
        )

    current_status = row[0]
    if current_status == "Processing":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "msg": "Cannot delete a lead that is currently Processing.",
                "lead_id": lead_id,
                "status": current_status,
            },
        )

    settings = load_settings()
    await delete_lead_cascade(pool, lead_id, recordings_root=settings.recordings_root)


# ─── Block lead endpoint ──────────────────────────────────────────────────────


class BlockSuccessResponse(BaseModel):
    """Response when a lead is successfully blocked."""

    message: str = Field(
        default="Customer has been blocked.",
        description="Confirmation message.",
    )


@router.post(
    "/leads/{lead_id:int}/block",
    response_model=BlockSuccessResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Block a lead (admin)",
    description=(
        "Insert the lead into the block_list using their canonical phone, email, "
        "and a name snapshot.  Re-blocking is idempotent (no-op if already blocked). "
        "Blocking a Processing lead does NOT terminate the active call. "
        "Session-gated and CSRF-protected."
    ),
)
async def block_lead(
    lead_id: int,
    _: Session = Depends(get_current_admin),
    pool: Annotated[object, Depends(get_pool)] = ...,  # type: ignore[assignment]
) -> BlockSuccessResponse:
    """
    Block a lead by ID.

    Auth
    ────
    ``get_current_admin`` enforces a valid session cookie and CSRF token.

    Guard: does NOT terminate active calls
    ─────────────────────────────────────
    A lead in ``Processing`` status is actively handled by a call worker.
    Blocking simply prevents future submissions — it has no effect on the
    current call.  This is intentional: Issue 009 already handles the
    submission block, and interrupting an active call would corrupt state.

    Idempotency
    ───────────
    ``INSERT OR IGNORE`` against the UNIQUE ``phone`` column means re-blocking
    the same phone is a no-op (no error, no duplicate row).  The name and
    email snapshot are preserved from the first block.
    """
    from backend.database._blocks import block_customer

    async with pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT phone, email, name FROM leads WHERE id = ?",
            (lead_id,),
        )
        row = await cursor.fetchone()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Lead {lead_id} not found.",
        )

    canonical_phone, canonical_email, name_snapshot = row

    await block_customer(
        pool=pool,
        canonical_phone=canonical_phone,
        canonical_email=canonical_email,
        name_snapshot=name_snapshot,
    )

    return BlockSuccessResponse(message="Customer has been blocked.")


# ─── Blocked customers endpoints ───────────────────────────────────────────────


class BlockedCustomerResponse(BaseModel):
    """Single blocked customer in the GET /api/blocks response."""

    id: int = Field(..., description="Block entry ID")
    phone: str = Field(..., description="E.164 canonical phone")
    email: str = Field(..., description="Lowercased/trimmed email")
    name: str | None = Field(None, description="Name snapshot at block time")
    created_at: str = Field(
        ...,
        description="Block time in ISO-8601 UTC (Z-suffixed)",
    )


class BlockedListResponse(BaseModel):
    """Response envelope for GET /api/blocks."""

    blocked: list[BlockedCustomerResponse] = Field(
        ...,
        description="All blocked customers, newest-first",
    )
    total: int = Field(..., description="Total number of blocked customers")


@router.get(
    "/blocks",
    response_model=BlockedListResponse,
    summary="List blocked customers (admin)",
    description=(
        "Return all blocked customers ordered newest-first. "
        "Session-gated but CSRF-exempt (read-only)."
    ),
)
async def list_blocked(
    _: Session = Depends(get_current_admin),
    pool: Annotated[object, Depends(get_pool)] = ...,  # type: ignore[assignment]
) -> BlockedListResponse:
    """
    Serve the admin dashboard blocked customers list.

    Auth
    ────
    ``get_current_admin`` enforces a valid session cookie.  GET is exempt from
    CSRF checks (no state-changing side-effects).

    Timestamp contract
    ──────────────────
    ``created_at`` is the raw SQLite UTC string.  ``to_iso_z`` normalises it
    to ISO-8601 with an explicit ``Z`` suffix.
    """
    from backend.database._blocks import fetch_blocked

    blocked = await fetch_blocked(pool)

    return BlockedListResponse(
        blocked=[
            BlockedCustomerResponse(
                id=b.id,
                phone=b.phone,
                email=b.email,
                name=b.name,
                created_at=to_iso_z(b.created_at),
            )
            for b in blocked
        ],
        total=len(blocked),
    )


class UnblockSuccessResponse(BaseModel):
    """Response when a customer is unblocked."""

    message: str = Field(
        default="Customer has been unblocked.",
        description="Confirmation message.",
    )


@router.delete(
    "/blocks/{phone}",
    response_model=UnblockSuccessResponse,
    status_code=status.HTTP_200_OK,
    summary="Unblock a customer (admin)",
    description=(
        "Remove a blocked customer by exact E.164 phone number. "
        "Session-gated and CSRF-protected. "
        "Returns 404 if the phone is not currently blocked."
    ),
)
async def unblock_customer(
    phone: str,
    _: Session = Depends(get_current_admin),
    pool: Annotated[object, Depends(get_pool)] = ...,  # type: ignore[assignment]
) -> UnblockSuccessResponse:
    """
    Unblock a customer by exact canonical phone.

    Auth
    ────
    ``get_current_admin`` enforces a valid session cookie and CSRF token.

    Edge case: shared email
    ─────────────────────────
    If two different blocked phones share the same email, unblocking one
    phone does NOT affect the other.  Each phone is an independent block_list
    entry with its own row.
    """
    from backend.database._blocks import unblock_customer as _unblock

    was_unblocked = await _unblock(pool, phone)

    if not was_unblocked:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Phone {phone} is not blocked.",
        )

    return UnblockSuccessResponse(message="Customer has been unblocked.")
