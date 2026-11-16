"""
backend/domain/submission.py — Lead submission business logic (Issues 008 / 009).

Extractable, unit-testable handler module for the guard pipeline that runs before
any lead row is inserted.  The pipeline enforces:

    1. Block-list check  (Issue 009)
       - Queries ``block_list`` for the canonical phone OR canonical email.
       - If found → silently drop (no lead inserted) and return the neutral ack.

    2. Deduplication check  (Issue 008)
       - Within the submission transaction, SELECT checks for an existing open
         lead (New or Processing) for the canonical phone.
       - If found → return the neutral ack WITHOUT inserting.
       - If only terminal leads (Completed / Failed) exist, the new row is
         inserted normally.

    3. Insert the new lead row.

The response is byte-identical for fresh, deduplicated, and blocked submissions
(201 + NEUTRAL_MESSAGE).  This leak-proof contract prevents a malicious actor
from probing phone numbers to determine block or dedupe state.

Pool injection
─────────────
All functions accept the DB pool as a parameter so callers can inject it via
``Depends(get_pool)`` from the FastAPI layer.  No module calls ``load_settings()``
at import time.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from backend.database import transaction

# ─── Module-level constant ──────────────────────────────────────────────────────

NEUTRAL_MESSAGE = "Thanks, we've received your request and will be in touch."
"""
Fixed string used as the response body.  Defined once here so it can also be
asserted in tests without duplicating the literal.  Reproducibility is paramount:
slices 008/009 must return this byte-identical shape and status code to prevent
state leakage.
"""


# ─── Data classes ────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CanonicalLead:
    """
    Fully canonicalized lead data ready for business-logic checks and storage.
    """

    name: str
    phone: str       # E.164 (+91XXXXXXXXXX)
    email: str       # lowercased / trimmed
    city: str
    language: str
    vehicle_id: int | None


# ─── Public API ────────────────────────────────────────────────────────────────


async def is_blocked(
    pool,
    canonical_phone: str,
    canonical_email: str,
) -> bool:
    """
    Check whether the canonical phone OR canonical email matches a block-list entry.

    Parameters
    ----------
    pool            : ConnectionPool
    canonical_phone : str  — E.164 form
    canonical_email : str  — lowercased / trimmed

    Returns
    -------
    True  → customer is blocked; submission must be silently dropped.
    False → customer is not blocked; proceed with deduplication.
    """
    async with pool.connection() as conn:
        async with conn.execute(
            """
            SELECT 1 FROM block_list
            WHERE phone = ? OR email = ?
            LIMIT 1
            """,
            (canonical_phone, canonical_email),
        ) as cur:
            row = await cur.fetchone()
            return row is not None


async def submit_lead(
    pool,
    name: str,
    canonical_phone: str,
    canonical_email: str,
    city: str,
    language: str,
    vehicle_id: int | None,
) -> str:
    """
    Full submission pipeline: block check → dedupe check → insert.

    Parameters
    ----------
    pool             : ConnectionPool
    name             : str
    canonical_phone  : str  — already normalized to E.164 form
    canonical_email  : str  — already normalized to lower/trimmed form
    city             : str
    language         : str
    vehicle_id       : int | None

    Returns
    -------
    The neutral acknowledgment message string (byte-identical to NEUTRAL_MESSAGE).

    Raises
    ------
    ValueError — phone or email failed canonicalization (should not occur;
                 normalization is done in the API layer before calling this).
    sqlite3.IntegrityError — race-condition backstop (maps to neutral ack).
    """
    # ── Step 1: Block-list check (Issue 009) ────────────────────────────────────
    # Runs BEFORE deduplication — a blocked customer is dropped instantly
    # regardless of dedupe state.  No transaction needed for this read check.
    if await is_blocked(pool, canonical_phone, canonical_email):
        return NEUTRAL_MESSAGE

    # ── Step 2: Deduplication check + insert (Issue 008) ──────────────────────
    #
    # Within the transaction:
    #   1. SELECT checks for an existing open lead (New or Processing) for
    #      this canonical phone.  If found, return the neutral ack without
    #      inserting.
    #   2. The partial UNIQUE index acts as a race-condition backstop:
    #      ``sqlite3.IntegrityError`` is caught and mapped to the same
    #      neutral ack.
    #
    # Terminal leads (Completed / Failed) do NOT block a new submission.
    try:
        async with pool.connection() as conn:
            async with transaction(conn) as txn:
                async with txn.execute(
                    """
                    SELECT 1 FROM leads
                    WHERE phone = ? AND status IN ('New', 'Processing')
                    LIMIT 1
                    """,
                    (canonical_phone,),
                ) as cur:
                    open_lead_exists = await cur.fetchone() is not None

                if open_lead_exists:
                    # Duplicate — return the neutral ack without inserting.
                    return NEUTRAL_MESSAGE

                # No open lead found — insert the new row.
                await txn.execute(
                    """
                    INSERT INTO leads
                        (name, phone, email, city, language, vehicle_id,
                         status, attempt_count)
                    VALUES
                        (?, ?, ?, ?, ?, ?, 'New', 0)
                    """,
                    (
                        name,
                        canonical_phone,
                        canonical_email,
                        city,
                        language,
                        vehicle_id,
                    ),
                )
    except sqlite3.IntegrityError:
        # Race-condition backstop: the partial UNIQUE index caught a
        # simultaneous insert.  Map it to the same neutral acknowledgment.
        return NEUTRAL_MESSAGE

    return NEUTRAL_MESSAGE
