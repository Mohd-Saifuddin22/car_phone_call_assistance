"""
backend/database/_leads.py — Lead list repository (Issue 016).

Provides a single fetch function for the admin dashboard's leads list view.

Design
──────
A flat ``SELECT`` ordered by ``created_at DESC`` is the simplest, most efficient
approach for the MVP scale (no pagination needed yet).  The caller groups leads
by phone in memory to build the ``history`` field.

Why not a window function?
─────────────────────────
SQLite's ``json_group_array`` requires a separate CTE or subquery for the window
partition, and attaching the full history JSON to each row adds per-row overhead
that grows with history size.  For the expected lead volume (dozens to low hundreds
per admin session), fetching all rows once and grouping in Python is faster and
produces the same shape with zero SQLite JSON complexity.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict

from backend.database._pool import ConnectionPool


# ─── Data classes ───────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class LeadRecord:
    """
    Raw fields pulled directly from the ``leads`` table.

    ``created_at`` is the raw SQLite space-separated UTC string — call sites
    are responsible for passing it through ``to_iso_z`` before serialising to
    JSON.
    """

    id: int
    name: str
    phone: str
    email: str
    city: str
    language: str
    vehicle_id: int | None
    status: str
    failure_reason: str | None
    attempt_count: int
    created_at: str


@dataclass(frozen=True, slots=True)
class LeadHistoryEntry:
    """A single historical lead entry for the ``history`` array."""

    id: int
    status: str
    created_at: str


@dataclass(frozen=True, slots=True)
class LeadCallContext:
    """
    Minimal lead projection used to build the Mercury-2 customer data block.

    Only the fields the conversation driver needs to compose the
    ``<<CUSTOMER_DATA…>>`` block are projected; everything else (status,
    timestamps, etc.) is irrelevant to Riya's prompt.
    """

    name: str
    phone: str
    city: str
    language: str
    vehicle_id: int | None


@dataclass(frozen=True, slots=True)
class LeadWithHistory:
    """
    Fully hydrated lead row with its per-phone history attached.

    ``created_at`` is the raw SQLite space-separated UTC string — the API
    layer is responsible for applying ``to_iso_z`` before JSON serialisation.
    """

    id: int
    name: str
    phone: str
    email: str
    city: str
    language: str
    vehicle_id: int | None
    status: str
    failure_reason: str | None
    attempt_count: int
    created_at: str
    history: tuple[LeadHistoryEntry, ...]


# ─── Repository function ───────────────────────────────────────────────────────


async def fetch_leads(pool: ConnectionPool) -> tuple[LeadWithHistory, ...]:
    """
    Fetch all leads ordered newest-first, annotated with per-phone history.

    Each returned ``LeadWithHistory`` carries a ``history`` tuple listing every
    lead sharing the same phone number, ordered newest-first.  The entry for the
    lead itself is included in the history (so the frontend can render the full
    timeline without extra lookups).

    Parameters
    ----------
    pool : ConnectionPool
        Injected by the caller (never imported directly here).

    Returns
    -------
    tuple[LeadWithHistory, ...]
        All leads with history attached, ordered newest-first by ``created_at DESC, id DESC``.
        The ``id DESC`` tiebreaker ensures deterministic ordering when two leads share
        the same timestamp (SQLite's ``datetime('now')`` has second-level precision).
    """
    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            SELECT
                id,
                name,
                phone,
                email,
                city,
                language,
                vehicle_id,
                status,
                failure_reason,
                attempt_count,
                created_at
            FROM leads
            ORDER BY created_at DESC, id DESC
            """,
        )
        rows = await cursor.fetchall()

    if not rows:
        return ()

    # ── Parse raw rows ─────────────────────────────────────────────────────────
    raw: list[LeadRecord] = [
        LeadRecord(
            id=row[0],
            name=row[1],
            phone=row[2],
            email=row[3],
            city=row[4],
            language=row[5],
            vehicle_id=row[6],
            status=row[7],
            failure_reason=row[8],
            attempt_count=row[9],
            created_at=row[10],
        )
        for row in rows
    ]

    # ── Group by phone for history ────────────────────────────────────────────
    by_phone: dict[str, list[LeadRecord]] = defaultdict(list)
    for lead in raw:
        by_phone[lead.phone].append(lead)

    # ── Build history map (newest-first per phone) ───────────────────────────
    # raw is already ordered newest-first globally, so appending in iteration
    # order preserves that order within each phone group.
    history_map: dict[str, list[LeadHistoryEntry]] = {}
    for phone, phone_leads in by_phone.items():
        history_map[phone] = [
            LeadHistoryEntry(id=lead.id, status=lead.status, created_at=lead.created_at)
            for lead in phone_leads
        ]

    # ── Assemble final hydrated objects ──────────────────────────────────────
    return tuple(
        LeadWithHistory(
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
            created_at=lead.created_at,
            history=tuple(history_map[lead.phone]),
        )
        for lead in raw
    )


async def fetch_lead_for_call(
    pool: ConnectionPool,
    lead_id: int,
) -> LeadCallContext | None:
    """
    Fetch the minimal lead fields the conversation driver needs.

    Returns ``None`` if the lead does not exist (should not happen during
    a live call — the lead row is created and the supervisor is built from
    it in the arrange handler — but a defensive ``None`` keeps the driver
    safe against an edge case where the row was deleted mid-call).

    Parameters
    ----------
    pool : ConnectionPool
        Injected by the caller (never imported directly here).
    lead_id : int
        Primary key of the lead row to project.

    Returns
    -------
    LeadCallContext | None
        Projected fields, or ``None`` if no row matches.
    """
    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            SELECT name, phone, city, language, vehicle_id
            FROM leads
            WHERE id = ?
            """,
            (lead_id,),
        )
        row = await cursor.fetchone()

    if row is None:
        return None

    return LeadCallContext(
        name=row[0],
        phone=row[1],
        city=row[2],
        language=row[3],
        vehicle_id=row[4],
    )
