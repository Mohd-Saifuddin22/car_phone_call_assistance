"""
backend/database/_blocks.py — Block-list repository (Issue 019).

Provides repository functions for the block/unblock operations:
  - fetch_blocked: list all blocked customers
  - block_customer: insert a block-list entry (idempotent via UNIQUE constraint)
  - unblock_customer: remove a block-list entry by exact phone number

Design
──────
The ``block_list`` table already has a UNIQUE constraint on ``phone``, so
re-blocking the same phone is naturally idempotent (SQLite returns an
IntegrityError which we catch and swallow).  No upsert logic needed.

The repository returns domain dataclasses, never ORM model instances.

Pool injection
──────────────
All functions accept the DB pool as a parameter.  No module calls
``load_settings()`` at import time.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.database._pool import ConnectionPool


# ─── Data classes ─────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class BlockedCustomer:
    """
    A row from the ``block_list`` table.

    ``created_at`` is the raw SQLite UTC string — call sites are responsible
    for formatting it before serialising to JSON.
    """

    id: int
    phone: str       # E.164 canonical form
    email: str       # lowercased / trimmed
    name: str | None  # snapshot at block time
    created_at: str   # raw SQLite UTC string


# ─── Repository functions ─────────────────────────────────────────────────────


async def fetch_blocked(pool: ConnectionPool) -> tuple[BlockedCustomer, ...]:
    """
    Fetch all blocked customers ordered newest-first.

    Parameters
    ----------
    pool : ConnectionPool
        Injected by the caller.

    Returns
    -------
    tuple[BlockedCustomer, ...]
        All blocked customers, ordered by ``created_at DESC``.
    """
    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            SELECT id, phone, email, name, created_at
            FROM block_list
            ORDER BY created_at DESC
            """,
        )
        rows = await cursor.fetchall()

    return tuple(
        BlockedCustomer(
            id=row[0],
            phone=row[1],
            email=row[2],
            name=row[3],
            created_at=row[4],
        )
        for row in rows
    )


async def block_customer(
    pool: ConnectionPool,
    canonical_phone: str,
    canonical_email: str,
    name_snapshot: str | None,
) -> None:
    """
    Insert a block-list entry for the given canonical phone and email.

    Idempotent: if the phone is already blocked, the existing row is kept
    (UNIQUE constraint on ``phone`` silently prevents a duplicate insert).
    The ``name`` and ``email`` fields are NOT updated on re-block — the
    original snapshot is preserved.

    Parameters
    ----------
    pool             : ConnectionPool
    canonical_phone   : str — E.164 form
    canonical_email   : str — lowercased / trimmed
    name_snapshot     : str | None — name snapshot at block time
    """
    async with pool.connection() as conn:
        await conn.execute(
            """
            INSERT OR IGNORE INTO block_list
                (phone, email, name)
            VALUES (?, ?, ?)
            """,
            (canonical_phone, canonical_email, name_snapshot),
        )


async def unblock_customer(pool: ConnectionPool, canonical_phone: str) -> bool:
    """
    Remove a block-list entry by exact canonical phone.

    Parameters
    ----------
    pool           : ConnectionPool
    canonical_phone : str — E.164 form (exact match required)

    Returns
    -------
    True  — a row was deleted.
    False — no row matched (phone not blocked).
    """
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "DELETE FROM block_list WHERE phone = ?",
            (canonical_phone,),
        )
        return cursor.rowcount > 0
