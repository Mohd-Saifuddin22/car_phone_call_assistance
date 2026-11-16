"""
backend/database/_pragma.py — PRAGMA enforcement helpers.

Every connection obtained from the pool MUST have these PRAGMAs applied:
  1. journal_mode = WAL          — allows concurrent readers + one writer
  2. busy_timeout  (ms)          — seconds → ms; how long to wait on lock
  3. foreign_keys  = ON          — enforces FK constraints (off by default!)

These functions are safe to call on every connection acquisition.  Setting
a PRAGMA that is already in the desired state is a no-op.
"""

from __future__ import annotations

import aiosqlite


async def enforce_pragmas(conn: aiosqlite.Connection, busy_timeout_seconds: float) -> None:
    """
    Apply all three required PRAGMAs on *conn*.

    Parameters
    ----------
    conn                : an open aiosqlite connection
    busy_timeout_seconds: SQLite busy_timeout in SECONDS (converted to ms internally)
    """
    busy_ms = int(busy_timeout_seconds * 1_000)
    await conn.execute("PRAGMA journal_mode = WAL;")
    await conn.execute(f"PRAGMA busy_timeout = {busy_ms};")
    await conn.execute("PRAGMA foreign_keys = ON;")
    # aiosqlite buffers reads until first iteration; force it to flush
    await conn.commit()


async def apply_pragma_once(conn: aiosqlite.Connection, pragma: str) -> None:
    """
    Execute a PRAGMA statement unconditionally.

    Safe to call multiple times on the same connection — re-applying the same
    value is always a no-op at the SQLite level.
    """
    await conn.execute(f"PRAGMA {pragma};")
    await conn.commit()
