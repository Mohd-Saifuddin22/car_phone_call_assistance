"""
backend/database/_bootstrap.py — idempotent schema bootstrap.

On first boot (or when app.db is missing), this module creates the database
by executing the DDL in `database/app_schema.sql`.

Idempotency is achieved by checking whether any tables already exist before
running the DDL script.  If tables are present, the bootstrap is a no-op.
This means the bootstrap is safe to call on every startup.

IMPORTANT
─────────
This module intentionally does NOT call `load_settings()` at import time.
All paths are passed in as parameters so that tests can inject in-memory
databases or temporary files freely.
"""

from __future__ import annotations

import logging
from pathlib import Path

import aiosqlite

from backend.database._pragma import enforce_pragmas

log = logging.getLogger(__name__)

# WAL and foreign_keys are enforced on the bootstrap connection, just like
# any other connection, so the schema initialisation itself benefits from
# the same correctness guarantees.
_BOOTSTRAP_BUSY_TIMEOUT: float = 30.0  # seconds; bootstrap connections get extra patience


async def bootstrap_db(
    db_path: Path,
    schema_path: Path,
    *,
    busy_timeout_seconds: float = _BOOTSTRAP_BUSY_TIMEOUT,
) -> bool:
    """
    Create `db_path` from `schema_path` DDL if the database is missing or empty.

    Parameters
    ----------
    db_path         : destination SQLite file (created with parents if needed)
    schema_path     : path to the DDL SQL file (e.g. database/app_schema.sql)
    busy_timeout_seconds : PRAGMA busy_timeout applied to the bootstrap connection

    Returns
    -------
    True  — database was created (first boot)
    False — database already existed with tables (no-op)

    Raises
    ------
    FileNotFoundError — schema_path does not exist
    RuntimeError      — schema_path is unreadable
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Open connection and enforce PRAGMAs ────────────────────────────
    conn = await aiosqlite.connect(str(db_path), isolation_level=None)
    await enforce_pragmas(conn, busy_timeout_seconds)

    try:
        # ── 2. Idempotency check: are tables already present? ───────────────
        already_seeded = await _has_tables(conn)
        if already_seeded:
            log.info("bootstrap_db: %s already has tables — skipping", db_path)
            return False

        # ── 3. Read the DDL script ─────────────────────────────────────────
        try:
            raw_sql = schema_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(
                f"Cannot read schema file {schema_path!s}: {exc}"
            ) from exc

        # ── 4. Execute DDL ─────────────────────────────────────────────────
        log.info("bootstrap_db: initialising %s from %s", db_path, schema_path)
        await conn.executescript(raw_sql)
        await conn.commit()
        log.info("bootstrap_db: schema applied successfully")
        return True

    finally:
        await conn.close()


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

async def _has_tables(conn: aiosqlite.Connection) -> bool:
    """
    Return True if any tables have been created in the attached database.

    We check `sqlite_master` directly (no PRAGMA call needed) rather than
    introspecting sqlite_tables, which avoids a round-trip through
    information_schema lookups.
    """
    cursor = await conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%';"
    )
    row = await cursor.fetchone()
    return row is not None and row[0] > 0
