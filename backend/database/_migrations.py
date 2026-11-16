"""
backend/database/_migrations.py — lightweight PRAGMA user_version migration runner.

Design
──────
Migrations live in ``database/migrations/`` as numbered SQL files::

    database/migrations/001_phonetic_index.sql
    database/migrations/002_played_ms_column.sql
    ...

Each file contains **one or more** SQL statements that are safe to re-run
(typically ``ALTER TABLE ADD COLUMN`` or ``CREATE INDEX IF NOT EXISTS``).
The runner determines the current database version via
``PRAGMA user_version`` and applies every migration whose number is higher,
in ascending order, advancing the counter after each successful migration.

The ``CURRENT_VERSION`` constant reflects the highest migration number that
has been validated as a complete, shipped migration.  Update it whenever a
new migration file is added.

Idempotency
───────────
Every migration file must be written to tolerate re-execution (e.g. use
``IF NOT EXISTS`` for new columns/indexes, or ``CREATE OR REPLACE`` where
appropriate).  The runner itself is also idempotent: re-running it when
the database is already at ``CURRENT_VERSION`` is a no-op.

No module calls ``load_settings()`` at import time.  Paths are passed in
as parameters so tests can inject temp directories freely.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import aiosqlite

from backend.database._pragma import enforce_pragmas

log = logging.getLogger(__name__)

# ─── Version counter ─────────────────────────────────────────────────────────
# Bump this constant by 1 for every new migration file you add.
# The migration files must be named ``<3-digit-number>_<slug>.sql`` starting
# from 001 so the runner can discover them via filesystem sorting.
CURRENT_VERSION: int = 0

# Default busy timeout used when opening a connection for migration work.
_MIGRATION_BUSY_TIMEOUT: float = 60.0  # seconds


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

async def run_migrations(
    db_path: Path,
    migrations_dir: Path,
    *,
    busy_timeout_seconds: float = _MIGRATION_BUSY_TIMEOUT,
) -> None:
    """
    Bring the database at *db_path* up to ``CURRENT_VERSION``.

    Parameters
    ----------
    db_path         : path to app.db
    migrations_dir  : directory containing numbered ``<N>_<name>.sql`` files
    busy_timeout_seconds : PRAGMA busy_timeout for migration connections

    Raises
    ------
    FileNotFoundError — migrations_dir does not exist
    RuntimeError      — a migration script fails; the error is logged and
                        re-raised after the transaction is rolled back.
    """
    conn = await aiosqlite.connect(str(db_path), isolation_level=None)
    await enforce_pragmas(conn, busy_timeout_seconds)

    try:
        current = await _get_user_version(conn)
        log.info("run_migrations: db at version %d, target %d", current, CURRENT_VERSION)

        pending = _discover_pending(current, migrations_dir)
        if not pending:
            log.info("run_migrations: database is up-to-date")
            return

        for version, script_path in pending:
            log.info("run_migrations: applying migration %03d from %s", version, script_path.name)
            await _apply_migration(conn, version, script_path)

        log.info("run_migrations: all pending migrations applied (now at %d)", CURRENT_VERSION)

    finally:
        await conn.close()


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------

_MIGRATION_RE = re.compile(r"^(\d{3})_[a-z_]+\.sql$")


def _discover_pending(
    current: int,
    migrations_dir: Path,
) -> list[tuple[int, Path]]:
    """
    Return sorted list of (version, path) for migration files with a version
    number strictly greater than *current*.
    """
    pending: list[tuple[int, Path]] = []

    if not migrations_dir.is_dir():
        log.warning("Migrations directory %s does not exist — skipping", migrations_dir)
        return pending

    for entry in migrations_dir.iterdir():
        if not entry.is_file() or entry.suffix != ".sql":
            continue
        m = _MIGRATION_RE.match(entry.name)
        if m is None:
            log.debug("Skipping non-migration file: %s", entry.name)
            continue
        version = int(m.group(1))
        if version > current:
            pending.append((version, entry))

    pending.sort(key=lambda x: x[0])
    return pending


async def _get_user_version(conn: aiosqlite.Connection) -> int:
    cursor = await conn.execute("PRAGMA user_version;")
    row = await cursor.fetchone()
    return row[0] if row else 0


async def _apply_migration(
    conn: aiosqlite.Connection,
    version: int,
    script_path: Path,
) -> None:
    """
    Execute *script_path* in a transaction, then advance user_version to *version*.

    Each SQL statement in the script is executed individually via conn.execute()
    (NOT executescript, which auto-commits before running — breaking the
    BEGIN/ROLLBACK wrapper).  If any statement raises, the entire migration is
    rolled back and the counter is NOT advanced.

    Raises
    ------
    RuntimeError — the script could not be read or a statement failed
    """
    try:
        raw_sql = script_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Cannot read migration {script_path.name}: {exc}") from exc

    statements = _split_sql_statements(raw_sql)
    if not statements:
        log.warning("Migration %03d (%s) contained no executable statements.", version, script_path.name)
        return

    # Begin explicit transaction — all subsequent execute() calls are part of it.
    await conn.execute("BEGIN;")
    try:
        for stmt in statements:
            await conn.execute(stmt)
        await conn.execute(f"PRAGMA user_version = {version};")
        await conn.commit()
        log.info("Migration %03d applied and committed.", version)
    except Exception:
        await conn.execute("ROLLBACK;")
        log.error("Migration %03d failed — rolled back.", version)
        raise


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

# Regex to split a SQL script on semicolons, yielding individual statements.
# Strips whitespace and SQL-style comments from each fragment.
_SEMICOLON_RE = re.compile(r";")


def _split_sql_statements(raw_sql: str) -> list[str]:
    """
    Split *raw_sql* on semicolons and return only non-empty, non-comment
    statement strings in their original order.

    This is equivalent to what sqlite3.executescript() does internally,
    but we need to do it manually so that all statements stay within the
    same BEGIN…COMMIT/ROLLBACK transaction.
    """
    results: list[str] = []
    for fragment in _SEMICOLON_RE.split(raw_sql):
        # Strip line comments (-- … newline) and leading/trailing whitespace
        stripped = _STRIP_COMMENTS_RE.sub("", fragment).strip()
        if stripped:
            results.append(stripped)
    return results


# Pre-compile once at module load (matches "-- ..." up to end-of-line)
_STRIP_COMMENTS_RE = re.compile(r"--[^\n]*", re.MULTILINE)
