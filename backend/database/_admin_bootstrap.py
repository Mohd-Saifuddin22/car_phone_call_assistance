"""
backend/database/_admin_bootstrap.py — idempotent admin account seeding.

On first boot (empty `admin` table) this module hashes the admin password and
inserts exactly one row.  On subsequent boots it is a no-op.

IMPORTANT: No module calls load_settings() at import time.  All values are
passed in as parameters so that tests can inject fixtures freely.
"""

from __future__ import annotations

import logging
from pathlib import Path

import aiosqlite

from backend.database._auth import hash_password
from backend.database._pragma import enforce_pragmas

log = logging.getLogger(__name__)

_BOOTSTRAP_BUSY_TIMEOUT: float = 30.0  # seconds


async def bootstrap_admin(
    db_path: Path,
    admin_email: str,
    admin_password_plain: str,
    *,
    busy_timeout_seconds: float = _BOOTSTRAP_BUSY_TIMEOUT,
) -> bool:
    """
    Seed the `admin` table with one admin row if it is currently empty.

    Parameters
    ----------
    db_path                : path to app.db
    admin_email            : email address for the admin account
    admin_password_plain    : plaintext password (NEVER stored; only its hash is)
    busy_timeout_seconds   : PRAGMA busy_timeout applied to the bootstrap connection

    Returns
    -------
    True  — admin row was created (first boot)
    False — admin row already existed (no-op; existing row untouched)

    Security notes
    ──────────────
    The plaintext password is used only to derive the bcrypt hash, which is
    inserted into the DB.  The plaintext never appears in a log, a query
    parameter, or a returned value.
    """

    conn = await aiosqlite.connect(str(db_path), isolation_level=None)
    await enforce_pragmas(conn, busy_timeout_seconds)

    try:
        count = await _count_admins(conn)
        if count > 0:
            log.info(
                "bootstrap_admin: admin table has %d row(s) — skipping (no clobber).",
                count,
            )
            return False

        password_hash = hash_password(admin_password_plain)
        await conn.execute(
            "INSERT INTO admin (email, password_hash) VALUES (?, ?);",
            (admin_email, password_hash),
        )
        await conn.commit()
        log.info("bootstrap_admin: admin account seeded for %s.", admin_email)
        return True

    finally:
        await conn.close()


async def _count_admins(conn: aiosqlite.Connection) -> int:
    """Return the number of rows currently in the `admin` table."""
    cursor = await conn.execute("SELECT COUNT(*) FROM admin;")
    row = await cursor.fetchone()
    return row[0] if row else 0
