"""
backend/database/_sessions.py — server-side session repository.

Design
──────
Each session row stores:
  • SHA-256 hash of the session token (never the raw token).
  • A per-session CSRF synchronizer token.
  • A rolling `expires_at` timestamp (default: 7 days from creation, refreshed on
    each authenticated request).
  • A reference to the `admin` row it belongs to.

All timestamps are stored and compared using SQLite-native ``datetime('now')``
format: ``YYYY-MM-DD HH:MM:SS`` (space separator, no microseconds, no 'T').

Security invariants
──────────────────
  • Session tokens are **never** stored in plaintext — only their SHA-256 digest.
  • ``get_session()`` and ``refresh_session()`` **reject expired sessions at read
    time**, even if the row exists in the database.
  • ``create_session()`` generates cryptographically-random 32-byte tokens using
    ``secrets.token_bytes()`` and returns the raw token to the caller (who is
    responsible for setting it as a HttpOnly cookie).
  • ``bytes.fromhex()`` is guarded: malformed tokens return None/False instead of
    raising ValueError, preventing 500 errors on bad cookies.

IMPORTANT: No module calls ``load_settings()`` at import time.  All values are
passed in as parameters so that tests can inject fixtures freely.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from backend.database._pragma import enforce_pragmas

if TYPE_CHECKING:
    from backend.database._pool import ConnectionPool

# ─── Constants ────────────────────────────────────────────────────────────────

# How many bytes of entropy the session token should have.
_TOKEN_BYTES: int = 32

# Default rolling expiry window in days (used when the caller doesn't supply one).
_DEFAULT_EXPIRY_DAYS: int = 7


# ─── Public types ──────────────────────────────────────────────────────────────


@dataclass
class Session:
    """
    Represents an authenticated server-side session.

    Attributes
    ----------
    id                 : database primary key
    session_token_hash : SHA-256 digest of the raw token
    admin_id           : which admin row owns this session
    csrf_token         : per-session synchronizer CSRF token
    created_at         : creation timestamp in ``YYYY-MM-DD HH:MM:SS`` format
    expires_at         : expiry timestamp in ``YYYY-MM-DD HH:MM:SS`` format
    """

    id: int
    session_token_hash: str
    admin_id: int
    csrf_token: str
    created_at: str
    expires_at: str


# ─── Public API ──────────────────────────────────────────────────────────────


async def create_session(
    pool: "ConnectionPool",
    admin_id: int,
    *,
    csrf_token: str | None = None,
    expiry_days: int = _DEFAULT_EXPIRY_DAYS,
) -> tuple[Session, str]:
    """
    Create a new session for *admin_id* and return the Session object + raw token.

    Parameters
    ----------
    pool        : a live ConnectionPool (acquired via ``get_pool`` dependency)
    admin_id    : id of the admin row this session belongs to
    csrf_token  : optional per-session CSRF token; generated if not supplied
    expiry_days : rolling expiry window in days (default 7)

    Returns
    -------
    (Session, raw_token) — the Session object and the raw 32-byte token.
                           The caller MUST set the raw token as an HttpOnly cookie;
                           it is never stored in the database.

    Security notes
    ──────────────
    The raw token is generated with ``secrets.token_bytes(32)`` (cryptographically
    secure).  Only the SHA-256 hash of the token is stored.
    """
    raw_token = secrets.token_bytes(_TOKEN_BYTES)
    token_hash = hashlib.sha256(raw_token).hexdigest()
    csrf = csrf_token or secrets.token_hex(16)

    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            INSERT INTO sessions (session_token_hash, admin_id, csrf_token, expires_at)
            VALUES (
                ?,
                ?,
                ?,
                datetime('now', '+' || ? || ' days')
            );
            """,
            (token_hash, admin_id, csrf, expiry_days),
        )
        await conn.commit()
        session_id = cursor.lastrowid
        assert session_id is not None

        # Read back the DB-computed timestamps
        select = await conn.execute(
            "SELECT created_at, expires_at FROM sessions WHERE id = ?;",
            (session_id,),
        )
        row = await select.fetchone()

        session = Session(
            id=session_id,
            session_token_hash=token_hash,
            admin_id=admin_id,
            csrf_token=csrf,
            created_at=row[0],
            expires_at=row[1],
        )
        return session, raw_token.hex()


async def get_session(
    pool: "ConnectionPool",
    raw_token: str,
) -> Session | None:
    """
    Retrieve and validate a session by its raw token.

    Parameters
    ----------
    pool      : a live ConnectionPool
    raw_token : the raw session token (hex string) as presented by the client

    Returns
    -------
    Session | None — the session if it exists, its token hash matches, and it has
                     not yet expired.  Returns None for any failure (bad token,
                     wrong hash, expired, missing row).  Never raises on bad input.

    Security notes
    ──────────────
    Expired sessions are rejected at read time — the row may exist in the database
    but will not be returned.
    """
    try:
        token_bytes = bytes.fromhex(raw_token)
    except ValueError:
        return None  # malformed hex → clean rejection, not a 500

    token_hash = hashlib.sha256(token_bytes).hexdigest()

    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            SELECT id, session_token_hash, admin_id, csrf_token, created_at, expires_at
            FROM sessions
            WHERE session_token_hash = ? AND expires_at > datetime('now');
            """,
            (token_hash,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None

        return Session(
            id=row[0],
            session_token_hash=row[1],
            admin_id=row[2],
            csrf_token=row[3],
            created_at=row[4],
            expires_at=row[5],
        )


async def refresh_session(
    pool: "ConnectionPool",
    raw_token: str,
    *,
    expiry_days: int = _DEFAULT_EXPIRY_DAYS,
) -> Session | None:
    """
    Refresh an active session's expiry timestamp and return the updated Session.

    Parameters
    ----------
    pool        : a live ConnectionPool
    raw_token   : the raw session token (hex string)
    expiry_days : rolling expiry window in days (default 7)

    Returns
    -------
    Updated Session | None — the session with a new ``expires_at`` if it was
                              valid and not yet expired.  None if the session
                              does not exist, the token hash doesn't match, or
                              it has already expired.  Also returns None for
                              malformed token input.

    Security notes
    ──────────────
    An already-expired session is NOT refreshed; it is rejected outright (None).
    """
    try:
        token_bytes = bytes.fromhex(raw_token)
    except ValueError:
        return None  # malformed hex → clean rejection

    token_hash = hashlib.sha256(token_bytes).hexdigest()

    async with pool.connection() as conn:
        # First check the session is still valid (not expired)
        cursor = await conn.execute(
            """
            SELECT id, session_token_hash, admin_id, csrf_token, created_at, expires_at
            FROM sessions
            WHERE session_token_hash = ? AND expires_at > datetime('now');
            """,
            (token_hash,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None

        # Compute the new expiry in Python so we can return it
        new_expires_at = _future_utc(expiry_days)

        # Refresh the expiry in the DB
        await conn.execute(
            "UPDATE sessions SET expires_at = ? WHERE session_token_hash = ?;",
            (new_expires_at, token_hash),
        )
        await conn.commit()

        return Session(
            id=row[0],
            session_token_hash=row[1],
            admin_id=row[2],
            csrf_token=row[3],
            created_at=row[4],
            expires_at=new_expires_at,
        )


async def revoke_session(
    pool: "ConnectionPool",
    raw_token: str,
) -> bool:
    """
    Revoke (delete) a session by its raw token.

    Parameters
    ----------
    pool      : a live ConnectionPool
    raw_token : the raw session token (hex string)

    Returns
    -------
    True if a session was deleted, False if no matching session was found or
    the token was malformed.
    """
    try:
        token_bytes = bytes.fromhex(raw_token)
    except ValueError:
        return False  # malformed hex → clean rejection

    token_hash = hashlib.sha256(token_bytes).hexdigest()

    async with pool.connection() as conn:
        cursor = await conn.execute(
            "DELETE FROM sessions WHERE session_token_hash = ?;",
            (token_hash,),
        )
        await conn.commit()
        return cursor.rowcount > 0


async def sweep_expired(pool: "ConnectionPool") -> int:
    """
    Delete all rows from the sessions table whose ``expires_at`` is in the past.

    Parameters
    ----------
    pool : a live ConnectionPool

    Returns
    -------
    int — the number of rows deleted.
    """
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "DELETE FROM sessions WHERE expires_at <= datetime('now');",
        )
        await conn.commit()
        return cursor.rowcount


# ─── Internal helpers ────────────────────────────────────────────────────────


def _now_utc() -> str:
    """
    Return the current UTC timestamp in SQLite-native ``YYYY-MM-DD HH:MM:SS`` format.
    Used when a Python-side timestamp is needed (e.g. for return values that
    match what the DB would store).
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _future_utc(days: int) -> str:
    """
    Return a UTC timestamp *days* from now in SQLite-native
    ``YYYY-MM-DD HH:MM:SS`` format.
    """
    return (datetime.now(timezone.utc) + timedelta(days=days)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
