"""
backend/database/__init__.py — public API for the database access layer.

Modules (private internals):
    _pool            — aiosqlite connection pool
    _pragma          — PRAGMA helpers
    _bootstrap       — schema bootstrap (create app.db from SQL file)
    _admin_bootstrap — idempotent admin account seeding from env vars
    _migrations      — PRAGMA user_version migration runner
    _txn             — transaction context manager
    _auth            — bcrypt password hashing and verification
    _sessions        — server-side session repository
    _leads_deletion  — hard-delete cascade for a lead (file-first-then-row policy)

Public interface (this module):
    create_pool          — build and return a ConnectionPool
    acquire / pool_close — borrow / return a connection
    transaction          — context manager: one logical change = one transaction
    bootstrap_db         — create app.db from schema SQL if missing / empty
    bootstrap_admin      — seed admin table with one row from env (idempotent)
    run_migrations       — advance user_version and apply pending migration scripts
    CURRENT_VERSION      — integer: latest applied migration version
    verify_password      — verify a plaintext password against a bcrypt hash
    create_session       — create a new server-side session
    get_session          — retrieve and validate a session by raw token
    refresh_session      — extend a session's expiry timestamp
    revoke_session       — revoke (delete) a session
    sweep_expired        — purge all expired session rows
    Session              — session dataclass
"""

from __future__ import annotations

from backend.database._admin_bootstrap import bootstrap_admin
from backend.database._auth import verify_password
from backend.database._bootstrap import bootstrap_db
from backend.database._leads import fetch_lead_for_call
from backend.database._leads_deletion import delete_lead_cascade
from backend.database._migrations import CURRENT_VERSION, run_migrations
from backend.database._pragma import apply_pragma_once, enforce_pragmas
from backend.database._pool import ConnectionPool, acquire, create_pool, pool_close
from backend.database._sessions import (
    Session,
    create_session,
    get_session,
    refresh_session,
    revoke_session,
    sweep_expired,
)
from backend.database._txn import transaction

__all__ = [
    "ConnectionPool",
    "CURRENT_VERSION",
    "Session",
    "acquire",
    "apply_pragma_once",
    "bootstrap_admin",
    "bootstrap_db",
    "create_pool",
    "create_session",
    "delete_lead_cascade",
    "enforce_pragmas",
    "fetch_lead_for_call",
    "get_session",
    "pool_close",
    "refresh_session",
    "revoke_session",
    "run_migrations",
    "sweep_expired",
    "transaction",
    "verify_password",
]
