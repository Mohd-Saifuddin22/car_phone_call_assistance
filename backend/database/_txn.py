"""
backend/database/_txn.py — transaction context manager.

Usage
─────
    async with transaction(conn) as c:
        await c.execute("INSERT INTO leads ...")
        await c.execute("UPDATE leads ...")
    # ← commits on normal exit, rolls back if any exception propagates

The context manager issues ``BEGIN`` immediately on entry and either
``COMMIT`` or ``ROLLBACK`` on exit, depending on whether an exception was
raised inside the block.

No module calls ``load_settings()`` at import time.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import aiosqlite

log = logging.getLogger(__name__)


@asynccontextmanager
async def transaction(conn: aiosqlite.Connection) -> AsyncIterator[aiosqlite.Connection]:
    """
    All-or-nothing transaction wrapper for a single aiosqlite connection.

    Parameters
    ----------
    conn : an open aiosqlite connection

    Yields
    ------
    The same connection, ready for statements.

    Behaviour
    ---------
    * No exception  → ``COMMIT`` is issued and the connection is returned to
                     its prior autocommit state.
    * Any exception propagates out after ``ROLLBACK`` is issued.
    * The caller is responsible for the connection's lifecycle (open/close).

    Notes
    -----
    * ``BEGIN`` is issued explicitly so that the transaction boundary is always
      visible in the WAL log, making debugging easier.
    * Connections opened with ``isolation_level=None`` (as ours always are)
      start in autocommit mode; this context manager manages that state.
    """
    await conn.execute("BEGIN;")
    try:
        yield conn
    except BaseException:
        await conn.execute("ROLLBACK;")
        log.debug("Transaction rolled back due to exception.")
        raise
    else:
        await conn.execute("COMMIT;")
        log.debug("Transaction committed.")
