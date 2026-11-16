"""
backend/database/_pool.py — aiosqlite connection pool.

Design
──────
SQLite connections are not thread-safe and aiosqlite connections are not
safe to share across coroutines.  The pool implements a **per-operation**
borrow pattern: callers `await acquire(pool)` to get a connection, use it,
then `await pool_close(pool)` to return it.

The pool maintains a fixed-size queue of pre-opened, PRAGMA-enforced
connections.  When the queue is exhausted (all connections are checked out),
a new one is opened on demand.  Excess connections above the configured
pool size are closed when returned.

All connections are created with WAL journal mode, the configured busy_timeout,
and foreign_keys=ON — enforced by _pragma.enforce_pragmas() at open time.

CRITICAL: No module calls load_settings() at import time.  All config values
(db_path, busy_timeout_seconds, pool_size) are accepted as parameters.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

import aiosqlite

from backend.database._pragma import enforce_pragmas

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------

BusyTimeoutSeconds = float
PoolSize = int


@dataclass
class ConnectionPool:
    """
    Fixed-size async connection pool for aiosqlite.

    Attributes
    ----------
    db_path              : path to the SQLite file (used to open connections)
    busy_timeout_seconds : passed to PRAGMA busy_timeout on every connection
    max_size             : pool size (number of concurrent borrowers)
    _semaphore           : gates how many connections can be held simultaneously
    _connections        : asyncio.Queue of available (open) connections
    _opened              : total connections ever opened (for debug / metrics)
    _closed              : True once close() has been called
    """

    db_path: Path
    busy_timeout_seconds: BusyTimeoutSeconds
    max_size: PoolSize = 5

    _semaphore: asyncio.Semaphore = field(init=False, repr=False)
    _connections: asyncio.Queue[aiosqlite.Connection] = field(
        init=False, repr=False
    )
    _opened: int = field(default=0, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_semaphore", asyncio.Semaphore(self.max_size))
        object.__setattr__(
            self, "_connections", asyncio.Queue(maxsize=self.max_size)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[aiosqlite.Connection]:
        """
        Borrow a PRAGMA-enforced connection for the duration of the block.

        Usage
        -----
            pool = ConnectionPool(db_path, busy_timeout_seconds=5.0, max_size=5)
            async with pool.connection() as conn:
                async with conn.execute("SELECT 1") as cur:
                    ...

        The connection is always returned to the pool; it is never closed.
        """
        conn = await self.acquire()
        try:
            yield conn
        finally:
            await self.release(conn)

    async def acquire(self) -> aiosqlite.Connection:
        """
        Acquire a PRAGMA-enforced connection from the pool.

        Creates a new connection on demand if the pool is not yet at capacity.
        Raises RuntimeError if the pool has been closed.
        """
        if self._closed:
            raise RuntimeError("ConnectionPool has been closed")

        await self._semaphore.acquire()

        # Try to grab an already-open connection from the queue.
        while not self._connections.empty():
            conn = self._connections.get_nowait()
            # Silently skip connections that were closed by a prior error.
            try:
                # ping — any simple query that validates the conn is alive
                await conn.execute("SELECT 1")
                return conn
            except Exception as exc:  # pragma: no cover (defensive)
                log.warning("Discarding stale connection (%s). Re-opening.", exc)
                await self._open_one()  # replace it

        # Pool not yet at capacity — open a fresh connection.
        conn = await self._open_one()
        return conn

    async def release(self, conn: aiosqlite.Connection) -> None:
        """
        Return *conn* to the pool.

        If the pool has been closed, the connection is closed immediately.
        """
        if self._closed:
            await conn.close()
            self._semaphore.release()
            return

        try:
            # Verify the connection is still healthy before returning it.
            await conn.execute("SELECT 1")
            self._connections.put_nowait(conn)
        except Exception as exc:  # pragma: no cover (defensive)
            log.warning("Connection error on return (%s). Closing.", exc)
            await conn.close()
            self._opened -= 1

        self._semaphore.release()

    async def close(self) -> None:
        """
        Gracefully shut down the pool: drain the queue and close every connection.

        Safe to call multiple times (subsequent calls are no-ops).
        """
        if self._closed:
            return
        object.__setattr__(self, "_closed", True)

        # Drain and close all queued connections.
        while not self._connections.empty():
            try:
                conn = self._connections.get_nowait()
                await conn.close()
            except Exception as exc:  # pragma: no cover
                log.warning("Error closing pooled connection: %s", exc)

        log.info("ConnectionPool closed (opened=%d)", self._opened)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _open_one(self) -> aiosqlite.Connection:
        """Open a single aiosqlite connection and apply required PRAGMAs."""
        conn = await aiosqlite.connect(str(self.db_path), isolation_level=None)
        await enforce_pragmas(conn, self.busy_timeout_seconds)
        self._opened += 1
        log.debug("Opened connection %d (pool: %s)", self._opened, self.db_path)
        return conn


# ----------------------------------------------------------------------
# Module-level factory (convenience wrapper; accepts raw config values)
# ----------------------------------------------------------------------
# NOTE: this function is NOT a singleton — callers create their own pool.
# It is a convenience factory so that the typical call-site:
#     pool = create_pool(db_path=settings.app_db_path,
#                       busy_timeout_seconds=settings.busy_timeout_seconds,
#                       pool_size=settings.db_pool_size)
# does not have to know about ConnectionPool internals.
# ----------------------------------------------------------------------


def create_pool(
    db_path: Path,
    busy_timeout_seconds: BusyTimeoutSeconds,
    pool_size: PoolSize,
) -> ConnectionPool:
    """Factory: build a ConnectionPool with the given config values."""
    return ConnectionPool(
        db_path=db_path,
        busy_timeout_seconds=busy_timeout_seconds,
        max_size=pool_size,
    )


# ----------------------------------------------------------------------
# Standalone acquire / pool_close helpers (for non-context-manager use)
# ----------------------------------------------------------------------
# Usage: conn = await acquire(pool); ...; await pool_close(pool, conn)
# Prefer pool.connection() context manager when possible.
# ----------------------------------------------------------------------

async def acquire(pool: ConnectionPool) -> aiosqlite.Connection:
    """Alias for pool.acquire()."""
    return await pool.acquire()


async def pool_close(pool: ConnectionPool, conn: aiosqlite.Connection) -> None:
    """Alias for pool.release()."""
    await pool.release(conn)
