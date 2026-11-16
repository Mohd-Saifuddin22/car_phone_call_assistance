"""
backend/tests/test_db_access.py — verification tests for the database access layer.

Tests
─────
1. PRAGMAs  — every connection reads back journal_mode=WAL, busy_timeout (≈configured ms),
              foreign_keys=ON.
2. Bootstrap — app.db is created from schema if missing; idempotent if already present.
3. Transaction helper — commits on normal exit, rolls back on exception.
4. Concurrency       — concurrent readers + writers complete without "database is locked"
                       errors, proving the pool and busy_timeout are correctly wired.

No test calls load_settings() at import time.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

import pytest

# ── bare aiosqlite (used for raw introspection, not the public API) ────────────
import aiosqlite

import backend.database as db
from backend.database import (
    ConnectionPool,
    acquire,
    bootstrap_db,
    create_pool,
    enforce_pragmas,
    pool_close,
    run_migrations,
    transaction,
)

log = logging.getLogger(__name__)

# ── Shared helpers ─────────────────────────────────────────────────────────────


async def _pragma_value(conn: aiosqlite.Connection, key: str) -> str | int:
    """Return the current value of a PRAGMA as a string."""
    cursor = await conn.execute(f"PRAGMA {key};")
    row = await cursor.fetchone()
    return row[0] if row else ""


async def _pragma_int(conn: aiosqlite.Connection, key: str) -> int:
    val = await _pragma_value(conn, key)
    return int(val)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Return a temp SQLite path that is guaranteed fresh (no file on disk)."""
    return tmp_path / "test_app.db"


@pytest.fixture
def schema_path() -> Path:
    """Path to the actual app_schema.sql (relative to project root)."""
    return Path(__file__).parent.parent.parent / "database" / "app_schema.sql"


@pytest.fixture
async def pool(tmp_db: Path) -> ConnectionPool:
    """A pool pointing at a fresh temp database."""
    p = create_pool(
        db_path=tmp_db,
        busy_timeout_seconds=5.0,
        pool_size=5,
    )
    yield p
    await p.close()


@pytest.fixture
async def bootstrapped_pool(
    tmp_db: Path, schema_path: Path
) -> ConnectionPool:
    """A pool whose database has been bootstrapped from the schema."""
    created = await bootstrap_db(tmp_db, schema_path)
    assert created, "bootstrap should create a fresh DB"
    p = create_pool(db_path=tmp_db, busy_timeout_seconds=5.0, pool_size=5)
    yield p
    await p.close()


# ==============================================================================
# Test 1 — PRAGMAs
# ==============================================================================

class TestPragmas:
    """Verify all three required PRAGMAs are enforced on every connection."""

    @staticmethod
    async def test_wal_mode_on_fresh_connection(tmp_db: Path) -> None:
        conn = await aiosqlite.connect(str(tmp_db), isolation_level=None)
        await enforce_pragmas(conn, busy_timeout_seconds=5.0)
        try:
            mode = await _pragma_value(conn, "journal_mode")
            assert mode.lower() == "wal", f"Expected WAL, got {mode}"
        finally:
            await conn.close()

    @staticmethod
    async def test_wal_mode_on_pooled_connection(bootstrapped_pool: ConnectionPool) -> None:
        async with bootstrapped_pool.connection() as conn:
            mode = await _pragma_value(conn, "journal_mode")
            assert mode.lower() == "wal", f"Expected WAL, got {mode}"

    @staticmethod
    async def test_busy_timeout_on_pooled_connection(
        bootstrapped_pool: ConnectionPool,
    ) -> None:
        async with bootstrapped_pool.connection() as conn:
            ms = await _pragma_int(conn, "busy_timeout")
            # 5 seconds → 5000 ms; allow ±100ms tolerance for rounding
            assert 4_900 <= ms <= 5_100, f"Expected ~5000 ms, got {ms}"

    @staticmethod
    async def test_foreign_keys_on_pooled_connection(
        bootstrapped_pool: ConnectionPool,
    ) -> None:
        async with bootstrapped_pool.connection() as conn:
            fk = await _pragma_int(conn, "foreign_keys")
            assert fk == 1, f"Expected foreign_keys=ON (1), got {fk}"

    @staticmethod
    async def test_all_three_pragmas_together(
        bootstrapped_pool: ConnectionPool,
    ) -> None:
        """Comprehensive check: all three PRAGMAs on the same connection."""
        async with bootstrapped_pool.connection() as conn:
            mode = (await _pragma_value(conn, "journal_mode")).lower()
            ms = await _pragma_int(conn, "busy_timeout")
            fk = await _pragma_int(conn, "foreign_keys")
        assert mode == "wal", f"journal_mode = {mode}"
        assert 4_900 <= ms <= 5_100, f"busy_timeout = {ms}"
        assert fk == 1, f"foreign_keys = {fk}"

    @staticmethod
    async def test_pragma_reapplication_is_idempotent(tmp_db: Path) -> None:
        """Calling enforce_pragmas twice on the same connection must not error."""
        conn = await aiosqlite.connect(str(tmp_db), isolation_level=None)
        await enforce_pragmas(conn, busy_timeout_seconds=5.0)
        await enforce_pragmas(conn, busy_timeout_seconds=5.0)  # must not raise
        mode = (await _pragma_value(conn, "journal_mode")).lower()
        await conn.close()
        assert mode == "wal"


# ==============================================================================
# Test 2 — Bootstrap (idempotent)
# ==============================================================================

class TestBootstrap:
    """Verify schema creation is idempotent."""

    @staticmethod
    async def test_bootstrap_creates_db(tmp_db: Path, schema_path: Path) -> None:
        assert not tmp_db.exists()
        created = await bootstrap_db(tmp_db, schema_path)
        assert created
        assert tmp_db.exists()

    @staticmethod
    async def test_bootstrap_creates_tables(tmp_db: Path, schema_path: Path) -> None:
        await bootstrap_db(tmp_db, schema_path)
        conn = await aiosqlite.connect(str(tmp_db))
        try:
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
            )
            rows = await cursor.fetchall()
            names = [r[0] for r in rows]
            assert "leads" in names
            assert "call_attempts" in names
            assert "transcript_turns" in names
            assert "recommendations" in names
            assert "sessions" in names
            assert "block_list" in names
            assert "admin" in names
        finally:
            await conn.close()

    @staticmethod
    async def test_bootstrap_is_idempotent(tmp_db: Path, schema_path: Path) -> None:
        """Second bootstrap must return False and not corrupt existing data."""
        await bootstrap_db(tmp_db, schema_path)
        # Insert a row
        conn = await aiosqlite.connect(str(tmp_db), isolation_level=None)
        await conn.execute(
            "INSERT INTO admin (email, password_hash) VALUES ('test@test.com', 'x');"
        )
        await conn.commit()
        await conn.close()

        # Bootstrap again — must be no-op
        again = await bootstrap_db(tmp_db, schema_path)
        assert again is False

        # Data must still be there
        conn2 = await aiosqlite.connect(str(tmp_db))
        try:
            cursor = await conn2.execute("SELECT email FROM admin;")
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == "test@test.com"
        finally:
            await conn2.close()

    @staticmethod
    async def test_bootstrap_enforces_wal(tmp_db: Path, schema_path: Path) -> None:
        """Bootstrap connection must also set WAL pragma."""
        await bootstrap_db(tmp_db, schema_path)
        conn = await aiosqlite.connect(str(tmp_db))
        try:
            mode = (await _pragma_value(conn, "journal_mode")).lower()
            assert mode == "wal"
        finally:
            await conn.close()


# ==============================================================================
# Test 3 — Transaction helper
# ==============================================================================

class TestTransactionHelper:
    """Verify transaction() commits on success and rolls back on exception."""

    @staticmethod
    async def test_commits_on_normal_exit(bootstrapped_pool: ConnectionPool) -> None:
        async with bootstrapped_pool.connection() as conn:
            async with transaction(conn):
                await conn.execute(
                    "INSERT INTO admin (email, password_hash) VALUES (?, ?);",
                    ("txn_commit@test.com", "hash"),
                )
        # Verify committed
        conn2 = await aiosqlite.connect(str(bootstrapped_pool.db_path))
        try:
            cursor = await conn2.execute(
                "SELECT email FROM admin WHERE email = 'txn_commit@test.com';"
            )
            row = await cursor.fetchone()
            assert row is not None, "Row should have been committed"
        finally:
            await conn2.close()

    @staticmethod
    async def test_rolls_back_on_exception(bootstrapped_pool: ConnectionPool) -> None:
        async with bootstrapped_pool.connection() as conn:
            with pytest.raises(RuntimeError, match="boom"):
                async with transaction(conn):
                    await conn.execute(
                        "INSERT INTO admin (email, password_hash) VALUES (?, ?);",
                        ("txn_rollback@test.com", "hash"),
                    )
                    raise RuntimeError("boom")

        # Verify rolled back
        conn2 = await aiosqlite.connect(str(bootstrapped_pool.db_path))
        try:
            cursor = await conn2.execute(
                "SELECT email FROM admin WHERE email = 'txn_rollback@test.com';"
            )
            row = await cursor.fetchone()
            assert row is None, "Row should have been rolled back"
        finally:
            await conn2.close()

    @staticmethod
    async def test_multiple_statements_in_one_transaction(
        bootstrapped_pool: ConnectionPool,
    ) -> None:
        """All statements in a transaction succeed or none do."""
        async with bootstrapped_pool.connection() as conn:
            async with transaction(conn):
                await conn.execute(
                    "INSERT INTO admin (email, password_hash) VALUES (?, ?);",
                    ("multi1@test.com", "h1"),
                )
                await conn.execute(
                    "INSERT INTO admin (email, password_hash) VALUES (?, ?);",
                    ("multi2@test.com", "h2"),
                )
        conn2 = await aiosqlite.connect(str(bootstrapped_pool.db_path))
        try:
            cursor = await conn2.execute(
                "SELECT COUNT(*) FROM admin WHERE email LIKE 'multi%@test.com';"
            )
            row = await cursor.fetchone()
            assert row is not None and row[0] == 2
        finally:
            await conn2.close()


# ==============================================================================
# Test 4 — Concurrency (WAL + pool)
# ==============================================================================

class TestConcurrency:
    """Prove concurrent reads + writes succeed under WAL without 'database locked'."""

    @staticmethod
    @pytest.mark.asyncio
    async def test_concurrent_reads(tmp_db: Path, schema_path: Path) -> None:
        """
        Launch N concurrent readers on the same DB.  All must complete without error.
        """
        await bootstrap_db(tmp_db, schema_path)
        pool = create_pool(db_path=tmp_db, busy_timeout_seconds=5.0, pool_size=10)
        n_readers = 20

        try:
            errors: list[BaseException] = []

            async def reader(idx: int) -> None:
                try:
                    async with pool.connection() as conn:
                        cursor = await conn.execute("SELECT COUNT(*) FROM leads;")
                        await cursor.fetchone()
                        await asyncio.sleep(0.01)  # simulate a bit of work
                        cursor2 = await conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table';"
                        )
                        await cursor2.fetchall()
                except Exception as exc:  # pragma: no cover
                    errors.append(exc)

            await asyncio.gather(*(reader(i) for i in range(n_readers)))
            assert not errors, f"Reader errors: {errors}"
        finally:
            await pool.close()

    @staticmethod
    @pytest.mark.asyncio
    async def test_concurrent_readers_and_writers(tmp_db: Path, schema_path: Path) -> None:
        """
        Simultaneous readers and writers — the WAL + busy_timeout combination
        must prevent "database is locked" errors.
        """
        await bootstrap_db(tmp_db, schema_path)
        pool = create_pool(db_path=tmp_db, busy_timeout_seconds=5.0, pool_size=10)

        n_readers = 10
        n_writers = 5
        errors: list[BaseException] = []

        async def reader(idx: int) -> None:
            try:
                for _ in range(5):
                    async with pool.connection() as conn:
                        cursor = await conn.execute("SELECT COUNT(*) FROM admin;")
                        await cursor.fetchone()
                        await asyncio.sleep(0.005)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        async def writer(idx: int) -> None:
            try:
                for j in range(5):
                    async with pool.connection() as conn:
                        async with transaction(conn):
                            await conn.execute(
                                "INSERT INTO admin (email, password_hash) VALUES (?, ?);",
                                (f"conc_writer_{idx}_{j}@test.com", "hash"),
                            )
                        await asyncio.sleep(0.005)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        try:
            tasks = [
                *(reader(i) for i in range(n_readers)),
                *(writer(i) for i in range(n_writers)),
            ]
            await asyncio.gather(*tasks)
            assert not errors, f"Concurrency errors: {errors}"
        finally:
            await pool.close()

    @staticmethod
    @pytest.mark.asyncio
    async def test_concurrent_writers_different_rows(tmp_db: Path, schema_path: Path) -> None:
        """
        Multiple concurrent writers inserting different rows — each transaction
        is independent and must all succeed under WAL + busy_timeout.
        """
        await bootstrap_db(tmp_db, schema_path)
        pool = create_pool(db_path=tmp_db, busy_timeout_seconds=5.0, pool_size=5)
        n_writers = 8

        errors: list[BaseException] = []

        async def writer(idx: int) -> None:
            try:
                async with pool.connection() as conn:
                    async with transaction(conn):
                        await conn.execute(
                            "INSERT INTO leads (name, phone, email, city, language) "
                            "VALUES (?, ?, ?, ?, ?);",
                            (f"Caller {idx}", f"+91999900000{idx}", f"c{idx}@test.com", "Mumbai", "English"),
                        )
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        try:
            await asyncio.gather(*(writer(i) for i in range(n_writers)))
            assert not errors, f"Writer errors: {errors}"
        finally:
            await pool.close()

        # Verify all rows landed
        conn = await aiosqlite.connect(str(tmp_db))
        try:
            cursor = await conn.execute("SELECT COUNT(*) FROM leads;")
            row = await cursor.fetchone()
            assert row is not None and row[0] == n_writers, f"Expected {n_writers} rows, got {row[0]}"
        finally:
            await conn.close()


# ==============================================================================
# Test 5 — Migration runner
# ==============================================================================

class TestMigrationRunner:
    """Verify the user_version-based migration runner works end-to-end."""

    @staticmethod
    @pytest.mark.asyncio
    async def test_run_migrations_noop_when_current(
        tmp_db: Path,
        schema_path: Path,
        tmp_path: Path,
    ) -> None:
        """When db is already at CURRENT_VERSION, run_migrations() is a no-op."""
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        # Write a dummy migration that would fail if executed
        (migrations_dir / "001_dummy.sql").write_text("SELECT 1/0;")

        await bootstrap_db(tmp_db, schema_path)
        # Manually set user_version to CURRENT_VERSION so nothing is pending
        conn = await aiosqlite.connect(str(tmp_db), isolation_level=None)
        await conn.execute(f"PRAGMA user_version = {db.CURRENT_VERSION};")
        await conn.close()

        # Should not raise despite the dummy migration having bad SQL
        await run_migrations(tmp_db, migrations_dir)

    @staticmethod
    @pytest.mark.asyncio
    async def test_run_migrations_advances_counter(
        tmp_db: Path,
        schema_path: Path,
        tmp_path: Path,
    ) -> None:
        """A real migration script should advance user_version."""
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        # Write a migration that creates a marker table
        (migrations_dir / "001_marker.sql").write_text(
            "CREATE TABLE IF NOT EXISTS migration_marker (id INTEGER PRIMARY KEY);"
        )

        await bootstrap_db(tmp_db, schema_path)

        # Confirm marker table doesn't exist yet (version 0 doesn't have it)
        conn0 = await aiosqlite.connect(str(tmp_db))
        try:
            cursor = await conn0.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='migration_marker';"
            )
            row = await cursor.fetchone()
            assert row is None
        finally:
            await conn0.close()

        # Patch CURRENT_VERSION to 1 and run migrations
        import backend.database._migrations as _m
        original_version = _m.CURRENT_VERSION
        try:
            _m.CURRENT_VERSION = 1
            await run_migrations(tmp_db, migrations_dir)
        finally:
            _m.CURRENT_VERSION = original_version

        # Now the marker table should exist
        conn1 = await aiosqlite.connect(str(tmp_db))
        try:
            cursor = await conn1.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='migration_marker';"
            )
            row = await cursor.fetchone()
            assert row is not None, "Migration should have created the marker table"
        finally:
            await conn1.close()

    @staticmethod
    @pytest.mark.asyncio
    async def test_run_migrations_sequential(
        tmp_db: Path,
        schema_path: Path,
        tmp_path: Path,
    ) -> None:
        """Multiple migrations in order — version counter must advance correctly."""
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()

        (migrations_dir / "001_first.sql").write_text(
            "CREATE TABLE IF NOT EXISTS seq_001 (v TEXT);"
        )
        (migrations_dir / "002_second.sql").write_text(
            "CREATE TABLE IF NOT EXISTS seq_002 (v TEXT);"
        )

        await bootstrap_db(tmp_db, schema_path)

        import backend.database._migrations as _m
        orig = _m.CURRENT_VERSION
        try:
            _m.CURRENT_VERSION = 2
            await run_migrations(tmp_db, migrations_dir)
        finally:
            _m.CURRENT_VERSION = orig

        conn = await aiosqlite.connect(str(tmp_db))
        for tbl in ("seq_001", "seq_002"):
            cursor = await conn.execute(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name='{tbl}';"
            )
            row = await cursor.fetchone()
            assert row is not None, f"Table {tbl} should exist"
        await conn.close()

    @staticmethod
    @pytest.mark.asyncio
    async def test_run_migrations_rolls_back_on_midway_failure(
        tmp_db: Path,
        schema_path: Path,
        tmp_path: Path,
    ) -> None:
        """
        A migration whose second statement fails must roll back the entire script —
        proving that statement-by-statement execute() (not executescript) keeps
        changes inside the BEGIN/ROLLBACK block.
        """
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        # First statement succeeds, second is invalid SQL → whole migration rolls back.
        (migrations_dir / "001_fail_midway.sql").write_text(
            "CREATE TABLE IF NOT EXISTS rollback_ok (id INTEGER PRIMARY KEY);\n"
            "INVALID SQL SYNTAX HERE;"
        )

        await bootstrap_db(tmp_db, schema_path)
        # Manually set user_version to 0 so migration 001 is pending.
        conn = await aiosqlite.connect(str(tmp_db), isolation_level=None)
        await conn.execute("PRAGMA user_version = 0;")
        await conn.close()

        import backend.database._migrations as _m

        # Patch CURRENT_VERSION and verify the migration raises.
        orig = _m.CURRENT_VERSION
        with pytest.raises(Exception):
            _m.CURRENT_VERSION = 1
            await run_migrations(tmp_db, migrations_dir)
        _m.CURRENT_VERSION = orig

        # The successful first statement must NOT be present — full rollback.
        conn2 = await aiosqlite.connect(str(tmp_db))
        try:
            cursor = await conn2.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='rollback_ok';"
            )
            row = await cursor.fetchone()
            assert row is None, (
                "Failed migration should have rolled back — rollback_ok should not exist"
            )
        finally:
            await conn2.close()


# ==============================================================================
# Test 6 — Schema correctness (PRD reconciliation)
# ==============================================================================

class TestSchemaReconciliation:
    """Verify the schema DDL contains the PRD-specified changes."""

    @staticmethod
    async def test_transcript_turns_has_played_ms_and_total_ms(
        tmp_db: Path, schema_path: Path
    ) -> None:
        await bootstrap_db(tmp_db, schema_path)
        conn = await aiosqlite.connect(str(tmp_db))
        try:
            cursor = await conn.execute("PRAGMA table_info(transcript_turns);")
            # PRAGMA table_info returns: (cid, name, type, notnull, dflt_value, pk)
            rows = {row[1]: row for row in await cursor.fetchall()}
            played_row = rows.get("played_ms")
            total_row = rows.get("total_ms")
            assert played_row is not None, "transcript_turns.played_ms missing"
            assert total_row is not None, "transcript_turns.total_ms missing"
            # col 2 = type, col 3 = notnull (0 = nullable, 1 = NOT NULL)
            assert played_row[2] == "INTEGER", f"played_ms type should be INTEGER, got {played_row[2]}"
            assert total_row[2] == "INTEGER", f"total_ms type should be INTEGER, got {total_row[2]}"
            assert played_row[3] == 0, f"played_ms must be nullable (notnull=0), got {played_row[3]}"
            assert total_row[3] == 0, f"total_ms must be nullable (notnull=0), got {total_row[3]}"
        finally:
            await conn.close()

    @staticmethod
    async def test_recommendations_blurb_text_nullable(
        tmp_db: Path, schema_path: Path
    ) -> None:
        await bootstrap_db(tmp_db, schema_path)
        conn = await aiosqlite.connect(str(tmp_db))
        try:
            cursor = await conn.execute("PRAGMA table_info(recommendations);")
            columns = {row[1]: row for row in await cursor.fetchall()}
            blurb_row = columns.get("blurb_text")
            assert blurb_row is not None
            # notnull column (pydantic row[3]): 0 = nullable, 1 = NOT NULL
            assert blurb_row[3] == 0, "blurb_text should be nullable (notnull=0)"
        finally:
            await conn.close()

    @staticmethod
    async def test_leads_partial_unique_index(
        tmp_db: Path, schema_path: Path
    ) -> None:
        """The partial UNIQUE index on leads(phone) WHERE status IN ('New','Processing')
        must be present and must enforce uniqueness for active leads only."""
        await bootstrap_db(tmp_db, schema_path)
        conn = await aiosqlite.connect(str(tmp_db), isolation_level=None)
        try:
            cursor = await conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL;"
            )
            indexes = {row[0] for row in await cursor.fetchall()}
        finally:
            await conn.close()

        # At least one index should contain the WHERE clause for phone/status
        partial = [s for s in indexes if "phone" in s.lower() and "where" in s.lower()]
        assert partial, f"No partial phone index found. Indexes: {indexes}"

        # Verify it is UNIQUE
        assert any("unique" in s.lower() for s in partial), (
            f"Partial index is not UNIQUE: {partial}"
        )
