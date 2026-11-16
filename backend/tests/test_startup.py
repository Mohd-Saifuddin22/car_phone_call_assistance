"""
backend/tests/test_startup.py — Verification tests for the startup reconciliation sweep.

Issue 029 — Startup Reconciliation Sweep
──────────────────────────────────────
Acceptance criteria:
1. A startup sweep function exists that successfully flips all Processing leads
   to Failed (dropped) in one transaction.
2. Leads in terminal states (New, Completed, Failed) are completely untouched.
3. The test seeds a mix of terminal leads and zombie Processing leads, runs the
   sweep, and asserts the final post-boot state is correct.

No test calls ``load_settings()`` at import time.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from typing import AsyncGenerator

import aiosqlite
import pytest
import pytest_asyncio

from backend.database import create_pool
from backend.startup import reconcile_startup

# ─── Test-database bootstrap ───────────────────────────────────────────────────


def _schema_path() -> Path:
    """Absolute path to the application schema file."""
    return (
        Path(__file__).parent.parent.parent / "database" / "app_schema.sql"
    ).resolve()


def _create_temp_db() -> Path:
    """
    Create a temporary app.db from the application schema.

    Uses ``PRAGMA foreign_keys = ON`` so the schema's self-referential
    constraints are enforced immediately.

    Uses ``NamedTemporaryFile`` instead of ``mktemp`` to guarantee a unique path
    even when multiple tests call this simultaneously.
    """
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".db", prefix="car_test_"
    ) as fh:
        tmp = Path(fh.name)
    conn = sqlite3.connect(str(tmp))
    conn.execute("PRAGMA foreign_keys = ON;")
    with open(_schema_path(), encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.close()
    return tmp


# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    """Fresh settings cache for every test — no bleed between test runs."""
    from backend.config import load_settings

    load_settings.cache_clear()
    yield
    load_settings.cache_clear()


@pytest_asyncio.fixture
async def temp_db() -> AsyncGenerator[Path, None]:
    """Per-test isolated database backed by a temp file."""
    db_path = _create_temp_db()
    yield db_path
    try:
        db_path.unlink()
    except OSError:
        pass  # already gone


@pytest_asyncio.fixture
async def pool(temp_db: Path):
    """
    A pre-warmed connection pool pointing at the isolated temp DB.

    Connections are acquired and released sequentially so that
    ``PRAGMA journal_mode = WAL`` runs serially and is cached on the file,
    preventing "database is locked" errors in concurrent tests.
    """
    pool = create_pool(
        db_path=temp_db,
        busy_timeout_seconds=5.0,
        pool_size=5,
    )
    _warm: list = []
    for _ in range(pool.max_size):
        c = await pool.acquire()
        _warm.append(c)
    for c in _warm:
        await pool.release(c)
    yield pool
    await pool.close()


@pytest_asyncio.fixture
async def read_leads(temp_db: Path):
    """
    Async fixture that reads all leads from the test database.

    Uses aiosqlite directly so that any writes from the sweep are
    checkpointed and visible.
    """
    async def _read() -> list[dict]:
        conn = await aiosqlite.connect(str(temp_db))
        conn.row_factory = sqlite3.Row
        try:
            async with conn.execute(
                "SELECT id, name, phone, status, failure_reason, attempt_count, updated_at FROM leads ORDER BY id"
            ) as cur:
                rows = await cur.fetchall()
                return [dict(r) for r in rows]
        finally:
            await conn.close()

    return _read


@pytest_asyncio.fixture
async def read_attempts(temp_db: Path):
    """
    Async fixture that reads all call_attempts from the test database.

    Uses aiosqlite directly so that any writes from the sweep are
    checkpointed and visible.
    """
    async def _read() -> list[dict]:
        conn = await aiosqlite.connect(str(temp_db))
        conn.row_factory = sqlite3.Row
        try:
            async with conn.execute(
                "SELECT id, lead_id, attempt_number, twilio_call_sid, outcome, started_at, ended_at, recording_path FROM call_attempts ORDER BY id"
            ) as cur:
                rows = await cur.fetchall()
                return [dict(r) for r in rows]
        finally:
            await conn.close()

    return _read


# ─── Helpers ───────────────────────────────────────────────────────────────────


async def _seed_leads(conn: aiosqlite.Connection, rows: list[dict]) -> None:
    """
    Seed the leads table with the given rows.

    Each dict must have keys: name, phone, email, city, language, status,
    and optionally failure_reason and attempt_count.
    """
    for row in rows:
        await conn.execute(
            """
            INSERT INTO leads
                (name, phone, email, city, language, status, failure_reason, attempt_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                row["name"],
                row["phone"],
                row["email"],
                row["city"],
                row["language"],
                row["status"],
                row.get("failure_reason"),
                row.get("attempt_count", 0),
            ),
        )
    await conn.commit()


async def _seed_attempts(conn: aiosqlite.Connection, rows: list[dict]) -> None:
    """
    Seed the call_attempts table with the given rows.

    Each dict must have keys: lead_id, attempt_number, and optionally
    twilio_call_sid, outcome, started_at, ended_at, recording_path.
    outcome IS NULL marks an in-flight (orphaned) attempt.
    """
    for row in rows:
        await conn.execute(
            """
            INSERT INTO call_attempts
                (lead_id, attempt_number, twilio_call_sid, outcome, started_at, ended_at, recording_path)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (
                row["lead_id"],
                row["attempt_number"],
                row.get("twilio_call_sid"),
                row.get("outcome"),   # NULL = in-flight
                row.get("started_at", "2025-01-01 00:00:00"),
                row.get("ended_at"),  # NULL = in-flight
                row.get("recording_path"),
            ),
        )
    await conn.commit()


# ─── Tests ─────────────────────────────────────────────────────────────────────


class TestReconcileStartupSweep:
    """
    Tests for the startup reconciliation sweep.

    The sweep must:
    • Flip all Processing leads → Failed (dropped) in a single transaction.
    • Leave terminal-state leads (New, Completed, Failed) completely untouched.
    """

    @pytest.mark.asyncio
    async def test_processing_leads_transitioned_to_failed_dropped(
        self,
        pool,
        read_leads,
    ) -> None:
        """
        Leads in the Processing state must be flipped to Failed with reason 'dropped'.
        """
        async with pool.connection() as conn:
            await _seed_leads(conn, [
                {
                    "name": "Zombie One",
                    "phone": "+919500000001",
                    "email": "z1@example.com",
                    "city": "Mumbai",
                    "language": "English",
                    "status": "Processing",
                    "attempt_count": 1,
                },
                {
                    "name": "Zombie Two",
                    "phone": "+919500000002",
                    "email": "z2@example.com",
                    "city": "Delhi",
                    "language": "Hinglish",
                    "status": "Processing",
                    "attempt_count": 2,
                },
            ])

        # Run the sweep.
        updated = await reconcile_startup(pool)
        assert updated == 2, f"Expected 2 leads updated, got {updated}"

        # Verify final state.
        leads = await read_leads()
        assert len(leads) == 2
        for lead in leads:
            assert lead["status"] == "Failed", (
                f"Lead {lead['id']} should be Failed, got {lead['status']}"
            )
            assert lead["failure_reason"] == "dropped", (
                f"Lead {lead['id']} failure_reason should be 'dropped', got {lead['failure_reason']!r}"
            )
            assert lead["updated_at"] is not None, (
                f"Lead {lead['id']} updated_at must be bumped by the sweep"
            )

    @pytest.mark.asyncio
    async def test_new_leads_untouched(
        self,
        pool,
        read_leads,
    ) -> None:
        """
        Leads in the New state must not be modified by the sweep.
        """
        async with pool.connection() as conn:
            await _seed_leads(conn, [
                {
                    "name": "Fresh Lead",
                    "phone": "+919510000001",
                    "email": "new@example.com",
                    "city": "Bangalore",
                    "language": "English",
                    "status": "New",
                    "attempt_count": 0,
                },
            ])

        updated = await reconcile_startup(pool)
        assert updated == 0, "New leads should not be touched"

        leads = await read_leads()
        assert len(leads) == 1
        assert leads[0]["status"] == "New"
        assert leads[0]["failure_reason"] is None

    @pytest.mark.asyncio
    async def test_completed_leads_untouched(
        self,
        pool,
        read_leads,
    ) -> None:
        """
        Leads in the Completed state must not be modified by the sweep.
        """
        async with pool.connection() as conn:
            await _seed_leads(conn, [
                {
                    "name": "Done Lead",
                    "phone": "+919520000001",
                    "email": "done@example.com",
                    "city": "Chennai",
                    "language": "Hinglish",
                    "status": "Completed",
                    "attempt_count": 1,
                },
            ])

        updated = await reconcile_startup(pool)
        assert updated == 0, "Completed leads should not be touched"

        leads = await read_leads()
        assert len(leads) == 1
        assert leads[0]["status"] == "Completed"
        assert leads[0]["failure_reason"] is None

    @pytest.mark.asyncio
    async def test_failed_leads_untouched(
        self,
        pool,
        read_leads,
    ) -> None:
        """
        Leads already in a Failed state must not be modified by the sweep.
        """
        async with pool.connection() as conn:
            await _seed_leads(conn, [
                {
                    "name": "Old Failed",
                    "phone": "+919530000001",
                    "email": "failed@example.com",
                    "city": "Hyderabad",
                    "language": "English",
                    "status": "Failed",
                    "failure_reason": "no_answer",
                    "attempt_count": 3,
                },
            ])

        updated = await reconcile_startup(pool)
        assert updated == 0, "Failed leads should not be touched"

        leads = await read_leads()
        assert len(leads) == 1
        assert leads[0]["status"] == "Failed"
        assert leads[0]["failure_reason"] == "no_answer", (
            "Existing failure_reason must not be overwritten"
        )

    @pytest.mark.asyncio
    async def test_mixed_leads_correct_final_state(
        self,
        pool,
        read_leads,
    ) -> None:
        """
        Seed a mix of terminal leads and zombie Processing leads, run the sweep,
        and assert the final post-boot state is correct.
        """
        async with pool.connection() as conn:
            await _seed_leads(conn, [
                {
                    "name": "Zombie Alice",
                    "phone": "+919540000001",
                    "email": "alice@example.com",
                    "city": "Mumbai",
                    "language": "English",
                    "status": "Processing",
                    "attempt_count": 1,
                },
                {
                    "name": "New Bob",
                    "phone": "+919540000002",
                    "email": "bob@example.com",
                    "city": "Delhi",
                    "language": "Hinglish",
                    "status": "New",
                    "attempt_count": 0,
                },
                {
                    "name": "Zombie Carol",
                    "phone": "+919540000003",
                    "email": "carol@example.com",
                    "city": "Bangalore",
                    "language": "English",
                    "status": "Processing",
                    "attempt_count": 2,
                },
                {
                    "name": "Completed Dan",
                    "phone": "+919540000004",
                    "email": "dan@example.com",
                    "city": "Chennai",
                    "language": "English",
                    "status": "Completed",
                    "attempt_count": 1,
                },
                {
                    "name": "Failed Eve",
                    "phone": "+919540000005",
                    "email": "eve@example.com",
                    "city": "Hyderabad",
                    "language": "Hinglish",
                    "status": "Failed",
                    "failure_reason": "error",
                    "attempt_count": 3,
                },
            ])

        # Run the sweep.
        updated = await reconcile_startup(pool)
        assert updated == 2, f"Expected 2 Processing leads updated, got {updated}"

        # Verify final state.
        leads = await read_leads()
        assert len(leads) == 5

        by_phone: dict[str, dict] = {l["phone"]: l for l in leads}

        # Zombie leads → Failed (dropped)
        for phone in ("+919540000001", "+919540000003"):
            lead = by_phone[phone]
            assert lead["status"] == "Failed", (
                f"{phone}: expected Failed, got {lead['status']}"
            )
            assert lead["failure_reason"] == "dropped", (
                f"{phone}: expected failure_reason='dropped', got {lead['failure_reason']!r}"
            )

        # Terminal leads untouched
        assert by_phone["+919540000002"]["status"] == "New"
        assert by_phone["+919540000002"]["failure_reason"] is None

        assert by_phone["+919540000004"]["status"] == "Completed"
        assert by_phone["+919540000004"]["failure_reason"] is None

        assert by_phone["+919540000005"]["status"] == "Failed"
        assert by_phone["+919540000005"]["failure_reason"] == "error", (
            "Existing failure_reason must not be overwritten"
        )

    @pytest.mark.asyncio
    async def test_sweep_is_idempotent(
        self,
        pool,
        read_leads,
    ) -> None:
        """
        Calling the sweep twice is safe.  The second call finds zero Processing
        leads and returns 0, leaving the database unchanged.
        """
        async with pool.connection() as conn:
            await _seed_leads(conn, [
                {
                    "name": "Zombie",
                    "phone": "+919550000001",
                    "email": "z@example.com",
                    "city": "Mumbai",
                    "language": "English",
                    "status": "Processing",
                    "attempt_count": 1,
                },
            ])

        first = await reconcile_startup(pool)
        assert first == 1

        leads_before = await read_leads()

        second = await reconcile_startup(pool)
        assert second == 0, "Second sweep should find no Processing leads"

        leads_after = await read_leads()
        assert leads_before == leads_after, (
            "Second sweep must not modify any state"
        )

    @pytest.mark.asyncio
    async def test_sweep_with_no_leads(
        self,
        pool,
        read_leads,
    ) -> None:
        """
        Running the sweep on an empty database returns 0 and leaves nothing behind.
        """
        updated = await reconcile_startup(pool)
        assert updated == 0

        leads = await read_leads()
        assert leads == []

    @pytest.mark.asyncio
    async def test_orphaned_call_attempts_resolved(
        self,
        pool,
        read_leads,
        read_attempts,
    ) -> None:
        """
        A Processing lead always has an in-flight call_attempts row (outcome IS NULL)
        from the crash.  The sweep must resolve that orphaned attempt (set
        outcome='dropped', ended_at=NOW) before flipping the lead.

        A prior terminal attempt (outcome='completed') belonging to the same lead
        must be left untouched.
        """
        # Step 1 — seed one Processing lead.
        async with pool.connection() as conn:
            await _seed_leads(conn, [
                {
                    "name": "Zombie Victor",
                    "phone": "+919560000001",
                    "email": "victor@example.com",
                    "city": "Mumbai",
                    "language": "English",
                    "status": "Processing",
                    "attempt_count": 2,
                },
            ])

        # Retrieve the lead's ID so we can reference it when seeding attempts.
        async with pool.connection() as conn:
            async with conn.execute(
                "SELECT id FROM leads WHERE phone = '+919560000001';"
            ) as cur:
                lead_row = await cur.fetchone()
            lead_id: int = lead_row[0]

        # Step 2 — seed two call_attempts for that lead:
        #   • attempt 1: terminal (outcome='completed') — must stay unchanged
        #   • attempt 2: in-flight/orphaned (outcome IS NULL, ended_at IS NULL)
        async with pool.connection() as conn:
            await _seed_attempts(conn, [
                {
                    "lead_id": lead_id,
                    "attempt_number": 1,
                    "twilio_call_sid": "CA_prior_terminal",
                    "outcome": "completed",
                    "started_at": "2025-01-01 10:00:00",
                    "ended_at": "2025-01-01 10:05:00",
                },
                {
                    "lead_id": lead_id,
                    "attempt_number": 2,
                    "twilio_call_sid": "CA_zombie_live",
                    # outcome IS NULL  → in-flight / orphaned
                    # ended_at IS NULL → still running
                },
            ])

        # Run the sweep.
        updated = await reconcile_startup(pool)
        assert updated == 1

        # Verify lead state.
        leads = await read_leads()
        assert len(leads) == 1
        assert leads[0]["status"] == "Failed"
        assert leads[0]["failure_reason"] == "dropped"
        assert leads[0]["updated_at"] is not None, (
            "updated_at must be bumped by the sweep"
        )

        # Verify call_attempts state.
        attempts = await read_attempts()
        assert len(attempts) == 2

        by_number: dict[int, dict] = {a["attempt_number"]: a for a in attempts}

        # Prior terminal attempt — must be completely untouched.
        prior = by_number[1]
        assert prior["outcome"] == "completed", (
            "Prior terminal attempt must not be modified"
        )
        assert prior["ended_at"] == "2025-01-01 10:05:00", (
            "Prior attempt ended_at must not be overwritten"
        )

        # Orphaned in-flight attempt — must be resolved.
        zombie = by_number[2]
        assert zombie["outcome"] == "dropped", (
            f"Orphaned attempt outcome must be set to 'dropped', got {zombie['outcome']!r}"
        )
        assert zombie["ended_at"] is not None, (
            "Orphaned attempt ended_at must be set by the sweep"
        )
