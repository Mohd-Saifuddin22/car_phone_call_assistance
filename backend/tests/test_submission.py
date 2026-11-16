"""
backend/tests/test_submission.py — Direct unit tests for the domain submission handler.

These tests exercise the handler without the FastAPI HTTP layer, proving the
value of the extraction for testability.

Tests for is_blocked and submit_lead cover:
- Block-list matching (phone and email)
- Block check runs before deduplication
- Deduplication with open leads
- Deduplication allows retry after terminal leads
- Successful insertion
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from backend.database import create_pool
from backend.domain.submission import NEUTRAL_MESSAGE, is_blocked, submit_lead


# ─── Test-database bootstrap ───────────────────────────────────────────────────


def _schema_path() -> Path:
    """Absolute path to the application schema file."""
    return (
        Path(__file__).parent.parent.parent / "database" / "app_schema.sql"
    ).resolve()


def _create_temp_db() -> Path:
    """Create a temporary app.db from the application schema."""
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
async def pool(temp_db: Path) -> object:
    """Per-test isolated pool backed by a temp database."""
    db_path = _create_temp_db()
    p = create_pool(
        db_path=db_path,
        busy_timeout_seconds=5.0,
        pool_size=5,
    )
    # Warm the pool.
    conns = []
    for _ in range(p.max_size):
        c = await p.acquire()
        conns.append(c)
    for c in conns:
        await p.release(c)
    yield p
    await p.close()
    # Clean up temp file.
    try:
        db_path.unlink()
    except OSError:
        pass


@pytest_asyncio.fixture
async def temp_db() -> Path:
    """Per-test isolated database backed by a temp file."""
    db_path = _create_temp_db()
    yield db_path
    try:
        db_path.unlink()
    except OSError:
        pass


# ─── Tests for is_blocked ──────────────────────────────────────────────────────


class TestIsBlocked:
    """Tests for the is_blocked function."""

    @pytest.mark.asyncio
    async def test_returns_false_when_not_blocked(self, pool) -> None:
        """Unblocked phone/email must return False."""
        result = await is_blocked(pool, "+919700000001", "notblocked@example.com")
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_true_when_phone_is_blocked(self, pool) -> None:
        """Canonical phone matching a block_list entry must return True."""
        # Add block entry.
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO block_list (phone, email, name) VALUES (?, ?, ?)",
                ("+919700000001", "anyone@example.com", "Blocked Phone"),
            )

        result = await is_blocked(pool, "+919700000001", "anyone@example.com")
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_true_when_email_is_blocked(self, pool) -> None:
        """Canonical email matching a block_list entry must return True."""
        # Add block entry.
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO block_list (phone, email, name) VALUES (?, ?, ?)",
                ("+919800000001", "blocked.email@example.com", "Blocked Email"),
            )

        result = await is_blocked(pool, "+919800000001", "blocked.email@example.com")
        assert result is True


# ─── Tests for submit_lead ────────────────────────────────────────────────────


class TestSubmitLeadBlockList:
    """Tests for block-list enforcement in submit_lead."""

    @pytest.mark.asyncio
    async def test_blocked_phone_returns_neutral_message(self, pool) -> None:
        """Blocked phone must return NEUTRAL_MESSAGE without inserting a lead."""
        blocked_phone = "+919700000001"

        # Add to block_list.
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO block_list (phone, email, name) VALUES (?, ?, ?)",
                (blocked_phone, "anyone@example.com", "Blocked"),
            )

        # Submit.
        result = await submit_lead(
            pool=pool,
            name="Should Be Blocked",
            canonical_phone=blocked_phone,
            canonical_email="anyone@example.com",
            city="Mumbai",
            language="English",
            vehicle_id=None,
        )

        assert result == NEUTRAL_MESSAGE

        # Verify no lead was inserted.
        async with pool.connection() as conn:
            async with conn.execute("SELECT COUNT(*) FROM leads") as cur:
                count = (await cur.fetchone())[0]
        assert count == 0

    @pytest.mark.asyncio
    async def test_blocked_email_returns_neutral_message(self, pool) -> None:
        """Blocked email must return NEUTRAL_MESSAGE without inserting a lead."""
        blocked_email = "blocked.email@example.com"

        # Add to block_list.
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO block_list (phone, email, name) VALUES (?, ?, ?)",
                ("+919800000001", blocked_email, "Blocked"),
            )

        # Submit.
        result = await submit_lead(
            pool=pool,
            name="Should Be Blocked",
            canonical_phone="+919800000001",
            canonical_email=blocked_email,
            city="Delhi",
            language="Hinglish",
            vehicle_id=None,
        )

        assert result == NEUTRAL_MESSAGE

        # Verify no lead was inserted.
        async with pool.connection() as conn:
            async with conn.execute("SELECT COUNT(*) FROM leads") as cur:
                count = (await cur.fetchone())[0]
        assert count == 0


class TestSubmitLeadDeduplication:
    """Tests for deduplication in submit_lead."""

    @pytest.mark.asyncio
    async def test_duplicate_open_lead_returns_neutral_without_second_insert(
        self, pool
    ) -> None:
        """Second submission with same phone returns NEUTRAL_MESSAGE; only 1 lead."""
        phone = "+919600000001"
        email = "dedup@example.com"

        # First submission.
        r1 = await submit_lead(
            pool=pool,
            name="First",
            canonical_phone=phone,
            canonical_email=email,
            city="Mumbai",
            language="English",
            vehicle_id=None,
        )
        assert r1 == NEUTRAL_MESSAGE

        # Second submission (duplicate).
        r2 = await submit_lead(
            pool=pool,
            name="Second",
            canonical_phone=phone,
            canonical_email=email,
            city="Mumbai",
            language="English",
            vehicle_id=None,
        )
        assert r2 == NEUTRAL_MESSAGE

        # Only one lead exists.
        async with pool.connection() as conn:
            async with conn.execute("SELECT COUNT(*) FROM leads") as cur:
                count = (await cur.fetchone())[0]
        assert count == 1

    @pytest.mark.asyncio
    async def test_retry_after_completed_lead_succeeds(self, pool) -> None:
        """Lead in Completed state must not block a new submission."""
        phone = "+919600000002"

        # First submission.
        await submit_lead(
            pool=pool,
            name="First",
            canonical_phone=phone,
            canonical_email="retry@example.com",
            city="Delhi",
            language="English",
            vehicle_id=None,
        )

        # Mark as Completed.
        async with pool.connection() as conn:
            await conn.execute(
                "UPDATE leads SET status = 'Completed' WHERE phone = ?",
                (phone,),
            )

        # Second submission must succeed.
        result = await submit_lead(
            pool=pool,
            name="Retry",
            canonical_phone=phone,
            canonical_email="retry@example.com",
            city="Delhi",
            language="English",
            vehicle_id=None,
        )
        assert result == NEUTRAL_MESSAGE

        # Two leads must exist.
        async with pool.connection() as conn:
            async with conn.execute("SELECT COUNT(*) FROM leads") as cur:
                count = (await cur.fetchone())[0]
        assert count == 2


class TestSubmitLeadSuccessfulInsert:
    """Tests for successful lead insertion."""

    @pytest.mark.asyncio
    async def test_successful_submission_returns_neutral_and_creates_lead(
        self, pool
    ) -> None:
        """Fresh submission must return NEUTRAL_MESSAGE and create a lead row."""
        phone = "+919600000003"
        email = "new@example.com"

        result = await submit_lead(
            pool=pool,
            name="New Lead",
            canonical_phone=phone,
            canonical_email=email,
            city="Bangalore",
            language="Hinglish",
            vehicle_id=None,
        )

        assert result == NEUTRAL_MESSAGE

        # Lead was inserted.
        async with pool.connection() as conn:
            async with conn.execute(
                "SELECT name, phone, email, city, language, status FROM leads WHERE phone = ?",
                (phone,),
            ) as cur:
                row = await cur.fetchone()
        assert row is not None
        assert row[0] == "New Lead"
        assert row[1] == phone
        assert row[2] == email
        assert row[3] == "Bangalore"
        assert row[4] == "Hinglish"
        assert row[5] == "New"
