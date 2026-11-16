"""
backend/tests/test_sessions.py — hermetic unit tests for the sessions repository.

All tests use a bootstrapped temporary SQLite database (no I/O to app.db).
The fixture pattern mirrors test_admin_bootstrap.py for consistency.

Coverage
────────
  1. create_session — token hash is SHA-256, raw token is 64 hex chars, CSRF is set.
  2. get_session  — happy path, wrong token, expired session (DB row exists but read returns None).
  3. refresh_session — extends expires_at, rejects already-expired sessions.
  4. revoke_session — deletes the row, returns True; unknown token returns False.
  5. sweep_expired — deletes dead rows, returns count; clean DB returns 0.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from backend.database._pool import ConnectionPool
from backend.database._sessions import (
    Session,
    create_session,
    get_session,
    refresh_session,
    revoke_session,
    sweep_expired,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_db(tmp_path):
    """Fresh temp SQLite path (no file on disk)."""
    return tmp_path / "test_sessions.db"


@pytest.fixture
def schema_path():
    """Path to the actual app_schema.sql."""
    from pathlib import Path

    return Path(__file__).parent.parent.parent / "database" / "app_schema.sql"


@pytest.fixture
async def pool(tmp_db, schema_path):
    """
    A pool backed by a temp DB bootstrapped from the schema with an admin row seeded.
    Sessions require a valid admin FK, so we seed one row.
    """
    from backend.database import bootstrap_admin, bootstrap_db

    await bootstrap_db(tmp_db, schema_path)
    await bootstrap_admin(tmp_db, "sess-test@example.com", "pw")
    p = ConnectionPool(
        db_path=tmp_db,
        busy_timeout_seconds=30.0,
        max_size=5,
    )
    yield p
    await p.close()


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _is_sqlite_datetime(ts: str) -> bool:
    """
    Return True if ts is a valid SQLite-native ``YYYY-MM-DD HH:MM:SS`` timestamp.
    """
    try:
        datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        return True
    except ValueError:
        return False


async def _set_expired(pool: ConnectionPool, token_hex: str, hours_ago: int = 1) -> None:
    """
    Backdate a session row's expires_at to simulate expiry.
    Uses the pool to run the UPDATE in an async context.
    """
    token_hash = hashlib.sha256(bytes.fromhex(token_hex)).hexdigest()
    past = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE sessions SET expires_at = ? WHERE session_token_hash = ?;",
            (past, token_hash),
        )
        await conn.commit()


# =============================================================================
# Test 1 — create_session
# =============================================================================


class TestCreateSession:
    """Verify create_session produces valid session rows."""

    @staticmethod
    async def test_returns_session_and_raw_token(pool) -> None:
        """create_session must return a Session object and a raw hex token."""
        session, raw_token = await create_session(pool, admin_id=1)

        assert session is not None
        assert raw_token is not None
        assert isinstance(raw_token, str)
        assert len(raw_token) == 64  # 32 bytes = 64 hex chars

    @staticmethod
    async def test_raw_token_is_hex(pool) -> None:
        """The raw token must be a valid hexadecimal string."""
        _, raw_token = await create_session(pool, admin_id=1)
        int(raw_token, 16)  # raises ValueError if not valid hex
        assert True

    @staticmethod
    async def test_token_hash_is_sha256(pool) -> None:
        """The stored hash must be the SHA-256 digest of the raw token."""
        session, raw_token = await create_session(pool, admin_id=1)

        expected_hash = hashlib.sha256(bytes.fromhex(raw_token)).hexdigest()
        assert session.session_token_hash == expected_hash

    @staticmethod
    async def test_stored_hash_is_unique_per_call(pool) -> None:
        """Two calls with the same admin_id produce different token hashes."""
        session1, _ = await create_session(pool, admin_id=1)
        session2, _ = await create_session(pool, admin_id=1)

        assert session1.session_token_hash != session2.session_token_hash

    @staticmethod
    async def test_csrf_token_is_present(pool) -> None:
        """Each session must have a non-empty CSRF token."""
        session, _ = await create_session(pool, admin_id=1)
        assert session.csrf_token
        assert len(session.csrf_token) > 0

    @staticmethod
    async def test_csrf_token_can_be_provided(pool) -> None:
        """Passing a custom csrf_token must store it verbatim."""
        custom = "my-custom-csrf-token-32chars!!"
        session, _ = await create_session(pool, admin_id=1, csrf_token=custom)
        assert session.csrf_token == custom

    @staticmethod
    async def test_expires_at_is_in_future(pool) -> None:
        """The expires_at timestamp must be ahead of now."""
        session, _ = await create_session(pool, admin_id=1, expiry_days=7)

        assert _is_sqlite_datetime(session.expires_at)
        expires_dt = datetime.strptime(session.expires_at, "%Y-%m-%d %H:%M:%S")
        expires_dt = expires_dt.replace(tzinfo=timezone.utc)
        # Allow 2-second clock skew
        assert expires_dt > datetime.now(timezone.utc) - timedelta(seconds=2)

    @staticmethod
    async def test_expires_at_respects_expiry_days(pool) -> None:
        """A custom expiry_days value must be reflected in expires_at."""
        session, _ = await create_session(pool, admin_id=1, expiry_days=14)

        expires_dt = datetime.strptime(session.expires_at, "%Y-%m-%d %H:%M:%S")
        expires_dt = expires_dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        diff = expires_dt - now

        # Should be ~14 days (allow ±2 seconds for execution time)
        assert (
            timedelta(days=14) - timedelta(seconds=2)
            <= diff
            <= timedelta(days=14) + timedelta(seconds=2)
        )

    @staticmethod
    async def test_created_at_is_sqlite_datetime(pool) -> None:
        """The created_at timestamp must be a valid SQLite datetime string."""
        session, _ = await create_session(pool, admin_id=1)
        assert _is_sqlite_datetime(session.created_at)

    @staticmethod
    async def test_admin_id_is_stored(pool) -> None:
        """The admin_id passed in must appear in the returned Session."""
        session, _ = await create_session(pool, admin_id=1)
        assert session.admin_id == 1

    @staticmethod
    async def test_session_is_persisted_to_db(pool) -> None:
        """A created session must survive a new connection (i.e., it's in the DB)."""
        _, raw_token = await create_session(pool, admin_id=1)

        # Retrieve with a fresh connection (still same pool)
        retrieved = await get_session(pool, raw_token)
        assert retrieved is not None
        assert retrieved.admin_id == 1

    @staticmethod
    async def test_session_is_instance_of_dataclass(pool) -> None:
        """The returned object must be a Session dataclass instance."""
        session, _ = await create_session(pool, admin_id=1)
        assert isinstance(session, Session)
        # All expected fields are present
        assert hasattr(session, "id")
        assert hasattr(session, "session_token_hash")
        assert hasattr(session, "admin_id")
        assert hasattr(session, "csrf_token")
        assert hasattr(session, "created_at")
        assert hasattr(session, "expires_at")


# =============================================================================
# Test 2 — get_session
# =============================================================================


class TestGetSession:
    """Verify get_session validates tokens strictly."""

    @staticmethod
    async def test_get_valid_session_returns_session(pool) -> None:
        """A valid raw token must retrieve the correct session."""
        _, raw_token = await create_session(pool, admin_id=1)

        session = await get_session(pool, raw_token)
        assert session is not None
        assert session.admin_id == 1

    @staticmethod
    async def test_get_unknown_token_returns_none(pool) -> None:
        """A token that was never created must return None."""
        fake_token = "00" * 32  # 64 hex chars, but never stored
        assert await get_session(pool, fake_token) is None

    @staticmethod
    async def test_get_tampered_token_returns_none(pool) -> None:
        """Flipping one hex char in the token must cause retrieval to fail."""
        _, raw_token = await create_session(pool, admin_id=1)

        # Flip the last character
        tampered = raw_token[:-1] + ("f" if raw_token[-1] == "0" else "0")
        assert await get_session(pool, tampered) is None

    @staticmethod
    async def test_get_expired_session_returns_none(pool) -> None:
        """
        Even if a row exists with a valid hash, an expired session must NOT be returned.
        This is the core security invariant: expiry is checked at read time.
        """
        _, raw_token = await create_session(pool, admin_id=1, expiry_days=0)

        # Push expires_at into the past via direct DB update
        await _set_expired(pool, raw_token, hours_ago=1)

        # get_session must return None for the expired session
        result = await get_session(pool, raw_token)
        assert result is None

    @staticmethod
    async def test_get_session_returns_correct_csrf(pool) -> None:
        """The CSRF token stored at creation must be retrievable."""
        custom_csrf = "my-csrf-value-12345678901234"
        _, raw_token = await create_session(pool, admin_id=1, csrf_token=custom_csrf)

        session = await get_session(pool, raw_token)
        assert session is not None
        assert session.csrf_token == custom_csrf

    @staticmethod
    async def test_invalid_hex_returns_none(pool) -> None:
        """Passing a non-hex string must return None (not raise)."""
        result = await get_session(pool, "not-a-hex-string!!")
        assert result is None


# =============================================================================
# Test 3 — refresh_session
# =============================================================================


class TestRefreshSession:
    """Verify refresh_session extends expiry and rejects expired sessions."""

    @staticmethod
    async def test_refresh_extends_expires_at(pool) -> None:
        """A successful refresh must move expires_at forward."""
        session1, raw_token = await create_session(pool, admin_id=1, expiry_days=1)

        # Small delay so time passes
        await asyncio.sleep(0.01)

        session2 = await refresh_session(pool, raw_token, expiry_days=7)
        assert session2 is not None

        expires1 = datetime.strptime(session1.expires_at, "%Y-%m-%d %H:%M:%S")
        expires2 = datetime.strptime(session2.expires_at, "%Y-%m-%d %H:%M:%S")

        assert expires2 > expires1

    @staticmethod
    async def test_refresh_unknown_token_returns_none(pool) -> None:
        """Refreshing a token that doesn't exist must return None."""
        fake_token = "00" * 32
        assert await refresh_session(pool, fake_token) is None

    @staticmethod
    async def test_refresh_expired_session_returns_none(pool) -> None:
        """An already-expired session must NOT be refreshed (core security invariant)."""
        _, raw_token = await create_session(pool, admin_id=1, expiry_days=0)
        await _set_expired(pool, raw_token, hours_ago=1)

        result = await refresh_session(pool, raw_token)
        assert result is None

    @staticmethod
    async def test_refresh_preserves_csrf_and_admin_id(pool) -> None:
        """Refresh must not change csrf_token or admin_id."""
        custom_csrf = "preserved-csrf-token-value!!"
        _, raw_token = await create_session(pool, admin_id=1, csrf_token=custom_csrf)

        refreshed = await refresh_session(pool, raw_token)
        assert refreshed is not None
        assert refreshed.csrf_token == custom_csrf
        assert refreshed.admin_id == 1

    @staticmethod
    async def test_refresh_id_is_unchanged(pool) -> None:
        """Refresh must not change the session's primary key id."""
        session1, raw_token = await create_session(pool, admin_id=1)
        refreshed = await refresh_session(pool, raw_token)

        assert refreshed is not None
        assert refreshed.id == session1.id


# =============================================================================
# Test 4 — revoke_session
# =============================================================================


class TestRevokeSession:
    """Verify revoke_session deletes rows correctly."""

    @staticmethod
    async def test_revoke_returns_true(pool) -> None:
        """Revoking an existing session must return True."""
        _, raw_token = await create_session(pool, admin_id=1)
        result = await revoke_session(pool, raw_token)
        assert result is True

    @staticmethod
    async def test_revoke_deletes_row(pool) -> None:
        """After revocation, the session must no longer be retrievable."""
        _, raw_token = await create_session(pool, admin_id=1)

        await revoke_session(pool, raw_token)
        assert await get_session(pool, raw_token) is None

    @staticmethod
    async def test_revoke_unknown_token_returns_false(pool) -> None:
        """Revoking a non-existent token must return False."""
        fake_token = "00" * 32
        result = await revoke_session(pool, fake_token)
        assert result is False

    @staticmethod
    async def test_revoke_only_affects_target_session(pool) -> None:
        """Revoking one session must not affect another active session."""
        _, raw_token1 = await create_session(pool, admin_id=1)
        _, raw_token2 = await create_session(pool, admin_id=1)

        await revoke_session(pool, raw_token1)

        assert await get_session(pool, raw_token1) is None
        assert await get_session(pool, raw_token2) is not None


# =============================================================================
# Test 5 — sweep_expired
# =============================================================================


class TestSweepExpired:
    """Verify sweep_expired removes dead rows and leaves healthy ones."""

    @staticmethod
    async def test_sweep_clean_db_returns_zero(pool) -> None:
        """Sweeping a DB with no expired sessions must return 0."""
        count = await sweep_expired(pool)
        assert count == 0

    @staticmethod
    async def test_sweep_removes_expired_rows(pool) -> None:
        """Expired rows must be deleted and the count returned."""
        _, raw_token1 = await create_session(pool, admin_id=1, expiry_days=1)
        _, raw_token2 = await create_session(pool, admin_id=1, expiry_days=1)

        # Push both to the past
        await _set_expired(pool, raw_token1, hours_ago=1)
        await _set_expired(pool, raw_token2, hours_ago=1)

        count = await sweep_expired(pool)
        assert count == 2

        # Both sessions should now be gone
        assert await get_session(pool, raw_token1) is None
        assert await get_session(pool, raw_token2) is None

    @staticmethod
    async def test_sweep_preserves_healthy_sessions(pool) -> None:
        """Active sessions must survive a sweep."""
        # One active session
        _, raw_active = await create_session(pool, admin_id=1, expiry_days=7)
        # One expired session
        _, raw_expired = await create_session(pool, admin_id=1, expiry_days=0)
        await _set_expired(pool, raw_expired, hours_ago=1)

        count = await sweep_expired(pool)
        assert count == 1  # only the expired one removed

        # Active session must still be retrievable
        assert await get_session(pool, raw_active) is not None
        # Expired session must be gone
        assert await get_session(pool, raw_expired) is None

    @staticmethod
    async def test_sweep_returns_accurate_count(pool) -> None:
        """The return value must equal the number of rows deleted."""
        # Create 3 expired sessions
        raw_tokens = []
        for _ in range(3):
            _, raw = await create_session(pool, admin_id=1, expiry_days=0)
            raw_tokens.append(raw)

        # Backdate all three
        for raw in raw_tokens:
            await _set_expired(pool, raw, hours_ago=1)

        count = await sweep_expired(pool)
        assert count == 3

    @staticmethod
    async def test_sweep_does_not_remove_future_sessions(pool) -> None:
        """Sessions with future expiry must NOT be removed by sweep."""
        _, raw_active = await create_session(pool, admin_id=1, expiry_days=30)

        count = await sweep_expired(pool)
        assert count == 0
        assert await get_session(pool, raw_active) is not None
