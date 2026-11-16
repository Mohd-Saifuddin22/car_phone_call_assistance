"""
backend/tests/test_admin_bootstrap.py — tests for Issue 011 admin bootstrap and password hashing.

Tests
─────
1. hash_password / verify_password — correct match, wrong match, different hashes for same input.
2. First boot (empty admin table) creates exactly one row with a bcrypt hash.
3. Second boot (existing admin row) is a no-op; existing row is untouched.
4. Plaintext password never touches the database.

No test calls load_settings() at import time.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import aiosqlite

from backend.database import bootstrap_admin, verify_password
from backend.database._auth import hash_password

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Fresh temp SQLite path (no file on disk)."""
    return tmp_path / "test_app.db"


@pytest.fixture
def schema_path() -> Path:
    """Path to the actual app_schema.sql."""
    return Path(__file__).parent.parent.parent / "database" / "app_schema.sql"


@pytest.fixture
async def bootstrapped_db(tmp_db: Path, schema_path: Path) -> Path:
    """A temp database bootstrapped from the schema (admin table exists, is empty)."""
    from backend.database import bootstrap_db

    created = await bootstrap_db(tmp_db, schema_path)
    assert created, "Schema bootstrap should have created the DB"
    return tmp_db


# ==============================================================================
# Test 1 — Password hashing
# ==============================================================================

class TestPasswordHashing:
    """Verify hash_password and verify_password behave correctly."""

    @staticmethod
    def test_hash_produces_non_empty_bcrypt_digest() -> None:
        """hash_password must return a non-empty string."""
        h = hash_password("hunter2")
        assert h
        assert len(h) > 20
        assert h.startswith("$2")  # bcrypt prefix

    @staticmethod
    def test_hash_is_deterministic_in_salt_only() -> None:
        """Two hashes of the same password must differ (different salt)."""
        h1 = hash_password("hunter2")
        h2 = hash_password("hunter2")
        assert h1 != h2, "Same password should produce different hashes (different salts)"

    @staticmethod
    def test_verify_matches_correct_password() -> None:
        """verify_password must return True for the correct password."""
        pw = "My$ecureP@ssw0rd!"
        h = hash_password(pw)
        assert verify_password(pw, h) is True

    @staticmethod
    def test_verify_rejects_incorrect_password() -> None:
        """verify_password must return False for a wrong password."""
        h = hash_password("correct_password")
        assert verify_password("wrong_password", h) is False

    @staticmethod
    def test_verify_rejects_empty_candidate() -> None:
        """verify_password must reject an empty-string candidate."""
        h = hash_password("nonempty")
        assert verify_password("", h) is False

    @staticmethod
    def test_verify_raises_on_invalid_hash() -> None:
        """verify_password must raise ValueError / TypeError on a malformed hash."""
        import pytest

        with pytest.raises((ValueError, TypeError)):
            verify_password("any", "not-a-bcrypt-hash")

    @staticmethod
    def test_plaintext_not_recoverable_from_hash() -> None:
        """The hash must not contain the plaintext password."""
        pw = "SuperSecret123!"
        h = hash_password(pw)
        assert pw not in h


# ==============================================================================
# Test 2 — Admin bootstrap: first boot
# ==============================================================================

class TestAdminBootstrapFirstBoot:
    """Verify first boot (empty admin table) seeds exactly one admin row."""

    @staticmethod
    async def test_first_boot_inserts_one_row(
        bootstrapped_db: Path,
    ) -> None:
        """bootstrap_admin must return True and insert exactly one admin row."""
        result = await bootstrap_admin(
            bootstrapped_db,
            "admin@test.com",
            "hunter2",
        )
        assert result is True

        conn = await aiosqlite.connect(str(bootstrapped_db))
        try:
            cursor = await conn.execute("SELECT COUNT(*) FROM admin;")
            row = await cursor.fetchone()
            assert row is not None and row[0] == 1
        finally:
            await conn.close()

    @staticmethod
    async def test_first_boot_stores_email(
        bootstrapped_db: Path,
    ) -> None:
        """The seeded row must contain the correct email."""
        email = "boss@example.com"
        await bootstrap_admin(bootstrapped_db, email, "pw")

        conn = await aiosqlite.connect(str(bootstrapped_db))
        try:
            cursor = await conn.execute("SELECT email FROM admin;")
            row = await cursor.fetchone()
            assert row is not None and row[0] == email
        finally:
            await conn.close()

    @staticmethod
    async def test_first_boot_stores_hash_not_plaintext(
        bootstrapped_db: Path,
    ) -> None:
        """The stored password_hash must be a bcrypt digest, never the plaintext."""
        pw = "TopSecret!@#$%"
        await bootstrap_admin(bootstrapped_db, "a@b.com", pw)

        conn = await aiosqlite.connect(str(bootstrapped_db))
        try:
            cursor = await conn.execute("SELECT password_hash FROM admin;")
            row = await cursor.fetchone()
            assert row is not None
            stored = row[0]
            # Must be a bcrypt hash
            assert stored.startswith("$2"), f"Expected bcrypt hash, got {stored!r}"
            # Must not contain the plaintext
            assert pw not in stored
            # Must verify correctly
            assert verify_password(pw, stored) is True
        finally:
            await conn.close()

    @staticmethod
    async def test_first_boot_password_can_be_verified(
        bootstrapped_db: Path,
    ) -> None:
        """The seeded admin can be verified with the original password."""
        pw = "Verifiab1eP@ss"
        await bootstrap_admin(bootstrapped_db, "login@test.com", pw)

        conn = await aiosqlite.connect(str(bootstrapped_db))
        try:
            cursor = await conn.execute("SELECT password_hash FROM admin;")
            row = await cursor.fetchone()
            assert row is not None
            assert verify_password(pw, row[0]) is True
            assert verify_password("wrong_password", row[0]) is False
        finally:
            await conn.close()


# ==============================================================================
# Test 3 — Admin bootstrap: second boot (no clobber)
# ==============================================================================

class TestAdminBootstrapNoClobber:
    """Verify subsequent boots leave the existing admin row untouched."""

    @staticmethod
    async def test_second_boot_returns_false(
        bootstrapped_db: Path,
    ) -> None:
        """bootstrap_admin must return False when admin table already has a row."""
        await bootstrap_admin(bootstrapped_db, "first@test.com", "pw1")

        result = await bootstrap_admin(
            bootstrapped_db,
            "second@test.com",
            "pw2",
        )
        assert result is False

    @staticmethod
    async def test_second_boot_does_not_insert(
        bootstrapped_db: Path,
    ) -> None:
        """Second boot must not add a second admin row."""
        await bootstrap_admin(bootstrapped_db, "original@test.com", "pw1")
        await bootstrap_admin(bootstrapped_db, "intruder@test.com", "pw2")

        conn = await aiosqlite.connect(str(bootstrapped_db))
        try:
            cursor = await conn.execute("SELECT COUNT(*) FROM admin;")
            row = await cursor.fetchone()
            assert row is not None and row[0] == 1, "Must still have exactly 1 row"
        finally:
            await conn.close()

    @staticmethod
    async def test_second_boot_preserves_original_email(
        bootstrapped_db: Path,
    ) -> None:
        """Second boot must not overwrite the original email."""
        original_email = "keeper@test.com"
        await bootstrap_admin(bootstrapped_db, original_email, "pw1")
        await bootstrap_admin(bootstrapped_db, "other@test.com", "pw2")

        conn = await aiosqlite.connect(str(bootstrapped_db))
        try:
            cursor = await conn.execute("SELECT email FROM admin;")
            row = await cursor.fetchone()
            assert row is not None and row[0] == original_email
        finally:
            await conn.close()

    @staticmethod
    async def test_second_boot_preserves_original_password(
        bootstrapped_db: Path,
    ) -> None:
        """Second boot must not overwrite the original password hash."""
        original_pw = "KeepThis!"
        await bootstrap_admin(bootstrapped_db, "a@b.com", original_pw)

        # Capture the original hash
        conn0 = await aiosqlite.connect(str(bootstrapped_db))
        try:
            cursor0 = await conn0.execute("SELECT password_hash FROM admin;")
            row0 = await cursor0.fetchone()
            original_hash = row0[0]
        finally:
            await conn0.close()

        # Second boot attempt
        await bootstrap_admin(bootstrapped_db, "a@b.com", "DifferentP@ss!")

        # Hash must be unchanged
        conn1 = await aiosqlite.connect(str(bootstrapped_db))
        try:
            cursor1 = await conn1.execute("SELECT password_hash FROM admin;")
            row1 = await cursor1.fetchone()
            assert row1 is not None and row1[0] == original_hash
            # Original password still verifies
            assert verify_password(original_pw, row1[0]) is True
        finally:
            await conn1.close()

    @staticmethod
    async def test_multiple_boots_all_noop(
        bootstrapped_db: Path,
    ) -> None:
        """Three consecutive boots must all return False and keep exactly one row."""
        await bootstrap_admin(bootstrapped_db, "a@b.com", "pw1")
        await bootstrap_admin(bootstrapped_db, "c@d.com", "pw2")
        await bootstrap_admin(bootstrapped_db, "e@f.com", "pw3")

        conn = await aiosqlite.connect(str(bootstrapped_db))
        try:
            cursor = await conn.execute("SELECT COUNT(*) FROM admin;")
            row = await cursor.fetchone()
            assert row is not None and row[0] == 1
        finally:
            await conn.close()
