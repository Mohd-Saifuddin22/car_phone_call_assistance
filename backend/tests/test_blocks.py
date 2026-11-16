"""
backend/tests/test_blocks.py — Block/Unblock endpoint tests (Issue 019).

Covers:
  1. Happy-path block (POST /api/leads/{id}/block)
  2. Block idempotency — re-blocking the same lead returns 201, no duplicate rows
  3. Block 404 — non-existent lead ID
  4. Happy-path unblock (DELETE /api/blocks/{phone})
  5. Unblock 404 — phone not in block_list
  6. Shared-email edge case — two phones with the same email; unblocking one does NOT unblock the other
  7. GET /api/blocks — list all blocked customers
  8. GET /api/blocks returns 401 without session
  9. POST /api/leads/{id}/block returns 401 without session
  10. POST /api/leads/{id}/block returns 403 without CSRF token
  11. DELETE /api/blocks/{phone} returns 401 without session
  12. DELETE /api/blocks/{phone} returns 403 without CSRF token
  13. Blocking a Processing lead does NOT terminate the call (endpoint succeeds)

All tests use a bootstrapped temporary SQLite database so no real DB is touched.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from backend.database import bootstrap_admin, bootstrap_db
from backend.database._pool import ConnectionPool


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _schema_path() -> Path:
    return (
        Path(__file__).parent.parent.parent / "database" / "app_schema.sql"
    ).resolve()


def _create_temp_db() -> Path:
    """Create a bootstrapped temp app.db with one admin row seeded."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db", prefix="blocks_") as fh:
        tmp = Path(fh.name)

    async def _do():
        await bootstrap_db(tmp, _schema_path())
        await bootstrap_admin(tmp, "block-test@example.com", "hunter2")

    asyncio.run(_do())
    return tmp


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_db() -> Path:
    return _create_temp_db()


@pytest.fixture
def pool(tmp_db: Path) -> ConnectionPool:
    """Pool backed by the temp DB. Closed automatically after test."""
    p = ConnectionPool(db_path=tmp_db, busy_timeout_seconds=30.0, max_size=5)
    yield p
    asyncio.run(p.close())


@pytest.fixture
def admin_email() -> str:
    return "block-test@example.com"


@pytest.fixture
def admin_password() -> str:
    return "hunter2"


@pytest.fixture
def client(pool: ConnectionPool) -> TestClient:
    """
    Minimal FastAPI app with:
      - Shared pool from the pool fixture
      - Leads router (includes block/unblock endpoints)
      - Auth router (for login)
    """
    from fastapi import Depends, FastAPI

    from backend.api.auth import router as auth_router, get_current_admin
    from backend.api.leads import router as leads_router
    from backend.database._sessions import Session

    app = FastAPI()
    app.state.pool = pool
    app.include_router(auth_router)
    app.include_router(leads_router)

    # A dummy protected GET route for auth tests
    @app.get("/api/test-auth")
    async def test_auth(_: Session = Depends(get_current_admin)) -> dict:
        return {"message": "auth ok"}

    return TestClient(app)


@pytest.fixture
def auth_pair(client: TestClient, admin_email: str, admin_password: str) -> tuple[str, str]:
    """
    Single login — returns (session_token, csrf_token) from the SAME session.
    """
    resp = client.post(
        "/api/admin/login",
        json={"email": admin_email, "password": admin_password},
    )
    assert resp.status_code == 200
    return resp.cookies["session_token"], resp.json()["csrf_token"]


@pytest.fixture
def session_token(client: TestClient, admin_email: str, admin_password: str) -> str:
    """Log in and return the raw session token from the cookie."""
    resp = client.post(
        "/api/admin/login",
        json={"email": admin_email, "password": admin_password},
    )
    assert resp.status_code == 200
    return resp.cookies["session_token"]


@pytest.fixture
def csrf_token(client: TestClient, admin_email: str, admin_password: str) -> str:
    """Log in and return the CSRF token from the JSON body."""
    resp = client.post(
        "/api/admin/login",
        json={"email": admin_email, "password": admin_password},
    )
    assert resp.status_code == 200
    return resp.json()["csrf_token"]


@pytest.fixture
def lead_with_lead(pool: ConnectionPool) -> dict:
    """
    Insert a lead row and return its id and fields.
    """
    async def _do() -> dict:
        async with pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO leads (name, phone, email, city, language, status, attempt_count)
                VALUES ('Alice Smith', '+919876543210', 'alice@example.com', 'Mumbai', 'English', 'New', 0)
                """,
            )
            cursor = await conn.execute("SELECT last_insert_rowid()")
            row = await cursor.fetchone()
            return {"id": row[0], "phone": "+919876543210", "email": "alice@example.com"}

    return asyncio.run(_do())


@pytest.fixture
def lead_processing(pool: ConnectionPool) -> dict:
    """
    Insert a Processing lead row and return its id and fields.
    """
    async def _do() -> dict:
        async with pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO leads (name, phone, email, city, language, status, attempt_count)
                VALUES ('Bob Jones', '+919876543211', 'bob@example.com', 'Delhi', 'Hinglish', 'Processing', 1)
                """,
            )
            cursor = await conn.execute("SELECT last_insert_rowid()")
            row = await cursor.fetchone()
            return {"id": row[0], "phone": "+919876543211", "email": "bob@example.com"}

    return asyncio.run(_do())


# =============================================================================
# Test 1 — Happy-path block
# =============================================================================


class TestBlockHappyPath:
    """POST /api/leads/{id}/block — happy path."""

    @staticmethod
    def test_block_lead_returns_201_and_confirmation(
        client: TestClient,
        auth_pair: tuple[str, str],
        lead_with_lead: dict,
    ) -> None:
        session_token, csrf_token = auth_pair
        resp = client.post(
            f"/api/leads/{lead_with_lead['id']}/block",
            cookies={"session_token": session_token},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 201
        assert "blocked" in resp.json()["message"].lower()

    @staticmethod
    def test_block_creates_entry_in_block_list(
        client: TestClient,
        auth_pair: tuple[str, str],
        lead_with_lead: dict,
    ) -> None:
        session_token, csrf_token = auth_pair
        client.post(
            f"/api/leads/{lead_with_lead['id']}/block",
            cookies={"session_token": session_token},
            headers={"X-CSRF-Token": csrf_token},
        )
        # Verify via GET /api/blocks
        resp = client.get(
            "/api/blocks",
            cookies={"session_token": session_token},
        )
        assert resp.status_code == 200
        blocked = resp.json()["blocked"]
        assert any(b["phone"] == lead_with_lead["phone"] for b in blocked)


# =============================================================================
# Test 2 — Block idempotency
# =============================================================================


class TestBlockIdempotency:
    """Re-blocking the same lead is a no-op."""

    @staticmethod
    def test_block_same_lead_twice_returns_201_both_times(
        client: TestClient,
        auth_pair: tuple[str, str],
        lead_with_lead: dict,
    ) -> None:
        session_token, csrf_token = auth_pair
        # First block
        resp1 = client.post(
            f"/api/leads/{lead_with_lead['id']}/block",
            cookies={"session_token": session_token},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp1.status_code == 201

        # Second block — must also return 201 (not 409)
        resp2 = client.post(
            f"/api/leads/{lead_with_lead['id']}/block",
            cookies={"session_token": session_token},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp2.status_code == 201

    @staticmethod
    def test_block_idempotent_no_duplicate_rows(
        client: TestClient,
        auth_pair: tuple[str, str],
        lead_with_lead: dict,
    ) -> None:
        session_token, csrf_token = auth_pair
        # Block twice
        client.post(
            f"/api/leads/{lead_with_lead['id']}/block",
            cookies={"session_token": session_token},
            headers={"X-CSRF-Token": csrf_token},
        )
        client.post(
            f"/api/leads/{lead_with_lead['id']}/block",
            cookies={"session_token": session_token},
            headers={"X-CSRF-Token": csrf_token},
        )

        # Verify via GET /api/blocks — only ONE entry for this phone
        resp = client.get(
            "/api/blocks",
            cookies={"session_token": session_token},
        )
        blocked = resp.json()["blocked"]
        matches = [b for b in blocked if b["phone"] == lead_with_lead["phone"]]
        assert len(matches) == 1


# =============================================================================
# Test 3 — Block 404
# =============================================================================


class TestBlockNotFound:
    """Blocking a non-existent lead ID returns 404."""

    @staticmethod
    def test_block_nonexistent_lead_returns_404(
        client: TestClient,
        auth_pair: tuple[str, str],
    ) -> None:
        session_token, csrf_token = auth_pair
        resp = client.post(
            "/api/leads/99999/block",
            cookies={"session_token": session_token},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


# =============================================================================
# Test 4 — Block does NOT terminate active calls
# =============================================================================


class TestBlockProcessingLead:
    """Blocking a Processing lead succeeds (does NOT kill the call)."""

    @staticmethod
    def test_block_processing_lead_returns_201(
        client: TestClient,
        auth_pair: tuple[str, str],
        lead_processing: dict,
    ) -> None:
        session_token, csrf_token = auth_pair
        resp = client.post(
            f"/api/leads/{lead_processing['id']}/block",
            cookies={"session_token": session_token},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 201


# =============================================================================
# Test 5 — Happy-path unblock
# =============================================================================


class TestUnblockHappyPath:
    """DELETE /api/blocks/{phone} — happy path."""

    @staticmethod
    def test_unblock_returns_200_and_removes_entry(
        client: TestClient,
        auth_pair: tuple[str, str],
        lead_with_lead: dict,
    ) -> None:
        session_token, csrf_token = auth_pair

        # First block
        client.post(
            f"/api/leads/{lead_with_lead['id']}/block",
            cookies={"session_token": session_token},
            headers={"X-CSRF-Token": csrf_token},
        )

        # Then unblock
        resp = client.delete(
            f"/api/blocks/{lead_with_lead['phone']}",
            cookies={"session_token": session_token},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 200
        assert "unblocked" in resp.json()["message"].lower()

        # Verify via GET /api/blocks
        resp = client.get(
            "/api/blocks",
            cookies={"session_token": session_token},
        )
        blocked = resp.json()["blocked"]
        assert not any(b["phone"] == lead_with_lead["phone"] for b in blocked)


# =============================================================================
# Test 6 — Unblock 404
# =============================================================================


class TestUnblockNotFound:
    """Unblocking a phone not in block_list returns 404."""

    @staticmethod
    def test_unblock_nonexistent_phone_returns_404(
        client: TestClient,
        auth_pair: tuple[str, str],
    ) -> None:
        session_token, csrf_token = auth_pair
        resp = client.delete(
            "/api/blocks/+919988887777",
            cookies={"session_token": session_token},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 404


# =============================================================================
# Test 7 — Shared-email edge case
# =============================================================================


class TestUnblockSharedEmailEdgeCase:
    """
    Two different phones sharing the same email — unblocking one does NOT
    unblock the other.
    """

    @staticmethod
    def test_unblock_one_phone_does_not_affect_other(
        client: TestClient,
        pool: ConnectionPool,
        auth_pair: tuple[str, str],
    ) -> None:
        """
        Scenario:
          - Lead A: phone +91A, email shared@example.com → block
          - Lead B: phone +91B, email shared@example.com → block
          - Unblock +91A
          - Assert: +91A is gone, +91B is still blocked
        """
        session_token, csrf_token = auth_pair

        async def _insert_leads():
            async with pool.connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO leads (name, phone, email, city, language, status, attempt_count)
                    VALUES ('Alice', '+919900000001', 'shared@example.com', 'Mumbai', 'English', 'New', 0)
                    """,
                )
                await conn.execute(
                    """
                    INSERT INTO leads (name, phone, email, city, language, status, attempt_count)
                    VALUES ('Bob', '+919900000002', 'shared@example.com', 'Delhi', 'Hinglish', 'New', 0)
                    """,
                )
                cursor1 = await conn.execute("SELECT last_insert_rowid() - 1")
                cursor2 = await conn.execute("SELECT last_insert_rowid()")
                row1 = await cursor1.fetchone()
                row2 = await cursor2.fetchone()
                return row1[0], row2[0]

        lead_a_id, lead_b_id = asyncio.run(_insert_leads())

        # Block both leads
        client.post(
            f"/api/leads/{lead_a_id}/block",
            cookies={"session_token": session_token},
            headers={"X-CSRF-Token": csrf_token},
        )
        client.post(
            f"/api/leads/{lead_b_id}/block",
            cookies={"session_token": session_token},
            headers={"X-CSRF-Token": csrf_token},
        )

        # Unblock only lead A
        client.delete(
            "/api/blocks/+919900000001",
            cookies={"session_token": session_token},
            headers={"X-CSRF-Token": csrf_token},
        )

        # Verify: lead A is gone, lead B is still blocked
        resp = client.get(
            "/api/blocks",
            cookies={"session_token": session_token},
        )
        blocked = resp.json()["blocked"]
        phones = {b["phone"] for b in blocked}
        assert "+919900000001" not in phones, "Lead A phone should be unblocked"
        assert "+919900000002" in phones, "Lead B phone should still be blocked"


# =============================================================================
# Test 8 — GET /api/blocks
# =============================================================================


class TestListBlocked:
    """GET /api/blocks returns all blocked customers."""

    @staticmethod
    def test_list_blocked_returns_correct_total(
        client: TestClient,
        auth_pair: tuple[str, str],
    ) -> None:
        session_token, csrf_token = auth_pair
        resp = client.get(
            "/api/blocks",
            cookies={"session_token": session_token},
        )
        assert resp.status_code == 200
        assert "blocked" in resp.json()
        assert "total" in resp.json()
        assert resp.json()["total"] == len(resp.json()["blocked"])

    @staticmethod
    def test_list_blocked_returns_empty_when_nobody_blocked(
        client: TestClient,
        auth_pair: tuple[str, str],
    ) -> None:
        session_token, _ = auth_pair
        resp = client.get(
            "/api/blocks",
            cookies={"session_token": session_token},
        )
        assert resp.status_code == 200
        assert resp.json()["blocked"] == []
        assert resp.json()["total"] == 0


# =============================================================================
# Test 9 — Session gating
# =============================================================================


class TestSessionGating:
    """All endpoints require a valid session cookie."""

    @staticmethod
    def test_get_blocks_without_session_returns_401(client: TestClient) -> None:
        resp = client.get("/api/blocks")
        assert resp.status_code == 401

    @staticmethod
    def test_block_without_session_returns_401(client: TestClient, lead_with_lead: dict) -> None:
        resp = client.post(f"/api/leads/{lead_with_lead['id']}/block")
        assert resp.status_code == 401

    @staticmethod
    def test_unblock_without_session_returns_401(client: TestClient) -> None:
        resp = client.delete("/api/blocks/+919900000001")
        assert resp.status_code == 401


# =============================================================================
# Test 10 — CSRF enforcement
# =============================================================================


class TestCSRFEnforcement:
    """POST and DELETE require X-CSRF-Token header."""

    @staticmethod
    def test_block_without_csrf_returns_403(
        client: TestClient,
        session_token: str,
        lead_with_lead: dict,
    ) -> None:
        resp = client.post(
            f"/api/leads/{lead_with_lead['id']}/block",
            cookies={"session_token": session_token},
        )
        assert resp.status_code == 403

    @staticmethod
    def test_block_with_wrong_csrf_returns_403(
        client: TestClient,
        session_token: str,
        lead_with_lead: dict,
    ) -> None:
        resp = client.post(
            f"/api/leads/{lead_with_lead['id']}/block",
            cookies={"session_token": session_token},
            headers={"X-CSRF-Token": "wrong-token"},
        )
        assert resp.status_code == 403

    @staticmethod
    def test_unblock_without_csrf_returns_403(
        client: TestClient,
        session_token: str,
    ) -> None:
        resp = client.delete(
            "/api/blocks/+919900000001",
            cookies={"session_token": session_token},
        )
        assert resp.status_code == 403

    @staticmethod
    def test_unblock_with_wrong_csrf_returns_403(
        client: TestClient,
        session_token: str,
    ) -> None:
        resp = client.delete(
            "/api/blocks/+919900000001",
            cookies={"session_token": session_token},
            headers={"X-CSRF-Token": "wrong-token"},
        )
        assert resp.status_code == 403

    @staticmethod
    def test_get_blocks_is_csrf_exempt(
        client: TestClient,
        session_token: str,
    ) -> None:
        """GET does not require CSRF — only session."""
        resp = client.get(
            "/api/blocks",
            cookies={"session_token": session_token},
        )
        assert resp.status_code == 200
