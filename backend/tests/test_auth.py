"""
backend/tests/test_auth.py — Hermetic endpoint tests for Issue 012 login/logout.

All tests use a bootstrapped temporary SQLite database and a test app that
sets ``app.state.pool`` directly so the auth endpoints use the temp DB
without touching app.db.

Coverage
────────
  1. Login — success, wrong password, unknown user, validation errors.
  2. Login response — HttpOnly cookie is set, CSRF token returned in body.
  3. Logout — clears session, clears cookie, idempotent (no session cookie → still 200).
  4. Session lifecycle end-to-end via HTTP layer.
  5. GET /api/admin/session — rehydration, unauthenticated, expired, no-CSRF-required.
  6. Sweep mechanics (unit-level).

No test writes to the real app.db.
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from backend.database import bootstrap_admin, bootstrap_db
from backend.database._pool import ConnectionPool
from backend.database._sessions import create_session, get_session, sweep_expired


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _schema_path() -> Path:
    return (
        Path(__file__).parent.parent.parent / "database" / "app_schema.sql"
    ).resolve()


def _create_temp_db() -> Path:
    """
    Create a bootstrapped temp app.db with an admin row seeded.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db", prefix="auth_test_") as fh:
        tmp = Path(fh.name)

    conn = sqlite3.connect(str(tmp))
    conn.execute("PRAGMA foreign_keys = ON;")
    with open(_schema_path(), encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.close()

    async def _do():
        await bootstrap_db(tmp, _schema_path())
        await bootstrap_admin(tmp, "auth-test@example.com", "hunter2")

    asyncio.run(_do())
    return tmp


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_db() -> Path:
    return _create_temp_db()


@pytest.fixture
async def pool(tmp_db: Path) -> ConnectionPool:
    """Pool backed by the temp DB. Automatically closed after the test."""
    p = ConnectionPool(db_path=tmp_db, busy_timeout_seconds=30.0, max_size=5)
    yield p
    await p.close()


@pytest.fixture
def admin_email() -> str:
    return "auth-test@example.com"


@pytest.fixture
def admin_password() -> str:
    return "hunter2"


@pytest.fixture
def client(tmp_db: Path, admin_email: str, admin_password: str) -> TestClient:
    """
    Build a minimal FastAPI app that includes the auth router, seeded with a
    real pool on ``app.state`` so ``get_pool`` can find it.

    By setting ``app.state.pool`` directly we avoid any ambiguity around
    ``dependency_overrides`` callable identity.
    """
    from fastapi import FastAPI

    from backend.api.auth import router as auth_router

    # Build pool and attach it to app state before any requests are made.
    pool = ConnectionPool(db_path=tmp_db, busy_timeout_seconds=30.0, max_size=5)

    app = FastAPI()
    app.state.pool = pool
    app.include_router(auth_router)

    client_obj = TestClient(app)

    yield client_obj

    # Teardown: close the pool.
    try:
        loop = asyncio.get_running_loop()
        loop.run_until_complete(pool.close())
    except RuntimeError:
        asyncio.run(pool.close())


# =============================================================================
# Test 1 — POST /api/admin/login — success
# =============================================================================


class TestLoginSuccess:
    """Verify a correct login succeeds with a session cookie and CSRF token."""

    @staticmethod
    def test_returns_200(client: TestClient, admin_email: str, admin_password: str) -> None:
        response = client.post(
            "/api/admin/login",
            json={"email": admin_email, "password": admin_password},
        )
        assert response.status_code == 200, response.text

    @staticmethod
    def test_body_contains_csrf_token(client: TestClient, admin_email: str, admin_password: str) -> None:
        response = client.post(
            "/api/admin/login",
            json={"email": admin_email, "password": admin_password},
        )
        data = response.json()
        assert "csrf_token" in data
        assert data["csrf_token"]  # non-empty
        assert len(data["csrf_token"]) > 0

    @staticmethod
    def test_body_contains_success_message(client: TestClient, admin_email: str, admin_password: str) -> None:
        response = client.post(
            "/api/admin/login",
            json={"email": admin_email, "password": admin_password},
        )
        data = response.json()
        assert data["message"] == "Login successful."

    @staticmethod
    def test_session_cookie_is_set(client: TestClient, admin_email: str, admin_password: str) -> None:
        response = client.post(
            "/api/admin/login",
            json={"email": admin_email, "password": admin_password},
        )
        cookies = response.cookies
        assert "session_token" in cookies

    @staticmethod
    def test_session_cookie_is_httponly(client: TestClient, admin_email: str, admin_password: str) -> None:
        response = client.post(
            "/api/admin/login",
            json={"email": admin_email, "password": admin_password},
        )
        set_cookie = response.headers.get("set-cookie", "")
        assert "HttpOnly" in set_cookie, f"Expected HttpOnly in Set-Cookie: {set_cookie}"

    @staticmethod
    def test_session_cookie_path_is_root(client: TestClient, admin_email: str, admin_password: str) -> None:
        response = client.post(
            "/api/admin/login",
            json={"email": admin_email, "password": admin_password},
        )
        set_cookie = response.headers.get("set-cookie", "")
        assert "Path=/" in set_cookie, f"Expected Path=/ in Set-Cookie: {set_cookie}"

    @staticmethod
    def test_session_cookie_value_is_hex(client: TestClient, admin_email: str, admin_password: str) -> None:
        response = client.post(
            "/api/admin/login",
            json={"email": admin_email, "password": admin_password},
        )
        token = response.cookies["session_token"]
        int(token, 16)  # must be valid hex
        assert len(token) == 64  # 32 bytes = 64 hex chars


# =============================================================================
# Test 2 — POST /api/admin/login — failures
# =============================================================================


class TestLoginFailure:
    """Verify incorrect credentials are rejected with 401 (identical message)."""

    @staticmethod
    def test_wrong_password_returns_401(client: TestClient, admin_email: str) -> None:
        response = client.post(
            "/api/admin/login",
            json={"email": admin_email, "password": "wrong_password"},
        )
        assert response.status_code == 401

    @staticmethod
    def test_unknown_user_returns_401(client: TestClient) -> None:
        response = client.post(
            "/api/admin/login",
            json={"email": "nobody@example.com", "password": "any_password"},
        )
        assert response.status_code == 401

    @staticmethod
    def test_error_message_is_identical_for_unknown_user_and_wrong_password(
        client: TestClient, admin_email: str, admin_password: str
    ) -> None:
        """Identical 401 body prevents user-enumeration attacks."""
        r1 = client.post(
            "/api/admin/login",
            json={"email": "nobody@example.com", "password": "x"},
        )
        r2 = client.post(
            "/api/admin/login",
            json={"email": admin_email, "password": "wrong_password"},
        )
        assert r1.status_code == r2.status_code == 401
        assert r1.json() == r2.json()

    @staticmethod
    def test_empty_password_returns_422(client: TestClient, admin_email: str) -> None:
        response = client.post(
            "/api/admin/login",
            json={"email": admin_email, "password": ""},
        )
        assert response.status_code == 422

    @staticmethod
    def test_missing_password_returns_422(client: TestClient, admin_email: str) -> None:
        response = client.post(
            "/api/admin/login",
            json={"email": admin_email},
        )
        assert response.status_code == 422

    @staticmethod
    def test_missing_email_returns_422(client: TestClient) -> None:
        response = client.post(
            "/api/admin/login",
            json={"password": "hunter2"},
        )
        assert response.status_code == 422


# =============================================================================
# Test 3 — POST /api/admin/logout
# =============================================================================


class TestLogout:
    """Verify logout revokes the session and clears the cookie."""

    @staticmethod
    def test_logout_returns_200(client: TestClient, admin_email: str, admin_password: str) -> None:
        login_resp = client.post(
            "/api/admin/login",
            json={"email": admin_email, "password": admin_password},
        )
        assert login_resp.status_code == 200
        token = login_resp.cookies["session_token"]

        response = client.post("/api/admin/logout", cookies={"session_token": token})
        assert response.status_code == 200

    @staticmethod
    def test_logout_message(client: TestClient, admin_email: str, admin_password: str) -> None:
        login_resp = client.post(
            "/api/admin/login",
            json={"email": admin_email, "password": admin_password},
        )
        token = login_resp.cookies["session_token"]

        response = client.post("/api/admin/logout", cookies={"session_token": token})
        assert response.json() == {"message": "Logged out."}

    @staticmethod
    def test_logout_clears_cookie(client: TestClient, admin_email: str, admin_password: str) -> None:
        login_resp = client.post(
            "/api/admin/login",
            json={"email": admin_email, "password": admin_password},
        )
        token = login_resp.cookies["session_token"]

        response = client.post("/api/admin/logout", cookies={"session_token": token})
        set_cookie = response.headers.get("set-cookie", "")
        # delete_cookie sets Max-Age=0
        assert "Max-Age=0" in set_cookie or "expires" in set_cookie.lower()

    @staticmethod
    def test_logout_idempotent_no_cookie(client: TestClient) -> None:
        """Logout with no cookie must still return 200 (no state leakage)."""
        response = client.post("/api/admin/logout")
        assert response.status_code == 200

    @staticmethod
    def test_session_is_revoked_after_logout(
        client: TestClient, admin_email: str, admin_password: str, pool: ConnectionPool
    ) -> None:
        """After logout, the session must not be retrievable from the DB."""
        login_resp = client.post(
            "/api/admin/login",
            json={"email": admin_email, "password": admin_password},
        )
        token = login_resp.cookies["session_token"]

        client.post("/api/admin/logout", cookies={"session_token": token})

        # Verify session is gone from DB
        async def _check():
            return await get_session(pool, token)

        result = asyncio.run(_check())
        assert result is None

    @staticmethod
    def test_logout_with_fake_token_still_returns_200(client: TestClient) -> None:
        """Revoking a non-existent token must still return 200 (idempotent)."""
        response = client.post(
            "/api/admin/logout",
            cookies={"session_token": "00" * 32},
        )
        assert response.status_code == 200


# =============================================================================
# Test 4 — Session lifecycle end-to-end via HTTP layer
# =============================================================================


class TestSessionLifecycle:
    """Prove the full session lifecycle works end-to-end through the HTTP API."""

    @staticmethod
    def test_login_creates_session_in_db(
        client: TestClient, admin_email: str, admin_password: str, pool: ConnectionPool
    ) -> None:
        login_resp = client.post(
            "/api/admin/login",
            json={"email": admin_email, "password": admin_password},
        )
        token = login_resp.cookies["session_token"]

        async def _check():
            return await get_session(pool, token)

        session = asyncio.run(_check())
        assert session is not None
        assert session.admin_id == 1  # first (and only) admin row

    @staticmethod
    def test_csrf_token_from_login_is_per_session(
        client: TestClient, admin_email: str, admin_password: str
    ) -> None:
        r1 = client.post(
            "/api/admin/login",
            json={"email": admin_email, "password": admin_password},
        )
        r2 = client.post(
            "/api/admin/login",
            json={"email": admin_email, "password": admin_password},
        )
        csrf1 = r1.json()["csrf_token"]
        csrf2 = r2.json()["csrf_token"]
        assert csrf1 != csrf2  # each login creates a fresh CSRF token


# =============================================================================
# Test 5 — GET /api/admin/session  (SPA rehydration endpoint)
# =============================================================================


class TestGetSession:
    """Verify GET /api/admin/session returns correct auth state."""

    @staticmethod
    def test_authenticated_returns_csrf_and_email(
        client: TestClient, admin_email: str, admin_password: str
    ) -> None:
        """Valid session → 200 with authenticated=true, csrf_token, and email."""
        login_resp = client.post(
            "/api/admin/login",
            json={"email": admin_email, "password": admin_password},
        )
        assert login_resp.status_code == 200
        token = login_resp.cookies["session_token"]

        response = client.get("/api/admin/session", cookies={"session_token": token})
        assert response.status_code == 200
        body = response.json()
        assert body["authenticated"] is True
        assert body["csrf_token"] is not None
        assert body["csrf_token"] == login_resp.json()["csrf_token"]
        assert body["email"] == admin_email

    @staticmethod
    def test_no_cookie_returns_unauthenticated(client: TestClient) -> None:
        """No session cookie → 200 with authenticated=false and null fields."""
        response = client.get("/api/admin/session")
        assert response.status_code == 200
        body = response.json()
        assert body["authenticated"] is False
        assert body["csrf_token"] is None
        assert body["email"] is None

    @staticmethod
    def test_expired_session_returns_unauthenticated(
        client: TestClient, admin_email: str, admin_password: str
    ) -> None:
        """Expired/invalid session cookie → 200 with authenticated=false."""
        # Log in to get a valid token, then deliberately use a wrong one.
        client.post(
            "/api/admin/login",
            json={"email": admin_email, "password": admin_password},
        )
        fake_token = "0" * 64  # wrong-format token → not in DB

        response = client.get("/api/admin/session", cookies={"session_token": fake_token})
        assert response.status_code == 200
        body = response.json()
        assert body["authenticated"] is False

    @staticmethod
    def test_does_not_require_csrf(
        client: TestClient, admin_email: str, admin_password: str
    ) -> None:
        """
        GET /session must work WITHOUT the X-CSRF-Token header.
        This is the whole point: the SPA uses it to rehydrate before it
        has a CSRF token.
        """
        login_resp = client.post(
            "/api/admin/login",
            json={"email": admin_email, "password": admin_password},
        )
        token = login_resp.cookies["session_token"]

        # Explicitly do NOT send X-CSRF-Token — it must still work.
        response = client.get(
            "/api/admin/session",
            cookies={"session_token": token},
            headers={},  # no CSRF header
        )
        assert response.status_code == 200
        assert response.json()["authenticated"] is True


# =============================================================================
# Test 6 — Sweep mechanics (unit-level)
# =============================================================================


class TestSweepMechanics:
    """Prove sweep_expired removes dead rows and leaves healthy sessions."""

    @staticmethod
    @pytest.mark.asyncio
    async def test_sweep_removes_expired_session(pool: ConnectionPool) -> None:
        """Expired session rows must be removed by sweep."""
        import hashlib

        # Create a session that expires immediately (expiry_days=0 → past)
        _, raw_token = await create_session(pool, admin_id=1, expiry_days=0)

        # Manually push expires_at into the past (SQLite datetime format)
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        token_hash = hashlib.sha256(bytes.fromhex(raw_token)).hexdigest()
        async with pool.connection() as conn:
            await conn.execute(
                "UPDATE sessions SET expires_at = ? WHERE session_token_hash = ?;",
                (past, token_hash),
            )
            await conn.commit()

        count = await sweep_expired(pool)
        assert count == 1

    @staticmethod
    @pytest.mark.asyncio
    async def test_sweep_preserves_healthy_session(pool: ConnectionPool) -> None:
        """Healthy sessions must survive a sweep."""
        _, raw_token = await create_session(pool, admin_id=1, expiry_days=7)

        count = await sweep_expired(pool)
        assert count == 0

        result = await get_session(pool, raw_token)
        assert result is not None

    @staticmethod
    @pytest.mark.asyncio
    async def test_sweep_returns_accurate_count(pool: ConnectionPool) -> None:
        """Sweep return value must equal the number of rows deleted."""
        # Create 5 expired sessions
        for _ in range(5):
            await create_session(pool, admin_id=1, expiry_days=0)

        # Backdate all to the past (SQLite datetime format)
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        async with pool.connection() as conn:
            await conn.execute("UPDATE sessions SET expires_at = ?;", (past,))
            await conn.commit()

        count = await sweep_expired(pool)
        assert count == 5
