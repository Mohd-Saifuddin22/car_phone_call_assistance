"""
backend/tests/test_auth_security.py — Hermetic security tests for Issue 013.

Covers:
  1. CORS origin rejection (origin not in allow-list → 403).
  2. CORS origin allowlist acceptance (origin in allow-list → passes).
  3. Session cookie absent → 401 from get_current_admin.
  4. Session cookie invalid/expired → 401 from get_current_admin.
  5. State-changing request (POST) without X-CSRF-Token → 403.
  6. State-changing request (POST) with wrong X-CSRF-Token → 403.
  7. State-changing request (PUT) without X-CSRF-Token → 403.
  8. State-changing request (DELETE) without X-CSRF-Token → 403.
  9. State-changing request (PATCH) without X-CSRF-Token → 403.
 10. GET / protected endpoint → 200 (no CSRF required).
 11. POST / protected endpoint with correct CSRF → 200.
 12. Cookie flags: HttpOnly, Secure, SameSite=none on login response.

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
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db", prefix="auth_sec_") as fh:
        tmp = Path(fh.name)

    async def _do():
        await bootstrap_db(tmp, _schema_path())
        await bootstrap_admin(tmp, "sec-test@example.com", "hunter2")

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
    return "sec-test@example.com"


@pytest.fixture
def admin_password() -> str:
    return "hunter2"


@pytest.fixture
def client(pool: ConnectionPool) -> TestClient:
    """
    Minimal FastAPI app with:
      - Shared pool from the pool fixture (single pool per test)
      - Auth router (login / logout)
      - ACTUAL LazyCORSMiddleware mounted — exercises the shipped code, not a mock
      - CORS allow-list: ["https://allowed.example.com"]
      - Protected routes (GET, POST, PUT, DELETE, PATCH) gated by get_current_admin

    The pool is closed by the pool fixture teardown — no duplicate teardown here.
    """
    from fastapi import Depends, FastAPI

    from backend.api.auth import get_current_admin, router as auth_router
    from backend.main import LazyCORSMiddleware

    app = FastAPI()
    app.state.pool = pool
    # Populate cors_allowed_origins so LazyCORSMiddleware reads it per-request
    app.state.cors_allowed_origins = ["https://allowed.example.com"]
    # Mount the actual shipped middleware — pure ASGI, SSE-safe
    app.add_middleware(LazyCORSMiddleware)
    app.include_router(auth_router)

    @app.get("/api/protected/get")
    async def protected_get(session=Depends(get_current_admin)) -> dict:
        return {"message": "GET OK", "admin_id": session.admin_id}

    @app.post("/api/protected/post")
    async def protected_post(session=Depends(get_current_admin)) -> dict:
        return {"message": "POST OK", "admin_id": session.admin_id}

    @app.put("/api/protected/put")
    async def protected_put(session=Depends(get_current_admin)) -> dict:
        return {"message": "PUT OK", "admin_id": session.admin_id}

    @app.delete("/api/protected/delete")
    async def protected_delete(session=Depends(get_current_admin)) -> dict:
        return {"message": "DELETE OK", "admin_id": session.admin_id}

    @app.patch("/api/protected/patch")
    async def protected_patch(session=Depends(get_current_admin)) -> dict:
        return {"message": "PATCH OK", "admin_id": session.admin_id}

    return TestClient(app)


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
def auth_pair(client: TestClient, admin_email: str, admin_password: str) -> tuple[str, str]:
    """
    Single login — returns (session_token, csrf_token) from the SAME session.
    Use this when both values are needed together to avoid mismatched sessions.
    """
    resp = client.post(
        "/api/admin/login",
        json={"email": admin_email, "password": admin_password},
    )
    assert resp.status_code == 200
    return resp.cookies["session_token"], resp.json()["csrf_token"]


# =============================================================================
# Test 1 — CORS origin rejection
# =============================================================================


class TestCORSOriginRejection:
    """
    Origin NOT in allow-list → CORS header NOT set.

    For simple requests (GET without preflight), LazyCORSMiddleware lets the
    response through but omits Access-Control-Allow-Origin, which is the
    browser-enforced block.  We test that the header is absent rather than
    asserting 403, since browsers don't block the HTTP response itself.
    """

    @staticmethod
    def test_disallowed_origin_gets_no_access_control_header(
        client: TestClient,
    ) -> None:
        """
        A request from an unlisted origin must NOT have Access-Control-Allow-Origin.
        The browser will silently drop the response body, enforcing CORS at the JS layer.
        """
        resp = client.get(
            "/api/protected/get",
            headers={"Origin": "https://evil.example.com"},
        )
        # The backend processes the request (CORS doesn't block HTTP)
        # but the Access-Control-Allow-Origin header must be absent.
        assert "access-control-allow-origin" not in {
            k.lower() for k in resp.headers.keys()
        }, (
            "Access-Control-Allow-Origin should NOT be set for disallowed origins"
        )

    @staticmethod
    def test_disallowed_origin_preflight_returns_400(
        client: TestClient,
    ) -> None:
        """
        An OPTIONS preflight request from a disallowed origin MUST be rejected.
        Starlette's CORSMiddleware returns 400 with body 'Disallowed CORS origin'
        for disallowed preflight origins — the precise HTTP status is a framework
        detail; the essential assertion is that the origin is rejected.
        """
        resp = client.options(
            "/api/protected/get",
            headers={
                "Origin": "https://evil.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        # 400 Bad Request is Starlette's CORS preflight rejection; verify it IS rejected
        assert resp.status_code in {400, 403}, (
            f"Preflight should be rejected, got {resp.status_code}: {resp.text}"
        )
        assert "disallowed" in resp.text.lower()

    @staticmethod
    def test_allowed_origin_has_access_control_header(
        client: TestClient,
    ) -> None:
        """
        A request from an allowed origin MUST have Access-Control-Allow-Origin set
        (browser-visible CORS signal that the response is safe to expose to JS).
        """
        resp = client.get(
            "/api/protected/get",
            headers={"Origin": "https://allowed.example.com"},
        )
        assert "access-control-allow-origin" in {k.lower() for k in resp.headers.keys()}
        assert resp.headers.get("Access-Control-Allow-Origin") == "https://allowed.example.com"


# =============================================================================
# Test 2 — CORS origin acceptance
# =============================================================================


class TestCORSOriginAccepted:
    """Origin IN allow-list → request proceeds (route logic then applies)."""

    @staticmethod
    def test_cors_allowed_origin_with_valid_session(
        client: TestClient, session_token: str
    ) -> None:
        """Allowed origin + valid session + GET → 200."""
        resp = client.get(
            "/api/protected/get",
            cookies={"session_token": session_token},
            headers={"Origin": "https://allowed.example.com"},
        )
        assert resp.status_code == 200

    @staticmethod
    def test_cors_allowed_origin_no_session_still_401(
        client: TestClient,
    ) -> None:
        """Allowed origin but no session → CORS passes, auth fails with 401."""
        resp = client.get(
            "/api/protected/get",
            headers={"Origin": "https://allowed.example.com"},
        )
        assert resp.status_code == 401


# =============================================================================
# Test 3 — Unauthenticated routing rejection
# =============================================================================


class TestUnauthenticatedRejection:
    """Any protected route without a session cookie → 401."""

    @staticmethod
    def test_protected_get_without_cookie_returns_401(client: TestClient) -> None:
        resp = client.get("/api/protected/get")
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Authentication required."

    @staticmethod
    def test_protected_post_without_cookie_returns_401(client: TestClient) -> None:
        resp = client.post("/api/protected/post")
        assert resp.status_code == 401

    @staticmethod
    def test_protected_put_without_cookie_returns_401(client: TestClient) -> None:
        resp = client.put("/api/protected/put")
        assert resp.status_code == 401

    @staticmethod
    def test_protected_delete_without_cookie_returns_401(client: TestClient) -> None:
        resp = client.delete("/api/protected/delete")
        assert resp.status_code == 401

    @staticmethod
    def test_protected_patch_without_cookie_returns_401(client: TestClient) -> None:
        resp = client.patch("/api/protected/patch")
        assert resp.status_code == 401

    @staticmethod
    def test_invalid_session_token_returns_401(client: TestClient) -> None:
        resp = client.get(
            "/api/protected/get",
            cookies={"session_token": "deadbeef" * 8},
        )
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Invalid or expired session."

    @staticmethod
    def test_malformed_session_token_returns_401(client: TestClient) -> None:
        """Malformed hex (contains invalid chars) → clean 401, not 500."""
        resp = client.get(
            "/api/protected/get",
            cookies={"session_token": "xyz_not_hex"},
        )
        assert resp.status_code == 401


# =============================================================================
# Test 4 — CSRF absence on state-changing methods
# =============================================================================


class TestCSRFAbsent:
    """State-changing method without X-CSRF-Token header → 403 Forbidden."""

    @staticmethod
    def test_post_without_csrf_returns_403(
        client: TestClient, session_token: str
    ) -> None:
        resp = client.post(
            "/api/protected/post",
            cookies={"session_token": session_token},
        )
        assert resp.status_code == 403
        assert "X-CSRF-Token" in resp.json()["detail"]

    @staticmethod
    def test_put_without_csrf_returns_403(
        client: TestClient, session_token: str
    ) -> None:
        resp = client.put(
            "/api/protected/put",
            cookies={"session_token": session_token},
        )
        assert resp.status_code == 403

    @staticmethod
    def test_delete_without_csrf_returns_403(
        client: TestClient, session_token: str
    ) -> None:
        resp = client.delete(
            "/api/protected/delete",
            cookies={"session_token": session_token},
        )
        assert resp.status_code == 403

    @staticmethod
    def test_patch_without_csrf_returns_403(
        client: TestClient, session_token: str
    ) -> None:
        resp = client.patch(
            "/api/protected/patch",
            cookies={"session_token": session_token},
        )
        assert resp.status_code == 403


# =============================================================================
# Test 5 — CSRF mismatch on state-changing methods
# =============================================================================


class TestCSRFMismatch:
    """State-changing method with wrong X-CSRF-Token → 403 Forbidden."""

    @staticmethod
    def test_post_with_wrong_csrf_returns_403(
        client: TestClient, session_token: str
    ) -> None:
        resp = client.post(
            "/api/protected/post",
            cookies={"session_token": session_token},
            headers={"X-CSRF-Token": "wrong-token-value"},
        )
        assert resp.status_code == 403
        assert "Invalid CSRF token" in resp.json()["detail"]

    @staticmethod
    def test_put_with_wrong_csrf_returns_403(
        client: TestClient, session_token: str
    ) -> None:
        resp = client.put(
            "/api/protected/put",
            cookies={"session_token": session_token},
            headers={"X-CSRF-Token": "wrong-token-value"},
        )
        assert resp.status_code == 403

    @staticmethod
    def test_delete_with_wrong_csrf_returns_403(
        client: TestClient, session_token: str
    ) -> None:
        resp = client.delete(
            "/api/protected/delete",
            cookies={"session_token": session_token},
            headers={"X-CSRF-Token": "wrong-token-value"},
        )
        assert resp.status_code == 403

    @staticmethod
    def test_patch_with_wrong_csrf_returns_403(
        client: TestClient, session_token: str
    ) -> None:
        resp = client.patch(
            "/api/protected/patch",
            cookies={"session_token": session_token},
            headers={"X-CSRF-Token": "wrong-token-value"},
        )
        assert resp.status_code == 403

    @staticmethod
    def test_post_with_empty_csrf_returns_403(
        client: TestClient, session_token: str
    ) -> None:
        """Empty X-CSRF-Token header is also rejected."""
        resp = client.post(
            "/api/protected/post",
            cookies={"session_token": session_token},
            headers={"X-CSRF-Token": ""},
        )
        assert resp.status_code == 403


# =============================================================================
# Test 6 — Successful authenticated requests
# =============================================================================


class TestSuccessfulAuthenticatedRequests:
    """Valid session + correct CSRF (where required) → 200."""

    @staticmethod
    def test_get_exempt_from_csrf(
        client: TestClient, session_token: str
    ) -> None:
        """GET never requires CSRF."""
        resp = client.get(
            "/api/protected/get",
            cookies={"session_token": session_token},
        )
        assert resp.status_code == 200
        assert resp.json()["message"] == "GET OK"

    @staticmethod
    def test_post_with_correct_csrf(
        client: TestClient, auth_pair: tuple[str, str]
    ) -> None:
        session_token, csrf_token = auth_pair
        resp = client.post(
            "/api/protected/post",
            cookies={"session_token": session_token},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 200
        assert resp.json()["message"] == "POST OK"

    @staticmethod
    def test_put_with_correct_csrf(
        client: TestClient, auth_pair: tuple[str, str]
    ) -> None:
        session_token, csrf_token = auth_pair
        resp = client.put(
            "/api/protected/put",
            cookies={"session_token": session_token},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 200

    @staticmethod
    def test_delete_with_correct_csrf(
        client: TestClient, auth_pair: tuple[str, str]
    ) -> None:
        session_token, csrf_token = auth_pair
        resp = client.delete(
            "/api/protected/delete",
            cookies={"session_token": session_token},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 200

    @staticmethod
    def test_patch_with_correct_csrf(
        client: TestClient, auth_pair: tuple[str, str]
    ) -> None:
        session_token, csrf_token = auth_pair
        resp = client.patch(
            "/api/protected/patch",
            cookies={"session_token": session_token},
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 200

    @staticmethod
    def test_csrf_token_is_per_session(
        client: TestClient, admin_email: str, admin_password: str
    ) -> None:
        """Two login sessions must have different CSRF tokens."""
        r1 = client.post(
            "/api/admin/login",
            json={"email": admin_email, "password": admin_password},
        )
        r2 = client.post(
            "/api/admin/login",
            json={"email": admin_email, "password": admin_password},
        )
        assert r1.json()["csrf_token"] != r2.json()["csrf_token"]


# =============================================================================
# Test 7 — Cookie security flags (Issue 013)
# =============================================================================


class TestCookieSecurityFlags:
    """Login response must set all three security flags on the session cookie."""

    @staticmethod
    def test_cookie_has_httponly_flag(
        client: TestClient, admin_email: str, admin_password: str
    ) -> None:
        resp = client.post(
            "/api/admin/login",
            json={"email": admin_email, "password": admin_password},
        )
        set_cookie = resp.headers.get("set-cookie", "")
        assert "HttpOnly" in set_cookie, f"Expected HttpOnly in: {set_cookie}"

    @staticmethod
    def test_cookie_has_secure_flag(
        client: TestClient, admin_email: str, admin_password: str
    ) -> None:
        resp = client.post(
            "/api/admin/login",
            json={"email": admin_email, "password": admin_password},
        )
        set_cookie = resp.headers.get("set-cookie", "")
        assert "Secure" in set_cookie, f"Expected Secure in: {set_cookie}"

    @staticmethod
    def test_cookie_has_samesite_none(
        client: TestClient, admin_email: str, admin_password: str
    ) -> None:
        resp = client.post(
            "/api/admin/login",
            json={"email": admin_email, "password": admin_password},
        )
        set_cookie = resp.headers.get("set-cookie", "")
        # The cookie header uses lowercase 'samesite=none' per the cookie spec;
        # check case-insensitively.
        assert "samesite=none" in set_cookie.lower(), (
            f"Expected SameSite=None in: {set_cookie}"
        )

    @staticmethod
    def test_cookie_path_is_root(
        client: TestClient, admin_email: str, admin_password: str
    ) -> None:
        resp = client.post(
            "/api/admin/login",
            json={"email": admin_email, "password": admin_password},
        )
        set_cookie = resp.headers.get("set-cookie", "")
        assert "Path=/" in set_cookie, f"Expected Path=/ in: {set_cookie}"
