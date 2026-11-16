"""
backend/tests/test_leads_list.py — Endpoint tests for GET /api/leads (Issue 016).

Tests cover:
  1. Unauthenticated request → 401
  2. Authenticated request → 200 + correct JSON shape
  3. Timestamps are ISO-8601 with Z suffix
  4. Per-phone history is attached correctly
  5. GET is CSRF-exempt (no X-CSRF-Token required)
  6. Empty leads → 200 with {leads: [], total: 0}

No test writes to the real app.db.
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from backend.database import bootstrap_admin, bootstrap_db
from backend.database._pool import ConnectionPool


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _schema_path() -> Path:
    return (
        Path(__file__).parent.parent.parent / "database" / "app_schema.sql"
    ).resolve()


def _create_temp_db() -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db", prefix="leads_test_") as fh:
        tmp = Path(fh.name)

    conn = sqlite3.connect(str(tmp))
    conn.execute("PRAGMA foreign_keys = ON;")
    with open(_schema_path(), encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.close()

    async def _do():
        await bootstrap_db(tmp, _schema_path())
        await bootstrap_admin(tmp, "leads-test@example.com", "hunter2")

    asyncio.run(_do())
    return tmp


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_db() -> Path:
    return _create_temp_db()


@pytest.fixture
def pool(tmp_db: Path) -> ConnectionPool:
    """Pool backed by the temp DB; closed deterministically at teardown.

    The previous version yielded the pool without closing it — the pool's
    sqlite3 connections and any in-flight worker threads survived across
    the test session and contributed to pytest's "test passed but suite
    hangs at exit" pattern.  This fixture closes the pool via ``asyncio.run``
    immediately after the test returns, so every test gets a fresh,
    fully-released pool and pytest can exit cleanly.
    """
    p = ConnectionPool(db_path=tmp_db, busy_timeout_seconds=30.0, max_size=5)
    try:
        yield p
    finally:
        asyncio.run(p.close())


@pytest.fixture
def client(tmp_db: Path) -> TestClient:
    """
    Minimal FastAPI app that includes both the auth router and the leads router,
    with a pool wired to ``app.state`` so dependencies resolve correctly.

    The ``TestClient`` is used as a context manager so its anyio portal
    shuts down cleanly when the fixture is torn down — without this, the
    portal's internal event-loop thread keeps pytest alive long after the
    test passes, producing the well-known "test passed but suite hangs at
    exit" pattern.
    """
    from fastapi import FastAPI

    from backend.api.auth import router as auth_router
    from backend.api.leads import router as leads_router

    pool = ConnectionPool(db_path=tmp_db, busy_timeout_seconds=30.0, max_size=5)

    app = FastAPI()
    app.state.pool = pool
    app.include_router(auth_router)
    app.include_router(leads_router)

    with TestClient(app) as client_obj:
        yield client_obj

    try:
        loop = asyncio.get_running_loop()
        loop.run_until_complete(pool.close())
    except RuntimeError:
        asyncio.run(pool.close())


@pytest.fixture
def admin_credentials() -> tuple[str, str]:
    return ("leads-test@example.com", "hunter2")


def _login(client: TestClient, email: str, password: str) -> str:
    """Log in and return the session token cookie value."""
    response = client.post(
        "/api/admin/login",
        json={"email": email, "password": password},
    )
    assert response.status_code == 200, f"Login failed: {response.text}"
    return response.cookies["session_token"]


# =============================================================================
# Test 1 — Unauthenticated access → 401
# =============================================================================


class TestLeadsListAuth:
    """GET /api/leads must require a valid session."""

    def test_no_session_returns_401(self, client: TestClient) -> None:
        response = client.get("/api/leads")
        assert response.status_code == 401
        assert response.json()["detail"] == "Authentication required."

    def test_fake_session_returns_401(self, client: TestClient) -> None:
        response = client.get(
            "/api/leads",
            cookies={"session_token": "00" * 32},
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid or expired session."


# =============================================================================
# Test 2 — Authenticated access → 200 + correct shape
# =============================================================================


class TestLeadsListSuccess:
    """Authed requests must return 200 with a {leads, total} envelope."""

    def test_authed_returns_200(
        self, client: TestClient, admin_credentials: tuple[str, str]
    ) -> None:
        token = _login(client, *admin_credentials)
        response = client.get("/api/leads", cookies={"session_token": token})
        assert response.status_code == 200

    def test_response_is_object_with_leads_and_total(
        self, client: TestClient, admin_credentials: tuple[str, str]
    ) -> None:
        token = _login(client, *admin_credentials)
        response = client.get("/api/leads", cookies={"session_token": token})
        body = response.json()
        assert "leads" in body
        assert "total" in body
        assert isinstance(body["leads"], list)
        assert isinstance(body["total"], int)

    def test_total_matches_leads_length(
        self, client: TestClient, admin_credentials: tuple[str, str]
    ) -> None:
        token = _login(client, *admin_credentials)
        response = client.get("/api/leads", cookies={"session_token": token})
        body = response.json()
        assert body["total"] == len(body["leads"])


# =============================================================================
# Test 3 — Response shape with inserted leads
# =============================================================================


class TestLeadsListResponseShape:
    """Verify field names and types in the response body."""

    def _insert_lead(
        self,
        pool: ConnectionPool,
        phone: str,
        email: str,
        name: str = "Test User",
        status: str = "New",
        failure_reason: str | None = None,
        attempt_count: int = 0,
        created_at_offset: str | None = None,
    ) -> None:
        async def _do():
            async with pool.connection() as conn:
                if created_at_offset:
                    await conn.execute(
                        """
                        INSERT INTO leads
                            (name, phone, email, city, language, vehicle_id,
                             status, failure_reason, attempt_count,
                             created_at, updated_at)
                        VALUES
                            (?, ?, ?, 'Mumbai', 'English', NULL, ?, ?, ?,
                             datetime('now', ?), datetime('now', ?))
                        """,
                        (name, phone, email, status, failure_reason, attempt_count, created_at_offset, created_at_offset),
                    )
                else:
                    await conn.execute(
                        """
                        INSERT INTO leads
                            (name, phone, email, city, language, vehicle_id,
                             status, failure_reason, attempt_count)
                        VALUES (?, ?, ?, 'Mumbai', 'English', NULL, ?, ?, ?)
                        """,
                        (name, phone, email, status, failure_reason, attempt_count),
                    )
                await conn.commit()

        asyncio.run(_do())

    def test_lead_fields_are_present(
        self,
        pool: ConnectionPool,
        client: TestClient,
        admin_credentials: tuple[str, str],
    ) -> None:
        phone = "+919600000001"
        email = "shape@example.com"
        self._insert_lead(pool, phone, email)

        token = _login(client, *admin_credentials)
        response = client.get("/api/leads", cookies={"session_token": token})
        leads = response.json()["leads"]

        assert len(leads) == 1
        lead = leads[0]
        assert lead["name"] == "Test User"
        assert lead["phone"] == phone
        assert lead["email"] == email
        assert lead["city"] == "Mumbai"
        assert lead["language"] == "English"
        assert lead["vehicle_id"] is None
        assert lead["status"] == "New"
        assert lead["failure_reason"] is None
        assert lead["attempt_count"] == 0
        assert "created_at" in lead
        assert "history" in lead

    def test_multiple_leads_ordered_newest_first(
        self,
        pool: ConnectionPool,
        client: TestClient,
        admin_credentials: tuple[str, str],
    ) -> None:
        # Insert with explicit different timestamps (SQLite datetime('now') is second-precision,
        # so rapid inserts share timestamps — use explicit offsets instead).
        self._insert_lead(pool, "+919600000001", "first@example.com", name="First Lead")
        self._insert_lead(pool, "+919600000002", "second@example.com", name="Second Lead", created_at_offset="+10 seconds")

        token = _login(client, *admin_credentials)
        response = client.get("/api/leads", cookies={"session_token": token})
        leads = response.json()["leads"]

        assert len(leads) == 2
        # Newest first
        assert leads[0]["name"] == "Second Lead"
        assert leads[1]["name"] == "First Lead"


# =============================================================================
# Test 4 — ISO-8601 Z-suffix on timestamps
# =============================================================================


class TestLeadsListTimestampFormat:
    """created_at must be ISO-8601 with a trailing Z, not SQLite space-format."""

    def _insert_lead(self, pool: ConnectionPool, phone: str, email: str) -> None:
        async def _do():
            async with pool.connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO leads (name, phone, email, city, language)
                    VALUES (?, ?, ?, 'Delhi', 'English')
                    """,
                    ("Test", phone, email),
                )
                await conn.commit()

        asyncio.run(_do())

    def test_created_at_has_T_and_Z(
        self,
        pool: ConnectionPool,
        client: TestClient,
        admin_credentials: tuple[str, str],
    ) -> None:
        self._insert_lead(pool, "+919600000001", "ts@example.com")

        token = _login(client, *admin_credentials)
        response = client.get("/api/leads", cookies={"session_token": token})
        lead = response.json()["leads"][0]

        created_at = lead["created_at"]
        assert "T" in created_at, f"Expected 'T' in timestamp, got: {created_at}"
        assert created_at.endswith("Z"), f"Expected trailing 'Z', got: {created_at}"

    def test_history_created_at_has_T_and_Z(
        self,
        pool: ConnectionPool,
        client: TestClient,
        admin_credentials: tuple[str, str],
    ) -> None:
        self._insert_lead(pool, "+919600000001", "history@example.com")

        token = _login(client, *admin_credentials)
        response = client.get("/api/leads", cookies={"session_token": token})
        lead = response.json()["leads"][0]

        assert len(lead["history"]) == 1
        history_entry = lead["history"][0]
        created_at = history_entry["created_at"]
        assert "T" in created_at, f"Expected 'T' in history timestamp, got: {created_at}"
        assert created_at.endswith("Z"), f"Expected trailing 'Z' in history timestamp, got: {created_at}"


# =============================================================================
# Test 5 — Per-phone history grouping
# =============================================================================


class TestLeadsListHistoryGrouping:
    """Leads sharing a phone must have their history attached to each entry."""

    def _insert_leads_for_same_phone(
        self,
        pool: ConnectionPool,
        phone: str,
        count: int = 3,
    ) -> None:
        """Insert ``count`` leads for the same phone with staggered timestamps (oldest first, newest last)."""

        async def _do():
            async with pool.connection() as conn:
                for i in range(count):
                    # Stagger timestamps so lead {count-1} is newest and lead 0 is oldest.
                    # Each lead gets created 10 seconds after the previous.
                    offset = f"+{i * 10} seconds"
                    await conn.execute(
                        """
                        INSERT INTO leads
                            (name, phone, email, city, language, status,
                             created_at, updated_at)
                        VALUES
                            (?, ?, ?, 'Bangalore', 'English', 'Completed',
                             datetime('now', ?), datetime('now', ?))
                        """,
                        (f"User {i}", phone, f"user{i}@example.com", offset, offset),
                    )
                await conn.commit()

        asyncio.run(_do())

    def test_same_phone_leads_share_history(
        self,
        pool: ConnectionPool,
        client: TestClient,
        admin_credentials: tuple[str, str],
    ) -> None:
        phone = "+919600000001"
        self._insert_leads_for_same_phone(pool, phone, count=3)

        token = _login(client, *admin_credentials)
        response = client.get("/api/leads", cookies={"session_token": token})
        leads = response.json()["leads"]

        assert len(leads) == 3
        for lead in leads:
            assert lead["phone"] == phone
            # Each lead should have exactly 3 history entries (all leads for this phone)
            assert len(lead["history"]) == 3

    def test_different_phones_have_separate_histories(
        self,
        pool: ConnectionPool,
        client: TestClient,
        admin_credentials: tuple[str, str],
    ) -> None:
        self._insert_leads_for_same_phone(pool, "+919600000001", count=2)
        self._insert_leads_for_same_phone(pool, "+919600000002", count=3)

        token = _login(client, *admin_credentials)
        response = client.get("/api/leads", cookies={"session_token": token})
        leads = response.json()["leads"]

        assert len(leads) == 5

        phone1_leads = [lead for lead in leads if lead["phone"] == "+919600000001"]
        phone2_leads = [lead for lead in leads if lead["phone"] == "+919600000002"]

        for lead in phone1_leads:
            assert len(lead["history"]) == 2
        for lead in phone2_leads:
            assert len(lead["history"]) == 3

    def test_history_ordered_newest_first(
        self,
        pool: ConnectionPool,
        client: TestClient,
        admin_credentials: tuple[str, str],
    ) -> None:
        phone = "+919600000001"
        self._insert_leads_for_same_phone(pool, phone, count=3)

        token = _login(client, *admin_credentials)
        response = client.get("/api/leads", cookies={"session_token": token})
        leads = response.json()["leads"]

        # Top lead (newest, inserted last with highest ID) should have itself first in history
        top = leads[0]
        assert top["history"][0]["id"] == top["id"]
        # All history IDs should be distinct
        history_ids = [h["id"] for h in top["history"]]
        assert len(history_ids) == len(set(history_ids))
        # History should have 3 entries
        assert len(history_ids) == 3

    def test_history_entry_has_id_and_status(
        self,
        pool: ConnectionPool,
        client: TestClient,
        admin_credentials: tuple[str, str],
    ) -> None:
        phone = "+919600000001"
        self._insert_leads_for_same_phone(pool, phone, count=2)

        token = _login(client, *admin_credentials)
        response = client.get("/api/leads", cookies={"session_token": token})
        lead = response.json()["leads"][0]

        assert len(lead["history"]) == 2
        for entry in lead["history"]:
            assert "id" in entry
            assert "status" in entry
            assert "created_at" in entry
            assert isinstance(entry["id"], int)
            assert isinstance(entry["status"], str)


# =============================================================================
# Test 6 — CSRF exemption for GET
# =============================================================================


class TestLeadsListCSRFExemption:
    """GET /api/leads must work without the X-CSRF-Token header."""

    def test_get_without_csrf_token_succeeds(
        self, client: TestClient, admin_credentials: tuple[str, str]
    ) -> None:
        token = _login(client, *admin_credentials)

        # Explicitly do NOT send X-CSRF-Token — it must still work (GET is CSRF-exempt).
        response = client.get(
            "/api/leads",
            cookies={"session_token": token},
            headers={},
        )
        assert response.status_code == 200


# =============================================================================
# Test 7 — Empty leads
# =============================================================================


class TestLeadsListEmpty:
    """Empty database must return {leads: [], total: 0}."""

    def test_empty_returns_empty_list(
        self, client: TestClient, admin_credentials: tuple[str, str]
    ) -> None:
        token = _login(client, *admin_credentials)
        response = client.get("/api/leads", cookies={"session_token": token})

        assert response.status_code == 200
        body = response.json()
        assert body["leads"] == []
        assert body["total"] == 0
