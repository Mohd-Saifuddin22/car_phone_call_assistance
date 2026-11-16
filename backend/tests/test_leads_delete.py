"""
backend/tests/test_leads_delete.py — API tests for DELETE /api/leads/{id} (Issue 018).

Coverage:
  1. DELETE requires session auth → 401 without cookie.
  2. DELETE requires CSRF token → 403 without X-CSRF-Token header.
  3. DELETE returns 404 for a non-existent lead.
  4. DELETE returns 409 Conflict for a Processing lead.
  5. DELETE returns 204 and wipes the lead row for a New lead.
  6. DELETE returns 204 and wipes the lead row for a Completed lead.
  7. DELETE returns 204 and wipes the lead row for a Failed lead.
  8. Cascade: associated call_attempts, transcript_turns, recommendations are
     deleted alongside the lead.
"""

from __future__ import annotations

import asyncio
import contextlib
import tempfile
from pathlib import Path
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

# ─── Test-database bootstrap ───────────────────────────────────────────────────


def _schema_path() -> Path:
    return (
        Path(__file__).parent.parent.parent / "database" / "app_schema.sql"
    ).resolve()


def _create_temp_db() -> Path:
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".db", prefix="car_del_"
    ) as fh:
        tmp = Path(fh.name)
    import sqlite3

    conn = sqlite3.connect(str(tmp))
    conn.execute("PRAGMA foreign_keys = ON;")
    with open(_schema_path(), encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.close()
    return tmp


# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    from backend.config import load_settings

    load_settings.cache_clear()
    yield
    load_settings.cache_clear()


@pytest_asyncio.fixture
async def temp_db() -> AsyncGenerator[Path, None]:
    db_path = _create_temp_db()
    yield db_path
    try:
        db_path.unlink()
    except OSError:
        pass


@pytest_asyncio.fixture
async def client(temp_db: Path) -> AsyncGenerator[AsyncClient, None]:
    """
    Async HTTP client wired to the FastAPI app with an isolated test database.

    Identical pattern to the client fixture in test_leads.py — the DELETE
    endpoint uses the same pool override and auth flow.
    """
    from backend import main as main_module
    from backend.api.leads import get_pool as leads_get_pool
    from backend.catalogue import VehicleRecord, _by_id, vehicles
    from backend.config import Settings
    from backend.database import create_pool

    main_module._db_pool = None
    main_module.settings = None

    pool = create_pool(
        db_path=temp_db,
        busy_timeout_seconds=5.0,
        pool_size=10,
    )
    _warm: list = []
    for _ in range(pool.max_size):
        c = await pool.acquire()
        _warm.append(c)
    for c in _warm:
        await pool.release(c)

    # Seed the admin account so _login can authenticate.
    from backend.database import bootstrap_admin

    await bootstrap_admin(
        temp_db,
        "del-test@example.com",
        "hunter2",
        busy_timeout_seconds=5.0,
    )

    fake_settings = Settings(
        app_db_path=temp_db,
        vehicles_db_path=Path("E:/Projects/car_phone_call_assistance/database/vehicles.db"),
        recordings_root=Path("E:/Projects/car_phone_call_assistance/recordings"),
        deepgram_api_key="test-deepgram",
        mercury_two_api_key="test-mercury",
        gemini_api_key="test-gemini",
        sarvam_api_key="test-sarvam",
        admin_email="del-test@example.com",
        admin_password="hunter2",
    )

    fake_vehicle = VehicleRecord(
        id=1,
        name="Test Vehicle",
        price_lakhs=5.0,
        fuel_type="Petrol",
        seating_capacity=5,
        transmission="Manual",
        body_style="Hatchback",
        mileage_kmpl=20.0,
        description="Test vehicle",
        features=["Airbags"],
    )
    vehicles.clear()
    vehicles.append(fake_vehicle)
    _by_id.clear()
    _by_id[1] = fake_vehicle

    import unittest.mock as umock

    @contextlib.asynccontextmanager
    async def fake_lifespan(app):
        main_module._db_pool = pool
        main_module.settings = fake_settings
        app.state.pool = pool
        app.state.rate_limit_store = {}
        app.state.rate_limit_limit = 100
        yield
        await pool.close()

    with umock.patch.object(main_module, "lifespan", fake_lifespan):
        app = main_module.create_app()
        app.dependency_overrides[leads_get_pool] = lambda: pool
        async with LifespanManager(app):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as ac:
                yield ac

        app.dependency_overrides.clear()

    vehicles.clear()
    _by_id.clear()


# ─── Auth helpers ─────────────────────────────────────────────────────────────


async def _login(client: AsyncClient) -> str:
    """
    Log in as the seeded admin. Sets the session cookie on the client instance
    (the recommended httpx pattern — avoids deprecated per-request cookies=).
    Returns the CSRF token for the caller to pass as a header.
    """
    resp = await client.post(
        "/api/admin/login",
        json={"email": "del-test@example.com", "password": "hunter2"},
    )
    assert resp.status_code == 200, f"Login failed: {resp.text}"
    csrf_token = resp.json()["csrf_token"]
    # Set the session cookie directly on the client so subsequent requests
    # send it automatically (the recommended httpx pattern).
    session_cookie = resp.cookies.get("session_token", "")
    if session_cookie:
        client.cookies.set("session_token", session_cookie)
    return csrf_token


async def _insert_lead(
    pool,
    status: str,
    name: str = "Test Lead",
    phone: str = "+919900000001",
    email: str = "test@example.com",
) -> int:
    """Insert a lead with the given status and return its id."""
    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            INSERT INTO leads
                (name, phone, email, city, language, status, attempt_count)
            VALUES (?, ?, ?, 'Mumbai', 'English', ?, 0)
            """,
            (name, phone, email, status),
        )
        return cursor.lastrowid


async def _count_rows(pool, table: str) -> int:
    async with pool.connection() as conn:
        cursor = await conn.execute(f"SELECT COUNT(*) FROM {table}")
        row = await cursor.fetchone()
        return row[0] if row else 0


# ─── Tests ─────────────────────────────────────────────────────────────────────


class TestDeleteLeadRequiresAuth:
    """DELETE /api/leads/{id} requires a valid session cookie."""

    @pytest.mark.asyncio
    async def test_delete_without_session_returns_401(self, client: AsyncClient) -> None:
        """
        An unauthenticated DELETE request must return 401 Unauthorized.
        """
        resp = await client.delete("/api/leads/1")
        assert resp.status_code == 401, (
            f"Expected 401 for unauthenticated request, got {resp.status_code}"
        )


class TestDeleteLeadRequiresCSRF:
    """DELETE is a state-changing method — X-CSRF-Token is mandatory."""

    @pytest.mark.asyncio
    async def test_delete_without_csrf_returns_403(self, client: AsyncClient) -> None:
        """
        A DELETE request with a valid session but no X-CSRF-Token header
        must return 403 Forbidden.
        """
        await _login(client)
        resp = await client.delete("/api/leads/1")
        assert resp.status_code == 403, (
            f"Expected 403 without CSRF token, got {resp.status_code}"
        )
        assert "X-CSRF-Token" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_delete_with_wrong_csrf_returns_403(self, client: AsyncClient) -> None:
        """
        A DELETE request with an invalid CSRF token must return 403 Forbidden.
        """
        await _login(client)
        resp = await client.delete(
            "/api/leads/1",
            headers={"X-CSRF-Token": "definitely-wrong-token"},
        )
        assert resp.status_code == 403, (
            f"Expected 403 for wrong CSRF token, got {resp.status_code}"
        )


class TestDeleteLeadNotFound:
    """DELETE returns 404 when the lead does not exist."""

    @pytest.mark.asyncio
    async def test_delete_returns_404_for_nonexistent_lead(self, client: AsyncClient) -> None:
        """
        Deleting a lead ID that does not exist must return 404 Not Found.
        """
        csrf_token = await _login(client)
        resp = await client.delete(
            "/api/leads/99999",
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 404, (
            f"Expected 404 for non-existent lead, got {resp.status_code}: {resp.text}"
        )


class TestDeleteLeadProcessingConflict:
    """DELETE returns 409 Conflict for a lead in Processing status."""

    @pytest.mark.asyncio
    async def test_delete_returns_409_for_processing_lead(
        self, client: AsyncClient
    ) -> None:
        """
        A lead actively being handled by the call worker must NOT be deletable.
        The API returns 409 Conflict so the caller knows to wait instead.
        """
        csrf_token = await _login(client)
        pool = client._transport.app.state.pool

        # Insert a Processing lead directly via the pool.
        lead_id = await _insert_lead(pool, status="Processing")

        resp = await client.delete(
            f"/api/leads/{lead_id}",
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 409, (
            f"Expected 409 for Processing lead, got {resp.status_code}: {resp.text}"
        )
        detail = resp.json()["detail"]
        assert detail["status"] == "Processing"
        assert detail["lead_id"] == lead_id

        # The lead must still exist (delete was blocked).
        assert await _count_rows(pool, "leads") == 1


class TestDeleteLeadSuccess:
    """DELETE returns 204 and cascades for non-Processing leads."""

    @pytest.mark.asyncio
    async def test_delete_new_lead_returns_204_and_removes_row(
        self, client: AsyncClient
    ) -> None:
        """
        A New lead is deleted successfully: 204 No Content, lead row gone.
        """
        csrf_token = await _login(client)
        pool = client._transport.app.state.pool

        lead_id = await _insert_lead(pool, status="New")

        resp = await client.delete(
            f"/api/leads/{lead_id}",
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 204, (
            f"Expected 204 for deleted lead, got {resp.status_code}: {resp.text}"
        )
        assert await _count_rows(pool, "leads") == 0

    @pytest.mark.asyncio
    async def test_delete_completed_lead_returns_204(
        self, client: AsyncClient
    ) -> None:
        """A Completed lead is deleted successfully."""
        csrf_token = await _login(client)
        pool = client._transport.app.state.pool

        lead_id = await _insert_lead(pool, status="Completed")

        resp = await client.delete(
            f"/api/leads/{lead_id}",
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 204
        assert await _count_rows(pool, "leads") == 0

    @pytest.mark.asyncio
    async def test_delete_failed_lead_returns_204(
        self, client: AsyncClient
    ) -> None:
        """A Failed lead is deleted successfully."""
        csrf_token = await _login(client)
        pool = client._transport.app.state.pool

        lead_id = await _insert_lead(pool, status="Failed")

        resp = await client.delete(
            f"/api/leads/{lead_id}",
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 204
        assert await _count_rows(pool, "leads") == 0


class TestDeleteLeadCascade:
    """Cascade: call_attempts, transcript_turns, recommendations are deleted."""

    @pytest.mark.asyncio
    async def test_delete_cascades_to_call_attempts_and_transcript_turns(
        self, client: AsyncClient
    ) -> None:
        """
        Deleting a lead must also wipe associated call_attempts,
        transcript_turns, and recommendations via SQLite ON DELETE CASCADE.
        """
        csrf_token = await _login(client)
        pool = client._transport.app.state.pool

        async with pool.connection() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO leads
                    (name, phone, email, city, language, status, attempt_count)
                VALUES ('Cascade Test', '+919900000099', 'cascade@example.com',
                        'Delhi', 'Hinglish', 'New', 2)
                """
            )
            lead_id = cursor.lastrowid

            await conn.execute(
                """
                INSERT INTO call_attempts
                    (lead_id, attempt_number, outcome, recording_path)
                VALUES (?, 1, 'completed', '99/call_001.wav')
                """,
                (lead_id,),
            )
            await conn.execute(
                """
                INSERT INTO transcript_turns
                    (call_attempt_id, turn_index, speaker, text)
                VALUES (1, 0, 'customer', 'I want a test car.')
                """
            )
            await conn.execute(
                """
                INSERT INTO recommendations
                    (lead_id, items, preference_snapshot)
                VALUES (?, '[{"vehicle_id":1,"rank":1,"score":0.9}]', '{}')
                """,
                (lead_id,),
            )

        resp = await client.delete(
            f"/api/leads/{lead_id}",
            headers={"X-CSRF-Token": csrf_token},
        )
        assert resp.status_code == 204

        assert await _count_rows(pool, "leads") == 0
        assert await _count_rows(pool, "call_attempts") == 0
        assert await _count_rows(pool, "transcript_turns") == 0
        assert await _count_rows(pool, "recommendations") == 0

        # block_list (if any) must survive — tested separately in Issue 017
