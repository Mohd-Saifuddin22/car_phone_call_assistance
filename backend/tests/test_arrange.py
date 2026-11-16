"""
backend/tests/test_arrange.py — Tests for Arrange AI Call handler (Issue 030).

Test coverage:
  P1  Real Twilio placement: mock client.calls.create, verify to/from/callback/AMD params.
  P2  Sync error handling: on Twilio failure, resolve attempt + release slot in handler.
  P3  Atomic idempotency: conditional UPDATE prevents race; concurrent requests handled.
  P4  Webhook release: slot released on terminal outcomes (REMOVE-IN-032 note verified).
  P5  Blocking SDK off event loop: Twilio call runs via run_in_executor.
  P6  AMD unconditional + placeholder-twiml endpoint + matcher_usable placeholder comment.

No test calls load_settings() at import time.
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, AsyncGenerator
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    import httpx as _httpx  # noqa: F401 — for quoted annotations
    from backend.call_supervisor import CallSupervisor  # noqa: F401

import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from backend.api.arrange import (
    set_concurrency_manager,
    _twilio_create_call,
    _twilio_hangup_call,
)
from backend.concurrency import ConcurrencyManager

# ─── Test-database bootstrap ───────────────────────────────────────────────────


def _schema_path() -> Path:
    """Absolute path to the application schema file."""
    return (
        Path(__file__).parent.parent.parent / "database" / "app_schema.sql"
    ).resolve()


def _create_temp_db() -> Path:
    """Create a temporary app.db from the application schema."""
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".db", prefix="car_test_arrange_"
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
def clear_supervisor_registry() -> None:
    """
    Clear the in-memory active-calls registry between tests (Issue 032).

    The registry is module-level global state; without an autouse clear, a
    supervisor registered in one test would persist into the next, causing
    stale-looking lookups and slot-state leaks across tests.
    """
    from backend.call_supervisor import _active_calls

    _active_calls.clear()
    yield
    _active_calls.clear()


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    """Fresh settings cache for every test — no bleed between test runs."""
    from backend.config import load_settings

    load_settings.cache_clear()
    yield
    load_settings.cache_clear()


@pytest.fixture
def twilio_stub(monkeypatch: pytest.MonkeyPatch):
    """
    Opt-in ``twilio.rest.Client`` stub for tests that exercise the arrange
    flow without a real Twilio account.

    The stub returns a sentinel CallSid so arrange-leads can record a
    successful call_attempt row.  Tests that need to assert on the
    specific Twilio call shape (e.g. P1, P2) should still use
    ``with patch("twilio.rest.Client", mock)`` directly and skip this
    fixture.  Tests that do not request this fixture (and do not provide
    their own Twilio mock) MUST fail loudly when arrange-ai-call tries
    to construct a real Twilio client.
    """
    from unittest.mock import MagicMock

    sentinel_sid = "CA00000000000000000000000000000000"  # 34 chars, Twilio shape

    class _NoOpCall:
        sid = sentinel_sid
        status = "queued"
        to = "+15005550006"
        from_ = "+15005550007"

        @property
        def from__(self) -> str:
            return "+15005550007"

        def fetch(self) -> "_NoOpCall":
            return self

        def update(self, **_kwargs: object) -> "_NoOpCall":
            return self

    class _NoOpCallCreate:
        def __call__(self, *_args: object, **_kwargs: object) -> _NoOpCall:
            return _NoOpCall()

    class _NoOpCallsResource:
        def create(self, *_args: object, **_kwargs: object) -> _NoOpCall:
            return _NoOpCall()

        def __getattr__(self, name: str) -> MagicMock:
            return MagicMock()

    class _DefaultTwilioClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.calls = _NoOpCallsResource()
            self.recordings = MagicMock()
            self.messages = MagicMock()

        def __getattr__(self, name: str) -> MagicMock:
            return MagicMock()

    import twilio.rest as _twilio_rest

    monkeypatch.setattr(_twilio_rest, "Client", _DefaultTwilioClient)
    return _DefaultTwilioClient


@pytest_asyncio.fixture
async def temp_db() -> AsyncGenerator[Path, None]:
    """Per-test isolated database backed by a temp file."""
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
    """
    from backend import main as main_module
    from backend.api.arrange import get_pool as arrange_get_pool
    from backend.api.auth import get_pool as auth_get_pool
    from backend.config import Settings
    from backend.database import create_pool

    # Reset module-level state.
    main_module._db_pool = None
    main_module.settings = None

    pool = create_pool(
        db_path=temp_db,
        busy_timeout_seconds=5.0,
        pool_size=10,
    )
    # Pre-warm all slots.
    _warm: list = []
    for _ in range(pool.max_size):
        c = await pool.acquire()
        _warm.append(c)
    for c in _warm:
        await pool.release(c)

    fake_settings = Settings(
        app_db_path=temp_db,
        vehicles_db_path=Path("E:/Projects/car_phone_call_assistance/database/vehicles.db"),
        recordings_root=Path("E:/Projects/car_phone_call_assistance/recordings"),
        deepgram_api_key="test-deepgram",
        mercury_two_api_key="test-mercury",
        gemini_api_key="test-gemini",
        sarvam_api_key="test-sarvam",
        admin_email="test@example.com",
        admin_password="test",
        twilio_account_sid="FAKE_ACCOUNT_SID",
        twilio_auth_token="FAKE_AUTH_TOKEN",
        twilio_phone_number="+15550000000",
        max_concurrent_calls=2,
        max_attempts=3,
        max_call_duration_seconds=720,
        poll_interval_seconds=5.0,
        stt_no_first_response_timeout=3.0,
        stt_midstream_stall_timeout=1.5,
        stt_reconnect_attempts=2,
        deepgram_eager_eot_threshold=0.4,
        deepgram_eot_threshold=0.7,
        deepgram_eot_timeout_ms=5500,
        rate_limit_per_ip_per_hour=100,
        db_pool_size=10,
        session_expiry_days=7,
        session_token_hash_algorithm="sha256",
        cors_allowed_origins=[],
        host="0.0.0.0",
        port=8000,
        reload=False,
        public_hostname="test.example.com",
    )

    concurrency_manager = ConcurrencyManager(capacity=2)
    set_concurrency_manager(concurrency_manager)

    @contextlib.asynccontextmanager
    async def fake_lifespan(app):
        main_module._db_pool = pool
        main_module.settings = fake_settings
        app.state.pool = pool
        app.state.rate_limit_store = {}
        app.state.rate_limit_limit = 100
        app.state.cors_allowed_origins = []

        from backend.database import bootstrap_admin

        await bootstrap_admin(
            temp_db,
            "test@example.com",
            "test",
            busy_timeout_seconds=5.0,
        )

        yield
        await pool.close()

    import unittest.mock as umock

    with umock.patch.object(main_module, "lifespan", fake_lifespan):
        app = main_module.create_app()
        app.dependency_overrides[auth_get_pool] = lambda: pool
        app.dependency_overrides[arrange_get_pool] = lambda: pool
        async with LifespanManager(app):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as ac:
                yield ac

        app.dependency_overrides.clear()

    if hasattr(app.state, "rate_limit_store"):
        app.state.rate_limit_store.clear()


@pytest_asyncio.fixture
async def admin_session(client: AsyncClient) -> dict:
    """Log in as admin and return session cookies + CSRF token."""
    login_resp = await client.post(
        "/api/admin/login",
        json={"email": "test@example.com", "password": "test"},
    )
    assert login_resp.status_code == 200, login_resp.text
    csrf_token = login_resp.json()["csrf_token"]
    session_token = client.cookies.get("session_token")
    return {"token": session_token, "csrf": csrf_token}


@pytest_asyncio.fixture
async def read_leads(temp_db: Path):
    """Async fixture that reads all leads from the test database."""
    import aiosqlite

    async def _read() -> list[dict]:
        conn = await aiosqlite.connect(str(temp_db))
        conn.row_factory = sqlite3.Row
        try:
            async with conn.execute("SELECT * FROM leads ORDER BY id") as cur:
                rows = await cur.fetchall()
                return [dict(r) for r in rows]
        finally:
            await conn.close()

    return _read


@pytest_asyncio.fixture
async def read_attempts(temp_db: Path):
    """Async fixture that reads all call_attempts from the test database."""
    import aiosqlite

    async def _read() -> list[dict]:
        conn = await aiosqlite.connect(str(temp_db))
        conn.row_factory = sqlite3.Row
        try:
            async with conn.execute("SELECT * FROM call_attempts ORDER BY id") as cur:
                rows = await cur.fetchall()
                return [dict(r) for r in rows]
        finally:
            await conn.close()

    return _read


async def _insert_lead(
    pool,
    name: str = "Test Lead",
    phone: str = "+919900000001",
    email: str = "test@example.com",
    city: str = "Mumbai",
    language: str = "English",
    status: str = "New",
    attempt_count: int = 0,
) -> int:
    """Insert a lead directly into the database and return its ID."""
    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            INSERT INTO leads
                (name, phone, email, city, language, status, attempt_count)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (name, phone, email, city, language, status, attempt_count),
        )
        return cursor.lastrowid


async def arrange_lead(
    client: AsyncClient,
    lead_id: int,
    session: dict,
) -> _httpx.Response:
    """Helper to POST /api/admin/leads/{lead_id}/arrange with auth."""
    return await client.post(
        f"/api/admin/leads/{lead_id}/arrange",
        cookies={"session_token": session["token"]},
        headers={"X-CSRF-Token": session["csrf"]},
    )


# ─── Helper: register a CallSupervisor for an existing live call_attempt ────────


def register_supervisor_for_live_call(
    pool,
    *,
    call_sid: str,
    lead_id: int,
    attempt_id: int,
) -> "CallSupervisor":
    """
    Manually register a CallSupervisor for a live call_attempt row that the
    test inserted directly.

    Issue 032: webhooks are thin signalers — they look up the supervisor via
    ``get_call(call_sid)`` and signal it.  If no supervisor is registered,
    webhooks return 200 with no action (trusting the startup sweep).  Tests
    that want the webhook to actually do work must therefore pre-register a
    supervisor before posting to the webhook, mirroring what
    ``arrange_ai_call`` does in production after the Twilio call is placed.

    Returns the registered supervisor so tests can inspect latch state, etc.
    """
    from backend.api.arrange import get_concurrency_manager
    from backend.call_supervisor import CallSupervisor, register_call

    manager = get_concurrency_manager()
    supervisor = CallSupervisor(
        lead_id=lead_id,
        attempt_id=attempt_id,
        call_sid=call_sid,
        language="English",
        concurrency_manager=manager,
        pool=pool,
    )
    register_call(supervisor)
    return supervisor


# ─── Helper: assert Twilio calls.create was called with expected params ──────────


def _assert_twilio_call_params(mock_create, *, to, from_, url, async_amd, status_callback):
    """Assert client.calls.create was called with the expected Twilio parameters."""
    mock_create.assert_called_once()
    _, kwargs = mock_create.call_args
    assert kwargs.get("to") == to, f"Expected to={to}, got {kwargs.get('to')}"
    assert kwargs.get("from_") == from_, f"Expected from_={from_}, got {kwargs.get('from_')}"
    assert kwargs.get("url") == url
    assert kwargs.get("async_amd") == async_amd
    assert kwargs.get("status_callback") == status_callback
    assert kwargs.get("status_callback_event") == [
        "initiated",
        "ringing",
        "answered",
        "completed",
    ]


# ─── Test: CSRF protection ───────────────────────────────────────────────────────


class TestArrangeCSRF:
    """Arrange endpoint must enforce CSRF protection."""

    @pytest.mark.asyncio
    async def test_request_without_csrf_token_is_rejected(
        self,
        client: AsyncClient,
        admin_session: dict,
    ) -> None:
        """A POST without X-CSRF-Token header must return 403."""
        pool = client._transport.app.state.pool
        lead_id = await _insert_lead(pool)
        response = await client.post(
            f"/api/admin/leads/{lead_id}/arrange",
            cookies={"session_token": admin_session["token"]},
        )
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_request_with_invalid_csrf_token_is_rejected(
        self,
        client: AsyncClient,
        admin_session: dict,
    ) -> None:
        """A POST with an invalid X-CSRF-Token header must return 403."""
        pool = client._transport.app.state.pool
        lead_id = await _insert_lead(pool)
        response = await client.post(
            f"/api/admin/leads/{lead_id}/arrange",
            cookies={"session_token": admin_session["token"]},
            headers={"X-CSRF-Token": "invalid-token"},
        )
        assert response.status_code == 403


# ─── Test: Lead not found ────────────────────────────────────────────────────────


class TestArrangeNotFound:
    """Arrange endpoint must return 404 for unknown leads."""

    @pytest.mark.asyncio
    async def test_arrange_nonexistent_lead_returns_404(
        self,
        client: AsyncClient,
        admin_session: dict,
    ) -> None:
        """A request for a lead ID that doesn't exist must return 404."""
        response = await arrange_lead(client, lead_id=99999, session=admin_session)
        assert response.status_code == 404


# ─── Test: Idempotency (P3: atomic conditional UPDATE) ───────────────────────────


class TestArrangeAtomicIdempotency:
    """
    P3 FIX: Idempotency is now atomic — the conditional UPDATE inside the
    transaction prevents races.  When two concurrent requests arrive for the
    same already-Processing lead, only one can win the UPDATE (rowcount=1);
    the loser gets rowcount=0, releases its slot, and returns idempotent.
    """

    @pytest.mark.asyncio
    async def test_already_processing_lead_skips_without_taking_slot(
        self,
        client: AsyncClient,
        admin_session: dict,
        read_attempts,
    ) -> None:
        """
        When a lead is already Processing, arrange must return 200 with the
        existing attempt_id WITHOUT leaving a new attempt row.
        """
        pool = client._transport.app.state.pool

        # Insert a Processing lead with one existing call_attempt.
        async with pool.connection() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO leads
                    (name, phone, email, city, language, status, attempt_count)
                VALUES ('Processing Lead', '+919910000001', 'proc@example.com', 'Delhi', 'English', 'Processing', 1)
                """,
            )
            lead_id = cursor.lastrowid
            await conn.execute(
                "INSERT INTO call_attempts (lead_id, attempt_number) VALUES (?, 1);",
                (lead_id,),
            )

        from backend.api.arrange import get_concurrency_manager

        manager = get_concurrency_manager()
        initial_active = manager.active

        response = await arrange_lead(client, lead_id=lead_id, session=admin_session)

        # Must return 200 (idempotent).
        assert response.status_code == 200, (
            f"Expected 200 for already-Processing lead, got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert data["message"] == "Lead is already being processed."
        assert "attempt_id" in data

        # No slot was acquired.
        assert manager.active == initial_active, (
            f"Slot was acquired for already-Processing lead! active={manager.active}, expected={initial_active}"
        )

        # No new call_attempt row was created.
        attempts = await read_attempts()
        assert len(attempts) == 1, (
            f"Expected 1 attempt (idempotency violated), got {len(attempts)}"
        )

    @pytest.mark.asyncio
    async def test_idempotent_response_includes_existing_attempt_id(
        self,
        client: AsyncClient,
        admin_session: dict,
    ) -> None:
        """The idempotent response must include the existing call_attempts ID."""
        pool = client._transport.app.state.pool

        async with pool.connection() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO leads
                    (name, phone, email, city, language, status, attempt_count)
                VALUES ('ID Test Lead', '+919920000001', 'id@example.com', 'Mumbai', 'English', 'Processing', 1)
                """,
            )
            lead_id = cursor.lastrowid
            await conn.execute(
                "INSERT INTO call_attempts (lead_id, attempt_number) VALUES (?, 1);",
                (lead_id,),
            )

        response = await arrange_lead(client, lead_id=lead_id, session=admin_session)
        assert response.status_code == 200
        data = response.json()
        assert "attempt_id" in data
        assert isinstance(data["attempt_id"], int)


# ─── Test: Concurrency rejection ──────────────────────────────────────────────────


class TestArrangeConcurrencyRejection:
    """Arrange endpoint must reject at-capacity requests with 409."""

    @pytest.mark.asyncio
    async def test_at_capacity_returns_409_immediately(
        self,
        client: AsyncClient,
        admin_session: dict,
    ) -> None:
        """When all concurrency slots are taken, arrange must return 409 immediately."""
        pool = client._transport.app.state.pool

        from backend.api.arrange import get_concurrency_manager

        manager = get_concurrency_manager()

        # Fill all slots.
        for _ in range(manager.capacity):
            assert manager.acquire() is True

        assert manager.active == manager.capacity

        lead_id = await _insert_lead(pool, phone="+919930000001")
        response = await arrange_lead(client, lead_id=lead_id, session=admin_session)

        assert response.status_code == 409
        data = response.json()
        assert "Concurrency limit reached" in data["detail"]["msg"]

        # No slot was consumed by the rejected request.
        assert manager.active == manager.capacity

        for _ in range(manager.capacity):
            manager.release()

    @pytest.mark.asyncio
    async def test_slot_not_consumed_on_rejection(
        self,
        client: AsyncClient,
        admin_session: dict,
    ) -> None:
        """After a rejection, the manager's active count must still be 0 (slots not consumed)."""
        pool = client._transport.app.state.pool

        from backend.api.arrange import get_concurrency_manager

        manager = get_concurrency_manager()
        manager.reset()

        for _ in range(manager.capacity):
            manager.acquire()

        lead_id = await _insert_lead(pool, phone="+919940000001")
        response = await arrange_lead(client, lead_id=lead_id, session=admin_session)
        assert response.status_code == 409

        assert manager.active == manager.capacity

        for _ in range(manager.capacity):
            manager.release()


# ─── Test: Max attempts enforcement ──────────────────────────────────────────────────


class TestArrangeMaxAttempts:
    """Arrange endpoint must reject leads that have reached max_attempts."""

    @pytest.mark.asyncio
    async def test_lead_at_max_attempts_returns_422(
        self,
        client: AsyncClient,
        admin_session: dict,
    ) -> None:
        """A lead with attempt_count >= max_attempts must return 422."""
        pool = client._transport.app.state.pool
        lead_id = await _insert_lead(
            pool,
            phone="+919950000001",
            status="New",
            attempt_count=3,  # max_attempts = 3
        )

        response = await arrange_lead(client, lead_id=lead_id, session=admin_session)
        assert response.status_code == 422
        assert "maximum" in response.json()["detail"]["msg"].lower()

    @pytest.mark.asyncio
    async def test_lead_one_below_max_attempts_succeeds(
        self,
        twilio_stub,
        client: AsyncClient,
        admin_session: dict,
    ) -> None:
        """A lead with attempt_count = max_attempts - 1 must not return 422."""
        pool = client._transport.app.state.pool
        lead_id = await _insert_lead(
            pool,
            phone="+919960000001",
            status="New",
            attempt_count=2,  # max_attempts = 3
        )

        response = await arrange_lead(client, lead_id=lead_id, session=admin_session)
        # May be 200 or 409, but not 422.
        assert response.status_code != 422


# ─── Test: Atomic DB updates ─────────────────────────────────────────────────────


class TestArrangeAtomicDBUpdates:
    """Arrange endpoint must atomically update call_attempts and leads table."""

    @pytest.mark.asyncio
    async def test_call_attempt_row_is_inserted(
        self,
        twilio_stub,
        client: AsyncClient,
        admin_session: dict,
        read_attempts,
    ) -> None:
        """A successful arrange must insert a call_attempts row with attempt_number = 1."""
        pool = client._transport.app.state.pool
        lead_id = await _insert_lead(pool, phone="+919970000001", attempt_count=0)

        # Arrange succeeds (Twilio not configured in mock returns None → resolved as error).
        # The attempt row must be inserted regardless of Twilio outcome.
        await arrange_lead(client, lead_id=lead_id, session=admin_session)
        attempts = await read_attempts()
        assert len(attempts) == 1
        assert attempts[0]["lead_id"] == lead_id
        assert attempts[0]["attempt_number"] == 1

    @pytest.mark.asyncio
    async def test_lead_status_flipped_to_processing(
        self,
        client: AsyncClient,
        admin_session: dict,
        read_leads,
    ) -> None:
        """A successful arrange must flip the lead status to 'Processing'."""
        pool = client._transport.app.state.pool
        lead_id = await _insert_lead(
            pool,
            phone="+919980000001",
            status="New",
            attempt_count=0,
        )

        # Mock Twilio so the arrange completes successfully (call_sid stored).
        with patch(
            "backend.api.arrange._twilio_create_call",
            return_value="CA_PROCESSING_TEST_001",
        ):
            await arrange_lead(client, lead_id=lead_id, session=admin_session)

        leads = await read_leads()
        assert leads[0]["status"] == "Processing"

    @pytest.mark.asyncio
    async def test_attempt_count_is_bumped(
        self,
        twilio_stub,
        client: AsyncClient,
        admin_session: dict,
        read_leads,
    ) -> None:
        """A successful arrange must increment the lead's attempt_count."""
        pool = client._transport.app.state.pool
        lead_id = await _insert_lead(
            pool,
            phone="+919990000001",
            status="New",
            attempt_count=0,
        )

        await arrange_lead(client, lead_id=lead_id, session=admin_session)

        leads = await read_leads()
        assert leads[0]["attempt_count"] == 1

    @pytest.mark.asyncio
    async def test_updated_at_is_bumped(
        self,
        twilio_stub,
        client: AsyncClient,
        admin_session: dict,
        read_leads,
    ) -> None:
        """A successful arrange must update the lead's updated_at timestamp."""
        pool = client._transport.app.state.pool

        # Insert a lead with an old timestamp directly via the pool.
        async with pool.connection() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO leads
                    (name, phone, email, city, language, status, attempt_count, created_at, updated_at)
                VALUES ('Old Lead', '+919991000001', 'old@example.com', 'Mumbai', 'English', 'New', 0, '2020-01-01 00:00:00', '2020-01-01 00:00:00')
                """,
            )
            lead_id = cursor.lastrowid

        await arrange_lead(client, lead_id=lead_id, session=admin_session)

        leads = await read_leads()
        assert leads[0]["updated_at"] != "2020-01-01 00:00:00"


# ─── Test: Twilio call placement (P1 / P5 / P6) ─────────────────────────────────────


class TestArrangeTwilioPlacement:
    """
    P1 FIX: Real Twilio placement — phone fetched in handler, passed to
    _twilio_create_call which is called via run_in_executor.

    P5 FIX: client.calls.create() runs in a thread pool executor, not on
    the asyncio event loop.

    P6 FIX: AMD is unconditional (async_amd=True).  Status callback events
    are initiated, ringing, answered, completed.
    """

    @pytest.mark.asyncio
    async def test_arrange_without_twilio_config_returns_success_with_null_sid(
        self,
        client: AsyncClient,
        admin_session: dict,
        read_attempts,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        When Twilio credentials are empty, arrange still succeeds (Twilio is
        skipped) and the attempt is resolved as error.
        """
        # Force-empty Twilio config so this test is hermetic regardless of
        # whatever credentials ``.env`` happens to carry.  ``load_settings``
        # is ``lru_cache``-d; the cache was populated at the ``client``
        # fixture (lifespan startup) before this test body ran, so we must
        # clear it AFTER the env is patched.
        monkeypatch.setenv("TWILIO_ACCOUNT_SID", "")
        monkeypatch.setenv("TWILIO_AUTH_TOKEN", "")
        monkeypatch.setenv("TWILIO_PHONE_NUMBER", "")
        from backend.config import load_settings
        load_settings.cache_clear()

        pool = client._transport.app.state.pool
        lead_id = await _insert_lead(
            pool,
            phone="+919900000001",
            attempt_count=0,
        )

        response = await arrange_lead(client, lead_id=lead_id, session=admin_session)

        assert response.status_code == 200, response.text
        data = response.json()
        assert "attempt_id" in data
        # call_sid may be None when Twilio is not configured.

    @pytest.mark.asyncio
    async def test_arrange_with_mocked_twilio_places_call_with_correct_params(
        self,
        client: AsyncClient,
        admin_session: dict,
    ) -> None:
        """
        P1/P6 FIX: Arrange must call client.calls.create with the correct parameters:
          - to = lead's phone number
          - from_ = settings.twilio_phone_number
          - url = https://<public_hostname>/api/twilio/placeholder-twiml
          - async_amd = 'true' (unconditional)
          - machine_detection = 'Enable' (Issue 031)
          - async_amd_status_callback = https://<public_hostname>/api/twilio/amd-callback (Issue 031)
          - async_amd_status_callback_method = 'POST' (Issue 031)
          - status_callback = https://<public_hostname>/api/twilio/status-callback
          - status_callback_event = [initiated, ringing, answered, completed]
        """
        pool = client._transport.app.state.pool
        to_phone = "+919900000002"
        lead_id = await _insert_lead(
            pool,
            phone=to_phone,
            email="twilio@example.com",
            attempt_count=0,
        )

        # Mock _twilio_create_call (the synchronous function called via run_in_executor).
        with patch(
            "backend.api.arrange._twilio_create_call",
            return_value="CA1234567890abcdef",
        ) as mock_create:
            response = await arrange_lead(client, lead_id=lead_id, session=admin_session)

            assert response.status_code == 200, response.text
            data = response.json()
            assert data["call_sid"] == "CA1234567890abcdef"

            # Verify _twilio_create_call was called with the lead's phone.
            mock_create.assert_called_once()
            args = mock_create.call_args[0]
            assert args[2] == to_phone, (
                f"Expected to_phone={to_phone}, got {args[2]}"
            )

    @pytest.mark.asyncio
    async def test_twilio_create_call_function_signature(
        self,
    ) -> None:
        """
        Verify _twilio_create_call has the expected signature and returns CallSid on success.
        Patches at 'twilio.rest.Client' because that's where TwilioClient is imported inside
        _twilio_create_call (function-local import).
        """
        mock_call = MagicMock()
        mock_call.sid = "CA_UNIT_TEST_001"

        mock_client = MagicMock()
        mock_client.calls.create = MagicMock(return_value=mock_call)

        # Patch at twilio.rest.Client — the actual import site inside _twilio_create_call.
        with patch("twilio.rest.Client", return_value=mock_client):
            sid = _twilio_create_call(
                lead_id=1,
                attempt_id=1,
                to_phone="+919900000001",
                twilio_account_sid="AC fake",
                twilio_auth_token="fake",
                twilio_phone_number="+15550000000",
                public_hostname="test.example.com",
            )

        assert sid == "CA_UNIT_TEST_001"
        mock_client.calls.create.assert_called_once()
        _, kwargs = mock_client.calls.create.call_args

        # P6 FIX: AMD is unconditional.
        assert kwargs["async_amd"] == "true"
        assert kwargs["to"] == "+919900000001"
        assert kwargs["from_"] == "+15550000000"
        assert kwargs["status_callback"] == (
            "https://test.example.com/api/twilio/status-callback"
        )
        assert kwargs["status_callback_event"] == [
            "initiated",
            "ringing",
            "answered",
            "completed",
        ]
        assert kwargs["url"] == (
            "https://test.example.com/api/twilio/placeholder-twiml"
        )

        # Issue 031: dedicated async-AMD callback wired to /api/twilio/amd-callback
        # with machine detection enabled.  This is the new surface area for
        # AMD verdicts so the status-callback endpoint stays focused on terminal
        # CallStatus transitions.
        assert kwargs["machine_detection"] == "Enable"
        assert kwargs["async_amd_status_callback"] == (
            "https://test.example.com/api/twilio/amd-callback"
        )
        assert kwargs["async_amd_status_callback_method"] == "POST"

    @pytest.mark.asyncio
    async def test_twilio_create_call_returns_none_on_api_error(
        self,
    ) -> None:
        """_twilio_create_call must return None when TwilioRestException is raised."""
        from twilio.base.exceptions import TwilioRestException

        mock_client = MagicMock()
        mock_client.calls.create.side_effect = TwilioRestException(
            msg="Invalid phone number", status=400, uri="/2010-04-01/Accounts/ACfake/Calls.json"
        )

        with patch("twilio.rest.Client", return_value=mock_client):
            sid = _twilio_create_call(
                lead_id=1,
                attempt_id=1,
                to_phone="+919900000001",
                twilio_account_sid="AC fake",
                twilio_auth_token="fake",
                twilio_phone_number="+15550000000",
                public_hostname="test.example.com",
            )

        assert sid is None


# ─── Test: Sync error handling (P2) ───────────────────────────────────────────────


class TestArrangeSyncErrorHandling:
    """
    P2 FIX: If Twilio throws an error or returns no SID, the handler must
    synchronously resolve the attempt as 'error', flip the lead to Failed,
    and release the concurrency slot.
    """

    @pytest.mark.asyncio
    async def test_twilio_failure_resolves_attempt_as_error(
        self,
        client: AsyncClient,
        admin_session: dict,
    ) -> None:
        """
        When Twilio API fails (returns None), the attempt must be resolved
        with outcome='error' and the lead transitioned to Failed.
        """
        pool = client._transport.app.state.pool
        lead_id = await _insert_lead(
            pool,
            phone="+919900000003",
            attempt_count=0,
        )

        # Mock _twilio_create_call to return None (simulating Twilio failure).
        with patch(
            "backend.api.arrange._twilio_create_call",
            return_value=None,
        ):
            response = await arrange_lead(client, lead_id=lead_id, session=admin_session)

        # Arrange still returns 200 (the failure was handled internally).
        assert response.status_code == 200, response.text

        # Verify the attempt was resolved as 'error'.
        import aiosqlite

        conn = await aiosqlite.connect(str(pool.db_path))
        conn.row_factory = sqlite3.Row
        async with conn.execute(
            "SELECT outcome, ended_at FROM call_attempts WHERE lead_id = ?",
            (lead_id,),
        ) as cur:
            row = await cur.fetchone()
            assert row is not None
            assert row["outcome"] == "error"
            assert row["ended_at"] is not None
        await conn.close()

        # Verify the lead was transitioned to Failed.
        conn = await aiosqlite.connect(str(pool.db_path))
        conn.row_factory = sqlite3.Row
        async with conn.execute(
            "SELECT status, failure_reason FROM leads WHERE id = ?",
            (lead_id,),
        ) as cur:
            row = await cur.fetchone()
            assert row["status"] == "Failed"
            assert row["failure_reason"] == "error"
        await conn.close()

    @pytest.mark.asyncio
    async def test_twilio_failure_releases_concurrency_slot(
        self,
        client: AsyncClient,
        admin_session: dict,
    ) -> None:
        """
        When Twilio API fails, the concurrency slot must be released
        immediately in the handler (not leaked).
        """
        pool = client._transport.app.state.pool

        from backend.api.arrange import get_concurrency_manager

        manager = get_concurrency_manager()
        manager.reset()

        lead_id = await _insert_lead(
            pool,
            phone="+919900000004",
            attempt_count=0,
        )

        with patch(
            "backend.api.arrange._twilio_create_call",
            return_value=None,
        ):
            response = await arrange_lead(client, lead_id=lead_id, session=admin_session)

        assert response.status_code == 200

        # Slot must have been released (active count is 0).
        assert manager.active == 0, (
            f"Slot was leaked! active={manager.active}, expected=0"
        )


# ─── Test: Early non-Twilio exception cleanup (Issue 032 follow-up) ────────


class TestArrangeEarlyExceptionCleanup:
    """
    REGRESSION (Issue 032 follow-up):

    If ``_twilio_create_call`` raises a non-Twilio exception early in the
    placement flow (BEFORE the supervisor is created), the generic
    ``except Exception`` handler must clean up gracefully — resolve the
    attempt as Failed/error, flip the lead to Failed, and release the slot.

    Before the fix, the handler referenced ``supervisor`` without it having
    been initialised, which raised ``UnboundLocalError`` (masking the real
    exception), skipped cleanup, left a zombie Processing lead, and
    permanently leaked the concurrency slot.
    """

    @pytest.mark.asyncio
    async def test_early_runtime_error_during_twilio_create_is_cleaned_up(
        self,
        client: AsyncClient,
        admin_session: dict,
    ) -> None:
        """
        Patch ``_twilio_create_call`` to raise a non-Twilio RuntimeError
        before the supervisor exists.  The arrange handler must:
          1. Catch the exception (no UnboundLocalError masking).
          2. Resolve the attempt as Failed (failure_reason='error').
          3. Flip the lead to Failed.
          4. Release the concurrency slot.
          5. Re-raise the original RuntimeError to the caller.
        """
        pool = client._transport.app.state.pool

        from backend.api.arrange import get_concurrency_manager

        manager = get_concurrency_manager()
        manager.reset()

        lead_id = await _insert_lead(
            pool,
            phone="+919994000001",
            attempt_count=0,
        )

        # Patch _twilio_create_call to raise a non-Twilio RuntimeError
        # BEFORE the supervisor would be created.  This simulates a
        # transient error inside the run_in_executor — e.g., the Twilio
        # SDK raising before returning.
        with patch(
            "backend.api.arrange._twilio_create_call",
            side_effect=RuntimeError("simulated early non-Twilio failure"),
        ):
            # arrange_ai_call should re-raise the original RuntimeError.
            # Before the fix, it would raise UnboundLocalError instead.
            with pytest.raises(
                RuntimeError,
                match="simulated early non-Twilio failure",
            ):
                await arrange_lead(
                    client, lead_id=lead_id, session=admin_session
                )

        # 1. Attempt resolved as Failed (failure_reason=error).
        import aiosqlite

        conn = await aiosqlite.connect(str(pool.db_path))
        conn.row_factory = sqlite3.Row
        async with conn.execute(
            "SELECT status, failure_reason FROM leads WHERE id = ?",
            (lead_id,),
        ) as cur:
            lead = await cur.fetchone()
        async with conn.execute(
            "SELECT outcome FROM call_attempts WHERE lead_id = ? "
            "ORDER BY id DESC LIMIT 1",
            (lead_id,),
        ) as cur:
            attempt = await cur.fetchone()
        await conn.close()

        assert lead["status"] == "Failed", (
            "Lead must be transitioned to Failed on early exception cleanup"
        )
        assert lead["failure_reason"] == "error", (
            "failure_reason must be 'error' for early non-Twilio exceptions"
        )
        assert attempt["outcome"] == "error", (
            "Attempt must be resolved as 'error' on early exception cleanup"
        )

        # 2. Slot released (no leak).
        assert manager.active == 0, (
            f"Slot was leaked on early exception cleanup! active={manager.active}"
        )

        # 3. No supervisor was ever registered (so no orphan entry).
        from backend.call_supervisor import get_call
        # Any random call_sid — registry should be empty.
        assert get_call("CA_NONEXISTENT") is None


# ─── Test: Orphan window — Twilio live, SID-store write fails (Issue 032) ─────


class TestArrangeOrphanWindow:
    """
    Issue 032: there is a small window between Twilio call placement success
    and the SID-store DB write where the call is LIVE at Twilio but the DB
    doesn't yet know its CallSid.  If the SID-store write raises (e.g., DB
    transient error, FK violation, etc.), the supervisor must trigger
    teardown so the live call is ended and the lead is marked Failed/error.

    Without this, we'd leak a Twilio credit and have an orphaned Processing
    lead in the DB.
    """

    @pytest.mark.asyncio
    async def test_sid_store_failure_triggers_supervisor_teardown(
        self,
        client: AsyncClient,
        admin_session: dict,
    ) -> None:
        """
        If the SID-store DB write fails after the Twilio call was successfully
        placed, the supervisor must:
          1. End the live Twilio call (hangup).
          2. Mark the lead Failed (failure_reason=error).
          3. Release the concurrency slot.
          4. Unregister itself from the active-calls registry.
        The arrange handler must then re-raise the original DB error.
        """
        pool = client._transport.app.state.pool

        from backend.api.arrange import get_concurrency_manager

        manager = get_concurrency_manager()
        manager.reset()

        lead_id = await _insert_lead(
            pool,
            phone="+919993000001",
            attempt_count=0,
        )

        call_sid_to_return = "CA_ORPHAN_WINDOW_001"

        # Patch 1: _twilio_create_call returns a successful CallSid.
        # Patch 2: _twilio_hangup_call so we can assert it was called by the
        # supervisor (and to avoid a real Twilio call).
        # Patch 3: Make the SID-store DB write fail by raising from the
        # execute() of any connection acquired during the test, but only
        # for the "UPDATE call_attempts SET twilio_call_sid" statement.
        # The simplest way: monkey-patch the pool's connection method.

        original_connection_method = pool.connection

        # Wrap pool.connection so it returns a context manager whose execute()
        # raises ONLY for the SID-store UPDATE (UPDATE call_attempts SET twilio_call_sid).
        class _FailingConn:
            def __init__(self, real_conn):
                self._real = real_conn

            async def execute(self, sql, *args, **kwargs):
                # The SID-store UPDATE has this distinctive form.  Match on it
                # so other UPDATEs (none exist in arrange_ai_call's hot path,
                # but be defensive) still work.
                if isinstance(sql, str) and "UPDATE call_attempts" in sql \
                        and "SET twilio_call_sid" in sql:
                    raise RuntimeError("simulated orphan-window DB failure")
                return await self._real.execute(sql, *args, **kwargs)

            def __getattr__(self, name):
                # Delegate attribute access to the real connection.  We do
                # NOT make this async — async methods are normal attributes
                # on aiosqlite connections (you call them and then await the
                # returned coroutine).  Returning the attribute unchanged
                # preserves that.
                attr = getattr(self._real, name)
                return attr

        class _FailingConnCM:
            """Async context manager that yields a failing connection."""
            def __init__(self):
                self._cm = original_connection_method()

            async def __aenter__(self):
                real_conn = await self._cm.__aenter__()
                return _FailingConn(real_conn)

            async def __aexit__(self, *args):
                return await self._cm.__aexit__(*args)

        def patched_connection():
            return _FailingConnCM()

        pool.connection = patched_connection  # type: ignore[assignment]

        try:
            with patch(
                "backend.api.arrange._twilio_create_call",
                return_value=call_sid_to_return,
            ), patch(
                "backend.call_supervisor._twilio_hangup_call",
                return_value=True,
            ) as mock_hangup:
                # arrange_ai_call should re-raise the DB error after teardown.
                with pytest.raises(
                    RuntimeError,
                    match="simulated orphan-window DB failure",
                ):
                    await arrange_lead(
                        client, lead_id=lead_id, session=admin_session
                    )
        finally:
            pool.connection = original_connection_method  # type: ignore[assignment]

        # 1. Hangup was triggered by the supervisor (with the right CallSid).
        mock_hangup.assert_called_once()
        assert mock_hangup.call_args[0][0] == call_sid_to_return

        # 2. Lead is Failed (failure_reason=error).
        import aiosqlite

        conn = await aiosqlite.connect(str(pool.db_path))
        conn.row_factory = sqlite3.Row
        async with conn.execute(
            "SELECT status, failure_reason FROM leads WHERE id = ?",
            (lead_id,),
        ) as cur:
            lead = await cur.fetchone()
        await conn.close()

        assert lead["status"] == "Failed"
        assert lead["failure_reason"] == "error"

        # 3. Slot released.
        assert manager.active == 0, (
            f"Slot was not released by supervisor! active={manager.active}"
        )

        # 4. Supervisor unregistered from the registry (no orphan in-memory entry).
        from backend.call_supervisor import get_call
        assert get_call(call_sid_to_return) is None, (
            "Supervisor must unregister itself after orphan-window teardown"
        )


# ─── Test: Status callback ────────────────────────────────────────────────────────


class TestStatusCallback:
    """Status callback must update attempt and lead state correctly."""

    @pytest.mark.asyncio
    async def test_callback_no_answer_transitions_lead_to_failed(
        self,
        client: AsyncClient,
    ) -> None:
        """A 'no-answer' status callback must transition the lead to Failed.

        Issue 032: webhooks are thin signalers.  The supervisor must be
        registered (mimicking arrange_ai_call's post-Twilio-placement flow)
        so the status-callback can find it via ``get_call(call_sid)``.
        """
        pool = client._transport.app.state.pool

        async with pool.connection() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO leads
                    (name, phone, email, city, language, status, attempt_count)
                VALUES ('No Answer', '+919920000001', 'noanswer@example.com', 'Mumbai', 'English', 'Processing', 1)
                """,
            )
            lead_id = cursor.lastrowid
            await conn.execute(
                "INSERT INTO call_attempts (lead_id, attempt_number, twilio_call_sid) VALUES (?, 1, 'CA_NO_ANSWER_001');",
                (lead_id,),
            )

        # Issue 032: register the supervisor so the status-callback can signal it.
        register_supervisor_for_live_call(
            pool,
            call_sid="CA_NO_ANSWER_001",
            lead_id=lead_id,
            attempt_id=1,
        )

        response = await client.post(
            "/api/twilio/status-callback",
            data={
                "CallSid": "CA_NO_ANSWER_001",
                "CallStatus": "no-answer",
            },
        )
        assert response.status_code == 200

        import aiosqlite

        conn = await aiosqlite.connect(str(pool.db_path))
        conn.row_factory = sqlite3.Row
        async with conn.execute(
            "SELECT status, failure_reason FROM leads WHERE id = ?",
            (lead_id,),
        ) as cur:
            row = await cur.fetchone()
            assert row["status"] == "Failed"
            assert row["failure_reason"] == "no_answer"
        await conn.close()

    @pytest.mark.asyncio
    async def test_callback_completed_transitions_lead_to_completed(
        self,
        client: AsyncClient,
    ) -> None:
        """A 'completed' status callback must transition the lead to Completed.

        Issue 032: webhooks are thin signalers.  The supervisor must be
        registered (mimicking arrange_ai_call's post-Twilio-placement flow)
        so the status-callback can find it via ``get_call(call_sid)``.
        """
        pool = client._transport.app.state.pool

        async with pool.connection() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO leads
                    (name, phone, email, city, language, status, attempt_count)
                VALUES ('Completed', '+919930000001', 'completed@example.com', 'Bangalore', 'English', 'Processing', 1)
                """,
            )
            lead_id = cursor.lastrowid
            await conn.execute(
                "INSERT INTO call_attempts (lead_id, attempt_number, twilio_call_sid) VALUES (?, 1, 'CA_COMPLETED_001');",
                (lead_id,),
            )

        # Issue 032: register the supervisor so the status-callback can signal it.
        register_supervisor_for_live_call(
            pool,
            call_sid="CA_COMPLETED_001",
            lead_id=lead_id,
            attempt_id=1,
        )

        response = await client.post(
            "/api/twilio/status-callback",
            data={
                "CallSid": "CA_COMPLETED_001",
                "CallStatus": "completed",
                "AnsweredBy": "human",
            },
        )
        assert response.status_code == 200

        import aiosqlite

        conn = await aiosqlite.connect(str(pool.db_path))
        conn.row_factory = sqlite3.Row
        async with conn.execute(
            "SELECT status FROM leads WHERE id = ?",
            (lead_id,),
        ) as cur:
            row = await cur.fetchone()
            assert row["status"] == "Completed"
        await conn.close()

    # Issue 031: AMD verdict handling has moved to a dedicated
    # /api/twilio/amd-callback endpoint.  Tests for that behavior now live in
    # the TestAmdCallback class below.

    @pytest.mark.asyncio
    async def test_callback_stale_call_sid_is_ignored(
        self,
        client: AsyncClient,
    ) -> None:
        """A callback for an unknown CallSid (no supervisor registered) must
        return 200 without action — startup sweep is the authoritative resolver."""
        response = await client.post(
            "/api/twilio/status-callback",
            data={
                "CallSid": "CA_STALE_999999",
                "CallStatus": "completed",
            },
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_callback_already_resolved_attempt_is_ignored(
        self,
        client: AsyncClient,
    ) -> None:
        """Issue 032: 'already resolved' is now expressed by the supervisor
        being absent from the registry (it unregistered itself at the end of
        its teardown).  Webhooks hit the no-supervisor path and return 200."""
        pool = client._transport.app.state.pool

        async with pool.connection() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO leads
                    (name, phone, email, city, language, status,
                     failure_reason, attempt_count)
                VALUES ('Already Resolved', '+919960000001', 'resolved@example.com',
                        'Pune', 'English', 'Failed', 'no_answer', 1)
                """,
            )
            lead_id = cursor.lastrowid
            await conn.execute(
                "INSERT INTO call_attempts (lead_id, attempt_number, twilio_call_sid, outcome, ended_at) VALUES (?, 1, 'CA_RESOLVED_001', 'completed', datetime('now'));",
                (lead_id,),
            )

        # No supervisor registered — webhook hits the no-supervisor path.
        response = await client.post(
            "/api/twilio/status-callback",
            data={
                "CallSid": "CA_RESOLVED_001",
                "CallStatus": "completed",
            },
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_callback_unknown_call_sid_returns_200(
        self,
        client: AsyncClient,
    ) -> None:
        """A callback for an unknown CallSid must be ignored (not crash)."""
        response = await client.post(
            "/api/twilio/status-callback",
            data={
                "CallSid": "CA_STALE_999999",
                "CallStatus": "completed",
            },
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_callback_already_resolved_attempt_is_ignored_returns_200(
        self,
        client: AsyncClient,
    ) -> None:
        """A callback for an already-resolved attempt must be ignored."""
        pool = client._transport.app.state.pool

        async with pool.connection() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO leads
                    (name, phone, email, city, language, status, attempt_count)
                VALUES ('Already Resolved', '+919960000001', 'resolved@example.com', 'Pune', 'English', 'Completed', 1)
                """,
            )
            lead_id = cursor.lastrowid
            await conn.execute(
                "INSERT INTO call_attempts (lead_id, attempt_number, twilio_call_sid, outcome, ended_at) VALUES (?, 1, 'CA_RESOLVED_001', 'completed', datetime('now'));",
                (lead_id,),
            )

        response = await client.post(
            "/api/twilio/status-callback",
            data={
                "CallSid": "CA_RESOLVED_001",
                "CallStatus": "completed",
            },
        )
        assert response.status_code == 200


# ─── Test: Concurrency slot release on terminal callback (P4) ─────────────────────


class TestConcurrencyRelease:
    """
    Issue 032: slot release is owned by the CallSupervisor's teardown
    sequence.  Webhooks are thin signalers — they look up the supervisor and
    call ``trigger_teardown`` which releases the slot.  These tests verify
    that the slot is released exactly once when a terminal webhook signals
    a registered supervisor.
    """

    @pytest.mark.asyncio
    async def test_slot_released_on_failed_callback(
        self,
        client: AsyncClient,
    ) -> None:
        """A 'no-answer' callback signalling a registered supervisor must
        release the concurrency slot via the supervisor's teardown."""
        pool = client._transport.app.state.pool

        from backend.api.arrange import get_concurrency_manager

        manager = get_concurrency_manager()
        manager.reset()

        async with pool.connection() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO leads
                    (name, phone, email, city, language, status, attempt_count)
                VALUES ('Slot Release', '+919970000001', 'slot@example.com', 'Delhi', 'English', 'Processing', 1)
                """,
            )
            lead_id = cursor.lastrowid
            await conn.execute(
                "INSERT INTO call_attempts (lead_id, attempt_number, twilio_call_sid) VALUES (?, 1, 'CA_SLOT_RELEASE_001');",
                (lead_id,),
            )

        # Acquire a slot to simulate arrange taking one.
        manager.acquire()
        assert manager.active == 1

        # Issue 032: register the supervisor — slot ownership transfers to it.
        register_supervisor_for_live_call(
            pool,
            call_sid="CA_SLOT_RELEASE_001",
            lead_id=lead_id,
            attempt_id=1,
        )

        response = await client.post(
            "/api/twilio/status-callback",
            data={
                "CallSid": "CA_SLOT_RELEASE_001",
                "CallStatus": "no-answer",
            },
        )
        assert response.status_code == 200

        assert manager.active == 0, (
            f"Slot was not released! active={manager.active}"
        )

    @pytest.mark.asyncio
    async def test_slot_released_on_completed_callback(
        self,
        client: AsyncClient,
    ) -> None:
        """A 'completed' callback signalling a registered supervisor must
        release the concurrency slot via the supervisor's teardown."""
        pool = client._transport.app.state.pool

        from backend.api.arrange import get_concurrency_manager

        manager = get_concurrency_manager()
        manager.reset()

        async with pool.connection() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO leads
                    (name, phone, email, city, language, status, attempt_count)
                VALUES ('Slot Release Complete', '+919980000001', 'slotcomplete@example.com', 'Mumbai', 'English', 'Processing', 1)
                """,
            )
            lead_id = cursor.lastrowid
            await conn.execute(
                "INSERT INTO call_attempts (lead_id, attempt_number, twilio_call_sid) VALUES (?, 1, 'CA_SLOT_COMPLETE_001');",
                (lead_id,),
            )

        manager.acquire()
        assert manager.active == 1

        # Issue 032: register the supervisor — slot ownership transfers to it.
        register_supervisor_for_live_call(
            pool,
            call_sid="CA_SLOT_COMPLETE_001",
            lead_id=lead_id,
            attempt_id=1,
        )

        response = await client.post(
            "/api/twilio/status-callback",
            data={
                "CallSid": "CA_SLOT_COMPLETE_001",
                "CallStatus": "completed",
                "AnsweredBy": "human",
            },
        )
        assert response.status_code == 200

        assert manager.active == 0


# ─── Test: Arrange with different lead statuses ────────────────────────────────────


class TestArrangeDifferentStatuses:
    """Arrange endpoint must handle leads in different states correctly."""

    @pytest.mark.asyncio
    async def test_arrange_new_lead_succeeds(
        self,
        twilio_stub,
        client: AsyncClient,
        admin_session: dict,
    ) -> None:
        """A lead with status='New' must be arrangeable."""
        pool = client._transport.app.state.pool
        lead_id = await _insert_lead(
            pool,
            phone="+919990000001",
            status="New",
            attempt_count=0,
        )

        response = await arrange_lead(client, lead_id=lead_id, session=admin_session)

        assert response.status_code in (200, 409)

    @pytest.mark.asyncio
    async def test_arrange_completed_lead_is_allowed(
        self,
        twilio_stub,
        client: AsyncClient,
        admin_session: dict,
    ) -> None:
        """A lead with status='Completed' must be arrangeable (retry allowed)."""
        pool = client._transport.app.state.pool
        lead_id = await _insert_lead(
            pool,
            phone="+919991000001",
            status="Completed",
            attempt_count=1,
        )

        response = await arrange_lead(client, lead_id=lead_id, session=admin_session)

        assert response.status_code in (200, 409)

    @pytest.mark.asyncio
    async def test_arrange_failed_lead_is_allowed(
        self,
        twilio_stub,
        client: AsyncClient,
        admin_session: dict,
    ) -> None:
        """A lead with status='Failed' must be arrangeable (retry allowed)."""
        pool = client._transport.app.state.pool
        lead_id = await _insert_lead(
            pool,
            phone="+919992000001",
            status="Failed",
            attempt_count=1,
        )

        response = await arrange_lead(client, lead_id=lead_id, session=admin_session)

        assert response.status_code in (200, 409)


# ─── Test: Arrange response shape ────────────────────────────────────────────────


class TestArrangeResponseShape:
    """Arrange endpoint must return the correct response shape."""

    @pytest.mark.asyncio
    async def test_success_response_has_required_fields(
        self,
        twilio_stub,
        client: AsyncClient,
        admin_session: dict,
    ) -> None:
        """A successful arrange response must include message, attempt_id, and call_sid."""
        pool = client._transport.app.state.pool
        lead_id = await _insert_lead(pool, phone="+919993000001", attempt_count=0)

        response = await arrange_lead(client, lead_id=lead_id, session=admin_session)

        if response.status_code == 200:
            data = response.json()
            assert "message" in data
            assert "attempt_id" in data
            assert "call_sid" in data
            assert isinstance(data["attempt_id"], int)


# ─── Test: Multiple concurrent arranges (P3 verification) ──────────────────────────


class TestArrangeConcurrencyInterleaving:
    """Test arrange under concurrent conditions (P3 atomic idempotency)."""

    @pytest.mark.asyncio
    async def test_multiple_arranges_respect_capacity(
        self,
        twilio_stub,
        client: AsyncClient,
        admin_session: dict,
    ) -> None:
        """When many arrange requests arrive simultaneously, at most capacity succeed."""
        pool = client._transport.app.state.pool

        from backend.api.arrange import get_concurrency_manager

        manager = get_concurrency_manager()
        manager.reset()

        # Insert many leads.
        lead_ids = []
        for i in range(10):
            lead_id = await _insert_lead(
                pool,
                phone=f"+9199{i:07d}",
                status="New",
                attempt_count=0,
            )
            lead_ids.append(lead_id)

        # Fire many arrange requests concurrently.
        tasks = [
            arrange_lead(client, lead_id=lead_id, session=admin_session)
            for lead_id in lead_ids
        ]
        responses = await asyncio.gather(*tasks)

        # Count successes and rejections.
        successes = sum(1 for r in responses if r.status_code == 200)
        rejections = sum(1 for r in responses if r.status_code == 409)

        assert successes + rejections == len(lead_ids)
        assert successes <= manager.capacity
        assert manager.active <= manager.capacity

        # Clean up.
        for _ in range(manager.active):
            manager.release()


# ─── Test: Placeholder TwiML endpoint (P6) ─────────────────────────────────────


class TestPlaceholderTwiML:
    """GET /api/twilio/placeholder-twiml must return valid TwiML XML."""

    @pytest.mark.asyncio
    async def test_placeholder_twiml_returns_xml(
        self,
        client: AsyncClient,
    ) -> None:
        """The placeholder TwiML endpoint must return XML with Content-Type application/xml."""
        response = await client.get("/api/twilio/placeholder-twiml")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/xml")
        assert b"<?xml" in response.content
        assert b"<Response>" in response.content
        assert b"</Response>" in response.content

    @pytest.mark.asyncio
    async def test_placeholder_twiml_accepts_post(
        self,
        client: AsyncClient,
    ) -> None:
        """
        HITL REGRESSION (Issue 030, BUG #2): Twilio fetches the TwiML `url` via
        HTTP POST by default. The endpoint was originally GET-only, so the live
        call got 405 and the customer heard Twilio's "application error" instead
        of the greeting. POST must return the same valid TwiML.
        """
        response = await client.post("/api/twilio/placeholder-twiml")

        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("application/xml")
        assert b"<Response>" in response.content


# ─── Test: status callback must ignore NON-terminal Twilio events (HITL) ──────────


class TestStatusCallbackNonTerminal:
    """
    HITL REGRESSION (Issue 030, BUG #1): Twilio emits non-terminal callbacks
    (initiated / ringing / in-progress) *before* the call ends. The outcome
    classifier maps any non-terminal status to Failed/error via its in-flight
    fallback, so resolving on an early callback wrongly killed a live call on the
    first ring. The handler must ignore non-terminal callbacks and leave the
    attempt live (outcome NULL, lead Processing).
    """

    @pytest.mark.parametrize("non_terminal_status", ["initiated", "ringing", "in-progress"])
    @pytest.mark.asyncio
    async def test_non_terminal_callback_does_not_resolve(
        self,
        client: AsyncClient,
        non_terminal_status: str,
    ) -> None:
        pool = client._transport.app.state.pool

        async with pool.connection() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO leads
                    (name, phone, email, city, language, status, attempt_count)
                VALUES ('Live Call', '+919970000001', 'live@example.com', 'Mumbai', 'English', 'Processing', 1)
                """,
            )
            lead_id = cursor.lastrowid
            await conn.execute(
                "INSERT INTO call_attempts (lead_id, attempt_number, twilio_call_sid) "
                "VALUES (?, 1, 'CA_NON_TERMINAL_001');",
                (lead_id,),
            )

        response = await client.post(
            "/api/twilio/status-callback",
            data={"CallSid": "CA_NON_TERMINAL_001", "CallStatus": non_terminal_status},
        )
        assert response.status_code == 200

        import aiosqlite

        conn = await aiosqlite.connect(str(pool.db_path))
        conn.row_factory = sqlite3.Row
        async with conn.execute(
            "SELECT outcome, ended_at FROM call_attempts WHERE twilio_call_sid = 'CA_NON_TERMINAL_001'",
        ) as cur:
            attempt = await cur.fetchone()
        async with conn.execute(
            "SELECT status, failure_reason FROM leads WHERE id = ?", (lead_id,),
        ) as cur:
            lead = await cur.fetchone()
        await conn.close()

        # The call is still live — nothing should be resolved.
        assert attempt["outcome"] is None, (
            f"{non_terminal_status}: attempt was wrongly resolved to {attempt['outcome']!r}"
        )
        assert attempt["ended_at"] is None
        assert lead["status"] == "Processing"
        assert lead["failure_reason"] is None


# ─── Test: _twilio_hangup_call helper (Issue 031) ───────────────────────────────


class TestTwilioHangupCall:
    """
    Tests for the synchronous ``_twilio_hangup_call`` helper.

    The helper is the server-side half of the AMD callback: it sets the live
    Twilio call's status to "completed" so Twilio ends the call.  This stops
    the ring / talk-to-voicemail / credit burn when AMD reports machine/fax.
    """

    def test_hangup_returns_true_on_success(self) -> None:
        """Successful calls.update(status='completed') must return True."""
        mock_calls = MagicMock()
        mock_calls.update = MagicMock(return_value=MagicMock())

        mock_client = MagicMock()
        # MagicMock chaining: client.calls(sid) -> mock_calls; mock_calls.update(...) -> ok
        mock_client.calls = MagicMock(return_value=mock_calls)

        with patch("twilio.rest.Client", return_value=mock_client):
            ok = _twilio_hangup_call(
                call_sid="CA_HANGUP_OK_001",
                twilio_account_sid="AC fake",
                twilio_auth_token="fake",
            )

        assert ok is True
        mock_client.calls.assert_called_once_with("CA_HANGUP_OK_001")
        mock_calls.update.assert_called_once_with(status="completed")

    def test_hangup_returns_false_on_twilio_api_error(self) -> None:
        """A TwilioRestException from calls.update must return False (graceful)."""
        from twilio.base.exceptions import TwilioRestException

        mock_calls = MagicMock()
        mock_calls.update.side_effect = TwilioRestException(
            msg="Call not found", status=404,
            uri="/2010-04-01/Accounts/ACfake/Calls/CA_NOT_FOUND.json",
        )
        mock_client = MagicMock()
        mock_client.calls = MagicMock(return_value=mock_calls)

        with patch("twilio.rest.Client", return_value=mock_client):
            ok = _twilio_hangup_call(
                call_sid="CA_MISSING",
                twilio_account_sid="AC fake",
                twilio_auth_token="fake",
            )

        assert ok is False

    def test_hangup_returns_false_when_credentials_missing(self) -> None:
        """Missing Twilio credentials must return False without calling the API."""
        ok = _twilio_hangup_call(
            call_sid="CA_NO_CREDS",
            twilio_account_sid="",
            twilio_auth_token="",
        )
        assert ok is False

    def test_hangup_returns_false_when_call_sid_empty(self) -> None:
        """An empty CallSid must return False without calling the API."""
        ok = _twilio_hangup_call(
            call_sid="",
            twilio_account_sid="AC fake",
            twilio_auth_token="fake",
        )
        assert ok is False


# ─── Test: arrange_ai_call passes the lead's language to CallSupervisor ────────


class TestArrangePassesLanguage:
    """
    REGRESSION (Issue 035 follow-up): ``arrange_ai_call`` must read the
    lead's ``language`` from the database and pass it to ``CallSupervisor``
    so the Deepgram reader picks the correct model + language hints.

    Without this, every call would silently fall back to a default — and
    since English leads would land on the multilingual model, English
    transcription accuracy would measurably degrade.
    """

    @pytest.mark.asyncio
    async def test_arrange_passes_lead_language_to_supervisor(
        self,
        client: AsyncClient,
        admin_session: dict,
    ) -> None:
        """
        Insert a Hinglish lead, arrange a call, and assert the supervisor
        was constructed with language='Hinglish'.  We patch the
        ``CallSupervisor`` constructor in the arrange module's namespace
        so the real one is not invoked, but we capture the kwargs.

        We also mock ``_twilio_create_call`` to return a synthetic CallSid
        so the handler reaches the supervisor-construction step (otherwise
        the Twilio path errors out before the supervisor exists).
        """
        pool = client._transport.app.state.pool
        lead_id = await _insert_lead(
            pool,
            phone="+919900000088",
            email="hinglish-arrange@example.com",
            language="Hinglish",
            attempt_count=0,
        )

        from unittest.mock import patch

        captured_kwargs: dict = {}

        def _capturing_supervisor(*args, **kwargs):
            captured_kwargs.update(kwargs)
            # Return a sentinel that won't break the rest of arrange.
            from backend.call_supervisor import CallSupervisor
            cm = __import__("backend.api.arrange", fromlist=["get_concurrency_manager"]).get_concurrency_manager()
            return CallSupervisor(
                lead_id=kwargs.get("lead_id", 1),
                attempt_id=kwargs.get("attempt_id", 1),
                call_sid=kwargs.get("call_sid", "CA_SENTINEL"),
                language=kwargs["language"],
                concurrency_manager=cm,
                pool=pool,
            )

        with patch(
            "backend.api.arrange._twilio_create_call",
            return_value="CA_HINGLISH_SYNTH_001",
        ), patch(
            "backend.api.arrange.CallSupervisor",
            side_effect=_capturing_supervisor,
        ):
            response = await arrange_lead(client, lead_id=lead_id, session=admin_session)

        assert response.status_code == 200, response.text
        assert "language" in captured_kwargs, (
            "arrange_ai_call must pass language= to CallSupervisor"
        )
        assert captured_kwargs["language"] == "Hinglish", (
            f"Expected language='Hinglish' for a Hinglish lead, "
            f"got {captured_kwargs['language']!r}"
        )

    @pytest.mark.asyncio
    async def test_arrange_passes_english_lead_to_supervisor(
        self,
        client: AsyncClient,
        admin_session: dict,
    ) -> None:
        """
        Symmetric English case: an English lead must be wired to a
        supervisor with language='English' so the STT model is
        flux-general-en (not the multilingual one).
        """
        pool = client._transport.app.state.pool
        lead_id = await _insert_lead(
            pool,
            phone="+919900000077",
            email="english-arrange@example.com",
            language="English",
            attempt_count=0,
        )

        from unittest.mock import patch

        captured_kwargs: dict = {}

        def _capturing_supervisor(*args, **kwargs):
            captured_kwargs.update(kwargs)
            from backend.call_supervisor import CallSupervisor
            cm = __import__("backend.api.arrange", fromlist=["get_concurrency_manager"]).get_concurrency_manager()
            return CallSupervisor(
                lead_id=kwargs.get("lead_id", 1),
                attempt_id=kwargs.get("attempt_id", 1),
                call_sid=kwargs.get("call_sid", "CA_EN_SENTINEL"),
                language=kwargs["language"],
                concurrency_manager=cm,
                pool=pool,
            )

        with patch(
            "backend.api.arrange._twilio_create_call",
            return_value="CA_ENGLISH_SYNTH_001",
        ), patch(
            "backend.api.arrange.CallSupervisor",
            side_effect=_capturing_supervisor,
        ):
            response = await arrange_lead(client, lead_id=lead_id, session=admin_session)

        assert response.status_code == 200, response.text
        assert captured_kwargs.get("language") == "English"


# ─── Test: Async AMD callback endpoint (Issue 031) ──────────────────────────────


class TestAmdCallback:
    """
    Tests for the dedicated ``POST /api/twilio/amd-callback`` endpoint.

    The endpoint receives Twilio's async Answered Machine Detection verdict
    out-of-band.  On a non-human verdict (machine_* or fax), the live Twilio
    call is hung up via the REST API, the attempt is resolved as
    Failed/no_answer, and the concurrency slot is released.  On a human or
    unknown verdict, no action is taken — the call is allowed to proceed.
    """

    async def _insert_live_call(
        self,
        pool,
        *,
        call_sid: str,
        phone: str,
        email: str,
        name: str = "AMD Test Lead",
    ) -> tuple[int, int]:
        """Insert a Processing lead + a live call_attempt with the given CallSid."""
        async with pool.connection() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO leads
                    (name, phone, email, city, language, status, attempt_count)
                VALUES (?, ?, ?, 'Mumbai', 'English', 'Processing', 1)
                """,
                (name, phone, email),
            )
            lead_id = cursor.lastrowid
            cursor = await conn.execute(
                """
                INSERT INTO call_attempts (lead_id, attempt_number, twilio_call_sid)
                VALUES (?, 1, ?)
                """,
                (lead_id, call_sid),
            )
            attempt_id = cursor.lastrowid
        return lead_id, attempt_id

    @pytest.mark.asyncio
    @pytest.mark.parametrize("verdict", ["machine_start", "machine_end_beep", "fax"])
    async def test_machine_verdict_hangs_up_and_resolves_as_no_answer(
        self,
        client: AsyncClient,
        verdict: str,
    ) -> None:
        """
        A machine/fax AMD verdict must:
          1. Trigger an active hangup (client.calls(sid).update(status='completed')).
          2. Resolve the attempt as outcome='no_answer', ended_at set.
          3. Transition the lead to Failed (failure_reason='no_answer').
          4. Release the concurrency slot.

        Issue 032: the AMD callback now signals the CallSupervisor, which
        owns the teardown.  The test pre-registers a supervisor (mimicking
        arrange_ai_call's post-Twilio-placement flow) and then verifies the
        supervisor ran the full teardown sequence in response to the callback.
        """
        pool = client._transport.app.state.pool

        from backend.api.arrange import get_concurrency_manager

        manager = get_concurrency_manager()
        manager.reset()
        manager.acquire()  # simulate the slot taken by arrange
        assert manager.active == 1

        call_sid = f"CA_AMD_{verdict.upper()}_001"
        lead_id, attempt_id = await self._insert_live_call(
            pool,
            call_sid=call_sid,
            phone="+919960000001",
            email=f"amd-{verdict}@example.com",
        )

        # Issue 032: register the supervisor so the AMD callback can signal it.
        register_supervisor_for_live_call(
            pool,
            call_sid=call_sid,
            lead_id=lead_id,
            attempt_id=attempt_id,
        )

        # Mock the Twilio Client to avoid hitting the real Twilio API.
        # Issue 032: the supervisor imports _twilio_hangup_call directly from
        # backend.call_supervisor, so we patch the symbol where the
        # supervisor actually resolves it.
        with patch("backend.call_supervisor._twilio_hangup_call", return_value=True) as mock_hangup:
            response = await client.post(
                "/api/twilio/amd-callback",
                data={"CallSid": call_sid, "AnsweredBy": verdict},
            )

        assert response.status_code == 200, response.text

        # 1. Hangup was attempted with the right CallSid (by the supervisor).
        mock_hangup.assert_called_once()
        hangup_args = mock_hangup.call_args[0]
        assert hangup_args[0] == call_sid

        # 2 + 3. Attempt resolved as no_answer, lead transitioned to Failed.
        import aiosqlite

        conn = await aiosqlite.connect(str(pool.db_path))
        conn.row_factory = sqlite3.Row
        async with conn.execute(
            "SELECT outcome, ended_at FROM call_attempts WHERE id = ?",
            (attempt_id,),
        ) as cur:
            attempt = await cur.fetchone()
        async with conn.execute(
            "SELECT status, failure_reason FROM leads WHERE id = ?",
            (lead_id,),
        ) as cur:
            lead = await cur.fetchone()
        await conn.close()

        assert attempt["outcome"] == "no_answer"
        assert attempt["ended_at"] is not None
        assert lead["status"] == "Failed"
        assert lead["failure_reason"] == "no_answer"

        # 4. Concurrency slot released by the supervisor's teardown.
        assert manager.active == 0, (
            f"Slot was not released! active={manager.active}"
        )

        # 5. Supervisor unregistered itself from the active-calls registry.
        from backend.call_supervisor import get_call
        assert get_call(call_sid) is None, (
            "Supervisor must unregister itself after teardown"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("verdict", ["human", "unknown"])
    async def test_human_or_unknown_verdict_is_no_op(
        self,
        client: AsyncClient,
        verdict: str,
    ) -> None:
        """
        A human or unknown AMD verdict must NOT touch the DB or release the
        slot — the call is allowed to proceed to the consultation pipeline.
        """
        pool = client._transport.app.state.pool

        from backend.api.arrange import get_concurrency_manager

        manager = get_concurrency_manager()
        manager.reset()
        manager.acquire()
        assert manager.active == 1

        call_sid = f"CA_AMD_{verdict.upper()}_001"
        lead_id, attempt_id = await self._insert_live_call(
            pool,
            call_sid=call_sid,
            phone="+919970000001",
            email=f"amd-{verdict}@example.com",
        )

        # Hangup must NOT be called for human/unknown.
        with patch("backend.api.arrange._twilio_hangup_call") as mock_hangup:
            response = await client.post(
                "/api/twilio/amd-callback",
                data={"CallSid": call_sid, "AnsweredBy": verdict},
            )

        assert response.status_code == 200
        mock_hangup.assert_not_called()

        # Attempt and lead remain live.
        import aiosqlite

        conn = await aiosqlite.connect(str(pool.db_path))
        conn.row_factory = sqlite3.Row
        async with conn.execute(
            "SELECT outcome FROM call_attempts WHERE id = ?",
            (attempt_id,),
        ) as cur:
            attempt = await cur.fetchone()
        async with conn.execute(
            "SELECT status FROM leads WHERE id = ?",
            (lead_id,),
        ) as cur:
            lead = await cur.fetchone()
        await conn.close()

        assert attempt["outcome"] is None
        assert lead["status"] == "Processing"

        # Slot was NOT released.
        assert manager.active == 1, (
            f"Slot was wrongly released for {verdict} verdict! active={manager.active}"
        )

    @pytest.mark.asyncio
    async def test_machine_verdict_with_hangup_failure_still_resolves_attempt(
        self,
        client: AsyncClient,
    ) -> None:
        """
        If the Twilio hangup itself fails (returns False), the attempt must
        still be resolved as Failed/no_answer and the slot released.  Hangup
        is best-effort credit-saving, not the source of truth — the database
        is the source of truth.

        Issue 032: the supervisor runs the hangup; the DB persistence step is
        independent and must complete even if hangup fails.
        """
        pool = client._transport.app.state.pool

        from backend.api.arrange import get_concurrency_manager

        manager = get_concurrency_manager()
        manager.reset()
        manager.acquire()

        call_sid = "CA_AMD_HANGUP_FAIL_001"
        lead_id, attempt_id = await self._insert_live_call(
            pool,
            call_sid=call_sid,
            phone="+919980000001",
            email="hangup-fail@example.com",
        )

        # Issue 032: register the supervisor so the AMD callback can signal it.
        register_supervisor_for_live_call(
            pool,
            call_sid=call_sid,
            lead_id=lead_id,
            attempt_id=attempt_id,
        )

        # Hangup fails (e.g., Twilio API down or CallSid already gone).
        # Patch the supervisor's reference (not arrange.py) since the
        # supervisor imports _twilio_hangup_call directly.
        with patch("backend.call_supervisor._twilio_hangup_call", return_value=False):
            response = await client.post(
                "/api/twilio/amd-callback",
                data={"CallSid": call_sid, "AnsweredBy": "machine_start"},
            )

        assert response.status_code == 200

        import aiosqlite

        conn = await aiosqlite.connect(str(pool.db_path))
        conn.row_factory = sqlite3.Row
        async with conn.execute(
            "SELECT outcome FROM call_attempts WHERE id = ?",
            (attempt_id,),
        ) as cur:
            attempt = await cur.fetchone()
        async with conn.execute(
            "SELECT status, failure_reason FROM leads WHERE id = ?",
            (lead_id,),
        ) as cur:
            lead = await cur.fetchone()
        await conn.close()

        assert attempt["outcome"] == "no_answer"
        assert lead["status"] == "Failed"
        assert lead["failure_reason"] == "no_answer"
        # Supervisor released the slot even though hangup failed.
        assert manager.active == 0
        # And unregistered itself.
        from backend.call_supervisor import get_call
        assert get_call(call_sid) is None

    @pytest.mark.asyncio
    async def test_teardown_runs_in_specified_order(
        self,
        client: AsyncClient,
    ) -> None:
        """
        SEQUENCE INVARIANT (Issue 032):

        CallSupervisor.trigger_teardown runs the fixed sequence:
            1. End the Twilio call (hangup).
            ...
            4. Persist outcome to DB (resolve).
            ...
            6. Release concurrency slot.
            7. Unregister from active-calls registry.

        This test pins the ordering between hangup and DB resolve.

        Note on race-condition hardening
        ─────────────────────────────────
        In Issue 031 follow-up, the AMD callback wrote the DB BEFORE
        triggering the hangup so the trailing `completed` status callback
        safely hit the "already resolved → ignore" idempotent guard.  In
        Issue 032, that guard is replaced by the supervisor's one-shot
        latch: the trailing status callback hits the latch and no-ops.  So
        the ordering is no longer load-bearing for race prevention — but we
        still pin it as a sequence invariant so future refactors don't
        silently reorder the supervisor's teardown body.
        """
        pool = client._transport.app.state.pool

        from backend.api.arrange import get_concurrency_manager

        manager = get_concurrency_manager()
        manager.reset()
        manager.acquire()

        call_sid = "CA_AMD_ORDER_001"
        lead_id, attempt_id = await self._insert_live_call(
            pool,
            call_sid=call_sid,
            phone="+919992000001",
            email="amd-order@example.com",
        )

        # Register the supervisor so the AMD callback has something to signal.
        register_supervisor_for_live_call(
            pool,
            call_sid=call_sid,
            lead_id=lead_id,
            attempt_id=attempt_id,
        )

        # Shared recorder: each wrapped function appends a tag on entry.
        call_order: list[str] = []

        # Patch the helpers at their home in backend.call_supervisor.
        # Wrapping preserves the original behavior so the post-condition
        # assertions (DB write, slot release) hold.
        from backend.call_supervisor import _resolve_failed_attempt

        original_resolve = _resolve_failed_attempt

        async def _resolve_wrapper(*args, **kwargs):
            call_order.append("resolve")
            return await original_resolve(*args, **kwargs)

        def _hangup_wrapper(*args, **kwargs):
            call_order.append("hangup")
            return True  # mock the actual Twilio REST call

        with patch(
            "backend.call_supervisor._resolve_failed_attempt",
            side_effect=_resolve_wrapper,
        ) as mock_resolve, patch(
            "backend.call_supervisor._twilio_hangup_call",
            side_effect=_hangup_wrapper,
        ) as mock_hangup:
            response = await client.post(
                "/api/twilio/amd-callback",
                data={"CallSid": call_sid, "AnsweredBy": "machine_start"},
            )

        assert response.status_code == 200

        # Both must have been called exactly once.
        assert mock_resolve.call_count == 1
        assert mock_hangup.call_count == 1

        # THE sequence invariant: hangup (Step 1) runs before resolve (Step 4).
        # See CallSupervisor.trigger_teardown docstring for the full ordering.
        assert call_order == ["hangup", "resolve"], (
            f"Teardown sequence violated. "
            f"Got call order: {call_order}. "
            f"The supervisor must run hangup (Step 1) BEFORE DB resolve (Step 4) "
            f"per the Issue 032 fixed sequence.  Note: the race-vs-trailing-status-callback "
            f"is now handled by the supervisor's one-shot latch, not by this ordering."
        )

        # And the DB must end up in the expected terminal state — this also
        # confirms the wrapped _resolve_failed_attempt actually executed
        # against the real pool.
        import aiosqlite

        conn = await aiosqlite.connect(str(pool.db_path))
        conn.row_factory = sqlite3.Row
        async with conn.execute(
            "SELECT outcome FROM call_attempts WHERE id = ?",
            (attempt_id,),
        ) as cur:
            attempt = await cur.fetchone()
        async with conn.execute(
            "SELECT status, failure_reason FROM leads WHERE id = ?",
            (lead_id,),
        ) as cur:
            lead = await cur.fetchone()
        await conn.close()

        assert attempt["outcome"] == "no_answer"
        assert lead["status"] == "Failed"
        assert lead["failure_reason"] == "no_answer"
        assert manager.active == 0

    @pytest.mark.asyncio
    async def test_stale_call_sid_is_ignored(
        self,
        client: AsyncClient,
    ) -> None:
        """An AMD callback for an unknown CallSid (no supervisor in the
        registry) must be ignored.  Returns 200 without DB action — startup
        sweep is the authoritative resolver for orphaned leads."""
        with patch("backend.call_supervisor._twilio_hangup_call") as mock_hangup:
            response = await client.post(
                "/api/twilio/amd-callback",
                data={"CallSid": "CA_AMD_STALE_999999", "AnsweredBy": "machine_start"},
            )

        assert response.status_code == 200
        mock_hangup.assert_not_called()

    @pytest.mark.asyncio
    async def test_already_resolved_attempt_is_ignored(
        self,
        client: AsyncClient,
    ) -> None:
        """
        An AMD callback for an attempt that is already terminal must be a no-op.
        Covers the race where the status callback resolved the attempt first.
        """
        pool = client._transport.app.state.pool

        from backend.api.arrange import get_concurrency_manager

        manager = get_concurrency_manager()
        manager.reset()

        call_sid = "CA_AMD_ALREADY_RESOLVED_001"
        async with pool.connection() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO leads
                    (name, phone, email, city, language, status,
                     failure_reason, attempt_count)
                VALUES ('Already Resolved', '+919990000001', 'ar@example.com',
                        'Mumbai', 'English', 'Failed', 'no_answer', 1)
                """,
            )
            lead_id = cursor.lastrowid
            cursor = await conn.execute(
                """
                INSERT INTO call_attempts
                    (lead_id, attempt_number, twilio_call_sid, outcome, ended_at)
                VALUES (?, 1, ?, 'no_answer', datetime('now'))
                """,
                (lead_id, call_sid),
            )

        with patch("backend.call_supervisor._twilio_hangup_call") as mock_hangup:
            response = await client.post(
                "/api/twilio/amd-callback",
                data={"CallSid": call_sid, "AnsweredBy": "machine_start"},
            )

        assert response.status_code == 200
        mock_hangup.assert_not_called()
        # Slot untouched.
        assert manager.active == 0

    @pytest.mark.asyncio
    async def test_missing_fields_returns_200_without_action(
        self,
        client: AsyncClient,
    ) -> None:
        """
        A malformed callback missing CallSid or AnsweredBy must return 200
        (Twilio retries on non-2xx — we don't want to trigger retry storms on
        our own bug).  No DB action.
        """
        with patch("backend.call_supervisor._twilio_hangup_call") as mock_hangup:
            response = await client.post(
                "/api/twilio/amd-callback",
                data={"CallSid": "CA_MISSING_FIELDS"},  # no AnsweredBy
            )

        assert response.status_code == 200
        mock_hangup.assert_not_called()

    @pytest.mark.asyncio
    async def test_status_callback_no_longer_routes_voicemail_to_failed(
        self,
        client: AsyncClient,
    ) -> None:
        """
        Issue 031: the status-callback endpoint must NOT act on AnsweredBy
        anymore.  If Twilio sends a status callback with AnsweredBy='machine'
        (which can happen if AMD arrives first), the status callback routes
        purely by terminal CallStatus.  This test pins down the new separation
        so future refactors don't accidentally re-couple AMD to status.

        Issue 032: the supervisor's outcome classifier does NOT see AnsweredBy
        (it always passes None), so a machine verdict arriving via status
        callback does NOT trigger Failed/no_answer — only Completed via the
        matcher_usable=True placeholder.  This is correct because in
        production the AMD callback fires first and would have already hung
        up + Failed this attempt; this test simply confirms the status
        callback path doesn't independently act on AnsweredBy.
        """
        pool = client._transport.app.state.pool

        from backend.api.arrange import get_concurrency_manager

        manager = get_concurrency_manager()
        manager.reset()
        manager.acquire()

        call_sid = "CA_STATUS_WITH_AMD_FIELD_001"
        async with pool.connection() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO leads
                    (name, phone, email, city, language, status, attempt_count)
                VALUES ('Status With AMD', '+919991000001', 'sa@example.com',
                        'Mumbai', 'English', 'Processing', 1)
                """,
            )
            lead_id = cursor.lastrowid
            await conn.execute(
                "INSERT INTO call_attempts (lead_id, attempt_number, twilio_call_sid) "
                "VALUES (?, 1, ?)",
                (lead_id, call_sid),
            )

        # Issue 032: register the supervisor so the status callback can signal.
        register_supervisor_for_live_call(
            pool,
            call_sid=call_sid,
            lead_id=lead_id,
            attempt_id=1,
        )

        # Terminal 'completed' status + AnsweredBy='machine' — but on the status
        # callback, AMD is ignored.  The supervisor's classifier returns
        # Completed/completed_timeout (matcher_usable=True placeholder), so
        # the lead ends up Completed, NOT Failed.
        response = await client.post(
            "/api/twilio/status-callback",
            data={
                "CallSid": call_sid,
                "CallStatus": "completed",
                "AnsweredBy": "machine_start",
            },
        )

        assert response.status_code == 200

        import aiosqlite

        conn = await aiosqlite.connect(str(pool.db_path))
        conn.row_factory = sqlite3.Row
        async with conn.execute(
            "SELECT status, failure_reason FROM leads WHERE id = ?",
            (lead_id,),
        ) as cur:
            lead = await cur.fetchone()
        await conn.close()

        # Lead is Completed (NOT Failed) — the status callback no longer
        # treats AMD verdicts as no_answer.  The supervisor's classifier
        # ignores AnsweredBy (always passes None to the classifier).
        assert lead["status"] == "Completed", (
            "Status callback must not act on AnsweredBy. AMD is handled "
            "exclusively by /api/twilio/amd-callback."
        )
        # And the supervisor ran its teardown — slot released.
        assert manager.active == 0


# ─── Test: STT task is NOT spawned during arrange (Issue 035 follow-up) ────────


class TestArrangeDoesNotSpawnStt:
    """
    REGRESSION (Issue 035 follow-up): the Deepgram STT reader must be spawned
    by ``/api/twilio/media-stream`` on Twilio's ``start`` event, NOT by
    ``arrange_ai_call``.

    Why: the call goes through several phases (placed → ringing → answered →
    start event) before the customer is actually on the line.  If we open
    the STT socket during arrange, the socket idles while the phone is
    ringing and burns through the bounded fast-reconnect budget before any
    audio ever flows — risking a hangup before the call is even answered.
    """

    @pytest.mark.asyncio
    async def test_arrange_does_not_call_start_background_tasks(
        self,
        twilio_stub,
        client: AsyncClient,
        admin_session: dict,
    ) -> None:
        """
        arrange_ai_call must NOT call ``supervisor.start_background_tasks()``
        — the call belongs to media_stream.py now.  We assert by patching
        the supervisor's method and verifying it was not invoked after a
        successful arrange.
        """
        pool = client._transport.app.state.pool
        lead_id = await _insert_lead(
            pool,
            phone="+919900000099",
            email="stt-not-spawned@example.com",
            attempt_count=0,
        )

        from unittest.mock import patch

        # Patch the high-level method to catch any direct call from arrange.
        with patch(
            "backend.call_supervisor.CallSupervisor.start_background_tasks",
            autospec=True,
        ) as mock_start:
            response = await arrange_lead(client, lead_id=lead_id, session=admin_session)

        assert response.status_code == 200, response.text

        # arrange must not have spawned the STT task — that is media_stream's job.
        mock_start.assert_not_called()
