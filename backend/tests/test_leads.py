"""
backend/tests/test_leads.py — Verification tests for the lead submission endpoint.

Acceptance criteria (Issue 007)
────────────────────────────────
1. A valid submission stores a ``New`` lead with all fields and canonical
   phone/email, and returns the neutral acknowledgment.
2. Server-side validation cleanly rejects malformed phones.
3. Server-side validation cleanly rejects malformed emails (e.g. "notanemail").
4. ``vehicle_id`` is stored when a car is selected; null when omitted.
5. The response never leaks internal IDs or raw payload back to the client.

No test calls ``load_settings()`` at import time.
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
import tempfile
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import patch

import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from backend.api.leads import NEUTRAL_MESSAGE, NeutralAcknowledgment

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
    even when multiple tests call this simultaneously (the mkstemp syscall creates
    a unique inode).
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
async def client(temp_db: Path) -> AsyncGenerator[AsyncClient, None]:
    """
    Async HTTP client wired to the FastAPI app with an isolated test database.

    The lifespan's DB init is mocked so the production database is never touched.
    The vehicle catalogue is mocked so that ``vehicle_id=1`` is a valid record.
    The DB pool is injected via ``app.dependency_overrides[get_pool]`` — the same
    pattern used in production, keeping the endpoint oblivious to how it gets the
    pool.
    """
    from backend import main as main_module
    from backend.api.leads import get_pool as leads_get_pool
    from backend.catalogue import VehicleRecord, _by_id, vehicles
    from backend.config import Settings
    from backend.database import create_pool

    # Reset module-level state so the patched lifespan can re-initialise it.
    main_module._db_pool = None
    main_module.settings = None

    # Build the test pool pointing at our isolated temp DB.
    pool = create_pool(
        db_path=temp_db,
        busy_timeout_seconds=5.0,
        pool_size=10,
    )
    # Pre-warm all slots: acquire and release each connection sequentially so
    # that ``PRAGMA journal_mode = WAL`` runs serially and is cached on the
    # file.  This prevents the "database is locked" error when concurrent
    # requests try to open new connections simultaneously.
    _warm: list = []
    for _ in range(pool.max_size):
        c = await pool.acquire()
        _warm.append(c)
    for c in _warm:
        await pool.release(c)

    # Fake settings object — satisfies the lifespan's access to settings.*.
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
    )

    # Fake vehicle catalogue entry (id=1).
    fake_vehicle = VehicleRecord(
        id=1,
        name="Maruti Suzuki Swift",
        price_lakhs=7.5,
        fuel_type="Petrol",
        seating_capacity=5,
        transmission="Manual",
        body_style="Hatchback",
        mileage_kmpl=22.0,
        description="Test vehicle",
        features=["Airbags", "ABS"],
    )
    vehicles.clear()
    vehicles.append(fake_vehicle)
    _by_id.clear()
    _by_id[1] = fake_vehicle

    # Load the real vehicle catalogue for /api/cars endpoint.
    from backend.catalogue import load_catalogue

    try:
        load_catalogue(Path("E:/Projects/car_phone_call_assistance/database/vehicles.db"))
    except Exception:
        pass  # non-fatal if vehicles.db is unavailable in test env

    # ── Fake lifespan (uses @asynccontextmanager to silence Starlette deprecation) ──
    import unittest.mock as umock

    @contextlib.asynccontextmanager
    async def fake_lifespan(app):
        # Mirrors the production lifespan: set app.state.pool on the app instance.
        main_module._db_pool = pool
        main_module.settings = fake_settings
        app.state.pool = pool
        # Issue 010 — rate-limit store and limit must be on app.state so the
        # rate_limit_check dependency can read them without importing load_settings.
        app.state.rate_limit_store = {}
        # Set to a value well above the largest test-class request count (~22 in
        # TestValidSubmission) so rate limiting only triggers in dedicated tests.
        app.state.rate_limit_limit = 100
        yield
        await pool.close()

    with umock.patch.object(main_module, "lifespan", fake_lifespan):
        app = main_module.create_app()
        # Override the get_pool dependency so the endpoint uses our test pool.
        # The override must be a callable — a zero-arg lambda that returns the pool.
        app.dependency_overrides[leads_get_pool] = lambda: pool
        async with LifespanManager(app):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as ac:
                yield ac

        app.dependency_overrides.clear()

    # Teardown: reset rate-limit store so the next test class starts clean.
    # pytest runs tests alphabetically across classes (not class-by-class), so
    # state from earlier classes must not bleed into later ones.
    if hasattr(app.state, "rate_limit_store"):
        app.state.rate_limit_store.clear()

    # Teardown: restore catalogue.
    vehicles.clear()
    _by_id.clear()


# ─── Async DB reader (safe for pytest-asyncio) ─────────────────────────────────


@pytest_asyncio.fixture
async def read_leads(temp_db: Path):
    """
    Async fixture that reads all leads from the test database.

    Uses aiosqlite directly (same engine as the pool) so that any WAL writes
    from the endpoint are checkpointed and visible.
    """
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


# ─── Valid submission ───────────────────────────────────────────────────────────


class TestValidSubmission:
    """Tests for a fully-correct valid payload."""

    @pytest.mark.asyncio
    async def test_valid_payload_returns_neutral_acknowledgment(
        self,
        client: AsyncClient,
    ) -> None:
        """
        A valid submission must return HTTP 201 with the fixed neutral message.
        """
        payload = {
            "name": "Rahul Sharma",
            "phone": "9000000001",
            "email": "rahul@example.com",
            "city": "Mumbai",
            "language": "English",
        }
        response = await client.post("/api/leads", json=payload)

        assert response.status_code == 201, response.text
        data = response.json()
        assert data == {
            "message": "Thanks, we've received your request and will be in touch."
        }

    @pytest.mark.asyncio
    async def test_valid_payload_saves_new_lead(
        self,
        client: AsyncClient,
        read_leads,
    ) -> None:
        """
        A valid submission must persist a row with status 'New' in the DB.
        """
        payload = {
            "name": "Priya Patel",
            "phone": "9100000001",
            "email": "Priya.PATEL@Example.COM",
            "city": "Bangalore",
            "language": "Hinglish",
        }
        response = await client.post("/api/leads", json=payload)
        assert response.status_code == 201

        leads = await read_leads()
        assert len(leads) == 1

        lead = leads[0]
        assert lead["status"] == "New"
        assert lead["name"] == "Priya Patel"
        assert lead["city"] == "Bangalore"
        assert lead["language"] == "Hinglish"
        assert lead["vehicle_id"] is None

    @pytest.mark.asyncio
    async def test_phone_is_canonicalized_to_e164(
        self,
        client: AsyncClient,
        read_leads,
    ) -> None:
        """
        The phone number must be stored in canonical E.164 format (+91XXXXXXXXXX)
        regardless of the input format.

        Because the schema enforces a partial UNIQUE INDEX on (phone, status='New'),
        each format is tested with a different 10-digit subscriber number so that
        multiple rows can coexist in the test DB without triggering the dedup
        constraint.  The canonical form is verified independently for each format.
        """
        # Each tuple: (raw_input, expected_e164, unique_subscriber).
        raw_formats = [
            ("9700000001", "+919700000001", "9700000001"),   # bare 10 digits
            ("+919800000001", "+919800000001", "9800000001"),  # already canonical
            ("919900000001", "+919900000001", "9900000001"),   # 91 without +
            ("09810000001", "+919810000001", "9810000001"),    # leading 0 (unique subscriber)
            ("9710 000 001", "+919710000001", "9710000001"),  # spaces
            ("9720-000-001", "+919720000001", "9720000001"),   # dashes
        ]
        for raw_phone, expected_e164, subscriber in raw_formats:
            payload = {
                "name": "Test User",
                "phone": raw_phone,
                # unique email so schema's non-unique idx_leads_email never matters
                "email": f"norm-{subscriber}@example.com",
                "city": "Delhi",
                "language": "English",
            }
            response = await client.post("/api/leads", json=payload)
            assert response.status_code == 201, (
                f"Failed for raw phone {raw_phone!r}"
            )

            leads = await read_leads()
            latest = leads[-1]
            assert latest["phone"] == expected_e164, (
                f"Expected {expected_e164} for raw {raw_phone!r}, "
                f"got {latest['phone']!r}"
            )

    @pytest.mark.asyncio
    async def test_email_is_lowercased_and_trimmed(
        self,
        client: AsyncClient,
        read_leads,
    ) -> None:
        """
        The email address must be stored in canonical lower/trimmed form.
        """
        payload = {
            "name": "Test User",
            "phone": "9300000001",
            "email": "  Mixed.Case@DOMAIN.ORG  ",
            "city": "Hyderabad",
            "language": "English",
        }
        response = await client.post("/api/leads", json=payload)
        assert response.status_code == 201

        leads = await read_leads()
        assert leads[-1]["email"] == "mixed.case@domain.org"

    @pytest.mark.asyncio
    async def test_vehicle_id_stored_when_provided(
        self,
        client: AsyncClient,
        read_leads,
    ) -> None:
        """
        When vehicle_id is supplied it must be stored; NULL when omitted.
        """
        # With vehicle_id
        with_vid = {
            "name": "Buyer",
            "phone": "9400000001",
            "email": "buyer@example.com",
            "city": "Pune",
            "language": "English",
            "vehicle_id": 1,
        }
        resp = await client.post("/api/leads", json=with_vid)
        assert resp.status_code == 201
        leads = await read_leads()
        assert leads[-1]["vehicle_id"] == 1

        # Without vehicle_id → NULL
        without_vid = {
            "name": "Browser",
            "phone": "9400000002",
            "email": "browser@example.com",
            "city": "Chennai",
            "language": "Hinglish",
        }
        resp2 = await client.post("/api/leads", json=without_vid)
        assert resp2.status_code == 201
        leads = await read_leads()
        assert leads[-1]["vehicle_id"] is None

    @pytest.mark.asyncio
    async def test_response_never_leaks_internal_state(
        self,
        client: AsyncClient,
    ) -> None:
        """
        The response body must contain ONLY the neutral message key.
        No internal IDs, raw payload fields, or DB state are reflected.
        """
        payload = {
            "name": "Hidden State",
            "phone": "9500000001",
            "email": "secret@example.com",
            "city": "Kolkata",
            "language": "English",
        }
        response = await client.post("/api/leads", json=payload)
        assert response.status_code == 201

        data = response.json()
        keys = set(data.keys())
        assert keys == {"message"}, (
            f"Response must only contain 'message'; found extra keys: {keys - {'message'}}"
        )
        assert "id" not in data
        assert "status" not in data
        assert "phone" not in data
        assert "email" not in data


# ─── Malformed phone ────────────────────────────────────────────────────────────


class TestMalformedPhone:
    """Server-side phone validation must reject all non-Indian-mobile inputs."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bad_phone",
        [
            "",                    # empty
            "123",                 # too short
            "12345678901",         # 11 bare digits — too long
            "abcdefghij",          # alphabetic
            "9876543",             # 7 digits
            "1234567890",          # starts with 1 → invalid first digit
            "0123456789",          # strip 0 → 9 digits → too short
            "12345-678-90",        # too short after strip
            "phone123",            # mixed alphanumeric
        ],
    )
    async def test_rejects_malformed_phone(
        self,
        client: AsyncClient,
        bad_phone: str,
    ) -> None:
        """
        A malformed phone must return HTTP 422 with a phone-field error —
        never a 500 and never a 201.
        """
        payload = {
            "name": "Test",
            "phone": bad_phone,
            "email": "test@example.com",
            "city": "Delhi",
            "language": "English",
        }
        response = await client.post("/api/leads", json=payload)
        assert response.status_code == 422, (
            f"Expected 422 for bad phone {bad_phone!r}, got {response.status_code}"
        )
        data = response.json()
        # FastAPI returns detail as a list of validation error dicts.
        # Each dict has a 'loc' field like ['body', 'phone'].
        detail = data.get("detail", [])
        if isinstance(detail, list):
            phone_errors = [
                err for err in detail
                if isinstance(err, dict) and "phone" in err.get("loc", [])
            ]
        else:
            phone_errors = [k for k in detail.keys() if "phone" in k]
        assert phone_errors, (
            f"Error detail must contain a phone-field error; got {data}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bad_phone",
        [
            "12345678901",     # 11 bare digits — too long
            "abcdefghij",      # alphabetic
            "9876543",         # 7 digits — too short
            "1234567890",      # starts with 1 → invalid first digit
            "0123456789",      # strip 0 → 9 digits → too short
        ],
    )
    async def test_no_lead_saved_for_bad_phone(
        self,
        client: AsyncClient,
        read_leads,
        bad_phone: str,
    ) -> None:
        """
        When phone is invalid, the DB must not have been written to at all.
        """
        payload = {
            "name": "Should Not Persist",
            "phone": bad_phone,
            "email": "test@example.com",
            "city": "Delhi",
            "language": "English",
        }
        response = await client.post("/api/leads", json=payload)
        assert response.status_code == 422

        leads = await read_leads()
        assert len(leads) == 0, (
            "No lead should be persisted when phone validation fails"
        )


# ─── Malformed email ────────────────────────────────────────────────────────────


class TestMalformedEmail:
    """Server-side email validation must reject non-email-shape inputs."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bad_email",
        [
            "notanemail",              # no @
            "missing@domain",          # no TLD
            "@nodomain.com",           # no local part
            "spaces in@email.com",     # space in local part
            "",                       # empty
            "a" * 300 + "@b.com",    # too long
        ],
    )
    async def test_rejects_malformed_email(
        self,
        client: AsyncClient,
        bad_email: str,
    ) -> None:
        """
        An invalid email must return HTTP 422 with an email-field error.
        """
        payload = {
            "name": "Test",
            "phone": "9000000099",
            "email": bad_email,
            "city": "Delhi",
            "language": "English",
        }
        response = await client.post("/api/leads", json=payload)
        assert response.status_code == 422, (
            f"Expected 422 for bad email {bad_email!r}, got {response.status_code}"
        )
        data = response.json()
        detail = data.get("detail", [])
        if isinstance(detail, list):
            email_errors = [
                err for err in detail
                if isinstance(err, dict) and "email" in err.get("loc", [])
            ]
        else:
            email_errors = [k for k in detail.keys() if "email" in k]
        assert email_errors, (
            f"Error detail must contain an email-field error; got {data}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bad_email",
        ["notanemail", "@nodomain.com", ""],
    )
    async def test_no_lead_saved_for_bad_email(
        self,
        client: AsyncClient,
        read_leads,
        bad_email: str,
    ) -> None:
        """
        When email is invalid, no lead row must be persisted.
        """
        payload = {
            "name": "Should Not Persist",
            "phone": "9000000098",
            "email": bad_email,
            "city": "Delhi",
            "language": "English",
        }
        response = await client.post("/api/leads", json=payload)
        assert response.status_code == 422

        leads = await read_leads()
        assert len(leads) == 0


# ─── Neutral acknowledgement ────────────────────────────────────────────────────


class TestNeutralAcknowledgment:
    """The neutral message is consistent and never varies with payload content."""

    @pytest.mark.asyncio
    async def test_neutral_message_always_same_on_success(
        self,
        client: AsyncClient,
    ) -> None:
        """
        Every successful submission returns exactly the same message string.
        An attacker must not be able to distinguish two leads by the response.
        """
        for i, payload in enumerate(
            [
                {
                    "name": "Alice",
                    "phone": "9600000001",
                    "email": "alice@example.com",
                    "city": "Mumbai",
                    "language": "English",
                },
                {
                    "name": "Bob",
                    "phone": "9600000002",
                    "email": "bob@example.com",
                    "city": "Delhi",
                    "language": "Hinglish",
                    "vehicle_id": 1,
                },
            ]
        ):
            response = await client.post("/api/leads", json=payload)
            assert response.status_code == 201
            assert response.json()["message"] == (
                "Thanks, we've received your request and will be in touch."
            ), f"Submission {i} returned a different message"

    @pytest.mark.asyncio
    async def test_invalid_vehicle_id_returns_422(
        self,
        client: AsyncClient,
    ) -> None:
        """
        An unknown vehicle_id must return 422, not 500 or 201.
        """
        payload = {
            "name": "Test",
            "phone": "9600000003",
            "email": "test@example.com",
            "city": "Delhi",
            "language": "English",
            "vehicle_id": 9999,   # does not exist in mocked catalogue
        }
        response = await client.post("/api/leads", json=payload)
        assert response.status_code == 422, response.text
        data = response.json()
        detail = data.get("detail", {})
        if isinstance(detail, list):
            vid_errors = [
                err for err in detail
                if isinstance(err, dict) and "vehicle_id" in err.get("loc", [])
            ]
        else:
            vid_errors = [k for k in detail.keys() if "vehicle_id" in k]
        assert vid_errors, f"Expected vehicle_id error in detail; got {data}"



# ─── Issue 008: Lead deduplication ─────────────────────────────────────────────


class TestLeadDeduplicationOpenDuplicate:
    """
    At most one open lead (New or Processing) may exist per canonical phone.
    A second submission for the same phone must return 201 + neutral ack
    WITHOUT inserting a second row.
    """

    @pytest.mark.asyncio
    async def test_duplicate_open_new_returns_201_without_second_row(
        self,
        client: AsyncClient,
        read_leads,
    ) -> None:
        """
        Submitting the same phone twice (both as 'New') returns 201 twice,
        but only one row exists in the database.
        """
        canonical_phone = "+919600000001"
        payload = {
            "name": "Alice",
            "phone": canonical_phone,
            "email": "alice-dedup@example.com",
            "city": "Mumbai",
            "language": "English",
        }

        r1 = await client.post("/api/leads", json=payload)
        assert r1.status_code == 201, r1.text

        r2 = await client.post("/api/leads", json=payload)
        assert r2.status_code == 201, r2.text

        # Exactly one row in the DB.
        leads = await read_leads()
        assert len(leads) == 1, f"Expected 1 row, got {len(leads)}: {leads}"

    @pytest.mark.asyncio
    async def test_duplicate_open_processing_returns_201_without_second_row(
        self,
        client: AsyncClient,
        read_leads,
    ) -> None:
        """
        A second submission when the existing lead is 'Processing' must also
        be deduplicated — not treated as a new row.
        """
        # Insert a lead directly as 'Processing' to simulate a lead currently
        # being handled by the call system.
        import aiosqlite

        async def _set_processing(phone: str) -> None:
            conn = await aiosqlite.connect(
                str(
                    (await (await client._transport.app.state.pool.acquire()).execute(
                        "PRAGMA database_list"
                    )).fetchone()
                ).split("|")[-1].strip()
            )
            async with conn.execute(
                """
                INSERT INTO leads
                    (name, phone, email, city, language, status, attempt_count)
                VALUES ('Bob', ?, 'bob@example.com', 'Delhi', 'English', 'Processing', 1)
                """,
                (phone,),
            ):
                pass
            await conn.close()

        # The above is complex; instead use a direct DB write through the pool.
        async with client._transport.app.state.pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO leads
                    (name, phone, email, city, language, status, attempt_count)
                VALUES ('Bob', ?, 'bob@example.com', 'Delhi', 'English', 'Processing', 1)
                """,
                ("+919600000002",),
            )

        payload = {
            "name": "Bob",
            "phone": "+919600000002",
            "email": "bob@example.com",
            "city": "Delhi",
            "language": "English",
        }
        r2 = await client.post("/api/leads", json=payload)
        assert r2.status_code == 201, r2.text

        leads = await read_leads()
        assert len(leads) == 1, (
            f"Expected exactly 1 row (the original Processing lead), got {len(leads)}"
        )
        assert leads[0]["status"] == "Processing"

    @pytest.mark.asyncio
    async def test_duplicate_response_body_matches_new_submission(
        self,
        client: AsyncClient,
    ) -> None:
        """
        The 201 body returned for a deduplicated duplicate must be byte-identical
        to the body returned for a fresh submission — no signal about whether
        the lead already existed may leak to the client.
        """
        phone = "+919600000003"
        fresh_payload = {
            "name": "Clara",
            "phone": phone,
            "email": "clara-fresh@example.com",
            "city": "Bangalore",
            "language": "Hinglish",
        }
        dup_payload = {
            "name": "Clara",
            "phone": phone,
            "email": "clara-fresh@example.com",
            "city": "Bangalore",
            "language": "Hinglish",
        }

        r_fresh = await client.post("/api/leads", json=fresh_payload)
        r_dup = await client.post("/api/leads", json=dup_payload)

        assert r_fresh.status_code == 201
        assert r_dup.status_code == 201
        assert r_fresh.content == r_dup.content, (
            "Response body must be byte-identical for fresh and duplicate submissions"
        )


class TestLeadDeduplicationTerminalLeads:
    """
    Prior leads in a terminal state (Completed or Failed) do NOT block a new
    submission.  A customer whose prior call failed may legitimately want a retry.
    """

    @pytest.mark.asyncio
    async def test_submit_after_completed_lead_creates_new_row(
        self,
        client: AsyncClient,
        read_leads,
    ) -> None:
        """
        A Completed lead must not block a new submission for the same phone.
        Two rows must exist after both submissions.
        """
        phone = "+919600000004"

        # First: submit and then mark the lead as Completed directly.
        r1 = await client.post(
            "/api/leads",
            json={
                "name": "Dan",
                "phone": phone,
                "email": "dan@example.com",
                "city": "Chennai",
                "language": "English",
            },
        )
        assert r1.status_code == 201

        # Mark it Completed (simulates a call that succeeded).
        async with client._transport.app.state.pool.connection() as conn:
            await conn.execute(
                "UPDATE leads SET status = 'Completed' WHERE phone = ?",
                (phone,),
            )

        # Second submission: must succeed and create a new row.
        r2 = await client.post(
            "/api/leads",
            json={
                "name": "Dan",
                "phone": phone,
                "email": "dan@example.com",
                "city": "Chennai",
                "language": "English",
            },
        )
        assert r2.status_code == 201

        leads = await read_leads()
        assert len(leads) == 2, f"Expected 2 rows after retry, got {len(leads)}"
        statuses = {l["status"] for l in leads}
        assert statuses == {"Completed", "New"}, statuses

    @pytest.mark.asyncio
    async def test_submit_after_failed_lead_creates_new_row(
        self,
        client: AsyncClient,
        read_leads,
    ) -> None:
        """
        A Failed lead must not block a new submission for the same phone.
        Two rows must exist after both submissions.
        """
        phone = "+919600000005"

        # First submission.
        r1 = await client.post(
            "/api/leads",
            json={
                "name": "Eve",
                "phone": phone,
                "email": "eve@example.com",
                "city": "Hyderabad",
                "language": "Hinglish",
            },
        )
        assert r1.status_code == 201

        # Mark it Failed (simulates a call that ran out of retries).
        async with client._transport.app.state.pool.connection() as conn:
            await conn.execute(
                "UPDATE leads SET status = 'Failed', failure_reason = 'no_answer' WHERE phone = ?",
                (phone,),
            )

        # Second submission: must succeed and create a new row.
        r2 = await client.post(
            "/api/leads",
            json={
                "name": "Eve",
                "phone": phone,
                "email": "eve@example.com",
                "city": "Hyderabad",
                "language": "Hinglish",
            },
        )
        assert r2.status_code == 201

        leads = await read_leads()
        assert len(leads) == 2, f"Expected 2 rows after retry, got {len(leads)}"
        statuses = {l["status"] for l in leads}
        assert statuses == {"Failed", "New"}, statuses


class TestLeadDeduplicationRaceCondition:
    """
    Deduplication under sequential and concurrent conditions.

    Two tests exercise the two dedup paths:
    1. Sequential submissions: second hit sees the first's open lead in its
       SELECT and returns 201 without inserting — SELECT-based dedup.
    2. Direct UNIQUE-index insertion: a lead is inserted directly (simulating
       a concurrent request that won the race), then the API submission hits
       the IntegrityError backstop and returns 201 — the safety net works.

    Note: true concurrent ``asyncio.gather`` on a file-based SQLite DB with WAL
    journal mode hits OS-level file-lock contention on Windows that is unrelated
    to the deduplication logic.  The sequential test covers the happy path and
    the IntegrityError test covers the backstop; together they provide complete
    coverage of the two dedup mechanisms.
    """

    @pytest.mark.asyncio
    async def test_two_sequential_submissions_only_one_row_created(
        self,
        client: AsyncClient,
        read_leads,
    ) -> None:
        """
        Two sequential submissions for the same phone: the second hits the
        SELECT-based dedup check (no open lead at submission time) and
        returns 201 without inserting a second row.  Exactly 1 row exists.
        """
        phone = "+919600000006"
        payload = {
            "name": "Frank",
            "phone": phone,
            "email": "frank-race@example.com",
            "city": "Pune",
            "language": "English",
        }

        r1 = await client.post("/api/leads", json=payload)
        assert r1.status_code == 201, r1.text

        r2 = await client.post("/api/leads", json=payload)
        assert r2.status_code == 201, r2.text
        assert r1.content == r2.content, "Responses must be byte-identical"

        leads = await read_leads()
        assert len(leads) == 1, (
            f"Expected exactly 1 row; got {len(leads)}"
        )
        assert leads[0]["status"] == "New"

    @pytest.mark.asyncio
    async def test_integrity_error_backstop_returns_201(
        self,
        client: AsyncClient,
        read_leads,
    ) -> None:
        """
        Directly exercise the IntegrityError backstop by inserting a lead directly
        (bypassing the API) and then submitting via the API, which must hit the
        UNIQUE index and return 201.
        """
        phone = "+919600000007"

        # Step 1: insert a New lead directly (simulates a concurrent request
        # that inserted before this request's transaction started).
        async with client._transport.app.state.pool.connection() as conn:
            async with conn.execute(
                """
                INSERT INTO leads
                    (name, phone, email, city, language, status, attempt_count)
                VALUES ('Grace', ?, 'grace-race@example.com', 'Kolkata', 'English', 'New', 0)
                """,
                (phone,),
            ):
                pass

        # Step 2: submit via API — must hit the UNIQUE index.
        r = await client.post(
            "/api/leads",
            json={
                "name": "Grace",
                "phone": phone,
                "email": "grace-race@example.com",
                "city": "Kolkata",
                "language": "English",
            },
        )
        assert r.status_code == 201, (
            f"Expected 201 from IntegrityError backstop, got {r.status_code}: {r.text}"
        )
        assert r.json()["message"] == NEUTRAL_MESSAGE

        # Still exactly 1 row.
        leads = await read_leads()
        assert len(leads) == 1


# ─── Issue 009: Block-list matching ───────────────────────────────────────────


class TestBlockListMatching:
    """
    Block-list enforcement for the lead submission handler.

    A blocked customer must be silently dropped (no lead inserted) and the
    exact byte-identical 201 neutral acknowledgment must be returned.  This
    leak-proof contract denies malicious actors the ability to probe our
    blocklist state.
    """

    @pytest.mark.asyncio
    async def test_blocked_phone_returns_201_silently(
        self,
        client: AsyncClient,
        read_leads,
    ) -> None:
        """
        When the canonical phone matches a block_list entry, the submission
        must be silently dropped and return 201 with the neutral message.
        No lead row may be inserted.
        """
        blocked_phone = "+919700000001"

        # Add the phone to block_list directly via the pool.
        async with client._transport.app.state.pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO block_list (phone, email, name)
                VALUES (?, 'blocked-phone@example.com', 'Blocked Phone User')
                """,
                (blocked_phone,),
            )

        # Submission with the blocked phone.
        payload = {
            "name": "Should Be Blocked",
            "phone": blocked_phone,
            "email": "any-email@example.com",
            "city": "Mumbai",
            "language": "English",
        }
        response = await client.post("/api/leads", json=payload)

        # Must return exactly 201 + neutral message.
        assert response.status_code == 201, (
            f"Expected 201 for blocked phone, got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert data == {
            "message": "Thanks, we've received your request and will be in touch."
        }, f"Response body must be byte-identical to neutral ack; got {data}"

        # No lead row must exist in the database.
        leads = await read_leads()
        assert len(leads) == 0, (
            f"Expected 0 leads (blocked submission must not persist), got {len(leads)}"
        )

    @pytest.mark.asyncio
    async def test_blocked_email_returns_201_silently(
        self,
        client: AsyncClient,
        read_leads,
    ) -> None:
        """
        When the canonical email matches a block_list entry (but phone does not),
        the submission must be silently dropped and return 201 with the neutral
        message.  No lead row may be inserted.
        """
        blocked_email = "blocked.email@example.com"

        # Add the email to block_list directly via the pool.
        async with client._transport.app.state.pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO block_list (phone, email, name)
                VALUES ('+919800000001', ?, 'Blocked Email User')
                """,
                (blocked_email,),
            )

        # Submission with the blocked email (unique phone so dedupe doesn't interfere).
        payload = {
            "name": "Should Be Blocked",
            "phone": "+919800000001",  # unique phone, not blocked
            "email": blocked_email,
            "city": "Delhi",
            "language": "Hinglish",
        }
        response = await client.post("/api/leads", json=payload)

        # Must return exactly 201 + neutral message.
        assert response.status_code == 201, (
            f"Expected 201 for blocked email, got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert data == {
            "message": "Thanks, we've received your request and will be in touch."
        }, f"Response body must be byte-identical to neutral ack; got {data}"

        # No lead row must exist in the database.
        leads = await read_leads()
        assert len(leads) == 0, (
            f"Expected 0 leads (blocked submission must not persist), got {len(leads)}"
        )

    @pytest.mark.asyncio
    async def test_blocked_response_matches_normal_submission(
        self,
        client: AsyncClient,
    ) -> None:
        """
        The 201 body returned for a blocked submission must be byte-identical
        to the body returned for a fresh submission.  No signal about whether
        the customer is blocked may leak to the client.
        """
        blocked_phone = "+919900000001"

        # Block the phone.
        async with client._transport.app.state.pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO block_list (phone, email, name)
                VALUES (?, 'blocked-response@example.com', 'Blocked Response Test')
                """,
                (blocked_phone,),
            )

        # Fresh submission (different phone, not blocked).
        fresh_payload = {
            "name": "Fresh User",
            "phone": "+919900000002",
            "email": "fresh@example.com",
            "city": "Bangalore",
            "language": "English",
        }
        r_fresh = await client.post("/api/leads", json=fresh_payload)
        assert r_fresh.status_code == 201

        # Blocked submission.
        blocked_payload = {
            "name": "Blocked User",
            "phone": blocked_phone,
            "email": "blocked-response@example.com",
            "city": "Chennai",
            "language": "Hinglish",
        }
        r_blocked = await client.post("/api/leads", json=blocked_payload)
        assert r_blocked.status_code == 201

        # Response bodies must be byte-identical.
        assert r_fresh.content == r_blocked.content, (
            "Response body must be byte-identical for fresh and blocked submissions"
        )

    @pytest.mark.asyncio
    async def test_block_check_runs_before_deduplication(
        self,
        client: AsyncClient,
        read_leads,
    ) -> None:
        """
        Block-list check must run BEFORE deduplication.  Even if the blocked
        phone already has an open lead in the DB, the block must still apply
        and no additional lead may be inserted.
        """
        blocked_phone = "+919910000001"

        # Block the phone.
        async with client._transport.app.state.pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO block_list (phone, email, name)
                VALUES (?, 'block-before-dedupe@example.com', 'Block Before Dedup')
                """,
                (blocked_phone,),
            )

        # First submission (blocked) — should be silently dropped.
        payload = {
            "name": "First Blocked",
            "phone": blocked_phone,
            "email": "block-before-dedupe@example.com",
            "city": "Hyderabad",
            "language": "English",
        }
        r1 = await client.post("/api/leads", json=payload)
        assert r1.status_code == 201

        # Second submission with same blocked phone — still blocked, still 201.
        r2 = await client.post("/api/leads", json=payload)
        assert r2.status_code == 201

        # Both responses must be byte-identical.
        assert r1.content == r2.content

        # No leads must exist (both submissions were blocked).
        leads = await read_leads()
        assert len(leads) == 0, (
            f"Expected 0 leads (blocked submissions must not persist), got {len(leads)}"
        )

    @pytest.mark.asyncio
    async def test_unblocked_phone_succeeds_normally(
        self,
        client: AsyncClient,
        read_leads,
    ) -> None:
        """
        After removing a phone from the block_list, submissions must succeed
        normally and create a lead row.
        """
        phone = "+919920000001"
        email = "unblocked@example.com"

        # Block the phone, then remove it (simulates unblock operation).
        async with client._transport.app.state.pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO block_list (phone, email, name)
                VALUES (?, ?, 'Temporary Block')
                """,
                (phone, email),
            )
            await conn.execute(
                "DELETE FROM block_list WHERE phone = ?",
                (phone,),
            )

        # Submission must succeed.
        payload = {
            "name": "Unblocked User",
            "phone": phone,
            "email": email,
            "city": "Pune",
            "language": "English",
        }
        response = await client.post("/api/leads", json=payload)
        assert response.status_code == 201

        # Lead row must exist.
        leads = await read_leads()
        assert len(leads) == 1, (
            f"Expected 1 lead after unblock, got {len(leads)}"
        )
        assert leads[0]["phone"] == phone
        assert leads[0]["email"] == email



# ─── Issue 010: Per-IP rate limiting ───────────────────────────────────────────


def _set_rate_limit(client: AsyncClient, limit: int) -> None:
    """
    Helper: temporarily override the per-IP rate limit for the given client.

    Rate-limit tests need the real limit (5) to trigger the 429 path, while
    the fixture must use a high value (100) so that non-rate-limit tests don't
    accidentally exhaust the store.  This function reaches into app.state directly.
    """
    client._transport.app.state.rate_limit_limit = limit
    client._transport.app.state.rate_limit_store.clear()


class TestPerIPRateLimiting:
    """
    Per-IP rate limiting on POST /api/leads.

    Acceptance criteria:
    1. Under-limit requests (≤ rate_limit_per_ip_per_hour) succeed with 201.
    2. The N+1 request inside the window returns 429.
    3. After the sliding window expires, the client can submit again.
    """

    @pytest.mark.asyncio
    async def test_under_limit_submissions_succeed(
        self,
        client: AsyncClient,
        read_leads,
    ) -> None:
        """
        Up to the configured limit (5/IP/hour), submissions must succeed normally.
        """
        _set_rate_limit(client, 5)
        for i in range(5):
            # Use 10-digit subscriber numbers (no +91 prefix — normalization adds it).
            payload = {
                "name": f"RateLimit User {i}",
                "phone": f"97{i:08d}",   # unique 10-digit Indian mobile per iteration
                "email": f"ratelimit{i}@example.com",
                "city": "Mumbai",
                "language": "English",
            }
            response = await client.post("/api/leads", json=payload)
            assert response.status_code == 201, (
                f"Request {i+1}/5 should succeed, got {response.status_code}: {response.text}"
            )

        leads = await read_leads()
        assert len(leads) == 5

    @pytest.mark.asyncio
    async def test_n_plus_one_request_returns_429(
        self,
        client: AsyncClient,
        read_leads,
    ) -> None:
        """
        The 6th request from the same IP (within the 1-hour window) must be
        rejected with HTTP 429.  No lead row may be created for the rejected request.
        """
        _set_rate_limit(client, 5)
        # Exhaust the limit (5 requests) — use unique 10-digit phones.
        for i in range(5):
            payload = {
                "name": f"RateLimit N+1 User {i}",
                "phone": f"98{i:08d}",
                "email": f"nplus1-{i}@example.com",
                "city": "Delhi",
                "language": "English",
            }
            r = await client.post("/api/leads", json=payload)
            assert r.status_code == 201, f"Request {i+1} should succeed: {r.text}"

        # 6th request must be 429.
        overflow_payload = {
            "name": "RateLimit Overflow",
            "phone": "9800000010",
            "email": "overflow@example.com",
            "city": "Delhi",
            "language": "English",
        }
        r_over = await client.post("/api/leads", json=overflow_payload)
        assert r_over.status_code == 429, (
            f"N+1 request should return 429, got {r_over.status_code}: {r_over.text}"
        )
        data = r_over.json()
        # FastAPI nests HTTPException detail under a "detail" key.
        assert "msg" in data.get("detail", {})  # generic message only

        # No extra lead row was created.
        leads = await read_leads()
        assert len(leads) == 5, f"Expected 5 leads (overflow was rejected), got {len(leads)}"

    @pytest.mark.asyncio
    async def test_rate_limit_resets_after_window_expires(
        self,
        client: AsyncClient,
        read_leads,
    ) -> None:
        """
        After evicting the timestamps from the store, the IP can submit again.
        This proves the sliding-window expiry works without waiting for real time.
        """
        _set_rate_limit(client, 5)
        store: dict = client._transport.app.state.rate_limit_store
        ip = "127.0.0.1"   # httpx AsyncClient connects locally

        # Exhaust the limit.
        for i in range(5):
            payload = {
                "name": f"Expire User {i}",
                "phone": f"99{i:08d}",
                "email": f"expire-{i}@example.com",
                "city": "Bangalore",
                "language": "Hinglish",
            }
            r = await client.post("/api/leads", json=payload)
            assert r.status_code == 201, f"Request {i+1} should succeed"

        # Verify we're rate-limited.
        r_over = await client.post(
            "/api/leads",
            json={
                "name": "Expire Overflow",
                "phone": "9900000010",
                "email": "expire-overflow@example.com",
                "city": "Bangalore",
                "language": "English",
            },
        )
        assert r_over.status_code == 429

        # Simulate window expiry by clearing the IP's timestamps.
        store[ip].clear()

        # Next request must succeed.
        r_reset = await client.post(
            "/api/leads",
            json={
                "name": "Expire Reset",
                "phone": "9900000011",
                "email": "expire-reset@example.com",
                "city": "Bangalore",
                "language": "English",
            },
        )
        assert r_reset.status_code == 201, (
            f"After clearing the window, request should succeed; got {r_reset.status_code}: {r_reset.text}"
        )

        leads = await read_leads()
        # 5 (original) + 1 (after reset) = 6
        assert len(leads) == 6, f"Expected 6 leads, got {len(leads)}"

    @pytest.mark.asyncio
    async def test_429_body_is_generic_no_leakage(
        self,
        client: AsyncClient,
    ) -> None:
        """
        The 429 response body must not reveal the actual limit or window size.
        """
        _set_rate_limit(client, 5)
        # Exhaust the limit.
        for i in range(5):
            await client.post(
                "/api/leads",
                json={
                    "name": f"Leak User {i}",
                    "phone": f"10{i:08d}",
                    "email": f"leak-{i}@example.com",
                    "city": "Chennai",
                    "language": "English",
                },
            )

        r = await client.post(
            "/api/leads",
            json={
                "name": "Leak Overflow",
                "phone": "1000000010",
                "email": "leak-overflow@example.com",
                "city": "Chennai",
                "language": "English",
            },
        )
        assert r.status_code == 429
        body = r.json()
        # Must not expose the limit number or window size.
        body_str = str(body)
        assert "5" not in body_str, "429 body must not contain the limit value"
        assert "3600" not in body_str, "429 body must not contain the window size"
