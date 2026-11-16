"""
backend/tests/test_catalogue.py — verification tests for the in-memory vehicle catalogue.

Tests
─────
1. load_catalogue() loads exactly 35 rows.
2. get_vehicle_by_id() resolves known IDs correctly and returns None for unknown IDs.
3. A known vehicle (id=1, Tata Nexon) has all fields fully populated.
4. The features JSON column is parsed into a tuple[str, ...] at load time.
5. VehicleRecord is hashable (frozen dataclass).
6. Opening vehicles.db with ?mode=ro creates absolutely no -wal or -shm sidecar files.
7. GET /api/cars serves all 35 records directly from memory (no per-request DB query).
8. The endpoint returns ONLY {id, name} — preserving the frontend contract.

No test calls load_settings() at import time.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from backend.catalogue import VehicleRecord, get_vehicle_by_id, load_catalogue, vehicles
from backend.config import load_settings
from backend.main import create_app

# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    """
    Clear the settings cache so tests start with a clean slate.
    """
    load_settings.cache_clear()
    yield
    load_settings.cache_clear()


@pytest.fixture
def vehicles_db_path() -> Path:
    """
    Path to the real vehicles.db in the project database/ directory.
    """
    return (
        Path(__file__).parent.parent.parent
        / "database"
        / "vehicles.db"
    ).resolve()


@pytest.fixture
def database_dir(vehicles_db_path: Path) -> Path:
    """The parent directory of vehicles.db."""
    return vehicles_db_path.parent


@pytest_asyncio.fixture(scope="session")
async def client():
    """
    Async HTTP client with the full FastAPI lifespan, isolated to a temp DB.

    The lifespan is driven with ``APP_DB_PATH`` pointing at a session-scoped
    temp directory so the real ``database/`` directory is never touched.
    WAL/SHM files from the app DB end up in the temp dir and are cleaned
    up when the session ends.

    Cleanup is fully async (this fixture owns a running event loop):

      * ``session_pool`` is initialised to ``None`` BEFORE the lifespan
        enters so a startup failure leaves the outer finally with a
        well-defined fallback (``app.state.pool``).
      * The pool is closed in the outer ``finally`` block, on the existing
        event loop — no ``asyncio.run()``, no swallowing of close
        failures, no private ``_closed`` access.
      * ``app.state.pool`` is nulled after the close so the lifespan's
        own shutdown path is a no-op.
      * The temp directory is removed whenever the close succeeded,
        regardless of test-body outcome.  It is preserved only when
        ``pool.close()`` itself raised, for diagnosis.
    """
    # Session-scoped temp dir — created once, reused for all catalogue HTTP tests.
    tmpdir = Path(tempfile.mkdtemp(prefix="car_test_catalogue_"))
    app_db = tmpdir / "app.db"

    load_settings.cache_clear()
    old_env = os.environ.get("APP_DB_PATH")
    os.environ["APP_DB_PATH"] = str(app_db)

    app = create_app()

    # `session_pool` is initialised BEFORE the LifespanManager enters so
    # the outer finally block always has a well-defined value to fall
    # back on.  If startup fails before the lifespan installs the pool,
    # the fallback below reads it from `app.state.pool`.
    session_pool: object | None = None

    try:
        async with LifespanManager(app):
            # Capture the pool the lifespan installed so the fixture
            # owns its teardown deterministically.
            session_pool = getattr(app.state, "pool", None)

            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as ac:
                yield ac
    finally:
        # ── Determine the pool to close. Prefer the captured
        # session_pool; fall back to app.state.pool only if startup
        # failed before the lifespan installed the pool. ──────────
        if session_pool is None:
            pool_to_close = getattr(app.state, "pool", None)
        else:
            pool_to_close = session_pool

        # ── Close the pool on the existing event loop.  CAPTURE the
        # exception (do not re-raise here) so we can still restore the
        # environment, clear the settings cache, and decide whether to
        # preserve the temp directory for diagnosis. ─────────────
        close_error: BaseException | None = None
        try:
            if pool_to_close is not None:
                await pool_to_close.close()
        except BaseException as exc:  # noqa: BLE001
            close_error = exc

        # Always null the handle so a second close is a no-op.
        app.state.pool = None

        # ── Restore env + clear cache in ALL cases (even if close
        # raised) so subsequent tests / teardown see a clean state. ─
        if old_env is None:
            os.environ.pop("APP_DB_PATH", None)
        else:
            os.environ["APP_DB_PATH"] = old_env
        load_settings.cache_clear()

        # ── Remove the temp directory only when the close succeeded.
        # On a close failure, preserve the directory for diagnosis. ──
        if close_error is None:
            shutil.rmtree(tmpdir, ignore_errors=True)

        # Re-raise the close error AFTER the rest of the cleanup ran so
        # the test (or pytest) sees the real cause.
        if close_error is not None:
            raise close_error


# ==============================================================================
# Failure-path regression: inject an exception at the yield to prove
# the client fixture's cleanup runs correctly.
# ==============================================================================


def test_client_fixture_cleanup_handles_injected_exception(
    tmp_path: Path,
) -> None:
    """
    Failure-path regression: drive the **real** ``client`` fixture from
    a fresh subprocess, inject a ``RuntimeError`` at the yield via
    ``athrow``, and verify:

      * the subprocess exits with status 0,
      * the injected exception's message is captured inside the
        subprocess,
      * the temp directory the fixture created is removed
        (``car_test_catalogue_*`` directory absent after the run),
      * no ``coroutine … was never awaited`` / ``Event loop is closed``
        / ``PytestUnraisableExceptionWarning`` leaks are reported,
      * no ``asyncio.run()`` is invoked anywhere in the cleanup path.

    Because the fixture's lifespan opens a real aiosqlite connection
    pool, a clean exit proves the SQLite worker was closed correctly.
    """
    import subprocess
    import sys
    import textwrap

    # Use pytest's tmp_path as the subprocess TMPDIR so any leftover
    # directory the fixture creates is constrained to a known location
    # we can clean up.  The fixture's temp dir (prefix
    # ``car_test_catalogue_``) lives in the system default tmp; the
    # assertion below checks for its absence.
    driver = textwrap.dedent(
        r'''
        import asyncio
        import os
        import sys
        import warnings

        from backend.tests.test_catalogue import client


        async def main() -> None:
            # Promote unraisable exceptions to hard errors so a coroutine
            # leak would visibly fail the subprocess.
            warnings.filterwarnings("error")

            gen = client.__wrapped__()  # type: ignore[attr-defined]
            try:
                # Advance to the first yield.
                await gen.__anext__()
            except StopAsyncIteration:
                raise RuntimeError("fixture did not yield")

            # Inject a RuntimeError into the test body slot.  ``athrow``
            # propagates the exception to the yield point, which is the
            # correct way to fake a test-body failure for an async
            # generator fixture.
            injected = RuntimeError(
                "injected by failure-path regression test"
            )
            # The fixture MUST propagate the injected RuntimeError with
            # its exact message — the fixture is required to clean up
            # after a test-body failure and then re-raise.
            try:
                await gen.athrow(injected)
            except RuntimeError as exc:
                assert str(exc) == "injected by failure-path regression test", (
                    f"athrow did not propagate the injected exception: {exc!r}"
                )
                # Surface the message on stderr so the parent test can
                # assert the cleanup path actually fired.
                print(
                    f"INJECTED_PROPAGATED: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                # athrow returned a value — the fixture DID NOT propagate
                # the injected exception.  This is a contract violation.
                print(
                    "ERROR: athrow returned without raising; the fixture "
                    "did not propagate the injected RuntimeError",
                    file=sys.stderr,
                    flush=True,
                )
                sys.exit(2)
            # StopAsyncIteration is NOT a valid result here — the fixture
            # MUST propagate the injected RuntimeError, not silently
            # finish.


        if __name__ == "__main__":
            asyncio.run(main())
        '''
    )

    # Use pytest's tmp_path as the subprocess TMPDIR so any leftover
    # fixture-created directory is constrained to a known location we
    # can inspect afterwards.
    env = dict(os.environ)
    env["TMPDIR"] = str(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-W", "error", "-c", driver],
        capture_output=True,
        text=True,
        timeout=15,
        env=env,
    )

    assert proc.returncode == 0, (
        f"Failure-path subprocess exited {proc.returncode}.\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    # The injected exception's exact message must appear in stderr.
    assert "injected by failure-path regression test" in proc.stderr, (
        f"Expected the injected exception in stderr.\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    # The driver explicitly signals the propagated-RuntimeError path.
    assert "INJECTED_PROPAGATED" in proc.stderr, (
        f"Probe did not confirm RuntimeError propagation.\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )

    # The probe must NOT mention any of these — they would indicate a
    # leak the cleanup was supposed to prevent.
    forbidden = [
        "asyncio.run()",  # the only forbidden re: cleanup helpers
        "coroutine",  # generic: catches "coroutine was never awaited"
        "was never awaited",
        "Event loop is closed",
        "PytestUnraisableExceptionWarning",
        "ResourceWarning",
    ]
    combined = proc.stdout + "\n" + proc.stderr
    for marker in forbidden:
        assert marker not in combined, (
            f"Probe output contained {marker!r} (forbidden):\n{combined}"
        )

    # The fixture's temp dir (prefix ``car_test_catalogue_``) is created
    # in the system default temp location; with ``TMPDIR=tmp_path`` the
    # fixture inherits the pytest temp area, so we inspect THAT
    # directory — not the parent process's /tmp.
    leftover = sorted(
        str(p) for p in tmp_path.glob("car_test_catalogue_*")
    )
    assert not leftover, (
        f"Failure-path probe left car_test_catalogue_* directories: "
        f"{leftover}"
    )


# ==============================================================================
# Test 1 — Row count
# ==============================================================================

def test_catalogue_loads_exactly_35_rows(vehicles_db_path: Path) -> None:
    """
    load_catalogue() must populate the in-memory list with exactly 35 records.
    """
    load_catalogue(vehicles_db_path)
    assert len(vehicles) == 35, f"Expected 35 records, got {len(vehicles)}"


# ==============================================================================
# Test 2 — Known ID resolution (name only)
# ==============================================================================

def test_get_vehicle_by_id_resolves_known_ids(
    vehicles_db_path: Path, ensure_catalogue_loaded=None
) -> None:
    """
    get_vehicle_by_id() must return the correct VehicleRecord for every
    hardcoded known ID (name check).
    """
    load_catalogue(vehicles_db_path)

    known_mappings = [
        (1, "Tata Nexon"),
        (7, "Tata Punch"),
        (19, "Toyota Fortuner"),
        (34, "Tata Nexon EV"),
        (35, "BYD Seal"),
    ]

    for vehicle_id, expected_name in known_mappings:
        record = get_vehicle_by_id(vehicle_id)
        assert record is not None, f"No record found for id={vehicle_id}"
        assert isinstance(record, VehicleRecord)
        assert record.id == vehicle_id
        assert record.name == expected_name


def test_get_vehicle_by_id_returns_none_for_unknown_id(
    vehicles_db_path: Path, ensure_catalogue_loaded=None
) -> None:
    """
    get_vehicle_by_id() must return None when given an ID that does not exist.
    """
    load_catalogue(vehicles_db_path)

    unknown_ids = [0, -1, 36, 999, 10_000]
    for bad_id in unknown_ids:
        result = get_vehicle_by_id(bad_id)
        assert result is None, f"Expected None for unknown id={bad_id}, got {result!r}"


# ==============================================================================
# Test 3 — Full record fields (id=1, Tata Nexon)
# ==============================================================================

def test_vehicle_record_all_fields_populated_for_known_id(
    vehicles_db_path: Path, ensure_catalogue_loaded=None
) -> None:
    """
    VehicleRecord for id=1 (Tata Nexon) must have all fields correctly populated
    from the database — not None, not empty strings.
    """
    load_catalogue(vehicles_db_path)
    record = get_vehicle_by_id(1)

    assert record is not None

    # Core identity
    assert record.id == 1
    assert record.name == "Tata Nexon"

    # Scalar fields — no SQL NULLs for id=1
    assert record.price_lakhs == 8.0, f"Expected 8.0, got {record.price_lakhs}"
    assert record.fuel_type == "Petrol", f"Expected 'Petrol', got {record.fuel_type!r}"
    assert record.seating_capacity == 5, f"Expected 5, got {record.seating_capacity}"
    assert record.transmission == "Manual", f"Expected 'Manual', got {record.transmission!r}"
    assert record.body_style == "SUV", f"Expected 'SUV', got {record.body_style!r}"
    assert record.mileage_kmpl == 17.0, f"Expected 17.0, got {record.mileage_kmpl}"
    assert record.description is not None and record.description != ""
    assert isinstance(record.description, str)


# ==============================================================================
# Test 4 — features JSON parsed into tuple[str, ...]
# ==============================================================================

def test_features_parsed_into_tuple_of_strings(
    vehicles_db_path: Path, ensure_catalogue_loaded=None
) -> None:
    """
    The features column (JSON string in DB) must be stored as tuple[str, ...]
    on VehicleRecord so the frozen dataclass remains hashable.
    """
    load_catalogue(vehicles_db_path)
    record = get_vehicle_by_id(1)

    assert record is not None
    assert isinstance(record.features, tuple), (
        f"features must be a tuple, got {type(record.features)}"
    )
    assert all(isinstance(f, str) for f in record.features), (
        f"All features must be strings: {record.features!r}"
    )
    # Tata Nexon features (id=1) from vehicles.sql
    expected_features = {
        "Touchscreen", "Rear Camera", "5 Airbags", "ABS", "ESP",
        "Connected Car Tech", "Sunroof",
    }
    assert set(record.features) == expected_features, (
        f"Expected {expected_features}, got {set(record.features)}"
    )


def test_features_contains_expected_elements(
    vehicles_db_path: Path, ensure_catalogue_loaded=None
) -> None:
    """
    For id=1 (Tata Nexon), the features tuple must contain specific expected
    elements like "Sunroof" and "5 Airbags".
    """
    load_catalogue(vehicles_db_path)
    record = get_vehicle_by_id(1)

    assert record is not None
    # Spot-check a few key features
    assert "Sunroof" in record.features, (
        f"'Sunroof' not in features: {record.features}"
    )
    assert "5 Airbags" in record.features, (
        f"'5 Airbags' not in features: {record.features}"
    )
    assert "ABS" in record.features


def test_features_parsed_for_electric_vehicle(
    vehicles_db_path: Path, ensure_catalogue_loaded=None
) -> None:
    """
    id=5 (MG ZS EV) has "Electric Motor" in features — verify parsing works for
    a non-Petrol/Diesel vehicle too.
    """
    load_catalogue(vehicles_db_path)
    record = get_vehicle_by_id(5)

    assert record is not None
    assert record.fuel_type == "Electric"
    assert isinstance(record.features, tuple)
    assert "Electric Motor" in record.features
    assert "Fast Charging" in record.features


def test_vehicle_record_is_hashable(
    vehicles_db_path: Path, ensure_catalogue_loaded=None
) -> None:
    """
    VehicleRecord must be hashable since it is a frozen dataclass.
    """
    load_catalogue(vehicles_db_path)
    record = get_vehicle_by_id(1)
    assert record is not None

    # Must not raise TypeError
    h = hash(record)
    assert isinstance(h, int)

    # Should be usable as a dict key or set member
    s = {record}
    m = {record.id: record}
    assert m[1] is record
    assert record in s


# ==============================================================================
# Test 5 — No sidecar files (-wal, -shm)
# ==============================================================================

def test_no_wal_or_shm_sidecar_files_created(
    vehicles_db_path: Path, database_dir: Path
) -> None:
    """
    Opening vehicles.db with ?mode=ro must never create -wal or -shm sidecar files.
    Snapshot before/after and assert zero new files.
    """
    before = {
        f.name
        for f in database_dir.iterdir()
        if f.suffix in {".wal", ".shm"} or f.name.endswith("-wal") or f.name.endswith("-shm")
    }

    load_catalogue(vehicles_db_path)

    after = {
        f.name
        for f in database_dir.iterdir()
        if f.suffix in {".wal", ".shm"} or f.name.endswith("-wal") or f.name.endswith("-shm")
    }

    new_sidecars = after - before
    assert not new_sidecars, (
        f"Sidecar files created during catalogue load: {new_sidecars!r}. "
        "The ?mode=ro URI must prevent WAL sidecar creation."
    )


def test_no_wal_or_shm_sidecar_files_exist_at_all(vehicles_db_path: Path) -> None:
    """
    vehicles.db must never have WAL or SHM sidecar files alongside it.

    Unlike app.db (which uses WAL for concurrency), vehicles.db is opened with
    ?mode=ro and must therefore never generate sidecars.  This test is scoped
    strictly to vehicles.db's own path — not the whole database/ directory —
    so that app.db's legitimate WAL files (if any) do not cause a false failure.
    """
    load_catalogue(vehicles_db_path)

    vehicles_name = vehicles_db_path.name
    vehicles_dir = vehicles_db_path.parent

    sidecars = [
        f.name
        for f in vehicles_dir.iterdir()
        if (
            f.suffix in {".wal", ".shm"}
            or "-wal" in f.name
            or "-shm" in f.name
        )
        and (
            f.name.startswith(vehicles_name)
            or f.stem == vehicles_db_path.stem
        )
    ]
    assert not sidecars, f"vehicles.db has WAL/SHM sidecar files: {sidecars!r}"


# ==============================================================================
# Test 6 — GET /api/cars endpoint
# ==============================================================================

@pytest.mark.asyncio
async def test_api_cars_returns_200(client: AsyncClient) -> None:
    """GET /api/cars must return HTTP 200."""
    response = await client.get("/api/cars")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_api_cars_returns_all_35_vehicles(client: AsyncClient) -> None:
    """The endpoint must return exactly 35 records, pulled from the in-memory catalogue."""
    response = await client.get("/api/cars")
    data = response.json()
    assert isinstance(data, list), f"Expected a list, got {type(data)}"
    assert len(data) == 35, f"Expected 35 vehicles, got {len(data)}"
    returned_ids = {rec["id"] for rec in data}
    assert returned_ids == set(range(1, 36)), (
        f"Expected IDs {{1..35}}, got {sorted(returned_ids)}"
    )


@pytest.mark.asyncio
async def test_api_cars_returns_only_id_and_name(client: AsyncClient) -> None:
    """
    The endpoint must project and return ONLY {id, name} — the frontend contract.
    No additional fields (price, features, etc.) must be present.
    """
    response = await client.get("/api/cars")
    data = response.json()

    assert isinstance(data, list) and len(data) > 0

    for idx, record in enumerate(data):
        keys = set(record.keys())
        assert keys == {"id", "name"}, (
            f"Record at index {idx} must have ONLY {{'id', 'name'}}, "
            f"got {keys!r}: {record!r}"
        )
        assert isinstance(record["id"], int), f"id at index {idx} must be int"
        assert isinstance(record["name"], str), f"name at index {idx} must be str"


@pytest.mark.asyncio
async def test_api_cars_returns_correct_known_names(client: AsyncClient) -> None:
    """Known vehicles must appear with the correct id and name from the HTTP response."""
    response = await client.get("/api/cars")
    data = response.json()
    by_id = {rec["id"]: rec["name"] for rec in data}

    known = [
        (1, "Tata Nexon"),
        (7, "Tata Punch"),
        (19, "Toyota Fortuner"),
        (35, "BYD Seal"),
    ]
    for vehicle_id, expected_name in known:
        assert vehicle_id in by_id, f"id={vehicle_id} missing from response"
        assert by_id[vehicle_id] == expected_name


@pytest.mark.asyncio
async def test_api_cars_serves_from_memory_not_disk(client: AsyncClient) -> None:
    """
    Endpoint data must match the catalogue module's vehicles list — proving it
    serves from the in-memory copy, not a fresh per-request query.
    """
    response = await client.get("/api/cars")
    endpoint_data = {rec["id"]: rec["name"] for rec in response.json()}

    for vehicle in vehicles:
        assert vehicle.id in endpoint_data
        assert endpoint_data[vehicle.id] == vehicle.name
