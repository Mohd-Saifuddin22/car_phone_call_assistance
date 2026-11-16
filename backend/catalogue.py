"""
backend/catalogue.py — in-memory car catalogue.

Design
──────
vehicles.db is a read-only reference database containing the 35-row vehicle
catalogue.  It is loaded once at application startup (inside the FastAPI
lifespan) and held entirely in memory as plain Python objects.

Strict Read-Only Enforcement
───────────────────────────
The DB is opened via the SQLite URI with ``?mode=ro`` appended, e.g.:

    file:///E:/.../vehicles.db?mode=ro

This guarantees:
  1. SQLite never acquires a write lock — no WAL / -wal / -shm sidecar files.
  2. Any attempt to mutate the catalogue raises ``OperationalError`` immediately.
  3. No connection can accidentally commit a write transaction.

The blocking synchronous read (``sqlite3``) is acceptable because it runs once
during startup, outside any hot path.

Public Interface
───────────────
load_catalogue(vehicles_db_path) -> None
    Call from the FastAPI lifespan after the app.db pool is ready.
    Raises ``FileNotFoundError`` if the DB file does not exist.

vehicles: list[VehicleRecord]
    All 35 rows, in insertion order.

get_vehicle_by_id(id: int) -> VehicleRecord | None
    O(1) dict lookup.  Returns None for unknown IDs — callers do not need
    to handle KeyError.

VehicleRecord: immutable dataclass with all columns; see class docstring.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["load_catalogue", "vehicles", "get_vehicle_by_id", "VehicleRecord"]


# ─── Types ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VehicleRecord:
    """
    Immutable snapshot of one row from vehicles.db.

    All fields are populated at load time so that callers (the UI dropdown,
    the AI scorer, and the call-time prompt compiler) never need to hit the
    database at runtime.

    Attributes
    ----------
    id              : int              — primary key
    name            : str              — display name
    price_lakhs     : float | None     — price in lakhs
    fuel_type       : str | None       — e.g. "Petrol", "Diesel", "Electric"
    seating_capacity: int | None       — number of seats
    transmission    : str | None       — e.g. "Manual", "Automatic"
    body_style      : str | None       — e.g. "SUV", "Hatchback", "Sedan"
    mileage_kmpl    : float | None     — km per litre (0.0 for electric)
    description     : str | None       — free-text description
    features        : tuple[str, ...]   — parsed JSON array of feature strings
    """

    id: int
    name: str
    price_lakhs: float | None
    fuel_type: str | None
    seating_capacity: int | None
    transmission: str | None
    body_style: str | None
    mileage_kmpl: float | None
    description: str | None
    features: tuple[str, ...]

    def __post_init__(self) -> None:
        # Parse the features JSON string into a tuple[str, ...] so that the
        # dataclass remains fully hashable (tuples are hashable; lists are not).
        # object.__setattr__ is required because the dataclass is frozen.
        if isinstance(self.features, str):
            object.__setattr__(
                self,
                "features",
                tuple(json.loads(self.features)),
            )
        elif isinstance(self.features, list):
            object.__setattr__(self, "features", tuple(self.features))


# ─── Module-level state (populated by load_catalogue) ────────────────────────────

vehicles: list[VehicleRecord] = []
"""
All vehicle records loaded from vehicles.db, in primary-key order.
Populated by ``load_catalogue()`` during FastAPI startup.
"""

_by_id: dict[int, VehicleRecord] = {}
"""
Dict-backed O(1) lookup: id -> VehicleRecord.
Populated alongside ``vehicles`` during ``load_catalogue()``.
"""


# ─── Loader ─────────────────────────────────────────────────────────────────────


def load_catalogue(vehicles_db_path: Path) -> None:
    """
    Load the entire vehicles.db into memory.

    This function is idempotent — calling it twice replaces the existing
    in-memory state.

    Parameters
    ----------
    vehicles_db_path : Path
        Absolute path to the read-only vehicles.db SQLite file.

    Raises
    ------
    FileNotFoundError
        The vehicles.db file does not exist at the given path.
    """
    global vehicles, _by_id

    if not vehicles_db_path.exists():
        raise FileNotFoundError(
            f"vehicles.db not found at {vehicles_db_path!s}. "
            "Ensure VEHICLES_DB_PATH is configured correctly."
        )

    # URI with ?mode=ro guarantees:
    #   - no WAL / -wal / -shm sidecar files are ever created
    #   - the connection is read-only — any write attempt raises OperationalError
    uri = f"file:///{vehicles_db_path.as_posix()}?mode=ro"

    conn = sqlite3.connect(uri, uri=True)  # uri=True enables the URI format
    try:
        conn.row_factory = sqlite3.Row  # named-column access

        cursor = conn.execute(
            "SELECT id, name, price_lakhs, fuel_type, seating_capacity, "
            "transmission, body_style, mileage_kmpl, description, features "
            "FROM vehicles ORDER BY id ASC;"
        )
        rows = cursor.fetchall()

        new_vehicles = [
            VehicleRecord(
                id=row["id"],
                name=row["name"],
                price_lakhs=_coerce(row["price_lakhs"], float),
                fuel_type=row["fuel_type"],
                seating_capacity=_coerce(row["seating_capacity"], int),
                transmission=row["transmission"],
                body_style=row["body_style"],
                mileage_kmpl=_coerce(row["mileage_kmpl"], float),
                description=row["description"],
                features=row["features"],
            )
            for row in rows
        ]
        new_by_id = {v.id: v for v in new_vehicles}

        # Mutate in-place so that any pre-existing references to the existing
        # `vehicles` / `_by_id` objects (e.g. `from backend.catalogue import vehicles`
        # in main.py) stay valid and see the new data.
        vehicles.clear()
        vehicles.extend(new_vehicles)
        _by_id.clear()
        _by_id.update(new_by_id)
    finally:
        conn.close()


def _coerce(value: Any, typ: type) -> Any:
    """Return None for SQL NULL, otherwise cast to the target type."""
    return typ(value) if value is not None else None


# ─── Accessors ──────────────────────────────────────────────────────────────────


def get_vehicle_by_id(id: int) -> VehicleRecord | None:
    """
    Return the vehicle with the given ``id``, or ``None`` if no such record exists.

    This is an O(1) dict lookup against the in-memory catalogue loaded at startup.
    """
    return _by_id.get(id)
