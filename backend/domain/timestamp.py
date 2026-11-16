"""
backend/domain/timestamp.py — SQLite UTC timestamp serialization (Issue 015).

Problem
───────
SQLite stores ``datetime('now')`` as UTC but without any timezone metadata:

    2026-06-28 14:00:00        ← space-separated, no T, no Z

When this raw string reaches the browser, JavaScript's ``Date`` constructor
interprets it as *local* time, causing the displayed time to drift by the
user's UTC offset.  A user in IST (UTC+5:30) would see 19:30 instead of 14:00.

Solution
────────
``to_iso_z`` enforces the invariant that every datetime string leaving the
backend carries an explicit ``Z`` suffix, marking it unambiguously as UTC.
Browsers that receive ``2026-06-28T14:00:00Z`` will parse it correctly
regardless of the user's local timezone.

Rules
─────
1. ``None``  → ``None``                          (None-safe)
2. Already ISO with ``Z``  → pass-through        (idempotent)
3. SQLite space-format  → ``T`` + ``Z`` added    (the core conversion)
"""

from __future__ import annotations


def to_iso_z(db_ts: str | None) -> str | None:
    """
    Convert a SQLite UTC timestamp to an ISO-8601 UTC string.

    Parameters
    ----------
    db_ts : str | None
        A timestamp from SQLite, e.g. ``"2026-06-28 14:00:00"``, or
        an already-normalised ISO-8601 UTC string like
        ``"2026-06-28T14:00:00Z"``.  May be ``None``.

    Returns
    -------
    str | None
        - ``None`` if the input is ``None``.
        - Pass-through unchanged if the input already contains a ``T`` **and**
          ends with ``Z``.
        - Otherwise replace the space with ``T`` and append ``Z``.

    Examples
    --------
    >>> to_iso_z(None)
    None
    >>> to_iso_z("2026-06-28T14:00:00Z")
    '2026-06-28T14:00:00Z'
    >>> to_iso_z("2026-06-28 14:00:00")
    '2026-06-28T14:00:00Z'
    """
    if db_ts is None:
        return None

    # Idempotent: already ISO-8601 with explicit UTC marker.
    if "T" in db_ts and db_ts.endswith("Z"):
        return db_ts

    # SQLite space-format → replace space with T and append Z.
    # Handles: "YYYY-MM-DD HH:MM:SS", "YYYY-MM-DD HH:MM:SS.ffffff", etc.
    return db_ts.replace(" ", "T") + "Z"
