"""
backend/scorer.py — Deterministic car ranking module (Issue 020).

A pure function that takes a preference set and an in-memory catalogue of vehicles
and returns the ranked top ~3–5 cars.  Hard constraints are never relaxed;
soft preferences are scored additively.

Public Interface
────────────────
score_and_rank(preferences: PreferenceSet, vehicles: list[VehicleRecord]) -> RankingResult

The function is:
  • Pure     — no side-effects, no I/O, no global state.
  • Instant  — O(n) over the catalogue (n ≤ 35).
  • Stable   — deterministic tie-breaking; same inputs always produce same output.
  • Testable — all edge cases are exercised in unit tests (no DB, no network).

Hard Constraints (never relaxed)
────────────────────────────────
  1. price_lakhs <= budget
  2. seating_capacity >= min_seats

Soft Scoring (additive points, no filtering)
────────────────────────────────────────────
  • fuel_type      — exact match: +50 pts
  • body_style     — exact match: +50 pts
  • transmission   — exact match: +30 pts
    • mileage_kmpl   — +6 pts per kmpl of excess over the minimum, capped at +30 pts;
                       below-minimum earns 0 pts (bonus-only, never negative)
  • features       — each feature keyword that is a case-insensitive substring of
                     any vehicle feature earns +10 pts (bonus only — never excludes)

Score ceiling: 100 pts (fuel+body+mileage+transmission), features add beyond the cap.

Output
──────
  RankingResult:
    cars       — list[ScoredCar] of length 1–5, sorted descending by score,
                 then ascending by price as tiebreaker.
    note       — non-empty only when hard-constraint fallback kicks in.

Fallback behaviour
───────────────────
If no vehicle satisfies both hard constraints simultaneously, the function
returns the up-to-5 closest-by-price vehicles (closest = minimum
|price_lakhs - budget|) with an explanatory note.  This guarantee ensures the
AI call always has something to read back to the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.catalogue import VehicleRecord

__all__ = ["score_and_rank", "PreferenceSet", "ScoredCar", "RankingResult"]


# ─── Input type ────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PreferenceSet:
    """
    The user's stated preferences extracted from the AI call transcript.

    ``budget`` and ``min_seats`` are hard constraints — never relaxed.
    All other fields are soft preferences — scored, never used as filters.

    A None / empty field means "no preference stated" and contributes 0 pts.
    """

    budget: float
    """Hard constraint: car price_lakhs must be <= this value."""

    min_seats: int
    """Hard constraint: car seating_capacity must be >= this value."""

    fuel_type: str | None = None
    """Soft: exact case-insensitive match earns +50 pts."""

    body_style: str | None = None
    """Soft: exact case-insensitive match earns +50 pts."""

    transmission: str | None = None
    """Soft: exact case-insensitive match earns +30 pts."""

    min_mileage_kmpl: float | None = None
    """Soft: car mileage_kmpl >= this earns up to +30 pts (capped)."""

    feature_keywords: tuple[str, ...] = field(default_factory=tuple)
    """Soft: each keyword matching a vehicle feature earns +10 pts."""


# ─── Intermediate type ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScoredCar:
    """
    A vehicle annotated with its computed score and the breakdown per criterion.
    """

    vehicle: VehicleRecord
    """The original vehicle record."""

    score: int
    """Total additive score (0–100+; features can push past 100)."""

    fuel_match: bool = False
    body_match: bool = False
    transmission_match: bool = False
    mileage_pts: int = 0
    feature_pts: int = 0


# ─── Output type ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RankingResult:
    """
    The ranked list returned to the caller (Step 3 of the AI Call pipeline).

    ``note`` is populated only in the hard-constraint fallback path.
    """

    cars: tuple[ScoredCar, ...]
    """Ranked cars, length 1–5, sorted by score desc then price asc."""

    note: str | None = None
    """Present when no car satisfied both hard constraints."""


# ─── Point constants ─────────────────────────────────────────────────────────

_FUEL_PTS = 50
_BODY_PTS = 50
_TRANS_PTS = 30
_MILEAGE_MAX_PTS = 30
_FEATURE_PTS = 10
_TOP_K = 5


# ─── Pure scoring function ─────────────────────────────────────────────────────


def score_and_rank(
    preferences: PreferenceSet,
    vehicles: list[VehicleRecord],
) -> RankingResult:
    """
    Score and rank vehicles against the given preference set.

    Parameters
    ----------
    preferences : PreferenceSet
        User's stated hard constraints and soft preferences.
    vehicles : list[VehicleRecord]
        The in-memory vehicle catalogue (typically ``catalogue.vehicles``).

    Returns
    -------
    RankingResult
        Top 3–5 ranked cars, or up to 5 closest-by-price cars if nothing
        satisfies the hard constraints.  ``cars`` is never empty.
    """
    hard_eligible: list[ScoredCar] = []
    all_by_price: list[tuple[float, ScoredCar]] = []  # for fallback

    for vehicle in vehicles:
        # ── Hard constraint 1: price ────────────────────────────────────────────
        if vehicle.price_lakhs is None:
            # No price known — treat as unknown; include in fallback pool
            all_by_price.append((float("inf"), _score_vehicle(vehicle, preferences)))
            continue
        price = vehicle.price_lakhs

        if price > preferences.budget:
            # Over-budget: excluded from ranking; included in fallback pool
            all_by_price.append((abs(price - preferences.budget), _score_vehicle(vehicle, preferences)))
            continue

        # ── Hard constraint 2: seating capacity ────────────────────────────────
        seats = vehicle.seating_capacity
        if seats is None or seats < preferences.min_seats:
            # Insufficient seats: excluded from ranking; included in fallback pool
            all_by_price.append((abs(price - preferences.budget), _score_vehicle(vehicle, preferences)))
            continue

        # ── Passed both hard constraints ───────────────────────────────────────
        scored = _score_vehicle(vehicle, preferences)
        hard_eligible.append(scored)

    # ── Ranked output ────────────────────────────────────────────────────────
    if hard_eligible:
        ranked = sorted(
            hard_eligible,
            key=lambda s: (-s.score, s.vehicle.price_lakhs or float("inf")),
        )
        return RankingResult(cars=tuple(ranked[:_TOP_K]))

    # ── Fallback: nothing satisfied hard constraints ─────────────────────────
    # Sort by |price - budget| ascending (closest first), return top 5
    all_by_price.sort(key=lambda x: x[0])
    fallback_cars = tuple(s for _, s in all_by_price[:_TOP_K])

    return RankingResult(
        cars=fallback_cars,
        note=(
            "No car satisfied both your budget and minimum-seat requirements. "
            "Showing the closest options by price instead."
        ),
    )


def _score_vehicle(vehicle: VehicleRecord, prefs: PreferenceSet) -> ScoredCar:
    """
    Compute per-criterion scores for one vehicle.

    Returns a ``ScoredCar`` snapshot — never mutates ``vehicle``.
    """
    score = 0
    fuel_match = False
    body_match = False
    trans_match = False
    mileage_pts = 0
    feature_pts = 0

    # ── Fuel type ────────────────────────────────────────────────────────────
    if prefs.fuel_type and vehicle.fuel_type:
        if prefs.fuel_type.strip().lower() == vehicle.fuel_type.strip().lower():
            score += _FUEL_PTS
            fuel_match = True

    # ── Body style ──────────────────────────────────────────────────────────
    if prefs.body_style and vehicle.body_style:
        if prefs.body_style.strip().lower() == vehicle.body_style.strip().lower():
            score += _BODY_PTS
            body_match = True

    # ── Transmission ────────────────────────────────────────────────────────
    if prefs.transmission and vehicle.transmission:
        if prefs.transmission.strip().lower() == vehicle.transmission.strip().lower():
            score += _TRANS_PTS
            trans_match = True

    # ── Mileage ─────────────────────────────────────────────────────────────
    if prefs.min_mileage_kmpl is not None and vehicle.mileage_kmpl is not None and vehicle.mileage_kmpl > 0:
        # Award points proportional to how much the vehicle exceeds the minimum.
        # Cap at _MILEAGE_MAX_PTS so it never overwhelms the other criteria.
        excess = vehicle.mileage_kmpl - prefs.min_mileage_kmpl
        raw_pts = int(excess * _MILEAGE_PTS_PER_KMPL)
        mileage_pts = max(0, min(raw_pts, _MILEAGE_MAX_PTS))  # floor at 0
        score += mileage_pts

    # ── Feature keywords (bonus only — never filters) ───────────────────────
    if prefs.feature_keywords:
        v_features_lower = {f.lower() for f in vehicle.features}
        for kw in prefs.feature_keywords:
            kw_lower = kw.strip().lower()
            if not kw_lower:
                continue
            # Check if keyword is a substring of any vehicle feature
            for vf in vehicle.features:
                if kw_lower in vf.lower():
                    feature_pts += _FEATURE_PTS
                    break  # each keyword scores at most once per car

    return ScoredCar(
        vehicle=vehicle,
        score=score + feature_pts,
        fuel_match=fuel_match,
        body_match=body_match,
        transmission_match=trans_match,
        mileage_pts=mileage_pts,
        feature_pts=feature_pts,
    )


# mileage pts per kmpl above min — chosen so that a 5 kmpl excess → +30 pts (cap)
_MILEAGE_PTS_PER_KMPL = _MILEAGE_MAX_PTS / 5.0  # 6.0 pts per kmpl
