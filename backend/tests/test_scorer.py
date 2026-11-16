"""
backend/tests/test_scorer.py — Issue 020 deterministic scorer tests.

Test plan
────────
T1  Worked example: Electric SUV under 20 L → Tata Nexon EV #1 (no tie).
T2  Hard exclusion: over-budget cars are excluded.
T3  Hard exclusion: insufficient-seat cars are excluded.
T4  Hard exclusion: both constraints violated → excluded.
T5  Soft fuel_type scoring: exact match → +50 pts.
T6  Soft body_style scoring: exact match → +50 pts.
T7  Soft transmission scoring: exact match → +30 pts.
T8  Soft mileage scoring: +pts proportional to kmpl above min, capped at +30.
T9  Feature substring bonus: matching keyword → +10 pts per keyword.
T10 Features never filter: a car with zero feature matches still appears.
T11 Additive scoring: fuel+body+transmission+mileage stack correctly.
T12 Relaxation-never-empty: if any car satisfies hard constraints → non-empty list.
T13 Fallback with note: when nothing satisfies hard constraints → closest-by-price + note.
T14 Fallback order: closest-by-price is sorted ascending by |price - budget|.
T15 Tie-breaking: same score → lower price ranks first.
T16 Stability: identical inputs always produce identical outputs.
T17 Top-K: never more than 5 cars returned.
T18 PreferenceSet fields are all optional (empty prefs = 0-score additive).
T19 None fields on VehicleRecord handled gracefully (no crash).
T20 Score + feature bonus: feature_pts can push score past 100.
T21 Multiple feature keywords each score independently.
T22 Case-insensitive matching on fuel, body, transmission.
T23 min_mileage_kmpl=None is a no-op (0 pts).
T24 Empty catalogue returns empty list (no cars at all → hard-constraint fallback
    returns empty as well, which is acceptable since there's nothing to score).

Fixtures
────────
catalogued_vehicles : loads real vehicles.db into memory before each test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.catalogue import VehicleRecord, load_catalogue
from backend.scorer import (
    PreferenceSet,
    RankingResult,
    ScoredCar,
    score_and_rank,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def catalogued_vehicles() -> list[VehicleRecord]:
    """
    Load the real vehicles.db once per test so that ScoredCar.vehicle
    comparisons use genuine VehicleRecord instances.
    """
    db_path = (
        Path(__file__).parent.parent.parent
        / "database"
        / "vehicles.db"
    ).resolve()
    load_catalogue(db_path)
    # Import here so the module-level vehicles list is populated
    from backend import catalogue

    return list(catalogue.vehicles)


# ─── Helpers ───────────────────────────────────────────────────────────────────


def _v(catalogued: list[VehicleRecord], name: str) -> VehicleRecord:
    """Find a vehicle by name in the fixture."""
    matches = [v for v in catalogued if v.name == name]
    if not matches:
        raise ValueError(f"No vehicle named {name!r}")
    if len(matches) > 1:
        raise ValueError(f"Duplicate name {name!r}")
    return matches[0]


def assert_scores_descending(result: RankingResult) -> None:
    """Assert that result.cars are sorted by score descending, price asc as tiebreak."""
    scores = [c.score for c in result.cars]
    assert scores == sorted(scores, reverse=True), (
        f"Scores not descending: {scores}"
    )
    for i in range(len(result.cars) - 1):
        if result.cars[i].score == result.cars[i + 1].score:
            p0 = result.cars[i].vehicle.price_lakhs
            p1 = result.cars[i + 1].vehicle.price_lakhs
            assert p0 <= p1, (
                f"Tied score but price not ascending: "
                f"{result.cars[i].vehicle.name}={p0} > "
                f"{result.cars[i+1].vehicle.name}={p1}"
            )


# =============================================================================
# T1 — Worked example: Electric SUV under 20 L ⇒ Tata Nexon EV ranks #1
# =============================================================================

def test_worked_example_electric_suv_under_20_l_nexon_ev_is_ranked_1(
    catalogued_vehicles: list[VehicleRecord],
) -> None:
    """
    PreferenceSet(budget=20, min_seats=5, fuel_type='Electric', body_style='SUV')
    applied to the real catalogue must put Tata Nexon EV (id=34, 14L) at rank 1.

    Expected score breakdown for Tata Nexon EV:
      fuel_type='Electric' → +50
      body_style='SUV'     → +50
      total = 100

    All other SUVs under 20L are either not Electric (Tata Nexon, Kia Seltos,
    Hyundai Creta, etc.) or exceed the budget (MG ZS EV = 23L).  No other car
    can tie the 100-point score.
    """
    prefs = PreferenceSet(
        budget=20.0,
        min_seats=5,
        fuel_type="Electric",
        body_style="SUV",
    )
    result = score_and_rank(prefs, catalogued_vehicles)

    assert result.note is None, "No fallback note expected — hard constraints were satisfied"

    assert len(result.cars) >= 1, "At least one car must match"
    assert result.cars[0].vehicle.name == "Tata Nexon EV", (
        f"Expected Tata Nexon EV at rank 1, got {result.cars[0].vehicle.name!r}"
    )
    assert result.cars[0].fuel_match is True, "Nexon EV must match fuel_type"
    assert result.cars[0].body_match is True, "Nexon EV must match body_style"
    assert result.cars[0].score == 100, f"Expected 100 pts, got {result.cars[0].score}"

    # No other car can score 100 (no other Electric SUV under 20L)
    for scored in result.cars[1:]:
        assert scored.score < 100, (
            f"{scored.vehicle.name} scored {scored.score} but should be < 100"
        )

    # MG ZS EV (23L) is over-budget and must not appear
    overbudget_names = {c.vehicle.name for c in result.cars if c.vehicle.price_lakhs and c.vehicle.price_lakhs > 20}
    assert not overbudget_names, f"Over-budget cars in result: {overbudget_names}"

    assert_scores_descending(result)


# =============================================================================
# T2 — Hard exclusion: over-budget
# =============================================================================

def test_over_budget_cars_are_excluded(catalogued_vehicles: list[VehicleRecord]) -> None:
    """
    With budget=10L and min_seats=5, Toyota Fortuner (33L) and BYD Seal (45L)
    must not appear in the result.
    """
    prefs = PreferenceSet(budget=10.0, min_seats=5)
    result = score_and_rank(prefs, catalogued_vehicles)

    result_names = {c.vehicle.name for c in result.cars}
    assert "Toyota Fortuner" not in result_names
    assert "BYD Seal" not in result_names
    assert "MG ZS EV" not in result_names  # 23L > 10L

    # Every returned car must be within budget
    for scored in result.cars:
        assert scored.vehicle.price_lakhs is not None
        assert scored.vehicle.price_lakhs <= 10.0, (
            f"{scored.vehicle.name} at {scored.vehicle.price_lakhs}L is over 10L budget"
        )


# =============================================================================
# T3 — Hard exclusion: insufficient seats
# =============================================================================

def test_insufficient_seats_are_excluded(catalogued_vehicles: list[VehicleRecord]) -> None:
    """
    With min_seats=7, no 5-seater may appear even if it's perfectly scored.
    Toyota Innova Crysta (7 seats) and Tata Safari (7 seats) must appear,
    while Tata Nexon (5 seats) must not.
    """
    prefs = PreferenceSet(budget=100.0, min_seats=7)
    result = score_and_rank(prefs, catalogued_vehicles)

    result_names = {c.vehicle.name for c in result.cars}
    assert "Tata Nexon" not in result_names
    assert "Tata Punch" not in result_names
    assert "Maruti Suzuki Swift" not in result_names

    # Every returned car must have sufficient seats
    for scored in result.cars:
        assert scored.vehicle.seating_capacity is not None
        assert scored.vehicle.seating_capacity >= 7, (
            f"{scored.vehicle.name} has {scored.vehicle.seating_capacity} seats < 7"
        )


# =============================================================================
# T4 — Hard exclusion: both constraints violated
# =============================================================================

def test_both_constraints_violated_excluded(catalogued_vehicles: list[VehicleRecord]) -> None:
    """
    Budget=8L AND min_seats=7: Toyota Fortuner (33L, 7 seats) must not appear.
    """
    prefs = PreferenceSet(budget=8.0, min_seats=7)
    result = score_and_rank(prefs, catalogued_vehicles)

    result_names = {c.vehicle.name for c in result.cars}
    assert "Toyota Fortuner" not in result_names


# =============================================================================
# T5 — Soft fuel_type scoring: exact match → +50 pts
# =============================================================================

def test_fuel_type_match_earns_50_pts(catalogued_vehicles: list[VehicleRecord]) -> None:
    """
    PreferenceSet(budget=100, min_seats=5, fuel_type='Electric').
    Tata Nexon EV (Electric) must score 50, MG ZS EV (Electric) must score 50.
    """
    prefs = PreferenceSet(budget=100.0, min_seats=5, fuel_type="Electric")
    result = score_and_rank(prefs, catalogued_vehicles)

    electric_cars = [c for c in result.cars if c.fuel_match]
    assert len(electric_cars) >= 2, "At least Nexon EV and MG ZS EV should match fuel_type"
    for c in electric_cars:
        assert c.score >= 50, f"{c.vehicle.name} should have fuel_match pts"
        assert c.fuel_match is True

    # Case insensitivity
    prefs_lower = PreferenceSet(budget=100.0, min_seats=5, fuel_type="electric")
    result_lower = score_and_rank(prefs_lower, catalogued_vehicles)
    assert result_lower.cars[0].fuel_match is True, "Case-insensitive match failed"


# =============================================================================
# T6 — Soft body_style scoring: exact match → +50 pts
# =============================================================================

def test_body_style_match_earns_50_pts(catalogued_vehicles: list[VehicleRecord]) -> None:
    """
    PreferenceSet(budget=100, min_seats=5, body_style='Sedan').
    Honda City, Skoda Slavia, Volkswagen Virtus should all score >= 50.
    """
    prefs = PreferenceSet(budget=100.0, min_seats=5, body_style="Sedan")
    result = score_and_rank(prefs, catalogued_vehicles)

    sedan_cars = [c for c in result.cars if c.body_match]
    assert len(sedan_cars) >= 3, "Should have multiple Sedans in result"
    for c in sedan_cars:
        assert c.score >= 50, f"{c.vehicle.name} should have body_match pts"
        assert c.body_match is True

    # Case insensitivity
    prefs_upper = PreferenceSet(budget=100.0, min_seats=5, body_style="SEDAN")
    result_upper = score_and_rank(prefs_upper, catalogued_vehicles)
    assert result_upper.cars[0].body_match is True, "Case-insensitive match failed"


# =============================================================================
# T7 — Soft transmission scoring: exact match → +30 pts
# =============================================================================

def test_transmission_match_earns_30_pts(catalogued_vehicles: list[VehicleRecord]) -> None:
    """
    PreferenceSet(budget=100, min_seats=5, transmission='Automatic').
    Kia Seltos (Automatic) should score 30+ for transmission alone.
    """
    prefs = PreferenceSet(budget=100.0, min_seats=5, transmission="Automatic")
    result = score_and_rank(prefs, catalogued_vehicles)

    auto_cars = [c for c in result.cars if c.transmission_match]
    assert len(auto_cars) >= 5, "Should have many Automatic cars"
    for c in auto_cars:
        assert c.transmission_match is True
        assert c.score >= 30, f"{c.vehicle.name} should have transmission pts"

    # Tata Nexon (Manual) should NOT have transmission_match
    nexon = _v(catalogued_vehicles, "Tata Nexon")
    nexon_scored = next((c for c in result.cars if c.vehicle is nexon), None)
    if nexon_scored:
        assert nexon_scored.transmission_match is False, (
            "Manual car should not match Automatic preference"
        )


# =============================================================================
# T8 — Soft mileage scoring: +pts proportional to kmpl above min, capped at +30
# =============================================================================

def test_mileage_scoring_proportional_and_capped(catalogued_vehicles: list[VehicleRecord]) -> None:
    """
    Verifies that mileage scoring is proportional to excess above min_mileage_kmpl
    and capped at +30 pts.

    Three scenarios (tested against the full catalogue, searching all results):
      1. Swift (22 kmpl) — excess 7 → raw 42 → capped at 30.
      2. Nexon (17 kmpl)  — excess 2 → raw 12 → 12 pts.
      3. Venue (18 kmpl) — excess 3 → raw 18 → 18 pts.

    NOTE: The 6L petrol cars (Swift, Dzire, Grand i10, Magnite) also score 30 pts
    with min_mileage_kmpl=15, so they crowd out Nexon/Venue from the top-5
    regardless of budget.  We search ALL result.cars (up to 5 in ranked order)
    rather than asserting a specific car is in the top-5 slot.
    """
    prefs = PreferenceSet(budget=8.0, min_seats=5, min_mileage_kmpl=15.0)
    result = score_and_rank(prefs, catalogued_vehicles)

    swift = next((c for c in result.cars if c.vehicle.name == "Maruti Suzuki Swift"), None)
    assert swift is not None, "Swift should be in results with budget=8L"
    assert swift.mileage_pts == 30, f"Swift expected 30 (capped), got {swift.mileage_pts}"

    # Nexon and Venue may not be in the top-5 (crowded by cheap 6L cars scoring 30).
    # Verify their mileage_pts by searching all returned cars.
    # If they happen to be in the top-5, the assertions still hold.
    # If not, we check the scoring directly via _score_vehicle.
    from backend.scorer import _score_vehicle

    nexon_vehicle = next((v for v in catalogued_vehicles if v.name == "Tata Nexon"), None)
    assert nexon_vehicle is not None
    nexon_scored = _score_vehicle(nexon_vehicle, prefs)
    assert nexon_scored.mileage_pts == 12, (
        f"Nexon mileage_pts expected 12 (2 kmpl excess × 6), got {nexon_scored.mileage_pts}"
    )

    venue_vehicle = next((v for v in catalogued_vehicles if v.name == "Hyundai Venue"), None)
    assert venue_vehicle is not None
    venue_scored = _score_vehicle(venue_vehicle, prefs)
    assert venue_scored.mileage_pts == 18, (
        f"Venue mileage_pts expected 18 (3 kmpl excess × 6), got {venue_scored.mileage_pts}"
    )


# =============================================================================
# T9 — Feature substring bonus: +10 pts per matching keyword
# =============================================================================

def test_feature_keyword_substring_bonus(catalogued_vehicles: list[VehicleRecord]) -> None:
    """
    PreferenceSet with feature_keywords=['sunroof'] and body_style='SUV'
    applied to the catalogue with budget=8L.

    Tata Nexon (SUV, has Sunroof) should earn:
      body_style match → +50
      'sunroof' keyword → +10
      total = 60 pts.

    Hyundai Venue (SUV, has Sunroof) should also earn 60 pts.
    Kia Sonet (SUV, has Sunroof) also earns 60 pts.

    With budget=8L + body_style='SUV', Nexon, Venue, and Sonet appear in
    the top 5 alongside Magnite and Kiger (also SUVs at 6L and 7L respectively,
    scoring 50+10=60).

    Note: 'adas' is excluded from keywords here so that the tied top-5
    includes Nexon; with 'adas' present, ADAS-equipped SUVs score 70 and
    crowd Nexon (no ADAS) out of the top 5.
    """
    prefs = PreferenceSet(
        budget=8.0,
        min_seats=5,
        body_style="SUV",
        feature_keywords=("sunroof",),
    )
    result = score_and_rank(prefs, catalogued_vehicles)

    nexon = next((c for c in result.cars if c.vehicle.name == "Tata Nexon"), None)
    assert nexon is not None, "Tata Nexon should be in results with body_style=SUV and budget=8L"
    assert nexon.feature_pts == 10, f"Nexon feature_pts expected 10, got {nexon.feature_pts}"
    assert nexon.body_match is True
    assert nexon.score == 60, f"Nexon expected score=60 (50+10), got {nexon.score}"

    venue = next((c for c in result.cars if c.vehicle.name == "Hyundai Venue"), None)
    assert venue is not None, "Venue should be in results"
    assert venue.feature_pts == 10, f"Venue feature_pts expected 10, got {venue.feature_pts}"
    assert venue.body_match is True

    sonet = next((c for c in result.cars if c.vehicle.name == "Kia Sonet"), None)
    assert sonet is not None, "Sonet should be in results"
    assert sonet.feature_pts == 10, f"Sonet feature_pts expected 10, got {sonet.feature_pts}"
    assert sonet.body_match is True


# =============================================================================
# T10 — Features never filter: a car with zero feature matches still appears
# =============================================================================

def test_features_never_filter(catalogued_vehicles: list[VehicleRecord]) -> None:
    """
    Even with feature_keywords that match nothing, valid cars must still appear.
    """
    prefs = PreferenceSet(
        budget=100.0,
        min_seats=5,
        feature_keywords=("nonexistent_gizmo_xyz",),
    )
    result = score_and_rank(prefs, catalogued_vehicles)

    assert len(result.cars) > 0, "Features are bonus-only; result must not be empty"
    for c in result.cars:
        assert c.feature_pts == 0, f"{c.vehicle.name} should have 0 feature_pts"


# =============================================================================
# T11 — Additive scoring: fuel+body+transmission+mileage stack correctly
# =============================================================================

def test_additive_scoring_stacks(catalogued_vehicles: list[VehicleRecord]) -> None:
    """
    Tata Nexon EV (Electric + SUV) with transmission='Automatic' and
    min_mileage_kmpl=0 scores:
      fuel match     → +50
      body match     → +50
      (no trans match, Nexon EV is Automatic so +30)
      mileage (0 kmpl → 0 excess) → 0
    Total = 130 pts.

    The total score is the sum of individual components.
    """
    prefs = PreferenceSet(
        budget=100.0,
        min_seats=5,
        fuel_type="Electric",
        body_style="SUV",
        transmission="Automatic",
        min_mileage_kmpl=0.0,
    )
    result = score_and_rank(prefs, catalogued_vehicles)

    nexon_ev = next(c for c in result.cars if c.vehicle.name == "Tata Nexon EV")

    # All three criteria match for Nexon EV
    assert nexon_ev.fuel_match is True
    assert nexon_ev.body_match is True
    assert nexon_ev.transmission_match is True

    expected = 50 + 50 + 30  # 130 pts from fuel+body+transmission
    assert nexon_ev.score >= expected, (
        f"Nexon EV expected score >= {expected}, got {nexon_ev.score}"
    )


# =============================================================================
# T12 — Relaxation-never-empty: if anything fits hard constraints → non-empty
# =============================================================================

def test_relaxation_never_empty_when_hard_constraints_satisfied(
    catalogued_vehicles: list[VehicleRecord],
) -> None:
    """
    With the widest possible hard constraints (budget=100, min_seats=1),
    every car in the catalogue satisfies both.  The result must contain
    all cars (or up to 5 of them) — never empty.
    """
    prefs = PreferenceSet(budget=100.0, min_seats=1)
    result = score_and_rank(prefs, catalogued_vehicles)

    assert len(result.cars) > 0, "Result must never be empty when hard constraints match"
    assert result.note is None, "No note expected in normal path"
    assert 1 <= len(result.cars) <= 5, f"Expected 1-5 cars, got {len(result.cars)}"


# =============================================================================
# T13 — Fallback with note: nothing satisfies hard constraints
# =============================================================================

def test_fallback_returns_closest_by_price_with_note(
    catalogued_vehicles: list[VehicleRecord],
) -> None:
    """
    With an absurdly low budget (1 L) and min_seats=7, no car satisfies
    both hard constraints.  The function must return up to 5 closest-by-price
    cars and set result.note.
    """
    prefs = PreferenceSet(budget=1.0, min_seats=7)
    result = score_and_rank(prefs, catalogued_vehicles)

    assert result.note is not None, "Fallback must set a note"
    assert "budget" in result.note.lower() or "price" in result.note.lower(), (
        f"Note should mention price/budget: {result.note!r}"
    )
    assert len(result.cars) > 0, "Fallback must return something"
    assert len(result.cars) <= 5, f"Expected ≤5 cars in fallback, got {len(result.cars)}"

    # The closest car to 1L is the cheapest one
    # (Maruti Swift at 6L, Maruti Dzire at 6L, etc. — all at distance 5)
    prices = [c.vehicle.price_lakhs for c in result.cars if c.vehicle.price_lakhs]
    assert prices, "Fallback cars must have prices"
    # All returned cars must be the cheapest available
    min_price = min(p for p in prices)
    assert all(abs(p - 1.0) == abs(min_price - 1.0) or p == min_price for p in prices), (
        "Fallback should show closest-by-price cars"
    )


# =============================================================================
# T14 — Fallback order: closest-by-price ascending
# =============================================================================

def test_fallback_order_is_closest_by_price(
    catalogued_vehicles: list[VehicleRecord],
) -> None:
    """
    With budget=8L and min_seats=7, no 7-seater is ≤8L.
    All 7-seaters are in the fallback pool sorted by |price - 8L| ascending:
      Ertiga (9L, dist=1) → Innova (20L, dist=12) → Safari (17L, dist=9) →
      Scorpio-N (15L, dist=7) → XUV700 (19L, dist=11) → Fortuner (33L, dist=25).
    The result must be ordered by price-distance ascending.
    """
    prefs = PreferenceSet(budget=8.0, min_seats=7)
    result = score_and_rank(prefs, catalogued_vehicles)

    assert result.note is not None, "Fallback note expected when no 7-seater ≤ 8L"
    assert len(result.cars) > 0

    prices = [c.vehicle.price_lakhs for c in result.cars if c.vehicle.price_lakhs]
    distances = [abs(p - 8.0) for p in prices]

    # Distances must be sorted ascending (closest first)
    assert distances == sorted(distances), (
        f"Expected distances sorted ascending, got {distances} from prices {prices}"
    )


# =============================================================================
# T15 — Tie-breaking: same score → lower price ranks first
# =============================================================================

def test_tie_break_lower_price_first(catalogued_vehicles: list[VehicleRecord]) -> None:
    """
    Two cars with identical scores must be ordered by price ascending.
    Maruti Swift (6L) and Tata Tiago (6L) with identical prefs should be
    ordered Swift before Tiago (or alphabetical if also same price).

    More generally: when two ScoredCars have the same score, the one with
    the lower price_lakhs must appear first.
    """
    prefs = PreferenceSet(budget=100.0, min_seats=5)
    result = score_and_rank(prefs, catalogued_vehicles)

    for i in range(len(result.cars) - 1):
        s0, s1 = result.cars[i], result.cars[i + 1]
        if s0.score == s1.score:
            p0 = s0.vehicle.price_lakhs or float("inf")
            p1 = s1.vehicle.price_lakhs or float("inf")
            assert p0 <= p1, (
                f"Tied score ({s0.score}) but "
                f"{s0.vehicle.name}({p0}L) ranked after "
                f"{s1.vehicle.name}({p1}L)"
            )


# =============================================================================
# T16 — Stability: identical inputs → identical outputs
# =============================================================================

def test_stability_identical_inputs_produce_identical_outputs(
    catalogued_vehicles: list[VehicleRecord],
) -> None:
    """
    Calling score_and_rank with the same arguments three times must produce
    byte-identical RankingResult objects.
    """
    prefs = PreferenceSet(budget=20.0, min_seats=5, fuel_type="Electric", body_style="SUV")

    r1 = score_and_rank(prefs, catalogued_vehicles)
    r2 = score_and_rank(prefs, catalogued_vehicles)
    r3 = score_and_rank(prefs, catalogued_vehicles)

    assert r1 == r2 == r3, "score_and_rank is not deterministic"


# =============================================================================
# T17 — Top-K: never more than 5 cars returned
# =============================================================================

def test_top_k_never_more_than_5(catalogued_vehicles: list[VehicleRecord]) -> None:
    """
    Even with no soft preferences (all 35 cars satisfy hard constraints),
    the result must contain at most 5 cars.
    """
    prefs = PreferenceSet(budget=100.0, min_seats=1)
    result = score_and_rank(prefs, catalogued_vehicles)

    assert len(result.cars) <= 5, f"Expected ≤5 cars, got {len(result.cars)}"
    assert len(result.cars) >= 1, "At least one car must be returned"


# =============================================================================
# T18 — Empty preferences: all cars scored at 0 are still ranked by price
# =============================================================================

def test_empty_preferences_ranks_all_cars_by_price(
    catalogued_vehicles: list[VehicleRecord],
) -> None:
    """
    PreferenceSet with no soft preferences scores every car at 0 pts.
    The tiebreaker (lower price first) should surface cheapest cars first.
    """
    prefs = PreferenceSet(budget=100.0, min_seats=1)
    result = score_and_rank(prefs, catalogued_vehicles)

    for c in result.cars:
        assert c.score == 0, f"Expected 0 score with no preferences, got {c.score} for {c.vehicle.name}"

    # Cheapest cars should appear first
    prices = [c.vehicle.price_lakhs for c in result.cars if c.vehicle.price_lakhs]
    assert prices == sorted(prices), (
        f"Without soft prefs, cars should be sorted by price: {prices}"
    )


# =============================================================================
# T19 — None fields on VehicleRecord handled gracefully
# =============================================================================

def test_none_fields_handled_without_crash(catalogued_vehicles: list[VehicleRecord]) -> None:
    """
    VehicleRecord.price_lakhs and VehicleRecord.mileage_kmpl can be None.
    The scorer must not raise TypeError or AttributeError.
    """
    # Build a synthetic vehicle with None fields
    synthetic = VehicleRecord(
        id=999,
        name="Test Mystery Car",
        price_lakhs=None,
        fuel_type=None,
        seating_capacity=None,
        transmission=None,
        body_style=None,
        mileage_kmpl=None,
        description=None,
        features=(),
    )

    prefs = PreferenceSet(budget=100.0, min_seats=1)
    # Must not raise TypeError/AttributeError
    result = score_and_rank(prefs, [synthetic])

    # The synthetic car has no seats (None) → fails min_seats=1 → not in results
    # result.cars is empty (no hard-eligible), fallback pool has it at infinity distance
    # but fallback returns empty tuple since we passed a single-element list
    assert isinstance(result, RankingResult)
    assert isinstance(result.cars, tuple)


# =============================================================================
# T20 — Feature bonus can push score past 100
# =============================================================================

def test_feature_pts_can_push_past_100(catalogued_vehicles: list[VehicleRecord]) -> None:
    """
    Kia Seltos with fuel_type='Petrol', body_style='SUV', transmission='Automatic',
    and feature_keywords=['sunroof', 'adas', 'bose', 'ventilated'] earns:
      fuel      → +50
      body      → +50
      trans     → +30
      features  → +40 (4 keywords × 10)
    Total = 170 pts.
    """
    prefs = PreferenceSet(
        budget=100.0,
        min_seats=5,
        fuel_type="Petrol",
        body_style="SUV",
        transmission="Automatic",
        feature_keywords=("sunroof", "adas", "bose", "ventilated"),
    )
    result = score_and_rank(prefs, catalogued_vehicles)

    seltos = next((c for c in result.cars if c.vehicle.name == "Kia Seltos"), None)
    assert seltos is not None, "Kia Seltos should be in result"
    assert seltos.score > 100, f"Expected score > 100 with features bonus, got {seltos.score}"


# =============================================================================
# T21 — Multiple feature keywords each score independently
# =============================================================================

def test_multiple_feature_keywords_score_independently(
    catalogued_vehicles: list[VehicleRecord],
) -> None:
    """
    Hyundai Creta (11L, has Panoramic Sunroof, Ventilated Seats, Wireless Charging,
    Air Purifier, 6 Airbags) with feature_keywords=['sunroof', 'ventilated', 'airbag']
    should earn +30 pts (10 per matching keyword).  Use budget=15L so Creta is in top 5.
    """
    prefs = PreferenceSet(
        budget=15.0,
        min_seats=5,
        feature_keywords=("sunroof", "ventilated", "airbag"),
    )
    result = score_and_rank(prefs, catalogued_vehicles)

    creta = next((c for c in result.cars if c.vehicle.name == "Hyundai Creta"), None)
    assert creta is not None, "Hyundai Creta should be in top 5 with budget=15L"
    assert creta.feature_pts == 30, (
        f"Creta expected 30 feature_pts (3 keywords × 10), got {creta.feature_pts}"
    )


# =============================================================================
# T22 — Case-insensitive matching on fuel, body, transmission
# =============================================================================

@pytest.mark.parametrize(
    "pref_attr,car_name,pref_value",
    [
        ("fuel_type", "Tata Nexon EV", "ELECTRIC"),
        ("body_style", "Tata Nexon", "SUV"),
        ("transmission", "Kia Seltos", "AUTOMATIC"),
    ],
)
def test_case_insensitive_matching(
    catalogued_vehicles: list[VehicleRecord],
    pref_attr: str,
    car_name: str,
    pref_value: str,
) -> None:
    """
    All three soft-match fields must match case-insensitively.
    Using the actual uppercase value of the field so the car IS in the top-5.
    """
    prefs = PreferenceSet(budget=100.0, min_seats=1, **{pref_attr: pref_value})
    result = score_and_rank(prefs, catalogued_vehicles)

    car = next((c for c in result.cars if c.vehicle.name == car_name), None)
    assert car is not None, f"{car_name} should be in result when {pref_attr}={pref_value!r}"

    if pref_attr == "fuel_type":
        assert car.fuel_match is True, f"fuel_type match failed with {pref_value!r}"
    elif pref_attr == "body_style":
        assert car.body_match is True, f"body_style match failed with {pref_value!r}"
    elif pref_attr == "transmission":
        assert car.transmission_match is True, f"transmission match failed with {pref_value!r}"


# =============================================================================
# T23 — min_mileage_kmpl=None is a no-op
# =============================================================================

def test_below_min_mileage_earns_zero_pts(catalogued_vehicles: list[VehicleRecord]) -> None:
    """
    Regression: below-minimum mileage must NOT produce negative mileage_pts.

    Toyota Fortuner (Diesel, 12 kmpl) with min_mileage_kmpl=20 earns 0 mileage_pts
    (excess = -8 kmpl → raw = -48 → floored to 0).

    Using min_seats=7 + fuel_type='Diesel' to ensure Fortuner appears in the top 5:
    the 7-seater diesel cars (Innova Crysta, Safari, Scorpio-N, Fortuner) all score 50
    (fuel match only, mileage=0), and Fortuner is the 3rd-ranked 7-seater Diesel by
    price tiebreak.  This proves the floor works in the full ranking path.
    """
    prefs = PreferenceSet(
        budget=35.0,
        min_seats=7,
        min_mileage_kmpl=20.0,
        fuel_type="Diesel",
    )
    result = score_and_rank(prefs, catalogued_vehicles)

    fortuner = next((c for c in result.cars if c.vehicle.name == "Toyota Fortuner"), None)
    assert fortuner is not None, "Fortuner should be in top 5 with diesel+7seater filter"
    assert fortuner.mileage_pts == 0, (
        f"Fortuner mileage_pts must be 0 (12 kmpl vs 20 min), got {fortuner.mileage_pts}"
    )
    assert fortuner.score >= 0, f"Score must never be negative, got {fortuner.score}"
    assert fortuner.fuel_match is True, "Fortuner should match fuel_type=Diesel"

    # Primary regression assertion: mileage_pts is floored at 0, never negative.
    from backend.scorer import _score_vehicle

    fortuner_vehicle = next((v for v in catalogued_vehicles if v.name == "Toyota Fortuner"), None)
    assert fortuner_vehicle is not None
    scored = _score_vehicle(fortuner_vehicle, prefs)
    assert scored.mileage_pts == 0, (
        f"mileage_pts must be floored at 0, got {scored.mileage_pts}"
    )
    assert scored.score >= 0, f"Total score must never be negative, got {scored.score}"


def test_min_mileage_none_is_noop(catalogued_vehicles: list[VehicleRecord]) -> None:
    """
    PreferenceSet with min_mileage_kmpl=None must not award mileage pts
    to any car (not even those with high mileage).
    """
    prefs = PreferenceSet(budget=100.0, min_seats=5, min_mileage_kmpl=None)
    result = score_and_rank(prefs, catalogued_vehicles)

    for c in result.cars:
        assert c.mileage_pts == 0, (
            f"{c.vehicle.name} should have 0 mileage_pts when min_mileage_kmpl=None, "
            f"got {c.mileage_pts}"
        )


# =============================================================================
# T24 — Empty catalogue returns empty result (no cars at all)
# =============================================================================

def test_empty_catalogue_returns_empty_result() -> None:
    """
    With an empty vehicle list, hard-constraint fallback has nothing to return.
    The function must return an empty cars tuple (no crash).
    """
    prefs = PreferenceSet(budget=100.0, min_seats=1)
    result = score_and_rank(prefs, [])

    assert result.cars == ()
    assert result.note is not None, "Fallback note expected for empty catalogue"


# =============================================================================
# T25 — Score components sum to total
# =============================================================================

def test_score_components_sum_to_total(catalogued_vehicles: list[VehicleRecord]) -> None:
    """
    For every ScoredCar, the per-criterion pts must sum to the total score
    (plus feature_pts which is not capped).
    """
    prefs = PreferenceSet(
        budget=100.0,
        min_seats=5,
        fuel_type="Electric",
        body_style="SUV",
        transmission="Automatic",
        min_mileage_kmpl=0.0,
        feature_keywords=("fast charging",),
    )
    result = score_and_rank(prefs, catalogued_vehicles)

    for c in result.cars:
        base_pts = sum(filter(None, [
            50 if c.fuel_match else 0,
            50 if c.body_match else 0,
            30 if c.transmission_match else 0,
            c.mileage_pts,
        ]))
        # feature_pts adds on top of base
        assert c.score == base_pts + c.feature_pts, (
            f"{c.vehicle.name}: score={c.score} != "
            f"base({base_pts}) + feature_pts({c.feature_pts})"
        )


# =============================================================================
# T26 — Return type is frozen dataclass (immutable)
# =============================================================================

def test_ranking_result_is_immutable(catalogued_vehicles: list[VehicleRecord]) -> None:
    """
    RankingResult and ScoredCar are frozen dataclasses — regular Python attribute
    assignment must raise FrozenInstanceError.
    """
    prefs = PreferenceSet(budget=20.0, min_seats=5, fuel_type="Electric", body_style="SUV")
    result = score_and_rank(prefs, catalogued_vehicles)

    # Normal attribute assignment (not object.__setattr__) raises FrozenInstanceError
    with pytest.raises(Exception):  # FrozenInstanceError
        result.note = "hacked"  # type: ignore[misc]

    # Also verify ScoredCar is immutable
    car = result.cars[0]
    with pytest.raises(Exception):
        car.score = 999  # type: ignore[misc]


# =============================================================================
# T27 — PreferenceSet is frozen and hashable
# =============================================================================

def test_preference_set_is_hashable() -> None:
    """
    PreferenceSet must be hashable so it can be used as a dict key
    or in a cache, and so the frozen dataclass is fully immutable.
    """
    prefs = PreferenceSet(budget=20.0, min_seats=5, fuel_type="Electric", body_style="SUV")
    h = hash(prefs)
    assert isinstance(h, int)

    # Should be usable in a set/dict
    s = {prefs}
    assert prefs in s


# =============================================================================
# T28 — CNG and other fuel types score correctly
# =============================================================================

def test_cng_fuel_type_scores(catalogued_vehicles: list[VehicleRecord]) -> None:
    """
    PreferenceSet(fuel_type='CNG') should match Tata Tiago CNG and Maruti Ertiga CNG.
    """
    prefs = PreferenceSet(budget=100.0, min_seats=5, fuel_type="CNG")
    result = score_and_rank(prefs, catalogued_vehicles)

    cng_cars = [c for c in result.cars if c.fuel_match]
    assert len(cng_cars) >= 2, "Should match at least Tata Tiago CNG and Maruti Ertiga CNG"
    for c in cng_cars:
        assert c.fuel_match is True
        assert c.score >= 50, f"{c.vehicle.name} should have >= 50 pts from CNG match"


# =============================================================================
# T29 — Diesel body_style=SUV with fuel_type=Diesel scores correctly
# =============================================================================

def test_diesel_suv_scores(catalogued_vehicles: list[VehicleRecord]) -> None:
    """
    Mahindra Thar is Diesel + SUV — should score 100 with fuel_type=Diesel, body_style=SUV.
    """
    prefs = PreferenceSet(budget=100.0, min_seats=5, fuel_type="Diesel", body_style="SUV")
    result = score_and_rank(prefs, catalogued_vehicles)

    thar = next((c for c in result.cars if c.vehicle.name == "Mahindra Thar"), None)
    assert thar is not None, "Mahindra Thar should be in result"
    assert thar.fuel_match is True
    assert thar.body_match is True
    assert thar.score == 100, f"Thar expected 100 pts (fuel+body), got {thar.score}"


# =============================================================================
# T30 — Work example with feature keyword boost
# =============================================================================

def test_worked_example_with_feature_keyword_boost(
    catalogued_vehicles: list[VehicleRecord],
) -> None:
    """
    PreferenceSet(budget=20, min_seats=5, fuel_type='Electric', body_style='SUV',
    feature_keywords=['fast charging']).

    Tata Nexon EV has "Fast Charging" in features → +10 pts.
    MG ZS EV (23L, over-budget) → excluded.

    Expected: Nexon EV at rank 1 with score = 100 + 10 = 110.
    """
    prefs = PreferenceSet(
        budget=20.0,
        min_seats=5,
        fuel_type="Electric",
        body_style="SUV",
        feature_keywords=("fast charging",),
    )
    result = score_and_rank(prefs, catalogued_vehicles)

    assert result.note is None
    nexon_ev = next(c for c in result.cars if c.vehicle.name == "Tata Nexon EV")
    assert nexon_ev.fuel_match is True
    assert nexon_ev.body_match is True
    assert nexon_ev.feature_pts == 10, f"Expected 10 feature_pts, got {nexon_ev.feature_pts}"
    assert nexon_ev.score == 110, f"Expected 110 pts (100+10), got {nexon_ev.score}"
    assert result.cars[0].vehicle.name == "Tata Nexon EV"
