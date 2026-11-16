"""
backend/tests/test_matcher_consumer.py — Unit tests for Issue 022.

Test plan
─────────
M1  Init: empty state on creation.
M2  Init: start() launches background task.
M3  State: get_current_preferences() returns None before first turn.
M4  State: get_top_cars() returns empty tuple before first turn.
M5  State: is_usable() returns False before first turn.
M6  None coercion: budget=None → float('inf').
M7  None coercion: min_seats=None → 1.
M8  None coercion: feature_keywords (list) → tuple.
M9  None coercion: all-None delta is handled gracefully (filler).
M10 Strict ordering: two sequential turns, latest wins.
M11 Strict ordering: three rapid contradicting turns, final state is last.
M12 Strict ordering: five rapid contradicting turns, final state is last (canonical).
M13 Strict ordering: concurrent ingestion — enqueue-all-then-drain, no out-of-order clobbering.
M14 Strict ordering: preserves non-contradicted fields.
M15 Usable: hard constraint (finite budget) → usable.
M16 Usable: hard constraint (min_seats > 1) → usable.
M17 Usable: ≥2 soft preferences → usable (no hard constraint needed).
M18 Usable: 0 hard + 1 soft → not usable.
M19 Usable: empty set → not usable.
M20 Usable: exactly budget=inf + min_seats=1 + 2 soft → usable.
M21 Usable: exactly budget=inf + min_seats=1 + 1 soft → not usable.
M22 Scoring: one turn produces non-empty ranked cars.
M23 Scoring: cars are returned as ScoredCar tuples.
M24 Scoring: filler turn (all-null) leaves previous scored cars unchanged.
M25 Scoring: new turn after filler updates cars correctly.
M26 Ranking note: present when hard constraints unsatisfiable.
M27 Ranking note: absent when hard constraints satisfiable.
M28 Drain: drain() waits for all queued turns to complete.
M29 Close: graceful shutdown drains pending turns.
M30 Reuse: fresh task created after close() + start().
M31 Soft preference count: feature_keywords non-empty counts as one.

Concurrency Testing Pattern
──────────────────────────
enqueue_turn() is NON-BLOCKING — it returns immediately after put_nowait.
Tests verify correct ordering by enqueueing all turns rapidly without awaiting,
then calling drain() to wait for the consumer to process them all, and finally
asserting the final state reflects the last enqueued turn.  This exercises the
true concurrent ingestion path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.catalogue import VehicleRecord, load_catalogue
from backend.matcher_consumer import MatcherConsumer
from backend.preference_extractor import (
    GeminiClient,
    _PREFERENCE_KEYS,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def catalogued_vehicles() -> list[VehicleRecord]:
    """Load the real vehicles.db once per test."""
    db_path = (
        Path(__file__).parent.parent.parent
        / "database"
        / "vehicles.db"
    ).resolve()
    load_catalogue(db_path)
    from backend import catalogue
    return list(catalogue.vehicles)


@pytest.fixture
def mock_extractor() -> MockGeminiClient:
    """Fresh MockGeminiClient per test."""
    return MockGeminiClient()


# ─── Mock extractor ─────────────────────────────────────────────────────────────


class MockGeminiClient(GeminiClient):
    """
    Record-and-replay mock for the preference extractor.

    Set ``responses`` to a list of dicts; each ``extract()`` call pops the next.
    If the list is exhausted the final response is returned (safe filler).
    """

    def __init__(self, responses: list[dict] | None = None) -> None:
        self._responses: list[dict] = list(responses) if responses is not None else []
        self.calls: list[tuple[str, dict | None]] = []

    def extract(self, utterance: str, current_prefs: dict | None) -> dict:
        self.calls.append((utterance, current_prefs))
        if self._responses:
            return self._responses.pop(0)
        return {k: None for k in _PREFERENCE_KEYS}


# ─── Test helpers ────────────────────────────────────────────────────────────────


def _delta(
    budget: float | None = None,
    min_seats: int | None = None,
    fuel_type: str | None = None,
    body_style: str | None = None,
    transmission: str | None = None,
    min_mileage_kmpl: float | None = None,
    feature_keywords: list[str] | None = None,
) -> dict:
    """Build a well-formed extractor delta dict."""
    return {
        "budget": budget,
        "min_seats": min_seats,
        "fuel_type": fuel_type,
        "body_style": body_style,
        "transmission": transmission,
        "min_mileage_kmpl": min_mileage_kmpl,
        "feature_keywords": feature_keywords if feature_keywords is not None else [],
    }


# =============================================================================
# M1-M5 — Initialization and empty state
# =============================================================================

class TestInitialization:
    """Consumer starts in a clean, empty state."""

    def test_empty_prefs_on_creation(
        self,
        catalogued_vehicles: list[VehicleRecord],
        mock_extractor: MockGeminiClient,
    ) -> None:
        """get_current_preferences() is None before any turn."""
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock_extractor,  # type: ignore[arg-type]
        )
        assert consumer.get_current_preferences() is None

    def test_empty_cars_on_creation(
        self,
        catalogued_vehicles: list[VehicleRecord],
        mock_extractor: MockGeminiClient,
    ) -> None:
        """get_top_cars() is empty tuple before any turn."""
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock_extractor,  # type: ignore[arg-type]
        )
        assert consumer.get_top_cars() == ()

    def test_not_usable_on_creation(
        self,
        catalogued_vehicles: list[VehicleRecord],
        mock_extractor: MockGeminiClient,
    ) -> None:
        """is_usable() is False before any turn."""
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock_extractor,  # type: ignore[arg-type]
        )
        assert consumer.is_usable() is False

    def test_ranking_note_none_on_creation(
        self,
        catalogued_vehicles: list[VehicleRecord],
        mock_extractor: MockGeminiClient,
    ) -> None:
        """get_ranking_note() is None before any turn."""
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock_extractor,  # type: ignore[arg-type]
        )
        assert consumer.get_ranking_note() is None

    @pytest.mark.asyncio
    async def test_consumer_task_launched(
        self,
        catalogued_vehicles: list[VehicleRecord],
        mock_extractor: MockGeminiClient,
    ) -> None:
        """start() creates a non-None background task."""
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock_extractor,  # type: ignore[arg-type]
        )
        consumer.start()
        assert consumer._task is not None
        assert not consumer._task.done()
        await consumer.close()


# =============================================================================
# M6-M9 — None coercion boundary
# =============================================================================

class TestNoneCoercion:
    """
    Before calling the scorer, the consumer coerces None hard constraints:

        budget=None          → float('inf')
        min_seats=None       → 1
        feature_keywords(list)→ tuple

    We verify this by checking the scoring result reflects the coerced values.
    """

    @pytest.mark.asyncio
    async def test_budget_none_coerced_to_inf(
        self,
        catalogued_vehicles: list[VehicleRecord],
    ) -> None:
        """
        When budget=None (all-null delta), the scorer receives budget=inf,
        so every car in the catalogue is within budget and the result is
        non-empty even for a very cheap car (e.g. Maruti Swift 6L).
        """
        mock = MockGeminiClient(responses=[_delta()])  # all-null
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock,  # type: ignore[arg-type]
        )
        consumer.start()

        consumer.enqueue_turn("hello riya")
        await consumer.drain()
        cars = consumer.get_top_cars()

        # With budget=inf, every car is in-bounds; at least Swift (6L) should appear
        names = {c.vehicle.name for c in cars}
        assert "Maruti Suzuki Swift" in names, (
            "With budget=inf, cheap cars must appear — budget=None was coerced to inf"
        )
        await consumer.close()

    @pytest.mark.asyncio
    async def test_min_seats_none_coerced_to_one(
        self,
        catalogued_vehicles: list[VehicleRecord],
    ) -> None:
        """
        When min_seats=None in the extractor delta, the scorer receives min_seats=1,
        so even 5-seaters satisfy the constraint.

        Note: with budget=8L + fuel_type=Petrol, the top-5 are the five cheapest
        petrol cars (all 6L).  Nexon at 8L loses the price tiebreaker to 6L cars,
        so we assert on a known 5-seater (Maruti Suzuki Swift = 5 seats, 6L, Petrol).
        """
        mock = MockGeminiClient(
            responses=[_delta(budget=8.0, min_seats=None, fuel_type="Petrol")]
        )
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock,  # type: ignore[arg-type]
        )
        consumer.start()

        consumer.enqueue_turn("budget 8 lakhs, petrol")
        await consumer.drain()
        prefs = consumer.get_current_preferences()

        # Extractor left min_seats as None
        assert prefs is not None
        assert prefs["min_seats"] is None

        cars = consumer.get_top_cars()
        # 5-seaters must appear because min_seats was coerced to 1
        names = {c.vehicle.name for c in cars}
        assert "Maruti Suzuki Swift" in names, (
            "5-seater Swift must appear — min_seats=None was coerced to 1. "
            f"Got: {names}"
        )
        await consumer.close()

    @pytest.mark.asyncio
    async def test_feature_keywords_list_coerced_to_tuple(
        self,
        catalogued_vehicles: list[VehicleRecord],
    ) -> None:
        """
        The extractor returns feature_keywords as a list; the consumer must
        coerce it to a tuple before passing to the frozen PreferenceSet.
        """
        mock = MockGeminiClient(
            responses=[_delta(
                budget=20.0,
                min_seats=5,
                fuel_type="Electric",
                body_style="SUV",
                feature_keywords=["sunroof", "fast charging"],
            )]
        )
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock,  # type: ignore[arg-type]
        )
        consumer.start()

        consumer.enqueue_turn("electric SUV with sunroof and fast charging")
        await consumer.drain()

        cars = consumer.get_top_cars()
        # Tata Nexon EV (Electric SUV, 14L) should be in top cars
        names = {c.vehicle.name for c in cars}
        assert "Tata Nexon EV" in names
        # Fast Charging bonus applied — Nexon EV has "Fast Charging" in features
        nexon = next(c for c in cars if c.vehicle.name == "Tata Nexon EV")
        assert nexon.feature_pts >= 10, (
            "Feature keyword 'fast charging' must score +10 pts — "
            "list was coerced to tuple correctly"
        )
        await consumer.close()

    @pytest.mark.asyncio
    async def test_filler_turn_handled_without_error(
        self,
        catalogued_vehicles: list[VehicleRecord],
    ) -> None:
        """All-null delta (filler) must not crash the consumer."""
        mock = MockGeminiClient(responses=[_delta()])
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock,  # type: ignore[arg-type]
        )
        consumer.start()

        # Should not raise
        consumer.enqueue_turn("okay thanks")
        await consumer.drain()
        assert consumer.get_current_preferences() is not None
        assert isinstance(consumer.get_top_cars(), tuple)
        await consumer.close()


# =============================================================================
# M10-M14 — Strict ordered processing / rapid contradicting turns
# =============================================================================

class TestStrictOrderedProcessing:
    """
    The consumer processes turns strictly in arrival order.  Rapid contradicting
    statements must resolve to the latest value without out-of-order clobbering.

    Tests use the enqueue-all-then-drain pattern to exercise true concurrent
    ingestion: enqueue_turn() returns immediately so all turns are placed on
    the queue before any processing begins.  drain() waits for the serialised
    consumer to finish processing them all, and the final state must reflect
    the last enqueued turn.
    """

    @pytest.mark.asyncio
    async def test_second_turn_overwrites_first(
        self,
        catalogued_vehicles: list[VehicleRecord],
    ) -> None:
        """
        Turn 1: "I want petrol"
        Turn 2: "actually diesel is fine"

        Final fuel_type must be Diesel.
        """
        mock = MockGeminiClient(responses=[
            _delta(fuel_type="Petrol"),
            _delta(fuel_type="Diesel"),
        ])
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock,  # type: ignore[arg-type]
        )
        consumer.start()

        consumer.enqueue_turn("I want petrol")
        consumer.enqueue_turn("actually diesel is fine")
        await consumer.drain()

        prefs = consumer.get_current_preferences()
        assert prefs is not None
        assert prefs["fuel_type"] == "Diesel", "Latest turn must win — fuel_type should be Diesel"
        await consumer.close()

    @pytest.mark.asyncio
    async def test_three_rapid_contradicting_turns_last_wins(
        self,
        catalogued_vehicles: list[VehicleRecord],
    ) -> None:
        """
        Three turns enqueued rapidly, drain() waits for all:
          1. "petrol car please"          → fuel_type=Petrol
          2. "actually make it diesel"   → fuel_type=Diesel
          3. "wait no, electric is fine" → fuel_type=Electric

        Final fuel_type must be Electric.
        """
        mock = MockGeminiClient(responses=[
            _delta(fuel_type="Petrol"),
            _delta(fuel_type="Diesel"),
            _delta(fuel_type="Electric"),
        ])
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock,  # type: ignore[arg-type]
        )
        consumer.start()

        consumer.enqueue_turn("petrol car please")
        consumer.enqueue_turn("actually make it diesel")
        consumer.enqueue_turn("wait no, electric is fine")
        await consumer.drain()

        prefs = consumer.get_current_preferences()
        assert prefs is not None
        assert prefs["fuel_type"] == "Electric", (
            "Third rapid turn must win — consumer processes strictly in order"
        )
        await consumer.close()

    @pytest.mark.asyncio
    async def test_rapid_contradicting_turns_no_out_of_order_clobbering(
        self,
        catalogued_vehicles: list[VehicleRecord],
    ) -> None:
        """
        Canonical test for the strict ordered processing contract.

        Five turns are enqueued in rapid succession (no awaiting between
        enqueues — enqueue_turn is non-blocking).  drain() waits for the
        serialised consumer to process them all in order.  The final state
        must be exactly the 5th turn's result; intermediate turns must not
        clobber the final state.
        """
        mock = MockGeminiClient(responses=[
            _delta(budget=10.0, fuel_type="Petrol"),
            _delta(budget=15.0, fuel_type="Diesel"),
            _delta(budget=12.0, fuel_type="Electric"),
            _delta(budget=20.0, fuel_type="CNG"),
            _delta(budget=8.0, fuel_type="Petrol"),
        ])
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock,  # type: ignore[arg-type]
        )
        consumer.start()

        # Enqueue all 5 turns without awaiting — this is the key concurrency test.
        # enqueue_turn returns immediately, so all items land on the queue in order
        # before the consumer has processed any of them.
        consumer.enqueue_turn("turn 0 — petrol 10L")
        consumer.enqueue_turn("turn 1 — diesel 15L")
        consumer.enqueue_turn("turn 2 — electric 12L")
        consumer.enqueue_turn("turn 3 — cng 20L")
        consumer.enqueue_turn("turn 4 — petrol 8L")
        await consumer.drain()

        final_prefs = consumer.get_current_preferences()
        assert final_prefs is not None

        # The 5th (final) turn's values must be the final state
        assert final_prefs["budget"] == 8.0, (
            f"Expected budget=8.0 from turn 5, got {final_prefs['budget']} — "
            "out-of-order clobbering detected"
        )
        assert final_prefs["fuel_type"] == "Petrol", (
            f"Expected fuel_type=Petrol from turn 5, got {final_prefs['fuel_type']} — "
            "out-of-order clobbering detected"
        )

        # Intermediate turns must NOT be observable after the 5th completes
        assert final_prefs["budget"] != 10.0, "Turn 1 budget must be overwritten"
        assert final_prefs["budget"] != 15.0, "Turn 2 budget must be overwritten"
        assert final_prefs["budget"] != 12.0, "Turn 3 budget must be overwritten"
        assert final_prefs["budget"] != 20.0, "Turn 4 budget must be overwritten"
        await consumer.close()

    @pytest.mark.asyncio
    async def test_preserves_non_contradicted_fields(
        self,
        catalogued_vehicles: list[VehicleRecord],
    ) -> None:
        """
        When a later turn only changes one field, other fields carry over.
        Turn 1: budget=15, fuel_type=Petrol
        Turn 2: fuel_type=Diesel (budget carries over)

        Final: budget=15, fuel_type=Diesel
        """
        mock = MockGeminiClient(responses=[
            _delta(budget=15.0, fuel_type="Petrol"),
            _delta(fuel_type="Diesel"),  # budget not mentioned — carries over
        ])
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock,  # type: ignore[arg-type]
        )
        consumer.start()

        consumer.enqueue_turn("budget 15 lakhs petrol")
        consumer.enqueue_turn("actually diesel")
        await consumer.drain()

        final = consumer.get_current_preferences()
        assert final is not None
        assert final["budget"] == 15.0, "Budget must carry over from turn 1"
        assert final["fuel_type"] == "Diesel", "Fuel type must be updated to Diesel"
        await consumer.close()


# =============================================================================
# M15-M21 — Usability policy
# =============================================================================

class TestUsabilityPolicy:
    """
    is_usable() returns True when the running set is actionable:
      - At least one hard constraint (finite budget OR min_seats > 1), OR
      - At least 2 soft preferences captured.
    """

    @pytest.mark.asyncio
    async def test_finite_budget_is_usable(
        self,
        catalogued_vehicles: list[VehicleRecord],
    ) -> None:
        """budget < inf makes the set usable (hard constraint)."""
        mock = MockGeminiClient(responses=[_delta(budget=15.0)])
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock,  # type: ignore[arg-type]
        )
        consumer.start()
        consumer.enqueue_turn("under 15 lakhs")
        await consumer.drain()
        assert consumer.is_usable() is True
        await consumer.close()

    @pytest.mark.asyncio
    async def test_min_seats_7_is_usable(
        self,
        catalogued_vehicles: list[VehicleRecord],
    ) -> None:
        """min_seats > 1 (i.e. explicitly 7) makes the set usable (hard constraint)."""
        mock = MockGeminiClient(responses=[_delta(min_seats=7)])
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock,  # type: ignore[arg-type]
        )
        consumer.start()
        consumer.enqueue_turn("need 7 seats")
        await consumer.drain()
        assert consumer.is_usable() is True
        await consumer.close()

    @pytest.mark.asyncio
    async def test_two_soft_prefs_usable(
        self,
        catalogued_vehicles: list[VehicleRecord],
    ) -> None:
        """≥2 soft preferences (no hard constraint) → usable."""
        mock = MockGeminiClient(responses=[
            _delta(fuel_type="Electric", body_style="SUV"),
        ])
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock,  # type: ignore[arg-type]
        )
        consumer.start()
        consumer.enqueue_turn("I want electric SUV")
        await consumer.drain()
        assert consumer.is_usable() is True
        await consumer.close()

    @pytest.mark.asyncio
    async def test_one_soft_pref_not_usable(
        self,
        catalogued_vehicles: list[VehicleRecord],
    ) -> None:
        """Exactly 1 soft preference → not usable."""
        mock = MockGeminiClient(responses=[_delta(fuel_type="Petrol")])
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock,  # type: ignore[arg-type]
        )
        consumer.start()
        consumer.enqueue_turn("petrol car")
        await consumer.drain()
        assert consumer.is_usable() is False
        await consumer.close()

    @pytest.mark.asyncio
    async def test_empty_set_not_usable(
        self,
        catalogued_vehicles: list[VehicleRecord],
    ) -> None:
        """All-null prefs (filler only) → not usable."""
        mock = MockGeminiClient(responses=[_delta()])
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock,  # type: ignore[arg-type]
        )
        consumer.start()
        consumer.enqueue_turn("okay")
        await consumer.drain()
        assert consumer.is_usable() is False
        await consumer.close()

    @pytest.mark.asyncio
    async def test_two_soft_including_feature_keywords_usable(
        self,
        catalogued_vehicles: list[VehicleRecord],
    ) -> None:
        """fuel_type + feature_keywords = 2 soft prefs → usable."""
        mock = MockGeminiClient(responses=[
            _delta(fuel_type="Petrol", feature_keywords=["sunroof"]),
        ])
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock,  # type: ignore[arg-type]
        )
        consumer.start()
        consumer.enqueue_turn("petrol with sunroof")
        await consumer.drain()
        assert consumer.is_usable() is True
        await consumer.close()

    @pytest.mark.asyncio
    async def test_min_seats_1_not_a_hard_constraint(
        self,
        catalogued_vehicles: list[VehicleRecord],
    ) -> None:
        """min_seats not set (None) + 1 soft → not usable (1 is the no-constraint sentinel)."""
        mock = MockGeminiClient(responses=[_delta(fuel_type="Diesel")])
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock,  # type: ignore[arg-type]
        )
        consumer.start()
        consumer.enqueue_turn("diesel car")
        await consumer.drain()
        assert consumer.is_usable() is False, (
            "min_seats=None is not a hard constraint; 1 soft pref is insufficient"
        )
        await consumer.close()

    @pytest.mark.asyncio
    async def test_budget_inf_and_min_seats_1_with_two_soft_usable(
        self,
        catalogued_vehicles: list[VehicleRecord],
    ) -> None:
        """No hard constraint set + 2 soft prefs → usable."""
        mock = MockGeminiClient(responses=[
            _delta(fuel_type="Petrol", body_style="SUV"),
        ])
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock,  # type: ignore[arg-type]
        )
        consumer.start()
        consumer.enqueue_turn("petrol SUV")
        await consumer.drain()
        assert consumer.is_usable() is True
        await consumer.close()

    @pytest.mark.asyncio
    async def test_budget_inf_and_min_seats_1_with_one_soft_not_usable(
        self,
        catalogued_vehicles: list[VehicleRecord],
    ) -> None:
        """No hard constraint set + 1 soft → not usable."""
        mock = MockGeminiClient(responses=[_delta(transmission="Automatic")])
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock,  # type: ignore[arg-type]
        )
        consumer.start()
        consumer.enqueue_turn("automatic car")
        await consumer.drain()
        assert consumer.is_usable() is False
        await consumer.close()


# =============================================================================
# M22-M27 — Scoring results
# =============================================================================

class TestScoringResults:
    """The consumer produces ranked cars and optional fallback notes."""

    @pytest.mark.asyncio
    async def test_one_turn_produces_ranked_cars(
        self,
        catalogued_vehicles: list[VehicleRecord],
    ) -> None:
        """A single complete turn produces a non-empty ranked car list."""
        mock = MockGeminiClient(responses=[
            _delta(budget=20.0, min_seats=5, fuel_type="Electric", body_style="SUV"),
        ])
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock,  # type: ignore[arg-type]
        )
        consumer.start()
        consumer.enqueue_turn("electric SUV under 20 lakhs")
        await consumer.drain()

        cars = consumer.get_top_cars()
        assert len(cars) >= 1
        assert cars == consumer.get_top_cars()  # deterministic
        await consumer.close()

    @pytest.mark.asyncio
    async def test_cars_are_scoredcar_tuples(
        self,
        catalogued_vehicles: list[VehicleRecord],
    ) -> None:
        """get_top_cars() returns a tuple of ScoredCar instances."""
        mock = MockGeminiClient(responses=[_delta(budget=15.0)])
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock,  # type: ignore[arg-type]
        )
        consumer.start()
        consumer.enqueue_turn("under 15 lakhs")
        await consumer.drain()

        cars = consumer.get_top_cars()
        assert isinstance(cars, tuple)
        for car in cars:
            assert hasattr(car, "score")
            assert hasattr(car, "vehicle")
            assert hasattr(car, "fuel_match")
        await consumer.close()

    @pytest.mark.asyncio
    async def test_filler_preserves_previous_cars(
        self,
        catalogued_vehicles: list[VehicleRecord],
    ) -> None:
        """A filler turn (all-null) must not clear the previously scored cars."""
        mock = MockGeminiClient(responses=[
            _delta(budget=15.0, min_seats=5, fuel_type="Electric", body_style="SUV"),
            _delta(),  # filler
        ])
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock,  # type: ignore[arg-type]
        )
        consumer.start()

        consumer.enqueue_turn("electric SUV under 15 lakhs")
        consumer.enqueue_turn("okay thanks")
        await consumer.drain()

        cars = consumer.get_top_cars()
        assert len(cars) >= 1, "Filler must not clear previously scored cars"
        await consumer.close()

    @pytest.mark.asyncio
    async def test_new_turn_updates_cars(
        self,
        catalogued_vehicles: list[VehicleRecord],
    ) -> None:
        """A new turn after filler updates the car list."""
        mock = MockGeminiClient(responses=[
            _delta(budget=15.0, min_seats=5, fuel_type="Electric"),
            _delta(),  # filler
            _delta(budget=15.0, min_seats=5, fuel_type="Diesel"),
        ])
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock,  # type: ignore[arg-type]
        )
        consumer.start()

        consumer.enqueue_turn("electric under 15")
        consumer.enqueue_turn("okay")
        consumer.enqueue_turn("actually diesel")
        await consumer.drain()

        cars = consumer.get_top_cars()
        assert len(cars) >= 1
        # Top car should be a Diesel vehicle
        assert cars[0].fuel_match is True or any(c.vehicle.fuel_type == "Diesel" for c in cars), (
            "Final turn should produce Diesel-scored cars"
        )
        await consumer.close()

    @pytest.mark.asyncio
    async def test_ranking_note_absent_when_hard_constraints_satisfied(
        self,
        catalogued_vehicles: list[VehicleRecord],
    ) -> None:
        """When cars satisfy hard constraints, no fallback note is set."""
        mock = MockGeminiClient(responses=[
            _delta(budget=15.0, min_seats=5),
        ])
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock,  # type: ignore[arg-type]
        )
        consumer.start()
        consumer.enqueue_turn("under 15 lakhs")
        await consumer.drain()
        assert consumer.get_ranking_note() is None
        await consumer.close()

    @pytest.mark.asyncio
    async def test_ranking_note_present_when_hard_constraints_unsatisfiable(
        self,
        catalogued_vehicles: list[VehicleRecord],
    ) -> None:
        """When no car satisfies hard constraints, a fallback note is set."""
        mock = MockGeminiClient(responses=[
            _delta(budget=1.0, min_seats=7),
        ])
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock,  # type: ignore[arg-type]
        )
        consumer.start()
        consumer.enqueue_turn("1 lakh 7 seater")
        await consumer.drain()
        note = consumer.get_ranking_note()
        assert note is not None, (
            "Fallback note must be present when hard constraints are unsatisfiable"
        )
        assert "budget" in note.lower() or "price" in note.lower()
        await consumer.close()


# =============================================================================
# M28-M30 — Lifecycle / drain / close / reuse
# =============================================================================

class TestLifecycleAndConcurrency:
    """Consumer lifecycle: drain, close, and reuse."""

    @pytest.mark.asyncio
    async def test_drain_waits_for_queue(
        self,
        catalogued_vehicles: list[VehicleRecord],
    ) -> None:
        """drain() completes only after all enqueued turns are processed."""
        mock = MockGeminiClient(responses=[
            _delta(budget=10.0),
            _delta(min_seats=5),
            _delta(fuel_type="Petrol"),
        ])
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock,  # type: ignore[arg-type]
        )
        consumer.start()

        consumer.enqueue_turn("turn 0")
        consumer.enqueue_turn("turn 1")
        consumer.enqueue_turn("turn 2")
        await consumer.drain()

        # All three responses consumed
        assert len(mock.calls) == 3
        prefs = consumer.get_current_preferences()
        assert prefs is not None
        assert prefs["fuel_type"] == "Petrol"
        await consumer.close()

    @pytest.mark.asyncio
    async def test_close_is_graceful(
        self,
        catalogued_vehicles: list[VehicleRecord],
    ) -> None:
        """close() must not raise — even when the queue is empty."""
        mock = MockGeminiClient()
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock,  # type: ignore[arg-type]
        )
        consumer.start()
        # No turns enqueued — close should still be clean
        await consumer.close()
        assert consumer._closed is True

    @pytest.mark.asyncio
    async def test_pending_turns_drained_on_close(
        self,
        catalogued_vehicles: list[VehicleRecord],
    ) -> None:
        """close() must drain pending turns before stopping the loop."""
        mock = MockGeminiClient(responses=[_delta(budget=15.0)])
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock,  # type: ignore[arg-type]
        )
        consumer.start()
        consumer.enqueue_turn("15 lakhs")
        await consumer.close()
        assert consumer.get_current_preferences() is not None
        assert consumer.get_current_preferences()["budget"] == 15.0

    @pytest.mark.asyncio
    async def test_fresh_task_after_close_and_restart(
        self,
        catalogued_vehicles: list[VehicleRecord],
    ) -> None:
        """After close(), calling start() again creates a new task and resets _closed."""
        mock = MockGeminiClient()
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock,  # type: ignore[arg-type]
        )
        consumer.start()
        old_task = consumer._task
        await consumer.close()
        assert consumer._closed is True

        consumer.start()
        assert consumer._closed is False, "start() must reset _closed to False"
        assert consumer._task is not None
        assert consumer._task is not old_task
        await consumer.close()


# =============================================================================
# M31 — Feature keywords usability edge case
# =============================================================================

class TestFeatureKeywordsUsability:
    """Feature keywords count as exactly one soft preference."""

    @pytest.mark.asyncio
    async def test_feature_keywords_non_empty_counts_as_one_soft(
        self,
        catalogued_vehicles: list[VehicleRecord],
    ) -> None:
        """feature_keywords (non-empty) + 1 other soft → usable."""
        mock = MockGeminiClient(responses=[
            _delta(feature_keywords=["sunroof"], transmission="Automatic"),
        ])
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock,  # type: ignore[arg-type]
        )
        consumer.start()
        consumer.enqueue_turn("automatic with sunroof")
        await consumer.drain()
        assert consumer.is_usable() is True
        await consumer.close()

    @pytest.mark.asyncio
    async def test_only_feature_keywords_not_usable(
        self,
        catalogued_vehicles: list[VehicleRecord],
    ) -> None:
        """Only feature_keywords (1 soft pref) → not usable."""
        mock = MockGeminiClient(responses=[
            _delta(feature_keywords=["sunroof"]),
        ])
        consumer = MatcherConsumer(
            vehicles=catalogued_vehicles,
            extractor_cls=lambda: mock,  # type: ignore[arg-type]
        )
        consumer.start()
        consumer.enqueue_turn("I want a sunroof")
        await consumer.drain()
        assert consumer.is_usable() is False
        await consumer.close()
