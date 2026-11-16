"""
backend/tests/test_blurb_formatter.py — Unit tests for Issue 023.

The Gemini boundary is fully mocked via ``MockGeminiClient``.
No live API calls are made in this suite.

Test coverage:
  • Successful formatting with catalogue enrichment.
  • Empty ranked list returns None.
  • Non-fatal failure fallback: API error → None (ranked items still stored).
  • Non-fatal failure fallback: malformed output → None.
  • Catalogue fields are correctly resolved and passed to Gemini.
  • Rank order is preserved.
  • Score and match flags are included in the enriched data.
  • Feature list is always a list (not tuple).
  • Multiple cars are all enriched in order.
  • GeminiRestClient._clean_response strips markdown fences.
"""

from __future__ import annotations

import pytest

from backend.blurb_formatter import (
    GeminiClient,
    format_blurb,
    _enrich_scored_car,
    GeminiRestClient,
)
from backend.catalogue import VehicleRecord
from backend.scorer import ScoredCar


# ─── Fixtures ────────────────────────────────────────────────────────────────


def _make_vehicle(
    vid: int,
    name: str,
    price: float = 10.0,
    fuel: str = "Petrol",
    seats: int = 5,
    transmission: str = "Manual",
    body: str = "Hatchback",
    mileage: float = 15.0,
    description: str = "A great car.",
    features: list[str] | tuple[str, ...] | str = ("Sunroof", "Bluetooth"),
) -> VehicleRecord:
    """Helper to create a VehicleRecord for tests."""
    return VehicleRecord(
        id=vid,
        name=name,
        price_lakhs=price,
        fuel_type=fuel,
        seating_capacity=seats,
        transmission=transmission,
        body_style=body,
        mileage_kmpl=mileage,
        description=description,
        features=features,
    )


def _make_scored(
    vehicle: VehicleRecord,
    score: int = 80,
    fuel_match: bool = False,
    body_match: bool = False,
    transmission_match: bool = False,
    mileage_pts: int = 0,
    feature_pts: int = 0,
) -> ScoredCar:
    """Helper to create a ScoredCar for tests."""
    return ScoredCar(
        vehicle=vehicle,
        score=score,
        fuel_match=fuel_match,
        body_match=body_match,
        transmission_match=transmission_match,
        mileage_pts=mileage_pts,
        feature_pts=feature_pts,
    )


# ─── Mock ───────────────────────────────────────────────────────────────────


class MockGeminiClient(GeminiClient):
    """
    Record-and-replay mock that never touches a live API.

    Set ``response`` to the string Gemini should return.
    Set ``should_fail`` to True to simulate an API error.
    Set ``raise_on_format`` to an Exception class to simulate failure in ``format()``.
    """

    def __init__(
        self,
        response: str = "Here are your top picks:\n\n1. Swift VXi\n\n2. Baleno Delta",
        *,
        should_fail: bool = False,
        raise_on_format: type[Exception] | None = None,
    ) -> None:
        self.response = response
        self.should_fail = should_fail
        self.raise_on_format = raise_on_format
        self.calls: list[list[dict]] = []
        self.last_note: str | None = None

    def format(self, enriched_cars: list[dict], *, note: str | None = None) -> str | None:
        self.calls.append(list(enriched_cars))  # copy so mutation can't poison assertions
        self.last_note = note
        if self.raise_on_format is not None:
            raise self.raise_on_format("Simulated Gemini error")
        if self.should_fail:
            raise RuntimeError("Simulated API error")
        # Mirror GeminiRestClient._clean_response normalization:
        # empty / whitespace-only responses are treated identically to failures.
        raw = self.response
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            content_lines = [
                ln for ln in lines
                if not ln.strip().startswith("```") and ln.strip()
            ]
            text = "\n".join(content_lines).strip()
        return text if text else None


# ─── Tests: successful formatting ────────────────────────────────────────────


class TestSuccessfulFormatting:
    """format_blurb returns the Gemini output when the call succeeds."""

    def test_returns_gemini_output(self) -> None:
        """A successful Gemini call returns its text unchanged."""
        vehicle = _make_vehicle(1, "Maruti Swift VXi", price=7.5)
        ranked = [_make_scored(vehicle, score=80)]
        mock = MockGeminiClient(response="Great choice: Swift VXi at 7.5 lakhs.")

        result = format_blurb(ranked, client=mock)

        assert result == "Great choice: Swift VXi at 7.5 lakhs."

    def test_gemini_called_with_enriched_data(self) -> None:
        """The mock receives enriched car dicts, not raw ScoredCars."""
        vehicle = _make_vehicle(
            vid=3,
            name="Hyundai Creta 1.6 SX",
            price=14.2,
            fuel="Diesel",
            seats=5,
            transmission="Automatic",
            body="SUV",
            mileage=17.0,
            description="Premium SUV.",
            features=["Sunroof", "Rear Camera", "ADAS"],
        )
        scored = ScoredCar(
            vehicle=vehicle,
            score=130,
            fuel_match=True,
            body_match=True,
            transmission_match=True,
            mileage_pts=12,
            feature_pts=20,
        )
        mock = MockGeminiClient(response="Top pick: Creta 1.6 SX.")

        format_blurb([scored], client=mock)

        assert len(mock.calls) == 1
        [enriched] = mock.calls[0]
        assert enriched["rank"] == 1
        assert enriched["name"] == "Hyundai Creta 1.6 SX"
        assert enriched["price_lakhs"] == 14.2
        assert enriched["fuel_type"] == "Diesel"
        assert enriched["seating_capacity"] == 5
        assert enriched["transmission"] == "Automatic"
        assert enriched["body_style"] == "SUV"
        assert enriched["mileage_kmpl"] == 17.0
        assert enriched["description"] == "Premium SUV."
        assert enriched["features"] == ["Sunroof", "Rear Camera", "ADAS"]
        assert enriched["fuel_match"] is True
        assert enriched["body_match"] is True
        assert enriched["transmission_match"] is True
        assert enriched["mileage_pts"] == 12
        assert enriched["feature_pts"] == 20
        assert enriched["score"] == 130

    def test_multiple_cars_in_order(self) -> None:
        """Cars are passed to Gemini in rank order with correct rank numbers."""
        v1 = _make_vehicle(1, "Top Car", price=10.0)
        v2 = _make_vehicle(2, "Second Car", price=8.0)
        v3 = _make_vehicle(3, "Third Car", price=6.0)
        ranked = [
            _make_scored(v1, score=100),
            _make_scored(v2, score=80),
            _make_scored(v3, score=60),
        ]
        mock = MockGeminiClient(response="Three great picks.")

        format_blurb(ranked, client=mock)

        [enriched] = mock.calls
        assert len(enriched) == 3
        assert enriched[0]["rank"] == 1
        assert enriched[0]["name"] == "Top Car"
        assert enriched[1]["rank"] == 2
        assert enriched[1]["name"] == "Second Car"
        assert enriched[2]["rank"] == 3
        assert enriched[2]["name"] == "Third Car"

    def test_features_always_list(self) -> None:
        """VehicleRecord stores features as tuple; enrichment converts to list."""
        vehicle = _make_vehicle(1, "Test Car", features=("Sunroof", "Bluetooth"))
        ranked = [_make_scored(vehicle)]
        mock = MockGeminiClient()

        format_blurb(ranked, client=mock)

        assert len(mock.calls) == 1
        [enriched] = mock.calls[0]
        assert isinstance(enriched["features"], list)
        assert enriched["features"] == ["Sunroof", "Bluetooth"]

    def test_none_fields_preserved(self) -> None:
        """Null catalogue fields are passed as None (not omitted)."""
        vehicle = VehicleRecord(
            id=99,
            name="Budget Car",
            price_lakhs=None,
            fuel_type=None,
            seating_capacity=None,
            transmission=None,
            body_style=None,
            mileage_kmpl=None,
            description=None,
            features=(),
        )
        ranked = [_make_scored(vehicle)]
        mock = MockGeminiClient(response="Affordable option.")

        format_blurb(ranked, client=mock)

        [enriched] = mock.calls[0]
        assert enriched["price_lakhs"] is None
        assert enriched["fuel_type"] is None
        assert enriched["seating_capacity"] is None
        assert enriched["transmission"] is None
        assert enriched["body_style"] is None
        assert enriched["mileage_kmpl"] is None
        assert enriched["description"] is None
        assert enriched["features"] == []

    def test_features_parsed_from_json_string(self) -> None:
        """VehicleRecord parses a JSON-string features field; enrichment uses the tuple."""
        vehicle = _make_vehicle(1, "JSON Car", features='["Sunroof","Bluetooth","ADAS"]')
        ranked = [_make_scored(vehicle)]
        mock = MockGeminiClient(response="Feature-rich car.")

        format_blurb(ranked, client=mock)

        [enriched] = mock.calls[0]
        # VehicleRecord.__post_init__ parses the JSON string into a tuple;
        # _enrich_scored_car converts it to a list.
        assert enriched["features"] == ["Sunroof", "Bluetooth", "ADAS"]


# ─── Tests: empty list ───────────────────────────────────────────────────────


class TestEmptyList:
    """Empty ranked list is a non-error that returns None."""

    def test_empty_list_returns_none(self) -> None:
        """No cars to format → None (no Gemini call needed)."""
        mock = MockGeminiClient()
        result = format_blurb([], client=mock)
        assert result is None
        assert mock.calls == []  # Gemini must not be called


# ─── Tests: non-fatal failure fallback ───────────────────────────────────────


class TestNonFatalFailureFallback:
    """
    A cosmetic Gemini failure must never cause the loss of the actual
    deliverable (the ranked vehicle items). format_blurb returns None
    so the caller can still store the ranked items in the database.
    """

    def test_api_error_returns_none(self) -> None:
        """RuntimeError from Gemini API is caught; None is returned."""
        vehicle = _make_vehicle(1, "Swift VXi", price=7.5)
        ranked = [_make_scored(vehicle)]
        mock = MockGeminiClient(should_fail=True)

        result = format_blurb(ranked, client=mock)

        assert result is None

    def test_timeout_error_returns_none(self) -> None:
        """TimeoutError is caught; None is returned."""
        vehicle = _make_vehicle(1, "Swift VXi")
        ranked = [_make_scored(vehicle)]
        mock = MockGeminiClient(raise_on_format=TimeoutError)

        result = format_blurb(ranked, client=mock)

        assert result is None

    def test_arbitrary_exception_returns_none(self) -> None:
        """Any unexpected exception is caught and None is returned."""
        vehicle = _make_vehicle(1, "Swift VXi")
        ranked = [_make_scored(vehicle)]
        mock = MockGeminiClient(raise_on_format=KeyError)

        result = format_blurb(ranked, client=mock)

        assert result is None

    def test_caller_can_still_store_items_on_none(self) -> None:
        """
        Demonstrate the decoupled-storage contract: the caller receives None
        but still has the ranked_cars to persist.

        This test documents that format_blurb returning None does NOT raise,
        so the caller can proceed with database storage.
        """
        vehicle = _make_vehicle(1, "Swift VXi", price=7.5)
        ranked = [_make_scored(vehicle, score=80)]
        mock = MockGeminiClient(raise_on_format=RuntimeError)

        # This must NOT raise — the caller catches the None and stores items
        blurb = format_blurb(ranked, client=mock)

        assert blurb is None
        # The caller can now do:
        #   db.store_recommendations(lead_id, ranked_cars, blurb=None)
        # without losing ranked_cars.

    def test_gemini_not_called_on_failure(self) -> None:
        """When Gemini fails, the call is recorded (before raising) in mock.calls."""
        vehicle = _make_vehicle(1, "Swift VXi")
        ranked = [_make_scored(vehicle)]
        mock = MockGeminiClient(should_fail=True)

        result = format_blurb(ranked, client=mock)

        assert result is None
        # The mock was called before the exception was raised by should_fail
        assert len(mock.calls) == 1

    def test_gemini_not_called_for_empty_list_on_failure(self) -> None:
        """Empty list: no Gemini call, no failure path entered."""
        mock = MockGeminiClient(should_fail=True)

        result = format_blurb([], client=mock)

        assert result is None
        assert mock.calls == []


# ─── Tests: GeminiRestClient._clean_response ──────────────────────────────────


class TestCleanResponse:
    """GeminiRestClient._clean_response strips markdown code fences."""

    def test_strips_markdown_fence(self) -> None:
        """Triple-backtick fences are stripped, leaving only the inner content."""
        client = GeminiRestClient()
        raw = "```\nTop pick: Swift VXi.\n```"
        assert client._clean_response(raw) == "Top pick: Swift VXi."

    def test_strips_fence_with_language_tag(self) -> None:
        """Fence with a language label (e.g. ```text) is also stripped."""
        client = GeminiRestClient()
        raw = "```text\nGreat car.\n```"
        assert client._clean_response(raw) == "Great car."

    def test_strips_multiple_fences(self) -> None:
        """If the response contains multiple fences, only content between them is kept."""
        client = GeminiRestClient()
        raw = "```\nFirst.\n```\n```\nSecond.\n```"
        result = client._clean_response(raw)
        assert "First" in result
        assert "Second" in result
        assert "```" not in result

    def test_no_fence_unchanged(self) -> None:
        """Plain text without fences passes through unchanged."""
        client = GeminiRestClient()
        raw = "Top pick: Swift VXi at 7.5 lakhs."
        assert client._clean_response(raw) == "Top pick: Swift VXi at 7.5 lakhs."

    def test_whitespace_trimmed(self) -> None:
        """Leading/trailing whitespace is stripped."""
        client = GeminiRestClient()
        raw = "   \n  Top pick.  \n  "
        assert client._clean_response(raw) == "Top pick."

    def test_empty_after_strip(self) -> None:
        """Response that becomes empty after stripping is returned as None."""
        client = GeminiRestClient()
        raw = "```\n```"
        assert client._clean_response(raw) is None

    def test_whitespace_only_normalized_to_none(self) -> None:
        """Whitespace-only raw response normalizes to None."""
        client = GeminiRestClient()
        raw = "     \n\n   "
        assert client._clean_response(raw) is None

    def test_empty_string_normalized_to_none(self) -> None:
        """Empty string raw response normalizes to None."""
        client = GeminiRestClient()
        assert client._clean_response("") is None


# ─── Tests: note parameter ────────────────────────────────────────────────────


class TestNoteParameter:
    """The note parameter is forwarded to the Gemini client."""

    def test_note_passed_to_gemini(self) -> None:
        """format_blurb passes note to client.format()."""
        vehicle = _make_vehicle(1, "Swift VXi", price=7.5)
        ranked = [_make_scored(vehicle)]
        mock = MockGeminiClient(response="Top pick.")

        format_blurb(ranked, note="No car satisfied your budget and seat requirements.", client=mock)

        assert mock.last_note == "No car satisfied your budget and seat requirements."

    def test_note_none_not_passed(self) -> None:
        """When note is None, client.format receives None."""
        vehicle = _make_vehicle(1, "Swift VXi")
        ranked = [_make_scored(vehicle)]
        mock = MockGeminiClient(response="Great car.")

        format_blurb(ranked, note=None, client=mock)

        assert mock.last_note is None

    def test_note_default_is_none(self) -> None:
        """Default note value is None."""
        vehicle = _make_vehicle(1, "Swift VXi")
        ranked = [_make_scored(vehicle)]
        mock = MockGeminiClient(response="Great car.")

        format_blurb(ranked, client=mock)

        assert mock.last_note is None

    def test_empty_list_with_note_skips_gemini(self) -> None:
        """Empty list returns None before calling Gemini — note is irrelevant."""
        mock = MockGeminiClient()
        result = format_blurb([], note="some note", client=mock)
        assert result is None
        assert mock.calls == []


# ─── Tests: empty/whitespace Gemini response ──────────────────────────────────


class TestEmptyWhitespaceResponse:
    """Empty or whitespace-only Gemini responses are normalized to None."""

    def test_whitespace_response_returns_none(self) -> None:
        """Mock returns whitespace; format_blurb returns None."""
        vehicle = _make_vehicle(1, "Swift VXi")
        ranked = [_make_scored(vehicle)]
        mock = MockGeminiClient(response="   \n  \n  ")

        result = format_blurb(ranked, client=mock)

        assert result is None

    def test_empty_response_returns_none(self) -> None:
        """Mock returns empty string; format_blurb returns None."""
        vehicle = _make_vehicle(1, "Swift VXi")
        ranked = [_make_scored(vehicle)]
        mock = MockGeminiClient(response="")

        result = format_blurb(ranked, client=mock)

        assert result is None

    def test_fence_with_only_newlines_returns_none(self) -> None:
        """GeminiRestClient._clean_response normalizes fence-only-newlines to None."""
        client = GeminiRestClient()
        assert client._clean_response("```\n\n```") is None


# ─── Tests: logging on failure ────────────────────────────────────────────────


class TestLoggingOnFailure:
    """Swallowed exceptions are logged so failures are diagnosable in production."""

    def test_api_error_is_logged(self, caplog) -> None:
        """RuntimeError triggers a logger.error call with exc_info."""
        import logging

        vehicle = _make_vehicle(1, "Swift VXi")
        ranked = [_make_scored(vehicle)]
        mock = MockGeminiClient(should_fail=True)

        with caplog.at_level(logging.ERROR, logger="backend.blurb_formatter"):
            result = format_blurb(ranked, client=mock)

        assert result is None
        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.ERROR
        assert "Blurb formatting failed" in caplog.records[0].message

    def test_timeout_error_is_logged(self, caplog) -> None:
        """TimeoutError triggers a logger.error call with exc_info."""
        import logging

        vehicle = _make_vehicle(1, "Swift VXi")
        ranked = [_make_scored(vehicle)]
        mock = MockGeminiClient(raise_on_format=TimeoutError)

        with caplog.at_level(logging.ERROR, logger="backend.blurb_formatter"):
            result = format_blurb(ranked, client=mock)

        assert result is None
        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.ERROR

    def test_empty_list_does_not_log(self, caplog) -> None:
        """Empty list short-circuits before the try block — no log emitted."""
        import logging

        mock = MockGeminiClient(should_fail=True)

        with caplog.at_level(logging.ERROR, logger="backend.blurb_formatter"):
            result = format_blurb([], client=mock)

        assert result is None
        assert len(caplog.records) == 0


# ─── Tests: _enrich_scored_car ───────────────────────────────────────────────


class TestEnrichScoredCar:
    """Unit tests for the enrichment helper."""

    def test_rank_is_preserved(self) -> None:
        """The rank argument is written directly into the enriched dict."""
        vehicle = _make_vehicle(1, "Car", price=10.0)
        scored = _make_scored(vehicle, score=50)
        for expected_rank in [1, 2, 3]:
            result = _enrich_scored_car(scored, rank=expected_rank)
            assert result["rank"] == expected_rank

    def test_all_catalogue_fields_present(self) -> None:
        """Every VehicleRecord field appears in the enriched dict."""
        vehicle = VehicleRecord(
            id=5,
            name="Tata Nexon EV",
            price_lakhs=14.5,
            fuel_type="Electric",
            seating_capacity=5,
            transmission="Automatic",
            body_style="SUV",
            mileage_kmpl=0.0,
            description="Electric SUV.",
            features=("Sunroof", "ADAS"),
        )
        scored = ScoredCar(
            vehicle=vehicle,
            score=90,
            fuel_match=True,
            body_match=True,
            transmission_match=True,
            mileage_pts=0,
            feature_pts=20,
        )
        result = _enrich_scored_car(scored, rank=1)

        assert result["name"] == "Tata Nexon EV"
        assert result["price_lakhs"] == 14.5
        assert result["fuel_type"] == "Electric"
        assert result["seating_capacity"] == 5
        assert result["transmission"] == "Automatic"
        assert result["body_style"] == "SUV"
        assert result["mileage_kmpl"] == 0.0
        assert result["description"] == "Electric SUV."
        assert result["features"] == ["Sunroof", "ADAS"]
        assert result["score"] == 90
        assert result["fuel_match"] is True
        assert result["body_match"] is True
        assert result["transmission_match"] is True
        assert result["mileage_pts"] == 0
        assert result["feature_pts"] == 20

    def test_score_from_scored_car(self) -> None:
        """The score field comes from ScoredCar, not VehicleRecord."""
        vehicle = _make_vehicle(1, "Car", price=10.0)
        scored = ScoredCar(vehicle=vehicle, score=77)
        result = _enrich_scored_car(scored, rank=1)
        assert result["score"] == 77


# ─── Tests: default client ───────────────────────────────────────────────────


class TestDefaultClient:
    """When no client is passed, GeminiRestClient is used (lazy import)."""

    def test_default_client_is_adk(self) -> None:
        """Passing None uses GeminiRestClient (no mock)."""
        # We can't easily verify the class without mocking the import,
        # but we can verify the function signature accepts None and doesn't raise.
        # The actual GeminiRestClient creation would fail without google-adk installed,
        # but the contract is: passing client=None is valid.
        from backend.blurb_formatter import format_blurb

        # Verify None is accepted as client parameter (signature check)
        import inspect
        sig = inspect.signature(format_blurb)
        client_param = sig.parameters["client"]
        assert client_param.default is None


# ─── Tests: real-world-ish scenarios ────────────────────────────────────────


class TestRealWorldScenarios:
    """End-to-end-ish scenarios matching the PRD AI Call flow."""

    def test_single_top_match(self) -> None:
        """One car with a clear winner score."""
        vehicle = _make_vehicle(
            vid=2,
            name="Hyundai Venue S 1.2",
            price=9.5,
            fuel="Petrol",
            seats=5,
            transmission="Manual",
            body="SUV",
            mileage=16.0,
            description="Compact SUV.",
            features=["Rear Camera", "Touchscreen"],
        )
        ranked = [
            ScoredCar(
                vehicle=vehicle,
                score=100,
                fuel_match=True,
                body_match=True,
                transmission_match=False,
                mileage_pts=6,
                feature_pts=20,
            )
        ]
        mock = MockGeminiClient(
            response=(
                "Based on your preferences, the Hyundai Venue S 1.2 is our top pick. "
                "It's a Petrol SUV that fits your budget of under 10 lakhs, "
                "delivers 16 kmpl mileage, and comes with a Rear Camera and Touchscreen."
            )
        )

        blurb = format_blurb(ranked, client=mock)

        assert blurb is not None
        assert "Hyundai Venue" in blurb
        assert mock.calls[0][0]["score"] == 100

    def test_five_car_list(self) -> None:
        """Full ranked list of 5 cars is passed to Gemini in order."""
        vehicles = [
            _make_vehicle(i, f"Car {i}", price=5.0 + i)
            for i in range(1, 6)
        ]
        ranked = [
            _make_scored(vehicles[0], score=130),
            _make_scored(vehicles[1], score=100),
            _make_scored(vehicles[2], score=80),
            _make_scored(vehicles[3], score=60),
            _make_scored(vehicles[4], score=40),
        ]
        mock = MockGeminiClient(response="Five great options for you.")

        blurb = format_blurb(ranked, client=mock)

        assert blurb is not None
        assert len(mock.calls[0]) == 5
        assert [c["rank"] for c in mock.calls[0]] == [1, 2, 3, 4, 5]
        assert [c["name"] for c in mock.calls[0]] == ["Car 1", "Car 2", "Car 3", "Car 4", "Car 5"]

    def test_api_failure_does_not_raise(self) -> None:
        """
        The public contract: format_blurb must never raise.
        Any Gemini failure is swallowed and None is returned.
        """
        vehicle = _make_vehicle(1, "Test Car")
        ranked = [_make_scored(vehicle)]

        for exc_type in [RuntimeError, TimeoutError, ConnectionError, ValueError, KeyError]:
            mock = MockGeminiClient(raise_on_format=exc_type)
            # This must not raise
            result = format_blurb(ranked, client=mock)
            assert result is None, f"{exc_type.__name__} should return None, not raise"

    def test_storage_contract_example(self) -> None:
        """
        Example of how the decoupled storage contract works in practice.
        """
        vehicle = _make_vehicle(1, "Swift VXi", price=7.5)
        ranked = [_make_scored(vehicle, score=80)]
        mock = MockGeminiClient(raise_on_format=RuntimeError)

        # Simulate what the call-finalise pipeline does:
        # 1. Try to format the blurb
        blurb = format_blurb(ranked, client=mock)
        # 2. blurb may be None — that's fine, store null in DB
        assert blurb is None
        # 3. ranked_cars are intact — they are NOT affected by the Gemini failure
        assert len(ranked) == 1
        assert ranked[0].vehicle.name == "Swift VXi"
        # 4. The caller can now persist ranked items with blurb=None
        #    db.insert_recommendation(lead_id, ranked_cars, blurb_text=None)
        #    (This step is simulated — the test just documents the contract.)
