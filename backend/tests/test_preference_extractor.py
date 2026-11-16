"""
backend/tests/test_preference_extractor.py — Unit tests for Issue 021.

The Gemini boundary is fully mocked via ``MockGeminiClient``.
No live API calls are made in this suite.

Test conventions (matching test_scorer.py style):
  • One logical assertion per test — each documents one contract.
  • Helpers live at module level (no pytest fixtures needed at this scale).
  • All seven keys are always asserted present in every output.
"""

from __future__ import annotations

from typing import Any

import pytest

from backend.preference_extractor import (
    GeminiClient,
    extract_preferences,
    _PREFERENCE_KEYS,
)


# ─── Helpers ───────────────────────────────────────────────────────────────────

def all_keys_present(result: dict) -> bool:
    """Return True only when every required key is in result (even if None)."""
    return all(k in result for k in _PREFERENCE_KEYS)


def all_keys_none(result: dict) -> bool:
    """Return True when all preference keys are None / empty (filler result)."""
    return all(
        (result.get(k) is None if k != "feature_keywords" else result.get(k) == [])
        for k in _PREFERENCE_KEYS
    )


class MockGeminiClient(GeminiClient):
    """
    Record-and-replay mock that never touches a live API.

    Set ``responses`` to a list of dicts; each call pops and returns the next.
    If the list is exhausted the final response is returned (useful for
    "always return all-null" filler tests).
    """

    def __init__(self, responses: list[dict] | None = None) -> None:
        self._responses: list[dict] = responses if responses is not None else []
        self.calls: list[tuple[str, dict | None]] = []

    def extract(self, utterance: str, current_prefs: dict | None) -> dict:
        self.calls.append((utterance, current_prefs))
        if self._responses:
            return self._responses.pop(0)
        # Default: return all-null delta (safe filler)
        return {k: None for k in _PREFERENCE_KEYS}


# ─── Tests ─────────────────────────────────────────────────────────────────────

class TestFillerUtterance:
    """Filler / non-preference utterances must not alter the running set."""

    def test_filler_returns_all_null_delta(self) -> None:
        """The mock returns all-null; the merge should leave current prefs untouched."""
        mock = MockGeminiClient(responses=[{k: None for k in _PREFERENCE_KEYS}])
        current: dict = {
            "budget": 10.0,
            "min_seats": 5,
            "fuel_type": "Petrol",
            "body_style": "SUV",
            "transmission": "Manual",
            "min_mileage_kmpl": 15.0,
            "feature_keywords": ["sunroof"],
        }
        result = extract_preferences("yeah okay sounds good", current, client=mock)

        assert result == current, "Filler must not alter the running set"
        assert len(mock.calls) == 1

    @pytest.mark.parametrize(
        "utterance",
        [
            "okay",
            "hmm",
            "sure",
            "no thanks",
            "not sure about that",
            "maybe",
            "hello",
            "hi riya",
        ],
    )
    def test_common_filler_phrases_unchanged(self, utterance: str) -> None:
        """Every common filler returns all-null delta → running set unchanged."""
        mock = MockGeminiClient(responses=[{k: None for k in _PREFERENCE_KEYS}])
        current = {k: None for k in _PREFERENCE_KEYS}
        current["budget"] = 12.0
        current["min_seats"] = 7

        result = extract_preferences(utterance, current, client=mock)

        assert result["budget"] == 12.0
        assert result["min_seats"] == 7
        assert all_keys_present(result)


class TestSingleSignal:
    """A single actionable preference updates exactly that key."""

    def test_budget_signal(self) -> None:
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": 15.0,
                    "min_seats": None,
                    "fuel_type": None,
                    "body_style": None,
                    "transmission": None,
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }
            ]
        )
        result = extract_preferences("my budget is 15 lakhs", None, client=mock)

        assert result["budget"] == 15.0
        assert result["min_seats"] is None
        assert result["fuel_type"] is None
        assert all_keys_present(result)

    def test_fuel_type_signal(self) -> None:
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": None,
                    "fuel_type": "Diesel",
                    "body_style": None,
                    "transmission": None,
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }
            ]
        )
        result = extract_preferences("I want a diesel car", None, client=mock)

        assert result["fuel_type"] == "Diesel"
        assert all_keys_present(result)

    def test_body_style_signal(self) -> None:
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": None,
                    "fuel_type": None,
                    "body_style": "MUV",
                    "transmission": None,
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }
            ]
        )
        result = extract_preferences("I need a 7-seater van", None, client=mock)

        assert result["body_style"] == "MUV"
        assert all_keys_present(result)

    def test_min_seats_signal(self) -> None:
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": 7,
                    "fuel_type": None,
                    "body_style": None,
                    "transmission": None,
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }
            ]
        )
        result = extract_preferences("need room for 7 people", None, client=mock)

        assert result["min_seats"] == 7
        assert all_keys_present(result)

    def test_transmission_signal(self) -> None:
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": None,
                    "fuel_type": None,
                    "body_style": None,
                    "transmission": "Automatic",
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }
            ]
        )
        result = extract_preferences("I prefer AMT or auto", None, client=mock)

        assert result["transmission"] == "Automatic"
        assert all_keys_present(result)

    def test_mileage_signal(self) -> None:
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": None,
                    "fuel_type": None,
                    "body_style": None,
                    "transmission": None,
                    "min_mileage_kmpl": 18.0,
                    "feature_keywords": [],
                }
            ]
        )
        result = extract_preferences("I need at least 18 kmpl", None, client=mock)

        assert result["min_mileage_kmpl"] == 18.0
        assert all_keys_present(result)

    def test_feature_keywords_signal(self) -> None:
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": None,
                    "fuel_type": None,
                    "body_style": None,
                    "transmission": None,
                    "min_mileage_kmpl": None,
                    "feature_keywords": ["sunroof", "rear camera", "adas"],
                }
            ]
        )
        result = extract_preferences(
            "I want sunroof, rear camera and ADAS", None, client=mock
        )

        assert result["feature_keywords"] == ["sunroof", "rear camera", "adas"]
        assert isinstance(result["feature_keywords"], list)
        assert all_keys_present(result)


class TestMultipleSignals:
    """Multiple preferences in one utterance are all extracted."""

    def test_budget_and_fuel_type_together(self) -> None:
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": 20.0,
                    "min_seats": None,
                    "fuel_type": "Electric",
                    "body_style": None,
                    "transmission": None,
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }
            ]
        )
        result = extract_preferences(
            "I can spend up to 20 lakhs and want an EV",
            None,
            client=mock,
        )

        assert result["budget"] == 20.0
        assert result["fuel_type"] == "Electric"
        assert all_keys_present(result)


class TestContradictionLatestWins:
    """A contradicting statement completely overwrites, not accumulates."""

    def test_fuel_type_overwrites(self) -> None:
        # First turn: customer said petrol
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": None,
                    "fuel_type": "Petrol",
                    "body_style": None,
                    "transmission": None,
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }
            ]
        )
        first = extract_preferences("I want a petrol car", None, client=mock)

        assert first["fuel_type"] == "Petrol"

        # Second turn: customer changes mind
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": None,
                    "fuel_type": "Diesel",
                    "body_style": None,
                    "transmission": None,
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }
            ]
        )
        second = extract_preferences("actually diesel is fine", first, client=mock)

        assert second["fuel_type"] == "Diesel"
        assert second["budget"] is None  # never mentioned, carries over as None

    def test_body_style_overwrites(self) -> None:
        first = extract_preferences(
            "I was thinking a hatchback",
            None,
            client=MockGeminiClient(
                responses=[
                    {
                        "budget": None,
                        "min_seats": None,
                        "fuel_type": None,
                        "body_style": "Hatchback",
                        "transmission": None,
                        "min_mileage_kmpl": None,
                        "feature_keywords": [],
                    }
                ]
            ),
        )
        assert first["body_style"] == "Hatchback"

        second = extract_preferences(
            "actually an SUV would be better",
            first,
            client=MockGeminiClient(
                responses=[
                    {
                        "budget": None,
                        "min_seats": None,
                        "fuel_type": None,
                        "body_style": "SUV",
                        "transmission": None,
                        "min_mileage_kmpl": None,
                        "feature_keywords": [],
                    }
                ]
            ),
        )
        assert second["body_style"] == "SUV"

    def test_budget_overwrites(self) -> None:
        first = extract_preferences(
            "around 8 lakhs",
            None,
            client=MockGeminiClient(
                responses=[
                    {
                        "budget": 8.0,
                        "min_seats": None,
                        "fuel_type": None,
                        "body_style": None,
                        "transmission": None,
                        "min_mileage_kmpl": None,
                        "feature_keywords": [],
                    }
                ]
            ),
        )
        assert first["budget"] == 8.0

        second = extract_preferences(
            "actually i can go up to 12 lakhs",
            first,
            client=MockGeminiClient(
                responses=[
                    {
                        "budget": 12.0,
                        "min_seats": None,
                        "fuel_type": None,
                        "body_style": None,
                        "transmission": None,
                        "min_mileage_kmpl": None,
                        "feature_keywords": [],
                    }
                ]
            ),
        )
        assert second["budget"] == 12.0


class TestMergeOntoExistingPrefs:
    """New extractions merge alongside already-stated preferences."""

    def test_new_keys_added_existing_preserved(self) -> None:
        current = {
            "budget": 10.0,
            "min_seats": 5,
            "fuel_type": "Petrol",
            "body_style": None,
            "transmission": None,
            "min_mileage_kmpl": None,
            "feature_keywords": [],
        }
        # Gemini returns only the newly stated dimension
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": None,
                    "fuel_type": None,
                    "body_style": "Sedan",
                    "transmission": None,
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }
            ]
        )
        result = extract_preferences("I also want a Sedan", current, client=mock)

        assert result["budget"] == 10.0
        assert result["min_seats"] == 5
        assert result["fuel_type"] == "Petrol"
        assert result["body_style"] == "Sedan"
        assert all_keys_present(result)

    def test_min_seats_and_budget_merged(self) -> None:
        current = {
            "budget": 15.0,
            "min_seats": None,
            "fuel_type": "Electric",
            "body_style": None,
            "transmission": None,
            "min_mileage_kmpl": None,
            "feature_keywords": [],
        }
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": 7,
                    "fuel_type": None,
                    "body_style": None,
                    "transmission": None,
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }
            ]
        )
        result = extract_preferences("we are a family of 7", current, client=mock)

        assert result["budget"] == 15.0
        assert result["min_seats"] == 7
        assert result["fuel_type"] == "Electric"


class TestFirstTurnNone:
    """First call with current_prefs=None must work without error."""

    def test_none_current_prefs_no_crash(self) -> None:
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": 20.0,
                    "min_seats": 5,
                    "fuel_type": "Petrol",
                    "body_style": "Hatchback",
                    "transmission": "Manual",
                    "min_mileage_kmpl": 15.0,
                    "feature_keywords": ["bluetooth"],
                }
            ]
        )
        result = extract_preferences(
            "I want a petrol hatchback under 20 lakhs",
            None,
            client=mock,
        )

        assert result["budget"] == 20.0
        assert result["min_seats"] == 5
        assert result["fuel_type"] == "Petrol"
        assert result["body_style"] == "Hatchback"
        assert result["transmission"] == "Manual"
        assert result["min_mileage_kmpl"] == 15.0
        assert result["feature_keywords"] == ["bluetooth"]
        assert all_keys_present(result)


class TestNullabilityNoDefaults:
    """The extractor must never invent default values."""

    def test_empty_prefs_fully_null(self) -> None:
        mock = MockGeminiClient(
            responses=[{k: None for k in _PREFERENCE_KEYS}]
        )
        result = extract_preferences("hmm let me think", None, client=mock)

        assert all_keys_none(result)
        assert all_keys_present(result)
        assert result.get("budget") is None
        assert result.get("min_seats") is None
        assert result.get("fuel_type") is None

    def test_partial_utterance_only_stated_fields_non_null(self) -> None:
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": None,
                    "fuel_type": "CNG",
                    "body_style": None,
                    "transmission": None,
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }
            ]
        )
        result = extract_preferences("I'm interested in CNG", None, client=mock)

        assert result["fuel_type"] == "CNG"
        assert result["budget"] is None
        assert result["min_seats"] is None
        assert result["body_style"] is None


class TestVocabularyConstraints:
    """The validator must reject values outside the canonical vocabulary."""

    def test_invalid_fuel_type_becomes_none(self) -> None:
        """Validator strips unknown fuel_type strings."""
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": None,
                    "fuel_type": "Petrol-Hybrid",  # not in vocabulary
                    "body_style": None,
                    "transmission": None,
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }
            ]
        )
        result = extract_preferences("I want a hybrid", None, client=mock)

        assert result["fuel_type"] is None

    def test_invalid_body_style_becomes_none(self) -> None:
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": None,
                    "fuel_type": None,
                    "body_style": "Coupe",  # not in vocabulary
                    "transmission": None,
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }
            ]
        )
        result = extract_preferences("I want a coupe", None, client=mock)

        assert result["body_style"] is None

    def test_invalid_transmission_becomes_none(self) -> None:
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": None,
                    "fuel_type": None,
                    "body_style": None,
                    "transmission": "CVT",  # not in vocabulary
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }
            ]
        )
        result = extract_preferences("CVT gearbox preferred", None, client=mock)

        assert result["transmission"] is None


class TestOutputShape:
    """All seven keys are always present; feature_keywords is always a list."""

    def test_all_seven_keys_present(self) -> None:
        mock = MockGeminiClient(
            responses=[{k: None for k in _PREFERENCE_KEYS}]
        )
        result = extract_preferences("hello", None, client=mock)

        assert all_keys_present(result)
        assert set(result.keys()) == set(_PREFERENCE_KEYS)

    def test_feature_keywords_always_list(self) -> None:
        # Even if Gemini returns a non-list (edge case), normaliser forces a list
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": None,
                    "fuel_type": None,
                    "body_style": None,
                    "transmission": None,
                    "min_mileage_kmpl": None,
                    "feature_keywords": "sunroof",  # accidentally a string
                }
            ]
        )
        result = extract_preferences("sunroof please", None, client=mock)

        assert isinstance(result["feature_keywords"], list)
        assert result["feature_keywords"] == []

    def test_feature_keywords_preserved_when_returned(self) -> None:
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": None,
                    "fuel_type": None,
                    "body_style": None,
                    "transmission": None,
                    "min_mileage_kmpl": None,
                    "feature_keywords": ["sunroof", "bluetooth"],
                }
            ]
        )
        result = extract_preferences("sunroof and bluetooth", None, client=mock)

        assert isinstance(result["feature_keywords"], list)
        assert "sunroof" in result["feature_keywords"]
        assert "bluetooth" in result["feature_keywords"]


class TestVocabularySynonyms:
    """Canonical vocabulary values (mapped by the prompt/Gemini) are preserved."""

    def test_ev_maps_to_electric(self) -> None:
        # The prompt instructs Gemini to map "EV" → Electric
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": None,
                    "fuel_type": "Electric",
                    "body_style": None,
                    "transmission": None,
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }
            ]
        )
        result = extract_preferences("I want an EV", None, client=mock)

        assert result["fuel_type"] == "Electric"

    def test_suv_crossover(self) -> None:
        # "crossover" → SUV per prompt vocabulary
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": None,
                    "fuel_type": None,
                    "body_style": "SUV",
                    "transmission": None,
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }
            ]
        )
        result = extract_preferences("looking for a crossover", None, client=mock)

        assert result["body_style"] == "SUV"

    def test_7_seater_van_maps_to_muv(self) -> None:
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": 7,
                    "fuel_type": None,
                    "body_style": "MUV",
                    "transmission": None,
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }
            ]
        )
        result = extract_preferences("need a 7-seater van", None, client=mock)

        assert result["body_style"] == "MUV"
        assert result["min_seats"] == 7

    def test_amt_maps_to_automatic(self) -> None:
        # "AMT" → Automatic per prompt vocabulary
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": None,
                    "fuel_type": None,
                    "body_style": None,
                    "transmission": "Automatic",
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }
            ]
        )
        result = extract_preferences("I want AMT", None, client=mock)

        assert result["transmission"] == "Automatic"


class TestNumericFieldTypes:
    """Numeric fields are coerced to the correct Python type."""

    def test_budget_float(self) -> None:
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": "12.5",  # string from JSON — should be coerced
                    "min_seats": None,
                    "fuel_type": None,
                    "body_style": None,
                    "transmission": None,
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }
            ]
        )
        result = extract_preferences("12.5 lakhs budget", None, client=mock)

        assert isinstance(result["budget"], float)
        assert result["budget"] == 12.5

    def test_min_seats_int(self) -> None:
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": "7",  # string from JSON — should be coerced
                    "fuel_type": None,
                    "body_style": None,
                    "transmission": None,
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }
            ]
        )
        result = extract_preferences("family of 7", None, client=mock)

        assert isinstance(result["min_seats"], int)
        assert result["min_seats"] == 7

    def test_mileage_float(self) -> None:
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": None,
                    "fuel_type": None,
                    "body_style": None,
                    "transmission": None,
                    "min_mileage_kmpl": "20.5",
                    "feature_keywords": [],
                }
            ]
        )
        result = extract_preferences("at least 20.5 kmpl", None, client=mock)

        assert isinstance(result["min_mileage_kmpl"], float)
        assert result["min_mileage_kmpl"] == 20.5


class TestEndToEndConversationFlow:
    """Simulate a realistic multi-turn conversation."""

    def test_full_conversation_flow(self) -> None:
        # Turn 1: greeting — filler
        mock = MockGeminiClient(
            responses=[{k: None for k in _PREFERENCE_KEYS}]
        )
        prefs = extract_preferences("hi riya", None, client=mock)
        assert all_keys_none(prefs)

        # Turn 2: budget stated
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": 15.0,
                    "min_seats": None,
                    "fuel_type": None,
                    "body_style": None,
                    "transmission": None,
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }
            ]
        )
        prefs = extract_preferences("my budget is around 15 lakhs", prefs, client=mock)
        assert prefs["budget"] == 15.0

        # Turn 3: fuel and seats stated together
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": 7,
                    "fuel_type": "Petrol",
                    "body_style": None,
                    "transmission": None,
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }
            ]
        )
        prefs = extract_preferences(
            "petrol is fine, and we have 7 people at home", prefs, client=mock
        )
        assert prefs["budget"] == 15.0
        assert prefs["fuel_type"] == "Petrol"
        assert prefs["min_seats"] == 7

        # Turn 4: contradicting fuel type
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": None,
                    "fuel_type": "Diesel",
                    "body_style": None,
                    "transmission": None,
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }
            ]
        )
        prefs = extract_preferences("actually i think diesel is better", prefs, client=mock)
        assert prefs["fuel_type"] == "Diesel"
        assert prefs["budget"] == 15.0
        assert prefs["min_seats"] == 7

        # Turn 5: features + filler at end
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": None,
                    "fuel_type": None,
                    "body_style": "SUV",
                    "transmission": "Automatic",
                    "min_mileage_kmpl": None,
                    "feature_keywords": ["sunroof", "bluetooth"],
                }
            ]
        )
        prefs = extract_preferences(
            "I want an SUV, automatic, with sunroof and bluetooth please",
            prefs,
            client=mock,
        )
        assert prefs["body_style"] == "SUV"
        assert prefs["transmission"] == "Automatic"
        assert prefs["feature_keywords"] == ["sunroof", "bluetooth"]
        assert prefs["budget"] == 15.0
        assert prefs["fuel_type"] == "Diesel"
        assert prefs["min_seats"] == 7


class TestCaseInsensitiveVocabulary:
    """
    Regression: valid vocabulary values in any casing must be accepted and
    canonicalised to the exact strings the scorer expects.

    Bug: before the fix, str(fuel).strip() in _VALID_FUEL_TYPES was case-sensitive,
    so "electric" / "ELECTRIC" / "Electric" → None (silently dropped).
    Fix: lowercase→canonical lookup maps (_FUEL_CANON, _BODY_CANON, _TRANS_CANON).
    """

    @pytest.mark.parametrize(
        "raw_value,expected",
        [
            # Lowercase
            ("electric", "Electric"),
            ("diesel", "Diesel"),
            ("petrol", "Petrol"),
            ("cng", "CNG"),
        ],
    )
    def test_fuel_type_lowercase(self, raw_value: str, expected: str) -> None:
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": None,
                    "fuel_type": raw_value,
                    "body_style": None,
                    "transmission": None,
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }
            ]
        )
        result = extract_preferences("I want electric", None, client=mock)
        assert result["fuel_type"] == expected, f"{raw_value!r} should map to {expected!r}"

    @pytest.mark.parametrize(
        "raw_value,expected",
        [
            # Uppercase
            ("ELECTRIC", "Electric"),
            ("DIESEL", "Diesel"),
            ("PETROL", "Petrol"),
            ("CNG", "CNG"),
        ],
    )
    def test_fuel_type_uppercase(self, raw_value: str, expected: str) -> None:
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": None,
                    "fuel_type": raw_value,
                    "body_style": None,
                    "transmission": None,
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }
            ]
        )
        result = extract_preferences("I want electric", None, client=mock)
        assert result["fuel_type"] == expected, f"{raw_value!r} should map to {expected!r}"

    @pytest.mark.parametrize(
        "raw_value,expected",
        [
            # Mixed case
            ("Electric", "Electric"),
            ("Diesel", "Diesel"),
            ("eLeCtRiC", "Electric"),
            ("DiEsEl", "Diesel"),
        ],
    )
    def test_fuel_type_mixed_case(self, raw_value: str, expected: str) -> None:
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": None,
                    "fuel_type": raw_value,
                    "body_style": None,
                    "transmission": None,
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }
            ]
        )
        result = extract_preferences("I want electric", None, client=mock)
        assert result["fuel_type"] == expected, f"{raw_value!r} should map to {expected!r}"

    @pytest.mark.parametrize(
        "raw_value,expected",
        [
            # Lowercase
            ("suv", "SUV"),
            ("sedan", "Sedan"),
            ("hatchback", "Hatchback"),
            ("muv", "MUV"),
        ],
    )
    def test_body_style_lowercase(self, raw_value: str, expected: str) -> None:
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": None,
                    "fuel_type": None,
                    "body_style": raw_value,
                    "transmission": None,
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }
            ]
        )
        result = extract_preferences("I want an SUV", None, client=mock)
        assert result["body_style"] == expected, f"{raw_value!r} should map to {expected!r}"

    @pytest.mark.parametrize(
        "raw_value,expected",
        [
            # Uppercase
            ("SUV", "SUV"),
            ("SEDAN", "Sedan"),
            ("HATCHBACK", "Hatchback"),
            ("MUV", "MUV"),
        ],
    )
    def test_body_style_uppercase(self, raw_value: str, expected: str) -> None:
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": None,
                    "fuel_type": None,
                    "body_style": raw_value,
                    "transmission": None,
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }
            ]
        )
        result = extract_preferences("I want an SUV", None, client=mock)
        assert result["body_style"] == expected, f"{raw_value!r} should map to {expected!r}"

    @pytest.mark.parametrize(
        "raw_value,expected",
        [
            # Mixed case
            ("SuV", "SUV"),
            ("Sedan", "Sedan"),
            ("sUv", "SUV"),
        ],
    )
    def test_body_style_mixed_case(self, raw_value: str, expected: str) -> None:
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": None,
                    "fuel_type": None,
                    "body_style": raw_value,
                    "transmission": None,
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }
            ]
        )
        result = extract_preferences("I want an SUV", None, client=mock)
        assert result["body_style"] == expected, f"{raw_value!r} should map to {expected!r}"

    @pytest.mark.parametrize(
        "raw_value,expected",
        [
            # Lowercase
            ("automatic", "Automatic"),
            ("manual", "Manual"),
        ],
    )
    def test_transmission_lowercase(self, raw_value: str, expected: str) -> None:
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": None,
                    "fuel_type": None,
                    "body_style": None,
                    "transmission": raw_value,
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }
            ]
        )
        result = extract_preferences("I want automatic", None, client=mock)
        assert result["transmission"] == expected, f"{raw_value!r} should map to {expected!r}"

    @pytest.mark.parametrize(
        "raw_value,expected",
        [
            # Uppercase
            ("AUTOMATIC", "Automatic"),
            ("MANUAL", "Manual"),
        ],
    )
    def test_transmission_uppercase(self, raw_value: str, expected: str) -> None:
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": None,
                    "fuel_type": None,
                    "body_style": None,
                    "transmission": raw_value,
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }
            ]
        )
        result = extract_preferences("I want automatic", None, client=mock)
        assert result["transmission"] == expected, f"{raw_value!r} should map to {expected!r}"

    @pytest.mark.parametrize(
        "raw_value,expected",
        [
            # Mixed case
            ("Automatic", "Automatic"),
            ("aUtOmAtIc", "Automatic"),
            ("MaNuAl", "Manual"),
        ],
    )
    def test_transmission_mixed_case(self, raw_value: str, expected: str) -> None:
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": None,
                    "fuel_type": None,
                    "body_style": None,
                    "transmission": raw_value,
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }
            ]
        )
        result = extract_preferences("I want automatic", None, client=mock)
        assert result["transmission"] == expected, f"{raw_value!r} should map to {expected!r}"

    def test_all_fields_lowercase_together(self) -> None:
        """All three vocabulary fields in lowercase simultaneously."""
        mock = MockGeminiClient(
            responses=[
                {
                    "budget": None,
                    "min_seats": None,
                    "fuel_type": "electric",
                    "body_style": "suv",
                    "transmission": "automatic",
                    "min_mileage_kmpl": None,
                    "feature_keywords": [],
                }
            ]
        )
        result = extract_preferences(
            "I want electric SUV automatic",
            None,
            client=mock,
        )
        assert result["fuel_type"] == "Electric"
        assert result["body_style"] == "SUV"
        assert result["transmission"] == "Automatic"

