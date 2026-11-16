"""
backend/preference_extractor.py — Gemini Preference Extractor (Issue 021).

Architecture
───────────
The Gemini call is fully isolated behind the abstract ``GeminiClient`` interface.
``GeminiRestClient`` is the concrete production implementation (uses httpx + Gemini REST API).
Tests substitute a mock that never makes a live API call.

Public Interface
───────────────
extract_preferences(
    utterance: str,
    current_prefs: dict | None,
    client: GeminiClient | None = None,
) -> dict

Returns a well-formed preference-set dict merged onto ``current_prefs``.
Each key present in the returned delta (and not None) overwrites the prior value
("latest explicit statement wins").  Keys absent from the delta or explicitly
None are carried over unchanged.

The output dict always contains exactly the seven scorer-compatible keys:
  budget, min_seats, fuel_type, body_style, transmission,
  min_mileage_kmpl, feature_keywords

All values are nullable — the extractor NEVER invents defaults.
The downstream Matcher / scorer is responsible for coercing None to
float('inf') / 1 as required by the scorer.

Data Contracts
──────────────
Vocabulary (strict — the prompt constrains Gemini to these values):
  fuel_type    ∈ {Petrol, Diesel, Electric, CNG}
  body_style   ∈ {Hatchback, Sedan, SUV, MUV}
  transmission ∈ {Manual, Automatic}
  feature_keywords — free-text array

Nullability: all seven keys may be None.  Gemini is instructed explicitly
to emit null / None for any dimension the customer has not stated.

Implementation Note (spike 026)
──────────────────────────────
The original speculative ``google.adk.run`` import does not exist in any published
SDK.  The concrete implementation now uses httpx to call the Gemini REST API directly:
  POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
This was validated against the live API in spike_026_providers.py (Probe 4).
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import httpx

from backend.config import load_settings

__all__ = [
    "extract_preferences",
    "GeminiClient",
    "GeminiRestClient",  # renamed from ADKGeminiClient (spike 026 fix)
]

# ─── Constants ────────────────────────────────────────────────────────────────

# The model used in production — resolved lazily to avoid import-time side-effects
def _default_gemini_model() -> str:
    return load_settings().gemini_model

# Strict vocabulary for soft-match dimensions
_VALID_FUEL_TYPES = {"Petrol", "Diesel", "Electric", "CNG"}
_VALID_BODY_STYLES = {"Hatchback", "Sedan", "SUV", "MUV"}
_VALID_TRANSMISSIONS = {"Manual", "Automatic"}

# Case-insensitive lookup maps: lowercase variant → canonical string.
# Allows the normalizer to accept e.g. "electric", "ELECTRIC", "Electric"
# and canonicalise all of them to "Electric".
_FUEL_CANON = {v.lower(): v for v in _VALID_FUEL_TYPES}
_BODY_CANON = {v.lower(): v for v in _VALID_BODY_STYLES}
_TRANS_CANON = {v.lower(): v for v in _VALID_TRANSMISSIONS}

# All seven keys that must appear in every output dict
_PREFERENCE_KEYS = (
    "budget",
    "min_seats",
    "fuel_type",
    "body_style",
    "transmission",
    "min_mileage_kmpl",
    "feature_keywords",
)

# ─── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = f"""You are a structured-preference extraction module for a car dealership AI consultant ("Riya").

Your task: read the customer's latest spoken utterance and a current running preference set,
and return ONLY a valid JSON object that represents the UPDATED preference set.

--- RULES ---

1. VOCABULARY (hard constraints — never violate these):
   - fuel_type must be exactly one of: {', '.join(sorted(_VALID_FUEL_TYPES))}
     Examples: "EV" → Electric, "electric car" → Electric, "diesel" → Diesel
   - body_style must be exactly one of: {', '.join(sorted(_VALID_BODY_STYLES))}
     Examples: "7-seater van" → MUV, "crossover" → SUV, "hatchback" → Hatchback
   - transmission must be exactly one of: {', '.join(_VALID_TRANSMISSIONS)}
     Examples: "AMT" → Automatic, "manual" → Manual, "auto" → Automatic
   - feature_keywords: any free-text strings the customer mentions as desired features
     (e.g. "sunroof", "rear camera", "ADAS"). Store them exactly as spoken, lowercase.

2. NULLABILITY (critical):
   - Any dimension the customer has NOT mentioned must be emitted as JSON `null`.
   - NEVER invent a value.  If the customer says nothing about budget, budget is null.
   - NEVER invent a default for min_seats, fuel_type, or any other field.

3. FILLER vs. SIGNAL:
   - If the utterance contains no actionable car preference (greetings, acknowledgements,
     filler words like "okay", "hmm", "sure", "no thanks", "not sure" without a preference),
     return ALL keys as null — no change should be made to the running set.
   - Only explicit statements about a preference dimension should update that key.

4. OVERWRITE NOT ACCUMULATE (latest wins):
   - If a new utterance contradicts a previous one (e.g. customer said "petrol" before
     and now says "actually diesel"), the new explicit statement completely replaces
     the old value.  Do NOT merge or accumulate — latest explicit statement wins.

5. SEATING CONTEXT:
   - The dealership catalogue only has vehicles with seating_capacity of 5 or 7.
   - If the customer says "7-seater" or "need room for 7 people", set min_seats to 7.
   - If they say nothing specific, leave min_seats as null.

6. OUTPUT FORMAT:
   - Return ONLY a single JSON object on one line.  No markdown, no explanation.
   - The object MUST contain exactly these seven keys:
     {{
       "budget": null | <number in lakhs>,
       "min_seats": null | <integer 1-7>,
       "fuel_type": null | "Petrol" | "Diesel" | "Electric" | "CNG",
       "body_style": null | "Hatchback" | "Sedan" | "SUV" | "MUV",
       "transmission": null | "Manual" | "Automatic",
       "min_mileage_kmpl": null | <positive number>,
       "feature_keywords": [<string>, ...]   ← always an array (empty if none)
     }}
   - budget: customer budget in lakhs (e.g. "under 10 lakhs" → 10.0).  Null if not stated.
   - min_seats: minimum required seats.  Null if not stated.
   - min_mileage_kmpl: minimum acceptable mileage.  Null if not stated.
   - feature_keywords: array of feature names mentioned by the customer.  Empty array if none.
"""


# ─── Abstract interface ────────────────────────────────────────────────────────

class GeminiClient(ABC):
    """
    Abstract interface for the Gemini preference-extraction call.

    Implementations must be safe to mock in tests — no live API calls in the suite.
    """

    @abstractmethod
    def extract(self, utterance: str, current_prefs: dict | None) -> dict:
        """
        Call Gemini to extract / update preferences from the given utterance.

        Parameters
        ----------
        utterance : str
            The customer's most recent spoken statement (may include context
            from the full transcript, but the instruction is to act on the
            latest statement only).
        current_prefs : dict | None
            The running preference set so far.  May be None for the first turn.
            Keys: ``budget``, ``min_seats``, ``fuel_type``, ``body_style``,
            ``transmission``, ``min_mileage_kmpl``, ``feature_keywords``.
            All values may be None.

        Returns
        -------
        dict
            A delta dict containing the keys that should be updated.
            Keys present with a non-None value overwrite the corresponding
            ``current_prefs`` entry; None values and absent keys leave
            ``current_prefs`` unchanged.
            Always contains all seven keys even if all are None.
        """
        ...


# ─── Normalisation helpers ────────────────────────────────────────────────────


def _normalise_delta(raw: dict) -> dict:
    """
    Normalise and validate a raw delta dict returned from a GeminiClient.

    This runs in ``extract_preferences`` so it applies regardless of which
    concrete client is used (production ADK or test mock).

    Applies:
      - ``feature_keywords`` → always a list (empty list if absent / non-list)
      - ``budget``, ``min_mileage_kmpl`` → float (None if unparseable)
      - ``min_seats`` → int (None if unparseable)
      - ``fuel_type`` → canonical value or None
      - ``body_style`` → canonical value or None
      - ``transmission`` → canonical value or None
      - all seven keys guaranteed present
    """
    if not isinstance(raw, dict):
        raw = {}

    result: dict = {}

    # feature_keywords: always a list
    fk = raw.get("feature_keywords")
    result["feature_keywords"] = list(fk) if isinstance(fk, list) else []

    # Numeric coercion helpers
    def _to_float(val: object) -> float | None:
        if val is None:
            return None
        try:
            return float(val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    def _to_int(val: object) -> int | None:
        if val is None:
            return None
        try:
            return int(val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    # Numeric fields
    result["budget"] = _to_float(raw.get("budget"))
    result["min_mileage_kmpl"] = _to_float(raw.get("min_mileage_kmpl"))
    result["min_seats"] = _to_int(raw.get("min_seats"))

    # Vocabulary constraints — case-insensitive lookup via lowercase map
    fuel = raw.get("fuel_type")
    result["fuel_type"] = _FUEL_CANON.get(str(fuel).strip().lower()) if fuel is not None else None

    body = raw.get("body_style")
    result["body_style"] = _BODY_CANON.get(str(body).strip().lower()) if body is not None else None

    trans = raw.get("transmission")
    result["transmission"] = _TRANS_CANON.get(str(trans).strip().lower()) if trans is not None else None

    # Ensure all seven keys present
    for key in _PREFERENCE_KEYS:
        if key not in result:
            result[key] = None

    return {k: result.get(k) for k in _PREFERENCE_KEYS}


# ─── Concrete REST implementation ─────────────────────────────────────────────


class GeminiRestClient(GeminiClient):
    """
    Production implementation using the Gemini REST API via httpx.

    Validated against the live API in spike_026_providers.py (Probe 4).
    No google-adk dependency required — uses the Gemini generateContent REST endpoint.

    Requires
    --------
    httpx (already in project dependencies)

    The ``GEMINI_API_KEY`` environment variable (or ``GEMINI_API_KEY`` in
    ``.env``) must be set to a valid Google AI API key.
    """

    # Gemini REST API base
    _BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, model: str | None = None) -> None:
        self._model = model if model is not None else _default_gemini_model()
        self._api_key: str | None = None  # lazy-loaded

    def _get_api_key(self) -> str:
        if self._api_key is None:
            # Read from environment (matching how the app loads other secrets)
            self._api_key = os.environ.get("GEMINI_API_KEY", "").strip()
            if not self._api_key:
                from pydantic_settings import BaseSettings
                from backend.config import Settings

                settings = Settings()  # loads from .env automatically
                self._api_key = settings.gemini_api_key.get_secret_value()
        return self._api_key

    def extract(self, utterance: str, current_prefs: dict | None) -> dict:
        current_str = (
            json.dumps(current_prefs, default=list) if current_prefs else "null (first turn)"
        )

        user_message = (
            f"## Current running preference set (do not repeat — update only what the customer just stated):\n"
            f"{current_str}\n\n"
            f"## Customer's latest utterance:\n"
            f'"{utterance}"\n\n'
            f"## Your task:\n"
            f"Read the latest utterance above.  Return ONLY the updated JSON preference set following all rules."
        )

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_message}],
                }
            ],
            "systemInstruction": {
                "parts": [{"text": _SYSTEM_PROMPT}],
            },
            "generationConfig": {
                "temperature": 0.1,  # low temperature for structured extraction
                "maxOutputTokens": 512,
            },
        }

        url = f"{self._BASE_URL}/models/{self._model}:generateContent"

        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self._get_api_key(),
                },
            )
            response.raise_for_status()
            data = response.json()

        raw_text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )

        return self._parse_and_validate(raw_text or "")

    def _parse_and_validate(self, raw_text: str) -> dict:
        """
        Extract JSON from the model response and validate vocabulary constraints.
        """
        # Strip potential markdown code fences
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            json_lines = [ln for ln in lines if not ln.strip().startswith("```") and ln.strip()]
            text = "\n".join(json_lines).strip()

        try:
            raw_dict = json.loads(text)
        except json.JSONDecodeError:
            return {k: None for k in _PREFERENCE_KEYS}

        return _normalise_delta(raw_dict)


# ─── Public API ───────────────────────────────────────────────────────────────


def extract_preferences(
    utterance: str,
    current_prefs: dict | None = None,
    *,
    client: GeminiClient | None = None,
) -> dict:
    """
    Extract and merge a preference-set update from the customer's latest utterance.

    Parameters
    ----------
    utterance : str
        The customer's most recent spoken statement.
    current_prefs : dict | None
        The running preference set from prior turns.  May be None on the first call.
    client : GeminiClient | None
        The Gemini client to use.  If None a ``GeminiRestClient`` is created and used.
        Pass a mock subclass in tests to avoid live API calls.

    Returns
    -------
    dict
        The merged preference set.  Contains all seven keys
        (``budget``, ``min_seats``, ``fuel_type``, ``body_style``, ``transmission``,
        ``min_mileage_kmpl``, ``feature_keywords``).  All values are nullable.
        ``feature_keywords`` is always a list (possibly empty).

    Merge semantics
    ───────────────
    The Gemini delta dict is merged onto ``current_prefs`` using "latest wins":
    - Each key present in the delta with a non-None value overwrites the prior value.
    - Keys absent from the delta or explicitly None leave ``current_prefs`` unchanged.
    - ``feature_keywords`` is replaced entirely (not appended) — the LLM returns
      the complete array based on the full transcript context.
    """
    if client is None:
        client = GeminiRestClient()

    # Start from a clean base containing all seven keys (all None)
    base: dict = {k: None for k in _PREFERENCE_KEYS}
    if current_prefs is not None:
        for key in _PREFERENCE_KEYS:
            base[key] = current_prefs.get(key)

    # Get the raw Gemini delta, then normalise + validate it
    raw_delta = client.extract(utterance, current_prefs)
    delta = _normalise_delta(raw_delta)

    # Merge: delta non-None values overwrite; None / absent → keep prior
    # feature_keywords: non-empty list → replace.  Empty list / None → carry over
    # existing (filler = no change).  Special case: when base starts as None (first
    # turn) and delta is [], we want [] as the initial state.
    merged = dict(base)
    has_existing_features = isinstance(base.get("feature_keywords"), list) and len(base["feature_keywords"]) > 0

    for key, value in delta.items():
        if key in _PREFERENCE_KEYS:
            if key == "feature_keywords":
                if isinstance(value, list) and len(value) > 0:
                    merged[key] = value
                elif isinstance(value, list) and len(value) == 0 and not has_existing_features:
                    # First turn with no features stated yet: initialise to []
                    merged[key] = []
                # elif empty but base already has features: carry over (no change)
            elif value is not None:
                merged[key] = value
            # None / absent for other keys: leave merged[key] as-is (carry-over)

    return merged
