"""
backend/blurb_formatter.py — Gemini Blurb Formatter (Issue 023).

Architecture
──────────
The Gemini call is fully isolated behind the abstract ``GeminiClient`` interface.
``GeminiRestClient`` is the concrete production implementation (uses httpx + Gemini REST API).
Tests substitute a mock that never makes a live API call.

The formatting step is strictly DECOUPLED from the core recommendation storage logic.
If the Gemini call fails (timeout, API error, malformed output), this module catches the
failure and returns ``None`` so that the caller can still successfully store the ranked
items in the database. A cosmetic failure must never cause the loss of the actual
deliverable.

Public Interface
─────────────
format_blurb(
    ranked_cars: list[ScoredCar],
    note: str | None = None,
    client: GeminiClient | None = None,
) -> str | None

Given the finalized ranked car list from the Matcher Consumer, enriches each vehicle
with catalogue details and calls Gemini to produce a natural-language recommendation
readout for the customer. Returns the formatted string, or ``None`` if the Gemini
call fails — the caller stores items with a ``null`` blurb and retries later.

Implementation Note (spike 026)
──────────────────────────────
The original speculative ``google.adk.run`` import does not exist in any published
SDK.  The concrete implementation now uses httpx to call the Gemini REST API directly:
  POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
This was validated against the live API in spike_026_providers.py (Probe 4).
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import httpx

from backend.config import load_settings

__all__ = [
    "format_blurb",
    "GeminiClient",
    "GeminiRestClient",  # renamed from ADKGeminiClient (spike 026 fix)
]

# Module-level logger — used to surface swallowed Gemini failures in production.
_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from backend.catalogue import VehicleRecord
    from backend.scorer import ScoredCar

# ─── Constants ────────────────────────────────────────────────────────────────

# The model used in production — resolved lazily to avoid import-time side-effects
def _default_gemini_model() -> str:
    return load_settings().gemini_model

# ─── System prompt ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a friendly car dealership AI consultant ("Riya").
Your task: write a short, natural recommendation readout for the customer based on
their preferences and the top vehicle matches.

--- RULES ---

1. Tone: warm, professional, conversational. The admin will read this verbatim
   to the customer after the call, so it must sound natural when spoken aloud.
   Use short sentences and avoid jargon.

2. Content per car:
   - Lead with the car name (Make + Model).
   - Mention the key reasons it scored well: fuel type match, body style, mileage,
     transmission, and notable features.
   - If the car matches the budget exactly or comes in under budget, mention value.
   - Keep it to 2–4 sentences per car.

3. Order: present cars in the ranked order given (rank 1 first).

4. Output: a single plain-text paragraph per car, separated by blank lines.
   Do NOT use bullet points, markdown, or any special formatting.
   Do NOT invent information not present in the enriched data.
   If mileage or features are not available for a car, simply omit that detail.
"""


# ─── Abstract interface ──────────────────────────────────────────────────────


class GeminiClient(ABC):
    """
    Abstract interface for the Gemini blurb-formatting call.

    Implementations must be safe to mock in tests — no live API calls in the suite.
    """

    @abstractmethod
    def format(self, enriched_cars: list[dict], *, note: str | None = None) -> str | None:
        """
        Call Gemini to produce a natural-language recommendation readout.

        Parameters
        ----------
        enriched_cars : list[dict]
            List of enriched vehicle dictionaries, one per ranked car,
            in descending rank order. Each dict contains keys:
              - rank (int): 1-based rank
              - name (str): Make + Model
              - price_lakhs (float | None): price in lakhs
              - fuel_type (str | None)
              - body_style (str | None)
              - seating_capacity (int | None)
              - transmission (str | None)
              - mileage_kmpl (float | None)
              - description (str | None)
              - features (list[str])
              - fuel_match (bool): matched customer's fuel preference
              - body_match (bool): matched customer's body style preference
              - transmission_match (bool): matched customer's transmission preference
              - mileage_pts (int): bonus points earned from mileage
              - feature_pts (int): bonus points earned from features
              - score (int): total score
        note : str | None
            Optional fallback note from the scorer (e.g. when hard constraints
            could not be satisfied). If provided, Gemini is instructed to include
            a caveat in the readout. Passed through from ``RankingResult.note``.

        Returns
        -------
        str | None
            The Gemini-formatted natural-language readout, or ``None`` if the
            response is empty / whitespace-only.
        """
        ...


# ─── Enrichment helpers ──────────────────────────────────────────────────────


def _enrich_scored_car(
    scored_car: ScoredCar,
    rank: int,
) -> dict:
    """
    Build an enriched dict for one scored car, ready for the Gemini prompt.

    Parameters
    ----------
    scored_car : ScoredCar
        A scored car from the Matcher Consumer's ranked output.
    rank : int
        1-based rank in the recommendation list.

    Returns
    -------
    dict
        Enriched dictionary as described in ``GeminiClient.format``.
    """
    vehicle = scored_car.vehicle
    return {
        "rank": rank,
        "name": vehicle.name,
        "price_lakhs": vehicle.price_lakhs,
        "fuel_type": vehicle.fuel_type,
        "body_style": vehicle.body_style,
        "seating_capacity": vehicle.seating_capacity,
        "transmission": vehicle.transmission,
        "mileage_kmpl": vehicle.mileage_kmpl,
        "description": vehicle.description,
        "features": list(vehicle.features),
        "fuel_match": scored_car.fuel_match,
        "body_match": scored_car.body_match,
        "transmission_match": scored_car.transmission_match,
        "mileage_pts": scored_car.mileage_pts,
        "feature_pts": scored_car.feature_pts,
        "score": scored_car.score,
    }


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

    _BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, model: str | None = None) -> None:
        self._model = model if model is not None else _default_gemini_model()
        self._api_key: str | None = None  # lazy-loaded

    def _get_api_key(self) -> str:
        if self._api_key is None:
            self._api_key = os.environ.get("GEMINI_API_KEY", "").strip()
            if not self._api_key:
                from backend.config import Settings

                settings = Settings()
                self._api_key = settings.gemini_api_key.get_secret_value()
        return self._api_key

    def format(self, enriched_cars: list[dict], *, note: str | None = None) -> str | None:
        cars_json = json.dumps(enriched_cars, indent=2, default=str)

        note_block = (
            f"\n\n## Important caveat to include in the readout:\n{note}"
            if note
            else ""
        )

        user_message = (
            "## Ranked vehicle matches (in descending rank order):\n"
            f"{cars_json}"
            f"{note_block}\n\n"
            "## Your task:\n"
            "Write a natural-language recommendation readout for the customer "
            "following all rules. Return ONLY the formatted text."
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
                "temperature": 0.7,
                "maxOutputTokens": 1024,
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

        return self._clean_response(raw_text or "")

    def _clean_response(self, raw_text: str) -> str | None:
        """
        Strip potential markdown code fences and surrounding whitespace.
        Normalizes empty or whitespace-only output to None so that callers
        treat "blank" and "failed" identically.
        """
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            content_lines = [
                ln for ln in lines
                if not ln.strip().startswith("```") and ln.strip()
            ]
            text = "\n".join(content_lines).strip()
        # Normalize empty / whitespace-only to None
        return text if text else None


# ─── Public API ─────────────────────────────────────────────────────────────


def format_blurb(
    ranked_cars: list[ScoredCar],
    note: str | None = None,
    *,
    client: GeminiClient | None = None,
) -> str | None:
    """
    Format the ranked vehicle recommendations into a natural-language readout.

    This function is the BLURB FORMATTER — it is the second and final LLM step
    in the call pipeline. It runs exactly once at call close and its result is
    stored persistently in the database (``recommendations.blurb_text``); it is
    never regenerated on view.

    Decoupled Storage (Critical)
    ───────────────────────────
    This function catches ALL exceptions raised by the Gemini call and returns
    ``None`` instead. The caller is responsible for storing the ranked items
    in the database regardless of whether the blurb succeeded:

        blurb = format_blurb(ranked_cars, note=scoring_note, client=mock)
        db.store_recommendations(lead_id, items, blurb)  # blurb may be None

    A cosmetic Gemini failure must never cause the loss of the actual deliverable
    (the ranked vehicle items).

    Parameters
    ----------
    ranked_cars : list[ScoredCar]
        The ranked vehicle list from the Matcher Consumer. Typically 1–5 items.
    note : str | None
        Optional fallback note from the scorer (e.g. when hard constraints
        could not be satisfied). Included verbatim in the Gemini prompt so the
        readout contains the caveat. Passed through from ``RankingResult.note``.
    client : GeminiClient | None
        The Gemini client to use.  If None a ``GeminiRestClient`` is created and used.
        Pass a mock subclass in tests to avoid live API calls.

    Returns
    -------
    str | None
        The formatted natural-language readout, or ``None`` if the Gemini call
        failed (timeout, API error, malformed output, etc.) or returned an empty
        / whitespace-only response.  Callers must handle ``None`` and store a
        ``null`` blurb in the database.
    """
    # Fast path: empty list → nothing to format
    if not ranked_cars:
        return None

    if client is None:
        client = GeminiRestClient()

    # Enrich each scored car with catalogue details
    enriched_cars = [
        _enrich_scored_car(scored_car, rank=i + 1)
        for i, scored_car in enumerate(ranked_cars)
    ]

    # Isolated Gemini call — non-fatal failure returns None
    try:
        return client.format(enriched_cars, note=note)
    except Exception:
        # Any failure (API timeout, malformed response, network error,
        # JSON decode error, KeyError, TypeError, etc.) is treated as
        # non-fatal. The ranked items are still stored; the blurb is null.
        # Log so persistently failing blurbs are diagnosable in production.
        _logger.error("Blurb formatting failed", exc_info=True)
        return None
