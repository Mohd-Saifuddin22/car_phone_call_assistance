"""
backend/audio/lexicon.py — Deepgram Flux custom-vocabulary lexicon (Issue 035).

DEEPGRAM_KEYTERMS is the seed list of domain terms sent to Deepgram Flux via
the ``keyterm`` query parameter on the WebSocket URL (and via the ``keyterms``
field of the ``Configure`` control message). Deepgram boosts recognition
accuracy on these terms so the STT renders them correctly instead of mangling
them with its general-language acoustic model.

Why this list exists
────────────────────
The voice pipeline carries free-form Hinglish speech about cars. The customer
will say "seven seater SUV under fifteen lakh" and Deepgram's base model has
no reason to weight "lakh" or "SUV" higher than "lock" or "sue". Keyterms
fix that — they bias the language model toward our domain.

Constraints
───────────
* Hard cap of **100 entries** (Deepgram rejects more). The list is sized to
  stay well under that bound so future additions have room.
* All entries are single tokens (whitespace is significant to Deepgram — a
  phrase gets split into independent weighted tokens). We split multi-word
  brand/model names into individual tokens here so each is weighted.
* Case is preserved as written but Deepgram is case-insensitive at match
  time. Acronyms like "SUV" and "ADAS" stay uppercase to match the way the
  customer actually says them.

Categories
──────────
1. Body styles — what shape of car the customer wants.
2. Fuel types — the propulsion they're asking about.
3. Transmissions.
4. Indian-context money / finance words.
5. Common car features that customers name out loud.
6. Brand names from the catalogue.
7. Model names from the catalogue (truncated on whitespace — see above).

Stability
─────────
The list is hand-curated and intentionally small. Adding a term is a code
change (not a runtime config) so that the lexicon is auditable in code
review and a deliberate decision every time.
"""

from __future__ import annotations

from typing import Final

__all__ = ["DEEPGRAM_KEYTERMS", "DEEPGRAM_KEYTERMS_HINTS"]

# ─── Body styles ───────────────────────────────────────────────────────────────
#
# Deepgram renders "SUV" as "sue you vee" without a hint; the lexicon weights
# the canonical rendering so the transcript reads "SUV" cleanly.
_BODY_STYLES: Final[tuple[str, ...]] = (
    "SUV",
    "MUV",
    "Hatchback",
    "Sedan",
)

# ─── Fuel types ────────────────────────────────────────────────────────────────
#
# "EV" is the customer-facing shorthand for "Electric"; both must be in the
# lexicon because the customer uses either freely.
_FUEL_TYPES: Final[tuple[str, ...]] = (
    "Petrol",
    "petrol",  # lowercase alias — Deepgram is case-insensitive but listing
               # both is cheap and self-documenting.
    "Diesel",
    "diesel",
    "Electric",
    "electric",
    "EV",
    "ev",
    "CNG",
    "cng",
    # "Hybrid" deliberately omitted — the catalogue has no hybrid vehicles
    # (only Petrol / Diesel / Electric / CNG), and we're at the Deepgram
    # 100-keyterm cap.  Re-add if the catalogue expands.
)

# ─── Transmissions ─────────────────────────────────────────────────────────────
_TRANSMISSIONS: Final[tuple[str, ...]] = (
    "Manual",
    "Automatic",
    "AMT",
    "CVT",
    "DCT",
)

# ─── Indian-context money / finance ───────────────────────────────────────────
#
# "lakh" / "crore" are the units customers use for car prices in India.
# "EMI" is the monthly-installment finance term.
_INDIAN_FINANCE: Final[tuple[str, ...]] = (
    "lakh",
    "lakhs",
    "crore",
    "crores",
    "EMI",
    "downpayment",
    "exshowroom",
)

# ─── Common car features that customers name out loud ─────────────────────────
#
# These are features the customer specifically asks about ("mujhe sunroof
# chahiye"), so mishearing them would degrade the Matcher's preference
# extraction. AC/NCAP = India's safety rating; "ADAS" = advanced driver
# assistance systems; "ABS" / "ESP" = brake / stability control.
_FEATURES: Final[tuple[str, ...]] = (
    "Sunroof",
    "sunroof",  # lowercase alias — Deepgram is case-insensitive but listing
                # both is cheap and self-documenting.
    "Moonroof",
    "ADAS",
    "Touchscreen",
    "Airbags",
    "Airbag",
    "ABS",
    "ESP",
    "Cruise",
    "Bluetooth",
    "CarPlay",
    "AndroidAuto",
    "Reverse",
    "Camera",
    "Leather",
    "Ventilated",
    "Panoramic",
    "Wireless",
    "Charging",
    "Alloys",
    "Turbo",
    "NCAP",
)

# ─── Brand names from the catalogue ───────────────────────────────────────────
#
# These are the manufacturer names. They appear in transcripts as part of
# "Tata Nexon", "Maruti Swift", "Hyundai Creta", etc.
_BRANDS: Final[tuple[str, ...]] = (
    "Tata",
    "Maruti",
    "Suzuki",
    "Mahindra",
    "Kia",
    "MG",
    "Toyota",
    "Hyundai",
    "Honda",
    "Skoda",
    "Volkswagen",
    "Renault",
    "Nissan",
    "Citroen",
    "BYD",
)

# ─── Model names from the catalogue ────────────────────────────────────────────
#
# Multi-word model names ("Grand i10", "XUV700", "Scorpio-N") are split into
# individual tokens — Deepgram's keyterms are single-word weighted tokens, so
# a phrase like "Scorpio-N" would only match the literal "Scorpio-N" string,
# not "Scorpio" spoken on its own. Splitting keeps partial matches useful.
_MODELS: Final[tuple[str, ...]] = (
    "Nexon",
    "Swift",
    "Thar",
    "Seltos",
    "Punch",
    "Creta",
    "Baleno",
    "Tiago",
    "XUV300",
    "Venue",
    "Sonet",
    "Dzire",
    "City",
    "Kushaq",
    "Virtus",
    "Harrier",
    "Fortuner",
    "Ertiga",
    "Scorpio",
    "Safari",
    "Nios",
    "Kiger",
    "Magnite",
    "Hector",
    "Tigor",
    "Jimny",
    "Elevate",
    "Verna",
    "Slavia",
    "XUV700",
    "Innova",
    "Crysta",
    "Seal",
    "i10",
    "ZS",
)


# ─── Assembled lexicon ────────────────────────────────────────────────────────

DEEPGRAM_KEYTERMS: Final[tuple[str, ...]] = (
    *_BODY_STYLES,
    *_FUEL_TYPES,
    *_TRANSMISSIONS,
    *_INDIAN_FINANCE,
    *_FEATURES,
    *_BRANDS,
    *_MODELS,
)
"""
Seed list of single-token domain terms for Deepgram Flux's ``keyterm`` query
parameter.

Capped well under Deepgram's hard limit of 100 terms per session, so future
additions (other catalogue entries, new feature names) have room. Order is
not semantically meaningful — Deepgram treats each entry as an independent
weighted token.

Stability
─────────
Tuple, not list, so callers cannot accidentally mutate the shared lexicon.
Any change to this tuple is a code-reviewable event.


Examples
────────
>>> from backend.audio.lexicon import DEEPGRAM_KEYTERMS
>>> "SUV" in DEEPGRAM_KEYTERMS
True
>>> len(DEEPGRAM_KEYTERMS) <= 100
True
"""


DEEPGRAM_KEYTERMS_HINTS: Final[dict[str, str]] = {
    # ── Spelled-out / expansion hints (for code review only — not sent to
    #    Deepgram, which only takes single tokens). Documents what each
    #    acronym means in case a future maintainer is unfamiliar.
    "SUV": "Sport Utility Vehicle",
    "MUV": "Multi-Utility Vehicle",
    "EV": "Electric Vehicle",
    "CNG": "Compressed Natural Gas",
    "AMT": "Automated Manual Transmission",
    "CVT": "Continuously Variable Transmission",
    "DCT": "Dual-Clutch Transmission",
    "EMI": "Equated Monthly Installment",
    "ABS": "Anti-lock Braking System",
    "ESP": "Electronic Stability Program",
    "ADAS": "Advanced Driver Assistance Systems",
    "NCAP": "Global New Car Assessment Programme",
}
"""
Documentation-only expansion map for the acronym entries in
``DEEPGRAM_KEYTERMS``. Not sent to Deepgram; included here so the acronyms
in the seed list are immediately legible in code review.
"""