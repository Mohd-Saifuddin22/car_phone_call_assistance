"""
backend/contact.py — Pure contact normalization module (no I/O, no DB, no network).

Provides canonical forms for phone numbers and email addresses used across the
entire application.  These functions are the single source of truth applied
right at the API boundary before storage so that a stored lead and a block-list
entry can never drift out of match due to spacing or casing differences.

Public API
----------
validate_indian_mobile(phone: str) -> ValidationResult
    Checks whether *phone* is a valid Indian mobile number.
    Returns a ValidationResult with .is_valid and .reason.

normalize_indian_mobile(phone: str) -> str
    Idempotent normalizer.  Strips superficial characters (spaces, dashes),
    the +91/91 country code prefix (if present), the leading 0 trunk prefix
    (if present), and returns the E.164 form (+91XXXXXXXXXX).
    Raises ValueError if the input is invalid.

normalize_email(email: str) -> str
    Strictly lowercases and strips leading and trailing whitespace.
"""

from __future__ import annotations

from dataclasses import dataclass
from re import sub

# ─── E.164 prefix for India (hardcoded for MVP) ─────────────────────────────────

_INDIA_CC = "+91"  # country code
_ALLOWED_FIRST_DIGITS = {"6", "7", "8", "9"}


# ─── Shared digit extraction ───────────────────────────────────────────────────
#
# Both validate_indian_mobile and normalize_indian_mobile must apply the same
# preprocessing steps so that validate(x) succeeds **iff** normalize(x) succeeds.
# Steps applied here:
#   1. Strip everything except digits.
#   2. Strip bare "91" (country code without +) if it makes the total exactly
#      12 digits — leaves exactly 10 subscriber digits.
#   3. Strip a leading "0" (India trunk prefix) if present.
#      This is a deliberate design choice to prevent spurious rejections.

def _extract_subscriber_digits(raw: str) -> str:
    """
    Strip and normalise a raw phone string to the 10-digit subscriber number.

    Steps (order matters!)
    -----
    1. Remove all non-digit characters.
    2. Strip the bare country code "91" if it results in exactly 10 remaining
       digits — this must run BEFORE the 0-strip so that "0919876543210"
       (trunk + CC + subscriber) is handled correctly.
    3. Strip a leading trunk prefix "0" if present.

    Returns
    -------
    str
        Exactly 10 digits (caller is responsible for validating length and
        starting digit).
    """
    digits = sub(r"\D", "", raw)

    # Strip bare country code "91" FIRST (must precede 0-strip for combined prefix)
    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]

    # Strip leading 0 trunk prefix (run after CC-strip so 0+91 combos work)
    if digits.startswith("0"):
        digits = digits[1:]

    return digits


# ─── Validation result ─────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ValidationResult:
    """
    Structured result for phone-number validation.

    Attributes
    ----------
    is_valid : bool
        True when the input satisfies all rules.
    reason : str | None
        Human-readable explanation when is_valid is False; None when valid.
    """

    is_valid: bool
    reason: str | None

    def __bool__(self) -> bool:
        """Allow `if validate_indian_mobile(x):` idiomatic checks."""
        return self.is_valid


# ─── Phone validation ──────────────────────────────────────────────────────────

def validate_indian_mobile(phone: str) -> ValidationResult:
    """
    Validate an Indian mobile number.

    This function and ``normalize_indian_mobile`` are guaranteed to agree:
    ``validate_indian_mobile(x).is_valid`` is True **iff**
    ``normalize_indian_mobile(x)`` succeeds without raising.

    Preprocessing (shared with normalize)
    -------------------------------------
    * Non-digit characters (spaces, dashes, etc.) are stripped.
    * A bare "91" country code is stripped if present.
    * A leading "0" trunk prefix is stripped if present.

    Rules
    -----
    * Exactly 10 digits after preprocessing.
    * The first digit must be 6, 7, 8, or 9.

    Parameters
    ----------
    phone : str
        Raw phone string — may contain spaces, dashes, +91, or a leading 0.

    Returns
    -------
    ValidationResult
        is_valid=True  → valid; reason=None
        is_valid=False → invalid; reason is a clear English sentence.
    """
    digits = _extract_subscriber_digits(phone)

    # ── Length check ────────────────────────────────────────────────────────────
    if len(digits) < 10:
        return ValidationResult(
            is_valid=False,
            reason=(
                f"Too short: found {len(digits)} digit(s); "
                "an Indian mobile number must contain exactly 10 digits "
                "(after stripping country code, trunk prefix, and separators)."
            ),
        )

    if len(digits) > 10:
        return ValidationResult(
            is_valid=False,
            reason=(
                f"Too long: found {len(digits)} digit(s); "
                "an Indian mobile number must contain exactly 10 digits "
                "(after stripping country code, trunk prefix, and separators)."
            ),
        )

    # ── First-digit check ──────────────────────────────────────────────────────
    assert len(digits) == 10
    if digits[0] not in _ALLOWED_FIRST_DIGITS:
        return ValidationResult(
            is_valid=False,
            reason=(
                f"Invalid first digit '{digits[0]}': "
                "Indian mobile numbers must start with 6, 7, 8, or 9."
            ),
        )

    return ValidationResult(is_valid=True, reason=None)


# ─── Phone normalisation ───────────────────────────────────────────────────────

def normalize_indian_mobile(phone: str) -> str:
    """
    Normalise *phone* to E.164 format (+91XXXXXXXXXX).

    Behaviour
    ---------
    * Strips all non-digit characters.
    * Strips bare "91" country code if present (12 bare digits → 10).
    * Strips leading "0" trunk prefix if present.
    * Validates the resulting 10-digit subscriber number
      (exactly 10 digits, first ∈ {6-9}).
    * Prepends ``+91`` and returns the canonical string.

    Idempotent
    ----------
    Calling this function on an already-canonical ``+919876543210`` returns the
    same string unchanged.

    Parameters
    ----------
    phone : str
        Raw phone string.

    Returns
    -------
    str
        E.164 number, e.g. ``"+919876543210"``.

    Raises
    ------
    ValueError
        When the input cannot be normalised — wraps the ``ValidationResult``.
    """
    digits = _extract_subscriber_digits(phone)

    # ── Validate ────────────────────────────────────────────────────────────────
    result = validate_indian_mobile(digits)  # digits is already bare 10-digit string
    if not result.is_valid:
        raise ValueError(result.reason)

    # ── Format ─────────────────────────────────────────────────────────────────
    return f"{_INDIA_CC}{digits}"


# ─── Email normalisation ───────────────────────────────────────────────────────

def normalize_email(email: str) -> str:
    """
    Return the canonical form of *email*.

    Canonicalisation steps
    ----------------------
    1. Strip leading and trailing whitespace.
    2. Strictly lowercase the local part and domain.

    Parameters
    ----------
    email : str
        Raw email address; may contain mixed casing or surrounding whitespace.

    Returns
    -------
    str
        Canonical form, e.g. ``"user@Example.COM"`` → ``"user@example.com"``.

    Raises
    ------
    ValueError
        If *email* is empty after trimming.
    """
    if not email:
        raise ValueError("Email address must not be empty.")

    canonical = email.strip().lower()

    if not canonical:
        raise ValueError("Email address must not be empty after normalisation.")

    return canonical
