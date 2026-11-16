"""
backend/tests/test_contact.py — Verification tests for the contact normalization module.

Coverage
--------
Phone validation (validate_indian_mobile)
    ✓ Accepts bare 10-digit numbers with valid starting digits (6–9)
    ✓ Accepts +91XXXXXXXXXX (already-canonical E.164)
    ✓ Accepts 91XXXXXXXXXX (country code without +)
    ✓ Accepts 0XXXXXXXXXX (leading 0 trunk prefix)
    ✓ Rejects numbers that are too short (after stripping all prefixes)
    ✓ Rejects numbers that are too long (without country code)
    ✓ Rejects numbers whose first digit is outside 6–9 (0–5)
    ✓ Rejects alphabetic strings
    ✓ Provides non-None reason on every invalid case

Phone normalisation (normalize_indian_mobile)
    ✓ Strips spaces, dashes, mixed separators
    ✓ Prepends +91 to bare 10-digit numbers
    ✓ Handles +91 prefix (strips the prefix before normalising)
    ✓ Handles 91 prefix (without +)
    ✓ Handles leading 0 trunk prefix
    ✓ Is idempotent on already-canonical +91XXXXXXXXXX numbers
    ✓ Is idempotent on bare 10-digit numbers
    ✓ Raises ValueError wrapping the ValidationResult.reason on invalid input

validate / normalize consistency
    ✓ validate(x).is_valid is True iff normalize(x) does not raise, for all
      canonical forms (+91, 91, 0, bare digits)

Email normalisation (normalize_email)
    ✓ Lowercases the local part and domain
    ✓ Strips leading/trailing whitespace
    ✓ Handles mixed casing throughout
    ✓ Returns the same string when already canonical
    ✓ Raises ValueError for empty input
    ✓ Raises ValueError for whitespace-only input

No test calls load_settings().
"""

from __future__ import annotations

import pytest

from backend.contact import (
    ValidationResult,
    normalize_email,
    normalize_indian_mobile,
    validate_indian_mobile,
)


# ==============================================================================
# Phone validation — valid inputs
# ==============================================================================

@pytest.mark.parametrize(
    "phone",
    [
        # Bare 10-digit numbers
        "9876543210",
        "9876543210",
        "9876543210",
        "6000000000",   # starts with 6 (minimum)
        "9999999999",   # starts with 9 (maximum)
        "7000000000",   # starts with 7
        "8000000000",   # starts with 8
        # With spaces and dashes
        "9876 543 210",
        "9876-543-210",
        "987 654 321 0",
        # Country code +91 (E.164)
        "+919876543210",
        # Country code 91 (without +)
        "919876543210",
        # Trunk prefix 0
        "09876543210",
    ],
)
def test_validate_indian_mobile_accepts_valid(phone: str) -> None:
    """All valid forms must be accepted by validate_indian_mobile."""
    result = validate_indian_mobile(phone)
    assert result.is_valid is True
    assert result.reason is None


# ==============================================================================
# Phone validation — invalid inputs
# ==============================================================================

class TestValidateRejectsTooShort:
    """Numbers with fewer than 10 digits must be rejected (after all stripping)."""

    @pytest.mark.parametrize(
        "phone",
        [
            "",                  # empty
            "1",                # 1 digit
            "12345",            # 5 digits
            "123456789",        # 9 digits
            "987654",           # 6 digits
            " 987654321 ",      # 9 digits (whitespace stripped)
            "0123456789",      # 10 bare digits, but starts with 0 → length-10 strip-0 → 9 digits
        ],
    )
    def test_too_short(self, phone: str) -> None:
        result = validate_indian_mobile(phone)
        assert result.is_valid is False
        assert result.reason is not None
        assert "Too short" in result.reason


class TestValidateRejectsTooLong:
    """Numbers with more than 10 bare digits (after country-code strip) must be rejected."""

    @pytest.mark.parametrize(
        "phone",
        [
            "12345678901",        # 11 bare digits (no prefix to strip)
            "1234567890123",      # 13 bare digits
            "00987654321099",     # country code + garbage after subscriber number
            "+919876543210999",  # +91 + 13 digits → strip 91 → 11 digits
        ],
    )
    def test_too_long(self, phone: str) -> None:
        result = validate_indian_mobile(phone)
        assert result.is_valid is False
        assert result.reason is not None
        assert "Too long" in result.reason


class TestValidateRejectsBadFirstDigit:
    """First digit must be 6, 7, 8, or 9 (after stripping trunk prefix)."""

    @pytest.mark.parametrize(
        "phone",
        [
            "0123456789",       # bare → strip 0 → 123456789 (9 digits) → too short
            "1234567890",       # strip nothing → starts with 1 → bad first digit
            "2345678901",       # starts with 2 → bad first digit
            "5000000000",       # starts with 5 → bad first digit
        ],
    )
    def test_first_digit_outside_6_9(self, phone: str) -> None:
        result = validate_indian_mobile(phone)
        assert result.is_valid is False
        assert result.reason is not None
        # 0123456789 hits the too-short branch; the rest hit the bad-first-digit branch
        if phone == "0123456789":
            assert "Too short" in result.reason
        else:
            assert "Invalid first digit" in result.reason


class TestValidateRejectsAlphabetical:
    """Purely alphabetic strings must be rejected."""

    @pytest.mark.parametrize(
        "phone",
        [
            "abcdefghij",
            "hello world",
            "phone-number",
            "abc",
            "NOTAPHONE",
        ],
    )
    def test_alphabetic_input(self, phone: str) -> None:
        result = validate_indian_mobile(phone)
        assert result.is_valid is False
        assert result.reason is not None


# ==============================================================================
# Phone normalisation — happy paths
# ==============================================================================

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Bare 10-digit numbers → +91XXXXXXXXXX
        ("9876543210", "+919876543210"),
        ("6000000000", "+916000000000"),
        ("9999999999", "+919999999999"),
        ("7000000000", "+917000000000"),
        ("8000000000", "+918000000000"),
        # With spaces / dashes
        ("9876 543 210", "+919876543210"),
        ("9876-543-210", "+919876543210"),
        ("987 654-321 0", "+919876543210"),
        # +91 prefix (E.164) — idempotent
        ("+919876543210", "+919876543210"),
        # 91 prefix (without +)
        ("919876543210", "+919876543210"),
            # Trunk prefix 0
            ("09876543210", "+919876543210"),
            # Leading/trailing whitespace
        ("  9876543210  ", "+919876543210"),
        ("\t+919876543210\n", "+919876543210"),
    ],
)
def test_normalize_indian_mobile_happy_paths(raw: str, expected: str) -> None:
    """Valid inputs must normalize to the expected E.164 string."""
    assert normalize_indian_mobile(raw) == expected


def test_normalize_idempotent_on_bare_digits() -> None:
    """Normalising a bare 10-digit number twice must yield the same result."""
    original = "9876543210"
    once = normalize_indian_mobile(original)
    twice = normalize_indian_mobile(once)
    assert once == twice == "+919876543210"


def test_normalize_idempotent_on_e164() -> None:
    """Normalising an already-canonical +91XXXXXXXXXX number must return it unchanged."""
    canonical = "+919876543210"
    assert normalize_indian_mobile(canonical) == canonical


def test_normalize_idempotent_on_trunk_prefix() -> None:
    """Normalising a number with leading 0 must also be idempotent."""
    original = "09876543210"
    once = normalize_indian_mobile(original)
    twice = normalize_indian_mobile(once)
    assert once == twice == "+919876543210"


# ==============================================================================
# Phone normalisation — error paths
# ==============================================================================

class TestNormalizeRaisesOnInvalid:
    """normalize_indian_mobile must raise ValueError with a clear message on bad input."""

    @pytest.mark.parametrize(
        "raw",
        [
            "",                  # empty
            "123",               # too short
            "0123456789",        # strip 0 → 9 digits → too short
            "abcdefghij",        # alphabetic
            "9876543210123",     # too long (13 digits)
        ],
    )
    def test_raises_valueerror(self, raw: str) -> None:
        with pytest.raises(ValueError, match=".+"):
            normalize_indian_mobile(raw)

    def test_valueerror_message_matches_validation_reason(self) -> None:
        """ValueError raised by normalize must carry the same reason as ValidationResult."""
        raw = "1234567890"   # starts with 1 → invalid first digit
        result = validate_indian_mobile(raw)
        assert not result.is_valid
        assert result.reason is not None

        with pytest.raises(ValueError) as exc_info:
            normalize_indian_mobile(raw)

        assert str(exc_info.value) == result.reason


# ==============================================================================
# validate / normalize consistency
# ==============================================================================

class TestValidateNormalizeConsistency:
    """
    validate_indian_mobile(x).is_valid must be True **iff**
    normalize_indian_mobile(x) succeeds without raising.

    This is the core invariant that the supervisor flagged as BLOCKING.
    """

    @pytest.mark.parametrize(
        "raw",
        [
            # All four valid canonical forms
            "9876543210",        # bare
            "+919876543210",     # E.164
            "919876543210",      # country code without +
            "09876543210",       # trunk prefix
            # With separators
            "9876 543 210",
            "+91 98765 43210",
            "  9876543210  ",
        ],
    )
    def test_validate_true_when_normalize_succeeds(self, raw: str) -> None:
        """If normalize succeeds, validate must also succeed."""
        # Normalize must not raise
        normalized = normalize_indian_mobile(raw)
        assert normalized == "+919876543210"

        # Validate must agree
        result = validate_indian_mobile(raw)
        assert result.is_valid is True, (
            f"validate_indian_mobile({raw!r}) returned invalid "
            f"but normalize_indian_mobile succeeded with {normalized!r}"
        )

    @pytest.mark.parametrize(
        "raw",
        [
            "",                  # empty
            "123",               # too short
            "abcdefghij",        # alphabetic
            "9876543210123",     # too long (13 bare digits)
            "1234567890",       # bad first digit (starts with 1)
            "0123456789",       # strip 0 → 9 digits → too short
        ],
    )
    def test_validate_false_when_normalize_raises(self, raw: str) -> None:
        """If normalize raises ValueError, validate must also return invalid."""
        with pytest.raises(ValueError):
            normalize_indian_mobile(raw)

        result = validate_indian_mobile(raw)
        assert not result.is_valid, (
            f"validate_indian_mobile({raw!r}) returned valid "
            f"but normalize_indian_mobile raised ValueError"
        )


# ==============================================================================
# Email normalisation
# ==============================================================================

class TestNormalizeEmailHappyPaths:
    """normalize_email must canonicalise casing and trim whitespace."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            # Casing variations
            ("User@Example.COM", "user@example.com"),
            ("USER@EXAMPLE.COM", "user@example.com"),
            ("user@example.com", "user@example.com"),
            ("UsEr@ExAmPlE.CoM", "user@example.com"),
            # Whitespace
            ("  user@example.com  ", "user@example.com"),
            ("\tuser@example.com\n", "user@example.com"),
            # Combined
            ("  USER@EXAMPLE.COM  ", "user@example.com"),
            (" Mixed.Case@Domain.ORG ", "mixed.case@domain.org"),
            # Plain
            ("simple@example.com", "simple@example.com"),
        ],
    )
    def test_cases(self, raw: str, expected: str) -> None:
        assert normalize_email(raw) == expected

    def test_already_canonical_returns_same_object(self) -> None:
        """Normalising an already-canonical email should return an equal string."""
        email = "test@example.org"
        result = normalize_email(email)
        assert result == email


class TestNormalizeEmailRejectsEmpty:
    """normalize_email must reject empty or whitespace-only inputs."""

    @pytest.mark.parametrize("raw", ["", "   ", "\t\n"])
    def test_raises_valueerror_for_empty(self, raw: str) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            normalize_email(raw)


# ==============================================================================
# ValidationResult dataclass interface
# ==============================================================================

def test_validation_result_is_frozen() -> None:
    """ValidationResult must be immutable (frozen=True)."""
    result = ValidationResult(is_valid=True, reason=None)
    with pytest.raises(AttributeError):
        result.is_valid = False  # type: ignore[assignment]


def test_validation_result_bool_true() -> None:
    """ValidationResult with is_valid=True must be truthy."""
    assert bool(ValidationResult(is_valid=True, reason=None)) is True


def test_validation_result_bool_false() -> None:
    """ValidationResult with is_valid=False must be falsy."""
    assert bool(ValidationResult(is_valid=False, reason="bad")) is False


def test_validation_result_equality() -> None:
    """Two ValidationResult objects with the same fields must be equal."""
    a = ValidationResult(is_valid=True, reason=None)
    b = ValidationResult(is_valid=True, reason=None)
    assert a == b


def test_validation_result_slots() -> None:
    """ValidationResult must use __slots__ for memory efficiency."""
    result = ValidationResult(is_valid=True, reason=None)
    assert not hasattr(result, "__dict__")
