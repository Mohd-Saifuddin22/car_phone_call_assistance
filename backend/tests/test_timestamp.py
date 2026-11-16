"""
backend/tests/test_timestamp.py — Unit tests for the timestamp serializer (Issue 015).

Tests cover:
- None-safety
- SQLite space-format conversion
- Subsecond precision preserved
- Idempotency (already ISO + Z passes through)
- Edge cases (empty string, weird formats)
"""

from __future__ import annotations

import pytest

from backend.domain.timestamp import to_iso_z


class TestToIsoZNoneSafety:
    """None inputs must return None."""

    def test_none_returns_none(self) -> None:
        assert to_iso_z(None) is None


class TestToIsoZIdempotency:
    """Already ISO-8601 UTC strings must pass through unchanged."""

    def test_already_iso_z_unchanged(self) -> None:
        result = to_iso_z("2026-06-28T14:00:00Z")
        assert result == "2026-06-28T14:00:00Z"

    def test_already_iso_z_midnight(self) -> None:
        result = to_iso_z("2026-01-01T00:00:00Z")
        assert result == "2026-01-01T00:00:00Z"

    def test_already_iso_z_with_subseconds(self) -> None:
        result = to_iso_z("2026-06-28T14:00:00.123456Z")
        assert result == "2026-06-28T14:00:00.123456Z"

    def test_already_iso_z_very_long_subseconds(self) -> None:
        result = to_iso_z("2026-06-28T14:00:00.123456789012345Z")
        assert result == "2026-06-28T14:00:00.123456789012345Z"


class TestToIsoZSqliteFormat:
    """SQLite space-separated format must be converted correctly."""

    def test_standard_format(self) -> None:
        result = to_iso_z("2026-06-28 14:00:00")
        assert result == "2026-06-28T14:00:00Z"

    def test_midnight(self) -> None:
        result = to_iso_z("2026-01-01 00:00:00")
        assert result == "2026-01-01T00:00:00Z"

    def test_with_subseconds(self) -> None:
        result = to_iso_z("2026-06-28 14:00:00.123456")
        assert result == "2026-06-28T14:00:00.123456Z"

    def test_with_microseconds(self) -> None:
        result = to_iso_z("2026-06-28 14:00:00.000001")
        assert result == "2026-06-28T14:00:00.000001Z"


class TestToIsoZEdgeCases:
    """Boundary and unusual inputs."""

    def test_empty_string_unchanged(self) -> None:
        """Empty string has no space to replace; Z is appended anyway."""
        result = to_iso_z("")
        assert result == "Z"

    def test_string_without_space_gets_z_appended(self) -> None:
        """Strings without a space (e.g. just a date) also get Z appended."""
        result = to_iso_z("2026-06-28")
        assert result == "2026-06-28Z"
