"""
backend/tests/test_startup_assets.py — Unit tests for
``backend.startup.validate_disclosure_assets`` (Issue 037 compliance gate).

Test plan
─────────

V1  Happy path: both frozen real assets present and non-empty → no exception.
V2  Custom assets_dir is respected — points at a temp dir with both files.
V3  Missing file (assets_dir has only English, no Hinglish) → FileNotFoundError.
V4  Missing file (assets_dir empty) → FileNotFoundError on the first language.
V5  Empty file (0 bytes) → RuntimeError.  Cannot boot with a zero-byte
    disclosure; that would mark-complete-without-playing in production.
V6  Mixed: one present+non-empty, one present-but-empty → RuntimeError
    (the empty one fails first, alphabetically).
V7  Custom languages tuple → only those files are required.

The asset validation is the compliance gate for Issue 037.  These
tests must be hermetic — they use temp dirs so the real frozen
``backend/audio/assets/`` files are NEVER touched (per the
disclosure_{en|hi}.ulaw "frozen" rule).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from backend.startup import (
    _DEFAULT_DISCLOSURE_ASSETS_DIR,
    _DISCLOSURE_LANGUAGES,
    validate_disclosure_assets,
)


# ─── V1: happy path (real frozen assets) ─────────────────────────────────────


class TestValidateDisclosureAssetsHappyPath:
    def test_v1_real_frozen_assets_validate(self) -> None:
        """The frozen backend/audio/assets/ files validate without error."""
        # Sanity: this test ASSUMES the real frozen files exist.  If a
        # prior test fixture deleted them, this test would fail loudly
        # instead of silently passing.
        assert _DEFAULT_DISCLOSURE_ASSETS_DIR.is_dir(), (
            f"Default disclosure assets dir missing: "
            f"{_DEFAULT_DISCLOSURE_ASSETS_DIR}"
        )
        for lang in ("en", "hi"):
            path = _DEFAULT_DISCLOSURE_ASSETS_DIR / f"disclosure_{lang}.ulaw"
            assert path.is_file() and path.stat().st_size > 0, (
                f"Frozen disclosure asset missing or empty: {path}"
            )
        # Should not raise.
        validate_disclosure_assets()

    def test_v2_custom_assets_dir_with_both_files(self, tmp_path: Path) -> None:
        """Custom assets_dir is honoured — files at that path are checked."""
        (tmp_path / "disclosure_en.ulaw").write_bytes(b"\x00" * 160)
        (tmp_path / "disclosure_hi.ulaw").write_bytes(b"\x00" * 160)
        # Should not raise.
        validate_disclosure_assets(tmp_path)

    def test_v7_custom_languages_tuple(self, tmp_path: Path) -> None:
        """Only the languages listed in the ``languages`` tuple are required."""
        # Write a Spanish file — would NOT be required by default.
        (tmp_path / "disclosure_es.ulaw").write_bytes(b"\x00" * 160)
        # Without "es" in languages, the default ("en", "hi") check fails
        # because neither exists.
        with pytest.raises(FileNotFoundError):
            validate_disclosure_assets(tmp_path)

        # Now add English only and pass languages=("en",) — Hinglish NOT required.
        (tmp_path / "disclosure_en.ulaw").write_bytes(b"\x00" * 160)
        validate_disclosure_assets(tmp_path, languages=("en",))

        # Add Spanish and pass languages=("en", "es") — both required.
        validate_disclosure_assets(tmp_path, languages=("en", "es"))


# ─── V3, V4: missing file → FileNotFoundError ────────────────────────────────


class TestValidateDisclosureAssetsMissingFile:
    def test_v3_missing_hinglish_file_raises_filenotfound(self, tmp_path: Path) -> None:
        """English present, Hinglish missing → FileNotFoundError on 'hi'."""
        (tmp_path / "disclosure_en.ulaw").write_bytes(b"\x00" * 160)
        with pytest.raises(FileNotFoundError) as exc_info:
            validate_disclosure_assets(tmp_path)

        msg = str(exc_info.value)
        assert "hi" in msg or "disclosure_hi.ulaw" in msg
        assert "missing" in msg.lower()

    def test_v4_assets_dir_empty_raises_filenotfound(self, tmp_path: Path) -> None:
        """Empty assets dir → FileNotFoundError on the FIRST language."""
        with pytest.raises(FileNotFoundError) as exc_info:
            validate_disclosure_assets(tmp_path)
        # "en" is the first language in the default tuple.
        assert "disclosure_en.ulaw" in str(exc_info.value)

    def test_default_languages_is_en_then_hi(self) -> None:
        """Pin the default tuple order — tests above depend on it."""
        assert _DISCLOSURE_LANGUAGES == ("en", "hi")


# ─── V5, V6: empty file → RuntimeError ───────────────────────────────────────


class TestValidateDisclosureAssetsEmptyFile:
    def test_v5_zero_byte_file_raises_runtimeerror(self, tmp_path: Path) -> None:
        """A 0-byte asset is present but unusable — backend must refuse boot."""
        (tmp_path / "disclosure_en.ulaw").write_bytes(b"")  # 0 bytes
        (tmp_path / "disclosure_hi.ulaw").write_bytes(b"\x00" * 160)
        with pytest.raises(RuntimeError) as exc_info:
            validate_disclosure_assets(tmp_path)
        msg = str(exc_info.value).lower()
        assert "empty" in msg
        assert "disclosure_en.ulaw" in str(exc_info.value)

    def test_v6_mixed_present_and_empty_raises_runtimeerror(
        self, tmp_path: Path
    ) -> None:
        """One valid + one empty → fails on the empty one."""
        (tmp_path / "disclosure_en.ulaw").write_bytes(b"\x00" * 160)
        (tmp_path / "disclosure_hi.ulaw").write_bytes(b"")  # empty
        with pytest.raises(RuntimeError) as exc_info:
            validate_disclosure_assets(tmp_path)
        assert "disclosure_hi.ulaw" in str(exc_info.value)


# ─── Path semantics ─────────────────────────────────────────────────────────


class TestValidateDisclosureAssetsPathSemantics:
    def test_missing_file_check_uses_is_file_not_is_dir(
        self, tmp_path: Path
    ) -> None:
        """A directory named disclosure_en.ulaw (weird but possible)
        must NOT pass the existence check — only a regular file counts.
        """
        (tmp_path / "disclosure_en.ulaw").mkdir()  # a DIRECTORY
        (tmp_path / "disclosure_hi.ulaw").write_bytes(b"\x00" * 160)
        with pytest.raises(FileNotFoundError):
            validate_disclosure_assets(tmp_path)