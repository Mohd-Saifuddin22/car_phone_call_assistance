"""
backend/tests/test_generate_disclosures.py — Issue 037 disclosure-audio tests.

Test plan
─────────
strip_wav_header (pure byte-level transform — no I/O, no async):

    H1  Pure μ-law input (no RIFF magic) passes through unchanged.
    H2  Standard 44-byte RIFF/WAVE with μ-law (AudioFormat=7) returns the
        raw payload only — header bytes removed, payload byte-identical.
    H3  RIFF/WAVE with an extra LIST sub-chunk before ``data`` still
        returns the raw payload (proves the chunk-walk handles offset
        shifts some encoders introduce).
    H4  RIFF/WAVE with PCM format (AudioFormat=1) raises ValueError —
        refuses to silently mislabel non-μ-law bytes as .ulaw.
    H5  RIFF/WAVE with no ``data`` sub-chunk raises ValueError — refuses
        to truncate blindly.
    H6  RIFF/WAVE with a malformed ``fmt `` payload (< 2 bytes) raises.

ulaw_duration_seconds:

    D1  16 000 bytes @ 8 kHz → 2.0 s (the standard 2-second disclosure).
    D2  Zero bytes → 0.0 s.
    D3  Custom sample rate (16 kHz) halves the duration for the same bytes.
    D4  Non-positive sample rate raises ValueError.

All assertions are byte-accurate; no audio libraries, no network, no
filesystem. The stripper is a deterministic pure function.
"""

from __future__ import annotations

import struct

import pytest

from backend.scripts.generate_disclosures import (
    _WAVE_FORMAT_MULAW,
    strip_wav_header,
    ulaw_duration_seconds,
)


# ─── Test helpers ────────────────────────────────────────────────────────────


def _make_wav(audio_format: int, payload: bytes, *, extra_pre_data: bytes = b"") -> bytes:
    """
    Build a minimal RIFF/WAVE byte string with the given AudioFormat and payload.

    Parameters
    ----------
    audio_format
        Value for the fmt chunk's AudioFormat field (1 = PCM, 7 = μ-law).
    payload
        Raw audio bytes that go inside the ``data`` sub-chunk.
    extra_pre_data
        Optional extra sub-chunks inserted between ``WAVE`` and ``data`` —
        used to simulate encoders that emit a ``LIST``/``INFO`` chunk
        before the audio data.

    Returns
    -------
    bytes
        A well-formed RIFF/WAVE buffer. The fmt chunk carries mono 8 kHz
        8-bit metadata matching the values Sarvam uses for μ-law output.
    """
    # fmt chunk payload (16 bytes): AudioFormat(2) | Channels(2) | Rate(4)
    #                                | ByteRate(4) | BlockAlign(2) | BitsPerSample(2)
    fmt_payload = struct.pack(
        "<HHIIHH",
        audio_format,
        1,            # channels (mono)
        8000,         # sample rate
        8000,         # byte rate (8 kB/s for μ-law)
        1,            # block align
        8,            # bits per sample
    )
    fmt_chunk = b"fmt " + struct.pack("<I", len(fmt_payload)) + fmt_payload

    data_chunk = b"data" + struct.pack("<I", len(payload)) + payload

    # RIFF size = 4 (WAVE) + everything inside
    riff_payload_size = 4 + len(fmt_chunk) + len(extra_pre_data) + len(data_chunk)
    return b"RIFF" + struct.pack("<I", riff_payload_size) + b"WAVE" + fmt_chunk + extra_pre_data + data_chunk


# ─── strip_wav_header: happy paths ───────────────────────────────────────────


def test_strip_wav_header_returns_pure_ulaw_unchanged_when_no_riff_magic() -> None:
    """H1: a buffer with no RIFF/WAVE header is treated as already-raw μ-law."""
    raw_ulaw = b"\x00\xff\x80\x7f" * 200  # 800 bytes of fake μ-law
    assert strip_wav_header(raw_ulaw) == raw_ulaw
    # Also covers empty / short inputs — the size guard returns early.
    assert strip_wav_header(b"") == b""
    assert strip_wav_header(b"RIFF") == b"RIFF"  # too short for WAVE check


def test_strip_wav_header_strips_standard_44_byte_wav_with_mulaw() -> None:
    """H2: standard μ-law RIFF/WAVE → only the data payload, byte-identical."""
    payload = b"\x00\xff" * 400  # 800 bytes of μ-law
    wav = _make_wav(_WAVE_FORMAT_MULAW, payload)

    # Sanity: input is the expected 44-byte header + 800-byte payload.
    assert len(wav) == 44 + len(payload)

    stripped = strip_wav_header(wav)
    assert stripped == payload
    assert len(stripped) == 800


def test_strip_wav_header_handles_extra_list_chunk_before_data() -> None:
    """H3: an extra LIST sub-chunk between WAVE and data must not shift the data read."""
    payload = b"\xab\xcd" * 300
    # 12-byte LIST chunk with arbitrary pad bytes.
    extra_list = b"LIST" + struct.pack("<I", 4) + b"INFO"

    wav = _make_wav(_WAVE_FORMAT_MULAW, payload, extra_pre_data=extra_list)
    # Without chunk-walking, a fixed-44-byte offset would land mid-LIST
    # and read garbage. This test guards against that regression.
    assert len(wav) == 56 + len(payload)  # 44 + 12 LIST

    stripped = strip_wav_header(wav)
    assert stripped == payload
    assert len(stripped) == len(payload)


# ─── strip_wav_header: format guard ──────────────────────────────────────────


def test_strip_wav_header_raises_on_pcm_format() -> None:
    """H4: AudioFormat=1 (PCM) must be refused — bytes are not μ-law."""
    payload = b"\x00\x00" * 100  # would look like 16-bit PCM silence
    wav = _make_wav(audio_format=1, payload=payload)

    with pytest.raises(ValueError) as exc_info:
        strip_wav_header(wav)

    # The error message must be actionable — it tells the operator exactly
    # what Sarvam sent and what was expected.
    msg = str(exc_info.value)
    assert "AudioFormat=1" in msg
    assert "WAVE_FORMAT_MULAW" in msg
    assert "7" in msg  # expected code


# ─── strip_wav_header: malformed input ───────────────────────────────────────


def test_strip_wav_header_raises_when_data_chunk_missing() -> None:
    """H5: a RIFF/WAVE with no 'data' sub-chunk raises instead of truncating."""
    # Build a RIFF with only a fmt chunk — no data.
    fmt_payload = struct.pack(
        "<HHIIHH", _WAVE_FORMAT_MULAW, 1, 8000, 8000, 1, 8
    )
    fmt_chunk = b"fmt " + struct.pack("<I", len(fmt_payload)) + fmt_payload
    riff_payload_size = 4 + len(fmt_chunk)
    wav = b"RIFF" + struct.pack("<I", riff_payload_size) + b"WAVE" + fmt_chunk

    with pytest.raises(ValueError) as exc_info:
        strip_wav_header(wav)

    assert "data" in str(exc_info.value).lower()


def test_strip_wav_header_raises_when_fmt_chunk_too_small() -> None:
    """H6: a fmt chunk with < 2 bytes of payload cannot carry AudioFormat."""
    # fmt chunk with only 1 byte of payload (malformed — should be 16).
    fmt_chunk = b"fmt " + struct.pack("<I", 1) + b"\x07"  # AudioFormat=7 but truncated
    data_chunk = b"data" + struct.pack("<I", 0)  # empty audio payload
    riff_payload_size = 4 + len(fmt_chunk) + len(data_chunk)
    wav = b"RIFF" + struct.pack("<I", riff_payload_size) + b"WAVE" + fmt_chunk + data_chunk

    with pytest.raises(ValueError) as exc_info:
        strip_wav_header(wav)

    assert "fmt" in str(exc_info.value).lower()


# ─── ulaw_duration_seconds ───────────────────────────────────────────────────


def test_ulaw_duration_seconds_at_8khz_for_two_seconds_of_audio() -> None:
    """D1: 16 000 μ-law bytes @ 8 kHz = exactly 2.0 seconds."""
    assert ulaw_duration_seconds(16_000) == 2.0


def test_ulaw_duration_seconds_zero_bytes_returns_zero() -> None:
    """D2: an empty μ-law buffer has zero playback duration."""
    assert ulaw_duration_seconds(0) == 0.0


def test_ulaw_duration_seconds_respects_custom_sample_rate() -> None:
    """D3: at 16 kHz the same byte count plays back in half the time."""
    assert ulaw_duration_seconds(16_000, sample_rate=16_000) == 1.0
    assert ulaw_duration_seconds(8000, sample_rate=16_000) == 0.5


def test_ulaw_duration_seconds_rejects_non_positive_sample_rate() -> None:
    """D4: a zero or negative sample rate is a programming error, not a runtime condition."""
    with pytest.raises(ValueError):
        ulaw_duration_seconds(1000, sample_rate=0)
    with pytest.raises(ValueError):
        ulaw_duration_seconds(1000, sample_rate=-1)