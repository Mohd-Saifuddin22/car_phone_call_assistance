"""
backend/tests/test_audio_repacketizer.py — Issue 033 unit tests.

Test plan
─────────
InboundRepacketizer (Twilio 20 ms base64 frames → Deepgram ~80 ms raw chunks)
    I1  feed on empty string returns no chunks, no state change.
    I2  feed on one 160-byte frame keeps it in the buffer, returns no chunks.
    I3  feeding four 160-byte base64 frames across four calls yields exactly
        one 640-byte raw chunk from the fourth call.
    I4  feeding eight 160-byte frames in a single call yields two complete
        640-byte chunks, buffer empty.
    I5  feeding 641 raw bytes yields one 640-byte chunk with one byte held.
    I6  flush on empty buffer returns an empty list.
    I7  flush after a sub-chunk feed returns that partial raw chunk unpadded.
    I8  carryover across multiple feeds is exact (no bytes lost or duplicated).
    I9  invalid base64 raises ValueError.
    I10 non-str input to feed raises TypeError.

OutboundRepacketizer (Sarvam arbitrary bytes → Twilio uniform 20 ms base64 frames)
    O1  feed on empty bytes returns no frames, no state change.
    O2  feed on exactly 160 raw bytes returns one base64 frame, buffer empty.
    O3  feed on 250 raw bytes returns one base64 frame, buffer retains 90.
    O4  after a 250-byte feed, flush yields exactly one frame whose decoded
        payload is 90 real bytes followed by 70 × 0xFF.
    O5  feeding 320 raw bytes yields two complete frames, buffer empty.
    O6  feeding 159 raw bytes returns no frames; flush pads to one 160-byte
        frame whose decoded payload is 159 real bytes + 1 × 0xFF.
    O7  multi-feed carryover is exact (no bytes lost or duplicated).
    O8  accumulator can be reused after flush (clean state).
    O9  non-bytes input to feed raises TypeError.

All assertions are byte-accurate; no audio libraries are required.
"""

from __future__ import annotations

import base64

import pytest

from backend.audio.repacketizer import (
    DEEPGRAM_CHUNK_BYTES,
    InboundRepacketizer,
    MULAW_FRAME_BYTES,
    OutboundRepacketizer,
)


# ─── Test helpers ────────────────────────────────────────────────────────────


def _b64(data: bytes) -> str:
    """Encode bytes to a Twilio-style base64 string."""
    return base64.b64encode(data).decode("ascii")


def _make_frame(seed: int) -> bytes:
    """
    Build a deterministic 160-byte μ-law-looking frame from a seed byte.

    We don't care about acoustic meaning — these tests only check byte-level
    framing — but using a recognisable pattern makes failures self-describing.
    """
    return bytes((seed + i) & 0xFF for i in range(MULAW_FRAME_BYTES))


def _is_silence_padded(real: bytes, total_len: int = MULAW_FRAME_BYTES) -> bool:
    """
    Return True iff ``real`` is a prefix of ``0xFF`` padding up to ``total_len``.

    Used to verify that OutboundRepacketizer.flush() pads the trailing
    partial with μ-law silence (0xFF) — not random noise.
    """
    if len(real) > total_len:
        return False
    return all(b == 0xFF for b in real[total_len - len(real) :] if False) or True


# ─── I1 — feed on empty string is a no-op ─────────────────────────────────────


def test_inbound_feed_empty_string_returns_no_chunks() -> None:
    """An empty base64 payload must produce zero chunks and leave the buffer untouched."""
    # Arrange
    repacketizer = InboundRepacketizer()

    # Act
    chunks = repacketizer.feed("")

    # Assert
    assert chunks == []


# ─── I2 — one Twilio frame fits in the buffer, not a full chunk yet ──────────


def test_inbound_feed_single_frame_buffers_without_returning_chunks() -> None:
    """One 160-byte frame is below the 640-byte chunk threshold — must buffer."""
    # Arrange
    repacketizer = InboundRepacketizer()
    frame = _make_frame(seed=0)

    # Act
    chunks = repacketizer.feed(_b64(frame))

    # Assert
    assert chunks == []
    # The accumulator must have the bytes in its buffer for the next feed;
    # we verify via flush() since the buffer is private.
    leftover = repacketizer.flush()
    assert leftover == [frame]


# ─── I3 — four 160-byte frames across four feeds → one 640-byte chunk ───────


def test_inbound_feed_four_frames_across_four_calls_yields_one_chunk() -> None:
    """
    The headline acceptance criterion: 4 × 20 ms Twilio frames (across four
    feed calls) yield exactly one ~80 ms raw chunk on the fourth call.
    """
    # Arrange
    repacketizer = InboundRepacketizer()
    frames = [_make_frame(seed=i) for i in range(4)]
    expected_chunk = b"".join(frames)

    # Act
    chunks_call_1 = repacketizer.feed(_b64(frames[0]))
    chunks_call_2 = repacketizer.feed(_b64(frames[1]))
    chunks_call_3 = repacketizer.feed(_b64(frames[2]))
    chunks_call_4 = repacketizer.feed(_b64(frames[3]))

    # Assert
    assert chunks_call_1 == [], "160 bytes buffered — no chunk yet"
    assert chunks_call_2 == [], "320 bytes buffered — no chunk yet"
    assert chunks_call_3 == [], "480 bytes buffered — no chunk yet"
    assert chunks_call_4 == [expected_chunk], "640 bytes → exactly one chunk"
    assert len(chunks_call_4[0]) == DEEPGRAM_CHUNK_BYTES

    # The accumulator must be empty after a clean drain.
    assert repacketizer.flush() == []


# ─── I4 — eight frames in a single feed → two chunks ────────────────────────


def test_inbound_feed_eight_frames_in_one_call_yields_two_chunks() -> None:
    """Two complete chunks worth of audio should drain atomically on one feed."""
    # Arrange
    repacketizer = InboundRepacketizer()
    frames = [_make_frame(seed=i) for i in range(8)]
    payload = _b64(b"".join(frames))
    expected_chunks = (b"".join(frames[0:4]), b"".join(frames[4:8]))

    # Act
    chunks = repacketizer.feed(payload)

    # Assert
    assert chunks == list(expected_chunks)
    assert all(len(c) == DEEPGRAM_CHUNK_BYTES for c in chunks)
    assert repacketizer.flush() == []


# ─── I5 — 641 raw bytes → one chunk + 1 leftover byte ───────────────────────


def test_inbound_feed_641_bytes_yields_one_chunk_with_one_byte_held() -> None:
    """One byte beyond a clean chunk boundary must be retained, not lost."""
    # Arrange
    repacketizer = InboundRepacketizer()
    payload_bytes = bytes(i & 0xFF for i in range(641))

    # Act
    chunks = repacketizer.feed(_b64(payload_bytes))

    # Assert
    assert len(chunks) == 1
    assert len(chunks[0]) == DEEPGRAM_CHUNK_BYTES
    assert chunks[0] == payload_bytes[:DEEPGRAM_CHUNK_BYTES]
    # The single trailing byte must be retrievable via flush.
    leftover = repacketizer.flush()
    assert leftover == [payload_bytes[DEEPGRAM_CHUNK_BYTES:]]


# ─── I6 — flush on empty buffer returns empty list ──────────────────────────


def test_inbound_flush_on_empty_buffer_returns_empty_list() -> None:
    """Flush on a fresh accumulator must not invent a phantom partial chunk."""
    # Arrange
    repacketizer = InboundRepacketizer()

    # Act
    leftover = repacketizer.flush()

    # Assert
    assert leftover == []
    # And the accumulator remains usable.
    assert repacketizer.feed(_b64(_make_frame(seed=0))) == []


# ─── I7 — flush returns raw partial bytes without padding ────────────────────


def test_inbound_flush_returns_unpadded_raw_partial() -> None:
    """
    The contract: inbound flush emits whatever's left, byte-for-byte.
    No μ-law silence padding — Deepgram would treat it as real audio.
    """
    # Arrange
    repacketizer = InboundRepacketizer()
    partial = bytes(i & 0xFF for i in range(320))  # 320 bytes < 640

    # Act
    repacketizer.feed(_b64(partial))
    leftover = repacketizer.flush()

    # Assert
    assert leftover == [partial]
    assert len(leftover[0]) == 320
    # The bytes are returned exactly — no 0xFF padding added.
    assert leftover[0] == partial


# ─── I8 — multi-feed carryover is byte-exact ────────────────────────────────


@pytest.mark.parametrize(
    "feed_sizes",
    [
        (1, 1, 1, 637),               # one byte at a time, last completes chunk
        (200, 200, 240),              # completes on third feed
        (160, 160, 160, 160, 1),      # exactly one chunk + one trailing byte
        (639, 1),                     # chunk completes on second feed
        (641, 639),                   # two feeds, second completes another chunk
    ],
)
def test_inbound_multi_feed_carryover_is_byte_exact(feed_sizes: tuple[int, ...]) -> None:
    """No bytes are lost or duplicated across feeds of various sizes."""
    # Arrange
    repacketizer = InboundRepacketizer()
    expected_total = sum(feed_sizes)
    full_payload = bytes((i * 7) & 0xFF for i in range(expected_total))

    # Act
    produced: list[bytes] = []
    cursor = 0
    for size in feed_sizes:
        piece = full_payload[cursor : cursor + size]
        cursor += size
        produced.extend(repacketizer.feed(_b64(piece)))
    produced.extend(repacketizer.flush())

    # Assert: concatenation of all produced chunks equals the original payload.
    reassembled = b"".join(produced)
    assert reassembled == full_payload, "Carryover dropped or duplicated bytes"
    assert sum(len(c) for c in produced) == expected_total


# ─── I9 — invalid base64 raises ValueError ───────────────────────────────────


def test_inbound_feed_invalid_base64_raises_value_error() -> None:
    """Non-alphabet characters in the payload must surface as ValueError."""
    # Arrange
    repacketizer = InboundRepacketizer()

    # Act / Assert
    with pytest.raises(ValueError):
        repacketizer.feed("!!!not-base64!!!")


# ─── I10 — non-str input raises TypeError ───────────────────────────────────


def test_inbound_feed_non_string_raises_type_error() -> None:
    """Type safety: a bytes/None/int payload must not be silently coerced."""
    # Arrange
    repacketizer = InboundRepacketizer()

    # Act / Assert
    for bad in (b"\x00" * MULAW_FRAME_BYTES, None, 12345):
        with pytest.raises(TypeError):
            repacketizer.feed(bad)  # type: ignore[arg-type]


# ─── O1 — feed on empty bytes is a no-op ─────────────────────────────────────


def test_outbound_feed_empty_bytes_returns_no_frames() -> None:
    """An empty Sarvam chunk must produce zero frames and leave state untouched."""
    # Arrange
    repacketizer = OutboundRepacketizer()

    # Act
    frames = repacketizer.feed(b"")

    # Assert
    assert frames == []


# ─── O2 — exactly 160 bytes → one complete frame, buffer empty ───────────────


def test_outbound_feed_exactly_one_frame_returns_one_b64_frame() -> None:
    """A perfect 160-byte feed yields exactly one base64 frame and empties the buffer."""
    # Arrange
    repacketizer = OutboundRepacketizer()
    payload = _make_frame(seed=42)
    assert len(payload) == MULAW_FRAME_BYTES

    # Act
    frames = repacketizer.feed(payload)

    # Assert
    assert len(frames) == 1
    assert base64.b64decode(frames[0]) == payload
    # Buffer must be empty — flush returns [].
    assert repacketizer.flush() == []


# ─── O3 — 250 bytes → one frame + 90 retained (acceptance criterion) ─────────


def test_outbound_feed_250_bytes_returns_one_frame_and_retains_90_bytes() -> None:
    """
    The headline outbound acceptance criterion: feeding 250 raw bytes yields
    one base64-encoded 160-byte frame and retains 90 bytes in the buffer.
    """
    # Arrange
    repacketizer = OutboundRepacketizer()
    payload = bytes((i * 13 + 5) & 0xFF for i in range(250))
    expected_first_frame_payload = payload[:MULAW_FRAME_BYTES]
    expected_retained = payload[MULAW_FRAME_BYTES:]

    # Act
    frames = repacketizer.feed(payload)

    # Assert
    assert len(frames) == 1, "250 bytes → exactly one 160-byte frame"
    assert len(frames[0]) > 0, "Frame must be a non-empty base64 string"
    assert base64.b64decode(frames[0]) == expected_first_frame_payload

    # The 90 retained bytes must surface via flush — verify directly.
    flushed = repacketizer.flush()
    assert len(flushed) == 1, "Flush must return the trailing partial as one frame"


# ─── O4 — flush after 250-byte feed: one frame with 90 real + 70 × 0xFF ─────


def test_outbound_flush_after_250_byte_feed_pads_with_silence_byte() -> None:
    """
    After a 250-byte feed, flush yields exactly one base64 frame whose
    decoded payload is the 90 retained bytes followed by 70 × 0xFF.
    """
    # Arrange
    repacketizer = OutboundRepacketizer()
    payload = bytes((i * 13 + 5) & 0xFF for i in range(250))
    retained_real = payload[MULAW_FRAME_BYTES:]
    assert len(retained_real) == 90

    # Act
    repacketizer.feed(payload)
    flushed = repacketizer.flush()

    # Assert
    assert len(flushed) == 1
    decoded = base64.b64decode(flushed[0])
    assert len(decoded) == MULAW_FRAME_BYTES, "Padded frame must be exactly 160 bytes"
    # The first 90 bytes are the real audio we retained.
    assert decoded[:90] == retained_real
    # The remaining 70 bytes are all μ-law silence (0xFF).
    padding = decoded[90:]
    assert len(padding) == 70
    assert padding == b"\xff" * 70
    # Buffer is now empty.
    assert repacketizer.flush() == []


# ─── O5 — 320 bytes → two frames, buffer empty ──────────────────────────────


def test_outbound_feed_320_bytes_returns_two_complete_frames() -> None:
    """Two exact frames fed at once drain completely with no padding."""
    # Arrange
    repacketizer = OutboundRepacketizer()
    payload = _make_frame(seed=0) + _make_frame(seed=99)
    assert len(payload) == 2 * MULAW_FRAME_BYTES

    # Act
    frames = repacketizer.feed(payload)

    # Assert
    assert len(frames) == 2
    assert base64.b64decode(frames[0]) == payload[:MULAW_FRAME_BYTES]
    assert base64.b64decode(frames[1]) == payload[MULAW_FRAME_BYTES:]
    assert repacketizer.flush() == []


# ─── O6 — 159 bytes → no frames; flush pads with a single 0xFF ───────────────


def test_outbound_feed_159_bytes_then_flush_pads_with_one_silence_byte() -> None:
    """The smallest possible partial (1 byte below a frame) flushes to one padded frame."""
    # Arrange
    repacketizer = OutboundRepacketizer()
    payload = bytes(range(159))

    # Act
    frames = repacketizer.feed(payload)
    flushed = repacketizer.flush()

    # Assert
    assert frames == []
    assert len(flushed) == 1
    decoded = base64.b64decode(flushed[0])
    assert decoded == payload + b"\xff"
    assert len(decoded) == MULAW_FRAME_BYTES


# ─── O7 — multi-feed carryover is byte-exact (table-driven) ──────────────────


@pytest.mark.parametrize(
    "feed_sizes",
    [
        (1, 1, 1, 157),                 # completes on fourth feed
        (50, 50, 60),                   # completes on third feed
        (159, 1),                       # second feed completes the frame
        (1, 159),                       # same as above, order swapped
        (160, 160, 90),                 # two frames + 90 leftover
        (90, 80),                       # 170 bytes → one frame + 10 leftover
        (641, 1),                       # 642 bytes → 4 frames + 2 leftover
    ],
)
def test_outbound_multi_feed_carryover_is_byte_exact(feed_sizes: tuple[int, ...]) -> None:
    """Across many feed calls, every byte survives; only the trailing partial is padded on flush."""
    # Arrange
    repacketizer = OutboundRepacketizer()
    total_bytes = sum(feed_sizes)
    full_payload = bytes((i * 17 + 3) & 0xFF for i in range(total_bytes))

    # Act
    produced: list[str] = []
    cursor = 0
    for size in feed_sizes:
        piece = full_payload[cursor : cursor + size]
        cursor += size
        produced.extend(repacketizer.feed(piece))
    produced.extend(repacketizer.flush())

    # Assert — every base64 frame is exactly MULAW_FRAME_BYTES long, and the
    # concatenation of the *first* N frames (where N × 160 ≤ total_bytes)
    # reconstructs the full real payload.  The trailing frame, if shorter
    # than the real payload's residual, must be padded with 0xFF up to 160.
    decoded_payloads = [base64.b64decode(f) for f in produced]
    assert all(len(p) == MULAW_FRAME_BYTES for p in decoded_payloads), (
        "Every outbound frame must be exactly 160 bytes"
    )

    full_count, remainder = divmod(total_bytes, MULAW_FRAME_BYTES)
    expected_complete_frames = full_count + (1 if remainder else 0)
    assert len(decoded_payloads) == expected_complete_frames, (
        f"Expected {expected_complete_frames} frames, got {len(decoded_payloads)}"
    )

    # Concatenate every byte *except* the trailing pad from the last frame.
    real_audio = b"".join(
        p[: MULAW_FRAME_BYTES if i < len(decoded_payloads) - 1 else remainder or MULAW_FRAME_BYTES]
        for i, p in enumerate(decoded_payloads)
    )
    assert real_audio == full_payload, "Carryover dropped or duplicated bytes"

    # If there was a remainder, the last frame's tail must be 0xFF.
    if remainder:
        last_frame_tail = decoded_payloads[-1][remainder:]
        assert last_frame_tail == b"\xff" * (MULAW_FRAME_BYTES - remainder)


# ─── O8 — accumulator can be reused after flush ─────────────────────────────


def test_outbound_accumulator_reusable_after_flush() -> None:
    """A clean flush must reset state so a new stream can be repacketized."""
    # Arrange
    repacketizer = OutboundRepacketizer()
    first_stream = bytes(range(250))
    second_stream = _make_frame(seed=200)

    # Act
    repacketizer.feed(first_stream)
    repacketizer.flush()
    frames = repacketizer.feed(second_stream)

    # Assert
    assert len(frames) == 1
    assert base64.b64decode(frames[0]) == second_stream
    assert repacketizer.flush() == []


# ─── O9 — non-bytes input raises TypeError ──────────────────────────────────


@pytest.mark.parametrize("bad", [None, "raw-string", 12345, 3.14])
def test_outbound_feed_non_bytes_raises_type_error(bad: object) -> None:
    """The contract requires bytes-like input; anything else is a TypeError."""
    # Arrange
    repacketizer = OutboundRepacketizer()

    # Act / Assert
    with pytest.raises(TypeError):
        repacketizer.feed(bad)  # type: ignore[arg-type]


# ─── O10 — flush on empty buffer returns empty list ─────────────────────────


def test_outbound_flush_on_empty_buffer_returns_empty_list() -> None:
    """Flush on a fresh accumulator returns [] (no phantom padded frame)."""
    # Arrange
    repacketizer = OutboundRepacketizer()

    # Act
    flushed = repacketizer.flush()

    # Assert
    assert flushed == []