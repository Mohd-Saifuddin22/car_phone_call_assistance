"""
backend/audio/repacketizer.py — Pure audio framing transforms (Issue 033).

The voice pipeline carries μ-law 8 kHz mono end-to-end, so no resampling or
transcoding is required — only deterministic byte-level aggregation and
re-slicing. Two directions need different framing:

  Inbound  — Twilio Media Streams ships 20 ms (160-byte) base64 frames.
             Deepgram Flux strongly prefers ~80 ms (640-byte) raw chunks.
             We aggregate four Twilio frames into one Deepgram chunk.

  Outbound — Sarvam's streaming WebSocket returns arbitrary-sized raw byte
             chunks. Twilio expects uniform 20 ms (160-byte) base64 frames.
             We re-slice Sarvam's output and pad the trailing partial frame
             with the μ-law silence byte so Twilio's playback never starves.

This module is intentionally I/O-free: it does not import any audio library
(stdlib ``base64`` only) and performs no I/O. Callers own the WebSocket
infrastructure; this module is a deterministic state machine that can be
unit-tested without any async plumbing or network stubs.

Critical correctness rules
──────────────────────────
* Inbound **never** pads. Inserting fake silence into a customer turn would
  poison Deepgram's acoustic model and corrupt turn detection.
* Outbound **always** emits uniform 160-byte frames, even for the final
  partial — Twilio's media-stream scheduler assumes every payload is exactly
  one 20 ms frame, and dropping the trailing audio would clip Riya's last
  syllable.
"""

from __future__ import annotations

import base64
from typing import Final

__all__ = [
    "InboundRepacketizer",
    "OutboundRepacketizer",
    "MULAW_FRAME_BYTES",
    "DEEPGRAM_CHUNK_BYTES",
]

# ─── Codec constants ──────────────────────────────────────────────────────────

# μ-law (PCMU) at 8 kHz, mono → 1 byte per sample → 8 000 B/s.
# 20 ms = 160 bytes (the size of one Twilio Media Streams frame).
MULAW_FRAME_BYTES: Final[int] = 160
"""
Size of exactly one 20 ms μ-law 8 kHz mono frame, in bytes.
This is the unit size Twilio sends and expects to receive.
"""

# Duration of one μ-law 8 kHz frame in milliseconds.
# Derived from MULAW_FRAME_BYTES / 8 000 B/s × 1 000 ms/s = 20 ms.
# Exposed so the outbound sender (Issue 036) can compute mark offsets
# without re-deriving the math in two places.
MULAW_FRAME_MS: Final[int] = 20
"""
Duration of one μ-law 8 kHz frame in milliseconds.
"""

# Deepgram Flux's strong recommendation: ~80 ms chunks for stable turn
# detection on continuous streams. 80 ms × 8 000 B/s = 640 bytes, which is
# exactly four Twilio frames — convenient, deterministic, byte-accurate.
DEEPGRAM_CHUNK_BYTES: Final[int] = MULAW_FRAME_BYTES * 4
"""
Size of one ~80 ms chunk fed to Deepgram Flux, in bytes.
Equals four 20 ms Twilio frames (160 × 4 = 640).
"""

# Standard μ-law "silence" byte (per RFC 3551 / G.711 §4): the value that
# decodes to the zero-amplitude segment. Padding the outbound trailing
# partial with this byte is inaudible silence — the customer cannot
# distinguish "Riya just stopped" from "Riya is mid-silence-frame".
_MULAW_SILENCE_BYTE: Final[bytes] = b"\xff"
"""
One byte of μ-law silence — used to pad outbound trailing partials so
every Twilio frame is exactly 20 ms (160 bytes).
"""


# ─── Inbound: Twilio 20 ms base64 frames → Deepgram ~80 ms raw chunks ────────


class InboundRepacketizer:
    """
    Stateful accumulator that re-packets Twilio Media Streams audio for Deepgram Flux.

    Twilio delivers audio as **base64-encoded μ-law 8 kHz mono frames of exactly
    160 bytes** (one 20 ms frame each). Deepgram Flux strongly prefers feeding
    on **640-byte (~80 ms) raw byte chunks** — aggregating four Twilio frames
    at a time. This class does only that aggregation; it does not interpret or
    transform the audio bytes themselves.

    Contract
    ────────
    * ``feed(payload_b64)`` accepts one Twilio payload string. The string is
      base64-decoded and appended to an internal buffer; as many **complete**
      640-byte chunks as the buffer now holds are returned (in arrival order).
      Anything less than 640 bytes is silently retained for the next call.
    * ``flush()`` returns whatever remains in the buffer as a single partial
      chunk — **without padding**. Deepgram must hear only real audio.

    State
    ─────
    The accumulator holds at most 639 bytes between ``feed`` calls. It is
    pure with respect to its inputs — no I/O, no logging, no globals.
    """

    __slots__ = ("_buffer",)

    def __init__(self) -> None:
        self._buffer: bytearray = bytearray()

    def feed(self, payload_b64: str) -> list[bytes]:
        """
        Append one base64-encoded Twilio frame and return any complete ~80 ms chunks.

        Parameters
        ──────────
        payload_b64
            The ``payload`` field of a Twilio Media Streams ``media`` event —
            a base64-encoded μ-law 8 kHz mono frame of exactly 160 bytes
            (modulo Twilio's own edge cases, which surface as ValueError).

        Returns
        ───────
        list[bytes]
            Each entry is exactly 640 raw bytes (one Deepgram-ready chunk).
            The list is empty if the buffer has not yet accumulated 640 bytes.

        Raises
        ──────
        TypeError
            ``payload_b64`` is not a ``str``.
        ValueError
            ``payload_b64`` is not valid base64, or the decode path itself
            fails (propagated from ``base64.b64decode`` with ``validate=True``).
        """
        if not isinstance(payload_b64, str):
            raise TypeError(
                f"feed() expects a base64 string, got {type(payload_b64).__name__}"
            )

        # validate=True: reject non-alphabet chars loudly instead of silently
        # skipping them — Twilio's payloads are well-formed, anything else
        # is a programming error worth surfacing.
        raw: bytes = base64.b64decode(payload_b64, validate=True)
        self._buffer.extend(raw)
        return self._drain_complete_chunks()

    def flush(self) -> list[bytes]:
        """
        Return any trailing partial bytes as a single raw chunk.

        Crucially, **the trailing chunk is not padded.** Deepgram performs
        acoustic and turn-boundary analysis on the audio we send; injecting
        silent samples would distort its VAD and EOT decisions. If the
        buffer is empty, the returned list is empty.

        After this call the internal buffer is empty; the accumulator is
        safe to reuse for a fresh stream.
        """
        if not self._buffer:
            return []

        chunk: bytes = bytes(self._buffer)
        self._buffer.clear()
        return [chunk]

    # ─── Internals ──────────────────────────────────────────────────────────

    def _drain_complete_chunks(self) -> list[bytes]:
        """Pop and return as many complete 640-byte chunks as the buffer allows."""
        chunks: list[bytes] = []
        while len(self._buffer) >= DEEPGRAM_CHUNK_BYTES:
            chunks.append(bytes(self._buffer[:DEEPGRAM_CHUNK_BYTES]))
            del self._buffer[:DEEPGRAM_CHUNK_BYTES]
        return chunks


# ─── Outbound: Sarvam arbitrary bytes → Twilio uniform 20 ms base64 frames ───


class OutboundRepacketizer:
    """
    Stateful accumulator that re-packets Sarvam's TTS audio for Twilio Media Streams.

    Sarvam's streaming WebSocket returns **arbitrary-sized raw byte chunks**
    (its internal scheduler decides the cadence). Twilio's Media Streams,
    however, require a uniform payload cadence: every ``media`` message must
    carry exactly one 20 ms μ-law frame (160 bytes) base64-encoded. Anything
    shorter under-runs Twilio's playback buffer; anything longer is rejected.

    This class re-slices Sarvam's bytes into exact 160-byte frames and pads
    the **final** trailing partial with μ-law silence so the customer hears
    Riya's last word intact — the silence pad is shorter than the audio
    decoder's smallest perceptual step and is inaudible in practice.

    Contract
    ────────
    * ``feed(raw_bytes)`` appends Sarvam's bytes to the internal buffer and
      returns as many **complete** 160-byte base64 frames as it now holds.
    * ``flush()`` pads any trailing partial (1–159 bytes) up to exactly
      160 bytes with the μ-law silence byte (``0xFF``), base64-encodes it,
      and returns it as a single frame. If the buffer is already empty,
      returns ``[]``.

    Why the silence pad is the right call
    ─────────────────────────────────────
    Dropping the partial frame would clip Riya's last syllable, which is
    audible and user-hostile. Emitting a *smaller* frame violates Twilio's
    20 ms contract and starves playback. Padding with μ-law silence keeps
    every frame at 160 bytes, preserves the trailing audio, and adds at most
    19 ms of digital silence — short enough to be imperceptible.
    """

    __slots__ = ("_buffer",)

    def __init__(self) -> None:
        self._buffer: bytearray = bytearray()

    def feed(self, raw_bytes: bytes) -> list[str]:
        """
        Append a Sarvam chunk and return any complete 20 ms frames.

        Parameters
        ──────────
        raw_bytes
            Raw μ-law 8 kHz mono bytes returned by Sarvam's streaming WebSocket.
            Length is unconstrained; Sarvam emits chunks of varying sizes.

        Returns
        ───────
        list[str]
            Each entry is the base64 encoding of exactly 160 raw bytes
            (one Twilio-ready ``media`` payload). Empty if the buffer has
            fewer than 160 bytes after the append.

        Raises
        ──────
        TypeError
            ``raw_bytes`` is not a ``bytes``-like object.
        """
        if not isinstance(raw_bytes, (bytes, bytearray)):
            raise TypeError(
                f"feed() expects bytes, got {type(raw_bytes).__name__}"
            )

        self._buffer.extend(raw_bytes)
        return self._drain_complete_frames()

    def flush(self) -> list[str]:
        """
        Pad any trailing partial to one full 20 ms frame and return it.

        Returns
        ───────
        list[str]
            * ``[]`` when the buffer is empty (nothing to flush).
            * A single-element list containing the base64 encoding of
              ``leftover + (160 − len(leftover)) × 0xFF`` when 1–159 bytes
              remain.

        After this call the internal buffer is empty.
        """
        if not self._buffer:
            return []

        leftover: int = len(self._buffer)
        padding_needed: int = MULAW_FRAME_BYTES - leftover
        padded: bytes = bytes(self._buffer) + (_MULAW_SILENCE_BYTE * padding_needed)
        self._buffer.clear()
        return [base64.b64encode(padded).decode("ascii")]

    # ─── Internals ──────────────────────────────────────────────────────────

    def _drain_complete_frames(self) -> list[str]:
        """Pop and base64-encode as many complete 160-byte frames as fit."""
        frames: list[str] = []
        while len(self._buffer) >= MULAW_FRAME_BYTES:
            chunk: bytes = bytes(self._buffer[:MULAW_FRAME_BYTES])
            frames.append(base64.b64encode(chunk).decode("ascii"))
            del self._buffer[:MULAW_FRAME_BYTES]
        return frames