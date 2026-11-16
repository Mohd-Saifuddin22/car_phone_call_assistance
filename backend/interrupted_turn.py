"""
backend/interrupted_turn.py — interrupted-turn metadata record (Issue 042).

Barge-in truncates one of Riya's turns mid-speech.  Word-accurate truncation
is impossible — the TTS provider returns no per-word timing — so instead of
guessing where the cut fell in the *text* we keep the **full intended text**
plus the **played / sent durations** recovered from Twilio ``mark`` accounting.
That is honest (never overstating what the customer actually heard) rather than
falsely precise, per the PRD's *Interrupted-turn metadata* decision.

This module owns only the data shape.  Issue 048 (transcript persistence)
consumes ``CallSupervisor.interrupted_turns`` and writes each record to the
``transcript_turns`` table (``interrupted = 1``, ``played_ms``, ``total_ms``,
``text`` = the full intended text).  Keeping the record here — an in-memory
seam on the supervisor — is what lets Issue 042 ship the barge-in state without
a speculative migration or DB write of its own.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "INTERRUPTED_MARKER",
    "InterruptedTurn",
    "build_interrupted_display_text",
]

#: Suffix appended to an interrupted turn's display/transcript text.  The
#: leading ellipsis makes the truncation visible to the admin without
#: implying a word-accurate cut point (which we cannot compute).
INTERRUPTED_MARKER: str = "… [interrupted]"


def build_interrupted_display_text(full_text: str) -> str:
    """
    Build the display/transcript string for an interrupted Riya turn.

    The result is the full intended text followed by the ``… [interrupted]``
    marker, so a reviewer sees everything Riya *meant* to say plus an explicit
    signal that she was cut off.  An empty/blank intended text yields the bare
    marker (barge-in fired before any words were generated).

    Parameters
    ----------
    full_text
        The complete text Riya was synthesizing when barge-in fired.

    Returns
    -------
    str
        Display text ending in ``… [interrupted]``.
    """
    text = (full_text or "").strip()
    if text:
        return f"{text} {INTERRUPTED_MARKER}"
    return INTERRUPTED_MARKER


@dataclass(frozen=True, slots=True)
class InterruptedTurn:
    """
    Immutable record of one Riya turn cut short by customer barge-in.

    Attributes
    ----------
    generation
        The per-call generation counter value of the interrupted utterance.
        Identifies exactly which outbound turn was abandoned.
    full_text
        The complete text Riya intended to speak (never truncated).
    display_text
        ``full_text`` with the ``… [interrupted]`` marker appended — the
        string Issue 050's transcript UI renders.
    played_ms
        Milliseconds of the turn Twilio confirmed as played, recovered from
        ``mark`` echoes (``played_duration_ms``).  Never estimated from text.
    total_ms
        Milliseconds of audio actually synthesized/sent for the turn, from
        outbound-frame accounting (``sent_duration_ms``).  ``played_ms`` is
        always ``<= total_ms`` in a healthy call.
    """

    generation: int
    full_text: str
    display_text: str
    played_ms: int
    total_ms: int

    @classmethod
    def from_snapshot(
        cls,
        *,
        generation: int,
        full_text: str,
        played_ms: int,
        total_ms: int,
    ) -> "InterruptedTurn":
        """
        Build a record from a barge-in snapshot, deriving ``display_text`` and
        clamping negative durations to zero (defensive against a mark echo
        that never arrived, which would leave the counters at their init 0).
        """
        text = full_text or ""
        return cls(
            generation=generation,
            full_text=text,
            display_text=build_interrupted_display_text(text),
            played_ms=max(0, played_ms),
            total_ms=max(0, total_ms),
        )
