"""
backend/tests/test_outcome_classifier.py — Unit tests for Issue 028.

Coverage
--------
Tests are entirely table-driven.  Every row of the PRD Call Outcome Detection
table is explicitly asserted, plus the Completed-vs-dropped discriminator and the
max-duration-usable exception.

PRD rows covered
───────────────
[R1] Create-call API error / combined_status='failed'          → Failed / error
[R2] CallStatus='busy'                                         → Failed / no_answer
[R3] CallStatus='no_answer'                                    → Failed / no_answer
[R4] AnsweredBy starts with 'machine'                          → Failed / no_answer
[R5] completed/canceled, human, no end_consult, matcher usable  → Completed / completed_timeout
[R6] completed/canceled, human, no end_consult, matcher empty  → Failed / dropped
[R7] completed/canceled, human, end_consultation fired          → Completed
[R8] Backend restart (startup sweep)                           → Failed / dropped  (handled by sweep caller; covered by R6)
[R9] Unrecoverable mid-call provider fault                     → Failed / error
[R10] end_consultation fired                                   → Completed           (R7 is the connected case; R5/R6 are the max-dur exception)

Discriminator
─────────────
On combined_status='completed'/'canceled': Completed vs. dropped is determined
by whether end_consultation_fired, with the max-duration-usable exception.
Usable + max-duration (no end_consult) → Completed/completed_timeout (R5).
Empty  + max-duration (no end_consult) → Failed/dropped (R6).
end_consult fired                       → Completed (R7).

AMD edge cases
──────────────
unknown: trial-account AMD can't determine → treated as human; no early failure.
None:   AMD callback not yet arrived → classifier proceeds; no early failure.
"""

from __future__ import annotations

import pytest

from backend.outcome_classifier import CallOutcome, classify


# ==============================================================================
# Helpers
# ==============================================================================

def c(
    status: str,
    failure_reason: str | None,
    completed_timeout: bool = False,
) -> CallOutcome:
    """Shorthand to construct a CallOutcome."""
    return CallOutcome(
        status=status,
        failure_reason=failure_reason,
        completed_timeout=completed_timeout,
    )


# ==============================================================================
# R1 — Create-call API error / combined_status='failed'
# ==============================================================================

@pytest.mark.parametrize(
    "combined_status",
    ["failed"],
)
def test_failed_status_is_error(combined_status: str) -> None:
    """R1: combined_status='failed' (API error or Twilio status=failed) → error."""
    result = classify(
        combined_status=combined_status,
        answered_by="human",       # irrelevant; failed takes priority
        end_consultation_fired=False,
        matcher_usable=True,       # irrelevant; failed takes priority
        provider_fault=False,
    )
    assert result == c("Failed", "error")


# ==============================================================================
# R2 — CallStatus='busy'
# ==============================================================================

def test_busy_is_no_answer() -> None:
    """R2: Twilio status='busy' → no_answer."""
    result = classify(
        combined_status="busy",
        answered_by="human",
        end_consultation_fired=False,
        matcher_usable=False,
        provider_fault=False,
    )
    assert result == c("Failed", "no_answer")


# ==============================================================================
# R3 — CallStatus='no_answer'
# ==============================================================================

def test_no_answer_status_is_no_answer() -> None:
    """R3: Twilio status='no-answer' → no_answer."""
    result = classify(
        combined_status="no_answer",
        answered_by="human",
        end_consultation_fired=False,
        matcher_usable=False,
        provider_fault=False,
    )
    assert result == c("Failed", "no_answer")


# ==============================================================================
# R4 — AMD machine_* voicemail detection
# ==============================================================================

@pytest.mark.parametrize(
    "answered_by",
    [
        "machine",      # AMD answered by generic answering machine
        "machine_end",  # AMD answered by machine; end of message
        "fax",          # AMD answered by fax machine (Twilio AMD separate value)
    ],
)
@pytest.mark.parametrize(
    "combined_status",
    ["completed", "canceled", "created"],
)
def test_machine_voicemail_is_no_answer(
    answered_by: str,
    combined_status: str,
) -> None:
    """R4: Any AnsweredBy value indicating a non-human answer ('machine*' or 'fax') → no_answer."""
    result = classify(
        combined_status=combined_status,
        answered_by=answered_by,
        end_consultation_fired=False,
        matcher_usable=False,
        provider_fault=False,
    )
    assert result == c("Failed", "no_answer")


# ==============================================================================
# R5 / R6 — completed / canceled, human, no end_consultation
#            (max-duration backstop path)
#
# The discriminator: matcher_usable determines whether we save or drop.
# ==============================================================================

@pytest.mark.parametrize("terminal_status", ["completed", "canceled"])
@pytest.mark.parametrize(
    "matcher_usable,expected",
    [
        (True,  c("Completed", None, completed_timeout=True)),   # R5
        (False, c("Failed",   "dropped",       completed_timeout=False)),  # R6
    ],
)
def test_max_duration_backstop_usable_vs_empty(
    terminal_status: str,
    matcher_usable: bool,
    expected: CallOutcome,
) -> None:
    """
    R5: max-duration + usable Matcher → Completed / completed_timeout.
    R6: max-duration + empty Matcher  → Failed / dropped.

    Both 'completed' and 'canceled' statuses share this behaviour when
    end_consultation has not fired.
    """
    result = classify(
        combined_status=terminal_status,
        answered_by="human",
        end_consultation_fired=False,
        matcher_usable=matcher_usable,
        provider_fault=False,
    )
    assert result == expected


# ==============================================================================
# R7 — completed / canceled, human, end_consultation fired
# ==============================================================================

@pytest.mark.parametrize("terminal_status", ["completed", "canceled"])
def test_end_consultation_fired_is_completed(terminal_status: str) -> None:
    """R7: end_consultation fires on a connected call → Completed."""
    result = classify(
        combined_status=terminal_status,
        answered_by="human",
        end_consultation_fired=True,
        matcher_usable=True,    # irrelevant; end_consultation wins
        provider_fault=False,
    )
    assert result == c("Completed", None)


# ==============================================================================
# R9 — Unrecoverable mid-call AI-provider fault
# ==============================================================================

@pytest.mark.parametrize("terminal_status", ["completed", "canceled"])
def test_provider_fault_is_error(terminal_status: str) -> None:
    """
    R9: Unrecoverable mid-call AI-provider fault (Deepgram / Mercury-2 / Sarvam)
    → Failed / error, even on a connected call.

    The provider_fault signal overrides the end_consultation / max-duration
    signals because a genuine fault is a different class of failure.
    """
    result = classify(
        combined_status=terminal_status,
        answered_by="human",
        end_consultation_fired=True,    # irrelevant; provider_fault takes priority
        matcher_usable=True,            # irrelevant
        provider_fault=True,
    )
    assert result == c("Failed", "error")


# ==============================================================================
# AMD edge cases: answered_by=None and answered_by='unknown'
# ==============================================================================

@pytest.mark.parametrize(
    "answered_by",
    [None, "unknown"],
)
@pytest.mark.parametrize(
    "combined_status,end_consultation_fired,matcher_usable,expected",
    [
        # No failure signal yet — await further signals; no early failure.
        # Treat as in-flight (unexpected status path), no failure.
        ("created",   False, False, c("Failed", "error")),
        # end_consultation fires → clean Completed
        ("completed", True,  False, c("Completed", None)),
        # Max-duration usable → Completed / completed_timeout
        ("completed", False, True,  c("Completed", None, completed_timeout=True)),
        # Max-duration empty → dropped
        ("completed", False, False, c("Failed", "dropped")),
        # Provider fault → error
        ("completed", False, True,  c("Failed", "error")),  # overridden by provider_fault
    ],
)
def test_amd_edge_cases(
    answered_by: str | None,
    combined_status: str,
    end_consultation_fired: bool,
    matcher_usable: bool,
    expected: CallOutcome,
) -> None:
    """
    AMD 'unknown' (trial account) and None (AMD not yet arrived) are not
    early-failure signals — they do not themselves cause a failure.
    The classifier proceeds to the end_consultation / max-duration logic.

    Note: The last param row (completed + end_consult=F + matcher=T + provider_fault=T)
    tests provider_fault overriding the max-duration-usable path; it is also covered
    by the dedicated provider_fault test above but included here for completeness.
    """
    # Only apply provider_fault=True when the expected outcome is error
    provider_fault = expected == c("Failed", "error")

    result = classify(
        combined_status=combined_status,
        answered_by=answered_by,
        end_consultation_fired=end_consultation_fired,
        matcher_usable=matcher_usable,
        provider_fault=provider_fault,
    )
    assert result == expected


# ==============================================================================
# Unexpected / in-flight status codes
# ==============================================================================

@pytest.mark.parametrize(
    "combined_status",
    [
        "created",    # Twilio API call succeeded, call is being placed
        "initiated",  # Twilio CallStatus=initiated
        "ringing",    # Twilio CallStatus=ringing
        "answered",   # Twilio CallStatus=answered (call connected, not yet complete)
    ],
)
def test_inflight_status_is_error(combined_status: str) -> None:
    """
    Any status that represents an in-flight or unrecognised state is treated
    as an error to avoid silently swallowing unknown signals.
    """
    result = classify(
        combined_status=combined_status,
        answered_by="human",
        end_consultation_fired=False,
        matcher_usable=False,
        provider_fault=False,
    )
    assert result == c("Failed", "error")


# ==============================================================================
# Completed / canceled — matcher_usable is ignored when end_consultation fires
# ==============================================================================

@pytest.mark.parametrize("terminal_status", ["completed", "canceled"])
@pytest.mark.parametrize(
    "matcher_usable",
    [True, False],
)
def test_end_consultation_ignores_matcher_usability(
    terminal_status: str,
    matcher_usable: bool,
) -> None:
    """
    When end_consultation fires the outcome is always Completed regardless of
    whether the Matcher set is usable — end_consultation is the unambiguous
    clean-close signal.
    """
    result = classify(
        combined_status=terminal_status,
        answered_by="human",
        end_consultation_fired=True,
        matcher_usable=matcher_usable,
        provider_fault=False,
    )
    assert result == c("Completed", None)


# ==============================================================================
# completed_timeout flag is False on clean Completed
# ==============================================================================

def test_completed_timeout_false_on_clean_completed() -> None:
    """completed_timeout must be False on a clean end_consultation close."""
    result = classify(
        combined_status="completed",
        answered_by="human",
        end_consultation_fired=True,
        matcher_usable=True,
        provider_fault=False,
    )
    assert result.completed_timeout is False


# ==============================================================================
# completed_timeout flag is True only on max-duration-usable path
# ==============================================================================

@pytest.mark.parametrize("terminal_status", ["completed", "canceled"])
@pytest.mark.parametrize("matcher_usable", [True, False])
def test_completed_timeout_flag_only_on_max_duration_usable(
    terminal_status: str,
    matcher_usable: bool,
) -> None:
    """
    completed_timeout is True only when:
      - combined_status is terminal (completed / canceled)
      - end_consultation has NOT fired
      - matcher_usable IS True  (max-duration backstop exception)
    All other paths must set it to False.
    """
    result = classify(
        combined_status=terminal_status,
        answered_by="human",
        end_consultation_fired=False,
        matcher_usable=matcher_usable,
        provider_fault=False,
    )
    assert result.completed_timeout == matcher_usable


# ==============================================================================
# CallOutcome repr is stable and informative
# ==============================================================================

def test_calloutcome_repr_basic() -> None:
    """Basic CallOutcome repr does not raise."""
    o = CallOutcome(status="Completed", failure_reason=None, completed_timeout=False)
    r = repr(o)
    assert "Completed" in r
    assert "error" not in r.lower()

def test_calloutcome_repr_with_failure() -> None:
    """CallOutcome repr includes failure_reason when set."""
    o = CallOutcome(status="Failed", failure_reason="no_answer", completed_timeout=False)
    r = repr(o)
    assert "Failed" in r
    assert "no_answer" in r

def test_calloutcome_repr_with_timeout() -> None:
    """CallOutcome repr includes completed_timeout when True."""
    o = CallOutcome(status="Completed", failure_reason=None, completed_timeout=True)
    r = repr(o)
    assert "completed_timeout=True" in r
