"""
backend/outcome_classifier.py — Call Outcome Classifier (Issue 028).

A pure mapping from telephony and system signals to a canonical call outcome.
No I/O, no side effects — the domain logic is fully isolated here so callers
(teardown, status callbacks, max-duration backstop) do not embed branching
logic in I/O-bound layers.

Public API
----------
class CallOutcome
    status            — 'Completed' | 'Failed'
    failure_reason    — None | 'no_answer' | 'dropped' | 'error'
    completed_timeout — bool  (True when Completed via the max-duration backstop)

classify(combined_status, answered_by, end_consultation_fired, matcher_usable, provider_fault)
    -> CallOutcome

Input parameters
----------------
combined_status : str
    The single unified signal that captures both the create-call result and the
    final Twilio CallStatus.  Values:

    'created'   — the Twilio API call succeeded and a CallSid was issued.
                   The call is in-progress or the final status has not yet arrived.
    'failed'    — either the Twilio API call itself errored (invalid number,
                   insufficient balance, rate limit, unverified-number on trial,
                   geo-permission) OR Twilio reported a terminal 'failed' status.
    'busy'      — Twilio CallStatus = 'busy'.
    'no_answer' — Twilio CallStatus = 'no-answer'.
    'completed' — Twilio CallStatus = 'completed' (customer was connected).
    'canceled'  — Twilio CallStatus = 'canceled' (post-answer teardown hangup
                   by the orchestrator).

answered_by : str | None
    AMD AnsweredBy verdict received asynchronously via the AMD callback.
    'human'   → human answered; proceed with the consultation.
    'machine_*' → voicemail / answering machine (any value starting with 'machine').
    'unknown'  → AMD could not determine (can occur on trial accounts — treated
                   as 'human' to proceed, per Spike 025 design note).
    None      → AMD callback has not yet arrived (normal before the verdict).

end_consultation_fired : bool
    True when Mercury-2 called the end_consultation tool and the closing TTS
    finish was acknowledged.  This is the unambiguous wrap-up trigger.

matcher_usable : bool
    True when the Matcher gathered a usable preference set:
    at least 1 hard constraint (finite budget OR min_seats > 1)
    OR at least 2 captured soft preferences.

provider_fault : bool
    True when an unrecoverable mid-call AI-provider fault occurred after the
    retry layer was exhausted (Deepgram / Mercury-2 / Sarvam failure).
    This is an orchestrator-detected signal, not a Twilio signal.

Outcome table
-------------
See PRD *Call Outcome Detection* and *Backend deep modules → Call outcome classifier*.

Twilio signal                                          | Outcome
-------------------------------------------------------|-------------------
Create-call API error / CallStatus = 'failed'          | Failed / error
CallStatus = 'busy' or 'no_answer'                    | Failed / no_answer
AnsweredBy starts with 'machine'                       | Failed / no_answer
Human connected, then disconnects before end_consultation | Failed / dropped
Max-duration hit, Matcher usable                        | Completed / completed_timeout
Max-duration hit, Matcher empty                         | Failed / dropped
Backend restart (startup sweep → zombie Processing)     | Failed / dropped
Unrecoverable mid-call AI-provider fault               | Failed / error
end_consultation fired                                  | Completed

Discriminator: on a 'completed' or 'canceled' status, Completed vs. dropped
is determined by whether end_consultation fired — with one exception: a
max-duration call that never fired end_consultation is still Completed
(completed_timeout) if the Matcher set is usable; otherwise it is dropped.
"""

from __future__ import annotations

from dataclasses import dataclass


# ─── Return type ───────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class CallOutcome:
    """
    Canonical call outcome emitted by the classifier.

    Attributes
    ----------
    status : str
        'Completed' — the call ran to a successful close and recommendations
        may be stored.
        'Failed'    — the call ended unsuccessfully; failure_reason is set.

    failure_reason : str | None
        None          — status is 'Completed', or the call is still in-flight
                         (classifier is not called in that case).
        'no_answer'  — busy / no-answer / AMD machine_*.
        'dropped'    — human disconnected before end_consultation; or
                        max-duration hit with empty Matcher set; or backend
                        restart detected by the startup sweep.
        'error'      — create-call API error; CallStatus=failed; or
                        unrecoverable mid-call AI-provider fault.

    completed_timeout : bool
        True when the call was marked Completed via the max-duration backstop
        (end_consultation did not fire but the Matcher gathered a usable set).
        Always False when status is 'Failed' or the call ended cleanly.
    """

    status: str
    failure_reason: str | None
    completed_timeout: bool

    def __repr__(self) -> str:
        parts = [f"status={self.status!r}"]
        if self.failure_reason is not None:
            parts.append(f"failure_reason={self.failure_reason!r}")
        if self.completed_timeout:
            parts.append("completed_timeout=True")
        return f"CallOutcome({', '.join(parts)})"


# ─── Classifier ───────────────────────────────────────────────────────────────

def classify(
    combined_status: str,
    answered_by: str | None,
    end_consultation_fired: bool,
    matcher_usable: bool,
    provider_fault: bool,
) -> CallOutcome:
    """
    Map telephony + system signals to a canonical CallOutcome.

    This function is pure — no I/O, no mutable state, no side effects.

    Parameters
    ----------
    combined_status : str
        As documented at the module level.
    answered_by : str | None
        As documented at the module level.
    end_consultation_fired : bool
        As documented at the module level.
    matcher_usable : bool
        As documented at the module level.
    provider_fault : bool
        As documented at the module level.

    Returns
    -------
    CallOutcome
        Canonical outcome with status, failure_reason, and completed_timeout.
    """
    # ── Create-call / API error ─────────────────────────────────────────────────
    if combined_status == "failed":
        return CallOutcome(
            status="Failed",
            failure_reason="error",
            completed_timeout=False,
        )

    # ── Twilio busy / no-answer ────────────────────────────────────────────────
    if combined_status in ("busy", "no_answer"):
        return CallOutcome(
            status="Failed",
            failure_reason="no_answer",
            completed_timeout=False,
        )

    # ── AMD voicemail / answering-machine / fax detection ─────────────────────
    # Twilio AMD AnsweredBy values: 'human', 'machine', 'machine_end', 'fax',
    # 'unknown'.  Any value indicating a non-human answer is treated as no_answer.
    # 'machine*' — voicemail / answering machine (starts with 'machine').
    # 'fax'      — fax machine (separate AMD value, also non-human).
    if answered_by is not None and (
        answered_by.startswith("machine") or answered_by == "fax"
    ):
        return CallOutcome(
            status="Failed",
            failure_reason="no_answer",
            completed_timeout=False,
        )

    # ── Terminal connected status: 'completed' or 'canceled' ──────────────────
    # Both mean the call was connected; the discriminator below applies.
    if combined_status in ("completed", "canceled"):
        # Unrecoverable mid-call AI-provider fault → error
        if provider_fault:
            return CallOutcome(
                status="Failed",
                failure_reason="error",
                completed_timeout=False,
            )

        # Max-duration backstop: neither end_consultation nor provider_fault fired.
        # The matcher decides whether the partial result is saved.
        if not end_consultation_fired:
            if matcher_usable:
                # Max-duration hit with a usable set → save with the timeout flag.
                return CallOutcome(
                    status="Completed",
                    failure_reason=None,
                    completed_timeout=True,
                )
            else:
                # Max-duration hit with nothing gathered → drop.
                return CallOutcome(
                    status="Failed",
                    failure_reason="dropped",
                    completed_timeout=False,
                )

        # end_consultation fired → clean close.
        return CallOutcome(
            status="Completed",
            failure_reason=None,
            completed_timeout=False,
        )

    # ── Unexpected / in-flight status ─────────────────────────────────────────
    # Any other status ('created', 'initiated', 'ringing', 'answered', etc.)
    # means the call is still in-flight or an unrecognised signal arrived.
    # Classify as error to avoid silently swallowing an unknown state.
    return CallOutcome(
        status="Failed",
        failure_reason="error",
        completed_timeout=False,
    )
