"""
backend/matcher_consumer.py — Matcher Consumer (Issue 022).

Architecture
────────────
The Matcher Consumer is the single, long-lived per-call asynchronous consumer
task that ties the Preference Extractor (Issue 021) and the Deterministic
Scorer (Issue 020) together.

It reads completed customer turns from an asyncio.Queue, processes them strictly
one at a time (extract → score → update state), and maintains the running
preference set and top vehicle recommendations.  The voice agent (Riya) must
never block waiting for the matcher — extraction and scoring happen entirely
off the conversation driver's critical path.

By processing turns in strict arrival order via a serialised consumer (rather
than launching parallel per-turn tasks), we guarantee that the "latest explicit
statement wins" rule holds perfectly true, even if the customer rapidly
changes their mind multiple times in a few seconds.

Concurrency Contract
──────────────────
enqueue_turn() is completely non-blocking:
  • It puts the turn on the queue with put_nowait and returns immediately.
  • The consumer task owns the authoritative running state.
  • If the customer outpaces extraction the queue backs up harmlessly.
  • Ordering is guaranteed by the single FIFO queue + single consumer.

Internal State Merge
───────────────────
_process_one reads self._running_prefs directly and passes it to the extractor
as the merge base.  This is what makes "latest statement wins" true even when
the producer (Riya) fires turns faster than the extractor can respond.

None Coercion Boundary
─────────────────────
The Extractor emits a fully nullable running set, but the Scorer's
``PreferenceSet`` requires concrete hard constraints.  Just before calling
the scorer, this consumer coerces absent hard constraints to "no-constraint"
values:

    budget=None           → float('inf')
    min_seats=None       → 1
    feature_keywords(list)→ tuple (immutable, required by frozen PreferenceSet)

Usability Policy
───────────────
A preference set is "usable" if it has:
  - At least one hard constraint (budget finite OR min_seats > 1), OR
  - At least 2 captured soft preferences (fuel_type, body_style, transmission,
    min_mileage_kmpl, feature_keywords non-empty).

Usage
─────
    consumer = MatcherConsumer(vehicles=vehicles)
    consumer.start()                        # starts background consumer task
    consumer.enqueue_turn("I want a petrol SUV")  # returns immediately
    # ... later ...
    prefs = consumer.get_current_preferences()
    cars  = consumer.get_top_cars()
    await consumer.drain()                  # wait for queue to clear
    await consumer.close()                  # graceful shutdown
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.catalogue import VehicleRecord
    from backend.preference_extractor import GeminiClient
    from backend.scorer import RankingResult, ScoredCar

__all__ = ["MatcherConsumer"]


# ─── Sentinel ──────────────────────────────────────────────────────────────────


class _ShutdownSentinel:
    """
    Dedicated shutdown marker placed on the queue to stop the consumer loop.

    An empty real utterance must never stop the consumer — use this sentinel
    instead of relying on an empty-string check.
    """
    __slots__ = ()

    def __repr__(self) -> str:
        return "<ShutdownSentinel>"


_SHUTDOWN = _ShutdownSentinel()


# ─── Queue item ────────────────────────────────────────────────────────────────


@dataclass
class TurnItem:
    """
    One completed customer turn queued for matching.

    Attributes
    ----------
    utterance : str
        The customer's spoken statement for this turn.
    """

    utterance: str


# ─── Matcher Consumer ───────────────────────────────────────────────────────────


class MatcherConsumer:
    """
    Single, ordered async consumer that extracts preferences and scores vehicles.

    Parameters
    ----------
    vehicles       : list[VehicleRecord]
        The in-memory vehicle catalogue (typically ``catalogue.vehicles``).
    extractor_cls  : type
        Concrete ``GeminiClient`` subclass to instantiate.  Defaults to
        ``GeminiRestClient`` (the production implementation).  Tests pass a
        ``MockGeminiClient`` to avoid live API calls.
    """

    def __init__(
        self,
        vehicles: list[VehicleRecord],
        *,
        extractor_cls: type | None = None,
    ) -> None:
        self._vehicles = vehicles
        self._queue: asyncio.Queue = asyncio.Queue()

        # Import here so the module is loadable without google-adk in dev/test.
        if extractor_cls is None:
            from backend.preference_extractor import GeminiRestClient
            extractor_cls = GeminiRestClient

        self._extractor: GeminiClient = extractor_cls()

        # ── Internal state (written only by the consumer task) ─────────────────
        self._running_prefs: dict | None = None
        self._top_cars: tuple[ScoredCar, ...] = ()
        self._ranking_note: str | None = None

        # ── Lifecycle ───────────────────────────────────────────────────────────
        self._task: asyncio.Task[None] | None = None
        self._closed = False

    # ── Public: lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Start the background consumer task.  Idempotent — safe to call more
        than once after ``close()`` (resets _closed and creates a fresh task).
        """
        self._closed = False
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def drain(self) -> None:
        """
        Wait for the queue to be fully drained — all pending turns processed.

        Use this in tests and at call-finalisation time to ensure the consumer
        has settled before reading state.
        """
        await self._queue.join()

    async def close(self) -> None:
        """
        Gracefully shut down the consumer.

        Places a shutdown sentinel on the queue so the consumer loop stops after
        draining any pending turns.  Waits for the background task to finish.
        """
        self._closed = True
        self._queue.put_nowait(_SHUTDOWN)
        if self._task is not None:
            await self._task

    # ── Public: ingestion ───────────────────────────────────────────────────────

    def enqueue_turn(self, utterance: str) -> None:
        """
        Enqueue a customer turn for asynchronous processing.

        NON-BLOCKING: returns immediately after placing the turn on the queue.
        The consumer task processes turns in strict FIFO order.

        Parameters
        ----------
        utterance : str
            The customer's latest spoken statement.
        """
        self._queue.put_nowait(TurnItem(utterance=utterance))

    # ── Public: state queries ──────────────────────────────────────────────────

    def get_current_preferences(self) -> dict | None:
        """
        Return a snapshot of the running preference set.

        Returns ``None`` if no turn has been processed yet.
        The returned dict is a copy — safe to read without locking.
        """
        return dict(self._running_prefs) if self._running_prefs is not None else None

    def get_top_cars(self) -> tuple[ScoredCar, ...]:
        """
        Return the current top-scored vehicles.

        Returns an empty tuple if no turn has been processed yet.
        """
        return self._top_cars

    def get_ranking_note(self) -> str | None:
        """
        Return the fallback note from the last ranking, if any.

        ``None`` when the last ranking satisfied hard constraints.
        """
        return self._ranking_note

    # ── Usability policy ───────────────────────────────────────────────────────

    def is_usable(self) -> bool:
        """
        Return True when the running preference set is "usable".

        A preference set is usable if it has:
          - At least one hard constraint: ``budget`` is finite (not inf) OR
            ``min_seats`` is greater than 1 (i.e. explicitly 7), OR
          - At least 2 captured soft preferences.

        Soft preference fields that count: ``fuel_type``, ``body_style``,
        ``transmission``, ``min_mileage_kmpl`` (not None), and
        ``feature_keywords`` (non-empty tuple/list).

        This method is used later by the call pipeline to decide whether to
        close the call due to timeout or to keep waiting for more input.
        """
        if self._running_prefs is None:
            return False

        prefs = self._running_prefs

        # ── Hard constraints ───────────────────────────────────────────────────
        budget = prefs.get("budget")
        min_seats = prefs.get("min_seats")

        has_hard = (
            (budget is not None and budget < float("inf"))  # noqa: PLR1714
            or (min_seats is not None and min_seats > 1)
        )
        if has_hard:
            return True

        # ── Soft preferences (need ≥ 2) ─────────────────────────────────────────
        soft_count = 0
        if prefs.get("fuel_type") is not None:
            soft_count += 1
        if prefs.get("body_style") is not None:
            soft_count += 1
        if prefs.get("transmission") is not None:
            soft_count += 1
        if prefs.get("min_mileage_kmpl") is not None:
            soft_count += 1
        fk = prefs.get("feature_keywords")
        if fk and len(fk) > 0:
            soft_count += 1

        return soft_count >= 2

    # ── Private: consumer loop ────────────────────────────────────────────────

    async def _run(self) -> None:
        """Background consumer task — runs until close()."""
        while True:
            item = await self._queue.get()
            try:
                if item is _SHUTDOWN:
                    # Drain any remaining real turns before exiting
                    while not self._queue.empty():
                        turn = self._queue.get_nowait()
                        if turn is _SHUTDOWN:
                            break
                        await self._process_one(turn)
                    break
                await self._process_one(item)
            finally:
                self._queue.task_done()

    async def _process_one(self, item: TurnItem) -> None:
        """
        Process a single turn: extract → coerce → score → update state.

        This method runs exclusively on the consumer task — never concurrently
        with itself.  The extractor reads self._running_prefs as the merge base,
        which is what makes "latest explicit statement wins" true even when
        turns arrive faster than extraction completes.
        """
        # Step 1: extract — pass the consumer's own running state as the base
        from backend.preference_extractor import extract_preferences

        merged = extract_preferences(
            utterance=item.utterance,
            current_prefs=self._running_prefs,
            client=self._extractor,
        )

        # Step 2: coerce None hard constraints to "no-constraint" values
        #   budget=None  → float('inf')
        #   min_seats=None → 1
        #   feature_keywords (list) → tuple  (PreferenceSet is frozen/hashable)
        coerced_budget = merged.get("budget")
        if coerced_budget is None:
            coerced_budget = float("inf")

        coerced_min_seats = merged.get("min_seats")
        if coerced_min_seats is None:
            coerced_min_seats = 1

        fk = merged.get("feature_keywords")
        coerced_fk: tuple[str, ...] = tuple(fk) if fk else ()

        # Step 3: build PreferenceSet and score
        from backend.scorer import PreferenceSet, score_and_rank

        prefs_for_scorer = PreferenceSet(
            budget=coerced_budget,
            min_seats=coerced_min_seats,
            fuel_type=merged.get("fuel_type"),
            body_style=merged.get("body_style"),
            transmission=merged.get("transmission"),
            min_mileage_kmpl=merged.get("min_mileage_kmpl"),
            feature_keywords=coerced_fk,
        )

        result: RankingResult = score_and_rank(prefs_for_scorer, self._vehicles)

        # Step 4: update authoritative state
        self._running_prefs = merged
        self._top_cars = result.cars
        self._ranking_note = result.note
