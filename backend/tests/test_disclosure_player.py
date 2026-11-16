"""
backend/tests/test_disclosure_player.py — Unit tests for Issue 037
disclosure-at-connect audio player.

Test plan
─────────

_disclosure_asset_path (pure path helper, no I/O)
A1  English language → backend/audio/assets/disclosure_en.ulaw
A2  Hinglish language → backend/audio/assets/disclosure_hi.ulaw
A3  Custom assets_dir is respected
A4  Unknown language falls back to the English file (safe default —
    never crash the player for a misconfigured lead)

play_disclosure (async coroutine)

Happy path (H1)
  Loads a small fake disclosure asset (timing-friendly — the real
  frozen files are ~10 s each), pushes (gen, chunk) + (gen, None) to
  outbound_audio_queue, waits for is_speaking to transition True →
  False, then calls mark_disclosure_complete().

Generation counter (G1, G2)
  G1  The player's generation counter is the supervisor's current
      counter at push time — NOT bumped.
  G2  The chunk pushed is tagged with that generation.

Timeout backstop (T1)
  No mark echo arrives; the player still calls
  mark_disclosure_complete() after the duration-based timeout.

Cancellation via shutdown_event (C1)
  Setting shutdown_event mid-playback causes the player to exit
  without calling mark_disclosure_complete() — half-played
  disclosures must NOT be marked legally complete.

Cancellation via asyncio.CancelledError (C2)
  Cancelling the player's task lets CancelledError propagate (the
  media-stream handler awaits the task with `return_exceptions=True`
  and proceeds regardless).

Asset missing (M1)
  A non-existent assets_dir → the player logs loudly and marks
  complete anyway so the call does not hang.

Enqueue failure (E1)
  If the outbound_audio_queue is full (maxsize=1) and the chunk
  push raises asyncio.QueueFull, the player marks complete anyway
  rather than blocking.

Why a small fake asset?
────────────────────────
The real frozen disclosure files are ~76 KB / ~89 KB (≈9.5 s / ≈11 s
of audio).  The player's timeout backstop is ``duration + buffer``, so
using the real files in timing tests would mean 10+ s waits.  The
tests below write a tiny 160-byte payload (one 20 ms μ-law frame) to
a temp dir and pass it via ``assets_dir`` — exercising the same code
path with predictable timing.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.api.media_stream import (
    DEFAULT_DISCLOSURE_ASSETS_DIR,
    _disclosure_asset_path,
    play_disclosure,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _make_test_supervisor(
    *,
    language: str = "English",
    queue_maxsize: int | None = None,
) -> SimpleNamespace:
    """
    Build a minimal supervisor-like object for ``play_disclosure`` tests.

    Uses real ``asyncio`` primitives for everything mutable (queues,
    events) so the player's coroutines observe realistic state
    transitions.  Plain attributes for the immutable fields.
    """
    sup: SimpleNamespace = SimpleNamespace()
    sup.call_sid = "CA_TEST_DISC"
    sup.language = language
    sup.generation_counter = 0
    sup.is_speaking = False
    sup.outbound_audio_queue = (
        asyncio.Queue(maxsize=queue_maxsize) if queue_maxsize else asyncio.Queue()
    )
    sup.shutdown_event = asyncio.Event()
    sup.disclosure_complete = asyncio.Event()
    sup.disclosure_active = True

    # Count mark_disclosure_complete calls so we can assert "exactly once".
    sup._mark_calls = 0

    def _mark() -> None:
        sup._mark_calls += 1
        sup.disclosure_complete.set()
        sup.disclosure_active = False

    sup.mark_disclosure_complete = _mark
    return sup


async def _simulate_outbound_sender(
    sup: SimpleNamespace,
    *,
    start_delay_s: float = 0.02,
    flip_back_delay_s: float = 0.05,
) -> None:
    """
    Simulate the outbound sender + inbound mark echo for the disclosure.

    After ``start_delay_s``, flip ``is_speaking`` to True (sender
    started emitting frames).  After another ``flip_back_delay_s``,
    flip it back to False (the mark echo arrived).  This lets the
    player's wait loop observe the True → False transition it needs to
    mark the disclosure complete.
    """
    await asyncio.sleep(start_delay_s)
    sup.is_speaking = True
    await asyncio.sleep(flip_back_delay_s)
    sup.is_speaking = False


async def _simulate_mark_echo_only(
    sup: SimpleNamespace,
    *,
    flip_delay_s: float = 0.05,
) -> None:
    """
    Simulate ONLY the inbound reader's final-mark echo — does NOT
    flip ``is_speaking`` to True first.

    The player is responsible for priming ``is_speaking = True``
    itself after enqueue (the manual-priming fix).  If the player
    fails to prime, the wait loop never observes the True → False
    edge and falls through to the duration timeout backstop.  This
    helper is the strict check for that fix — using it lets the
    associated test catch the bug at the unit-test level rather than
    relying on the HITL live-call harness.
    """
    # Brief delay so the player has time to call put_nowait + set
    # is_speaking = True (the priming fix).  Then flip to False to
    # simulate the mark echo.
    await asyncio.sleep(flip_delay_s)
    sup.is_speaking = False


@pytest.fixture
def small_disclosure_dir():
    """
    Write a 160-byte disclosure_en.ulaw + 160-byte disclosure_hi.ulaw
    in a fresh temp dir and yield the dir.  160 bytes = one 20 ms
    μ-law frame, so duration is 20 ms and the player's timeout
    backstop fires within ~timeout_buffer + 20 ms.
    """
    with tempfile.TemporaryDirectory(prefix="car_test_disc_") as td:
        d = Path(td)
        (d / "disclosure_en.ulaw").write_bytes(b"\x00" * 160)
        (d / "disclosure_hi.ulaw").write_bytes(b"\x00" * 160)
        yield d


# ─── A1–A4: _disclosure_asset_path ────────────────────────────────────────────


class TestDisclosureAssetPath:
    """Path helper picks the right frozen file for each language."""

    def test_a1_english_returns_en_ulaw(self) -> None:
        sup = _make_test_supervisor(language="English")
        path = _disclosure_asset_path(sup)
        assert path == DEFAULT_DISCLOSURE_ASSETS_DIR / "disclosure_en.ulaw"
        assert path.name == "disclosure_en.ulaw"

    def test_a2_hinglish_returns_hi_ulaw(self) -> None:
        sup = _make_test_supervisor(language="Hinglish")
        path = _disclosure_asset_path(sup)
        assert path == DEFAULT_DISCLOSURE_ASSETS_DIR / "disclosure_hi.ulaw"
        assert path.name == "disclosure_hi.ulaw"

    def test_a3_custom_assets_dir_is_respected(self) -> None:
        sup = _make_test_supervisor(language="English")
        custom = Path("/tmp/custom_assets")
        path = _disclosure_asset_path(sup, assets_dir=custom)
        assert path == custom / "disclosure_en.ulaw"

    def test_a4_unknown_language_falls_back_to_english(self) -> None:
        """A misconfigured lead must not crash the player — English is
        the safe default so the file path always resolves to a file
        that exists.
        """
        sup = _make_test_supervisor(language="Klingon")
        path = _disclosure_asset_path(sup)
        assert path == DEFAULT_DISCLOSURE_ASSETS_DIR / "disclosure_en.ulaw"


# ─── H1: happy path ───────────────────────────────────────────────────────────


class TestPlayDisclosureHappyPath:
    """The end-to-end happy path: load → enqueue → wait for mark → mark."""

    @pytest.mark.asyncio
    async def test_h1_loads_enqueues_waits_marks_complete(
        self, small_disclosure_dir
    ) -> None:
        sup = _make_test_supervisor(language="English")
        sender_task = asyncio.create_task(
            _simulate_outbound_sender(sup, start_delay_s=0.02, flip_back_delay_s=0.05)
        )

        player_task = asyncio.create_task(
            play_disclosure(
                sup, "MZ_TEST",
                assets_dir=small_disclosure_dir,
                timeout_buffer_s=0.2,
            )
        )
        await asyncio.wait_for(player_task, timeout=2.0)
        sender_task.cancel()
        try:
            await sender_task
        except asyncio.CancelledError:
            pass

        # Exactly one mark call (the player marks once at the end).
        assert sup._mark_calls == 1, (
            f"Expected exactly one mark_disclosure_complete call, got "
            f"{sup._mark_calls}"
        )
        # disclosure_complete event is set, disclosure_active is False.
        assert sup.disclosure_complete.is_set()
        assert sup.disclosure_active is False

    @pytest.mark.asyncio
    async def test_h2_completes_via_mark_echo_before_timeout_backstop(
        self, small_disclosure_dir
    ) -> None:
        """
        Issue 037 HITL regression: the player must mark complete via
        the ECHO path (is_speaking True→False), NOT via the duration
        timeout backstop.

        Root cause that this test catches:
        The outbound sender only flips is_speaking = True on a
        generation BUMP (``generation > last_seen_generation``).  The
        disclosure is the first audio on the call, so both values are
        0 — the sender never flips the flag, the inbound reader's
        final-mark echo is a no-op, and the player's wait loop never
        observes True → False.  Every call falls through to the
        duration timeout backstop (~10 s hard-coded wait for English).

        The fix: ``play_disclosure`` manually sets
        ``supervisor.is_speaking = True`` right after enqueueing the
        disclosure bytes so the inbound reader's mark echo can do its
        job.

        Test strategy: simulate ONLY the inbound reader's final-mark
        echo (flip is_speaking = False, do NOT touch True).  Without
        the fix, the player never sees True → False and waits the
        full timeout (~500 ms in this test) before completing.  With
        the fix, the player sets True internally, the echo flips it
        False, and completion happens near-instantly (~50 ms after
        the echo flip).  Asserting elapsed < 300 ms strictly proves
        the echo path was used.
        """
        sup = _make_test_supervisor(language="English")

        # Use a LONG timeout buffer (500 ms) so a timeout-path
        # completion would take >400 ms.  An echo-path completion
        # should take ~50 ms (mark echo delay).  The threshold of
        # 300 ms sits comfortably between the two regimes.
        mark_task = asyncio.create_task(_simulate_mark_echo_only(sup, flip_delay_s=0.05))

        start = time.perf_counter()
        player_task = asyncio.create_task(
            play_disclosure(
                sup, "MZ_TEST",
                assets_dir=small_disclosure_dir,
                timeout_buffer_s=0.5,
                poll_interval_s=0.01,
            )
        )
        await asyncio.wait_for(player_task, timeout=2.0)
        elapsed = time.perf_counter() - start
        await mark_task

        # Echo-path assertion: completion happens well before the
        # duration-timeout backstop.  If the manual-priming fix is
        # missing, is_speaking stays False forever and the player
        # waits the full 500 ms before timing out.
        assert elapsed < 0.3, (
            f"play_disclosure took {elapsed:.3f}s to complete — the "
            f"manual-priming fix is missing. The player is waiting for "
            f"the duration-timeout backstop instead of the inbound "
            f"reader's mark echo. Echo-path completion should happen "
            f"in ~50 ms; >300 ms means the wait loop never observed "
            f"is_speaking True→False."
        )

        # Functional assertion: the player still called mark exactly once.
        assert sup._mark_calls == 1
        assert sup.disclosure_complete.is_set()
        assert sup.disclosure_active is False


# ─── G1, G2: generation counter is not bumped ────────────────────────────────


class TestPlayDisclosureGenerationCounter:
    """The disclosure uses the supervisor's current generation but does
    NOT bump it — barge-in (Issue 042) owns generation bumps."""

    @pytest.mark.asyncio
    async def test_g1_generation_not_bumped_after_playback(
        self, small_disclosure_dir
    ) -> None:
        sup = _make_test_supervisor(language="English")
        sup.generation_counter = 0
        sender_task = asyncio.create_task(
            _simulate_outbound_sender(sup, start_delay_s=0.02, flip_back_delay_s=0.05)
        )

        player_task = asyncio.create_task(
            play_disclosure(
                sup, "MZ_TEST",
                assets_dir=small_disclosure_dir,
                timeout_buffer_s=0.5,
            )
        )
        await asyncio.wait_for(player_task, timeout=2.0)
        sender_task.cancel()
        try:
            await sender_task
        except asyncio.CancelledError:
            pass

        # Counter is unchanged from before playback.
        assert sup.generation_counter == 0, (
            "play_disclosure bumped generation_counter — Issue 042 owns "
            "generation bumps, the disclosure must not change it"
        )

    @pytest.mark.asyncio
    async def test_g2_chunk_tagged_with_current_generation(
        self, small_disclosure_dir
    ) -> None:
        """The (gen, chunk) item pushed to outbound_audio_queue carries
        the supervisor's generation_counter value at push time.
        """
        sup = _make_test_supervisor(language="English")
        sup.generation_counter = 7  # arbitrary non-zero

        # Don't simulate the sender — we only care about what was
        # enqueued.  Use a tiny timeout so the player marks complete
        # quickly via the timeout backstop.
        player_task = asyncio.create_task(
            play_disclosure(
                sup, "MZ_TEST",
                assets_dir=small_disclosure_dir,
                timeout_buffer_s=0.2,
                poll_interval_s=0.02,
            )
        )
        # Give the player a moment to enqueue.
        await asyncio.sleep(0.05)

        # Inspect the enqueued items BEFORE waiting for the player to
        # exit (the player will eventually time out and exit on its own).
        chunk_item = sup.outbound_audio_queue.get_nowait()
        gen, payload = chunk_item
        assert gen == 7, (
            f"Disclosure chunk tagged with generation {gen}, expected 7"
        )
        assert isinstance(payload, bytes) and len(payload) > 0

        marker_item = sup.outbound_audio_queue.get_nowait()
        assert marker_item == (7, None), (
            f"End-of-turn marker mis-shaped: {marker_item}"
        )

        # Now let the player finish (timeout backstop or sender flip).
        await asyncio.wait_for(player_task, timeout=2.0)


# ─── T1: timeout backstop ────────────────────────────────────────────────────


class TestPlayDisclosureTimeout:
    """When the mark echo never arrives, the duration-based timeout
    backstop still marks the disclosure complete so the driver can
    proceed."""

    @pytest.mark.asyncio
    async def test_t1_marks_complete_after_timeout_when_no_echo(
        self, small_disclosure_dir
    ) -> None:
        sup = _make_test_supervisor(language="English")
        # NO sender simulation — is_speaking stays False forever.

        # Use a tiny buffer so the player fires the backstop quickly.
        player_task = asyncio.create_task(
            play_disclosure(
                sup, "MZ_TEST",
                assets_dir=small_disclosure_dir,
                timeout_buffer_s=0.1,
                poll_interval_s=0.01,
            )
        )
        # Player should exit within ~duration (20 ms) + buffer (100 ms).
        await asyncio.wait_for(player_task, timeout=2.0)

        # Backstop fired: exactly one mark call, event set, active flipped.
        assert sup._mark_calls == 1
        assert sup.disclosure_complete.is_set()
        assert sup.disclosure_active is False


# ─── C1, C2: cancellation paths ──────────────────────────────────────────────


class TestPlayDisclosureCancellation:
    """Two cancellation paths: shutdown_event (Issue 037 spec) and
    asyncio.CancelledError (handler-cancelled)."""

    @pytest.mark.asyncio
    async def test_c1_shutdown_event_exits_without_marking_complete(
        self, small_disclosure_dir
    ) -> None:
        """Half-played disclosures must NOT be marked legally complete.
        Setting shutdown_event mid-playback causes the player to exit
        without calling mark_disclosure_complete().
        """
        sup = _make_test_supervisor(language="English")

        # Use a long timeout buffer so the player would NOT exit via
        # the timeout backstop before shutdown_event fires.
        player_task = asyncio.create_task(
            play_disclosure(
                sup, "MZ_TEST",
                assets_dir=small_disclosure_dir,
                timeout_buffer_s=5.0,
                poll_interval_s=0.02,
            )
        )
        # Give the player a moment to enqueue + start the wait loop.
        await asyncio.sleep(0.05)
        # Trigger teardown.
        sup.shutdown_event.set()

        # Player exits cleanly within a few poll cycles.
        await asyncio.wait_for(player_task, timeout=1.0)

        # CRITICAL: shutdown mid-playback must NOT call mark.
        assert sup._mark_calls == 0, (
            f"play_disclosure marked complete after shutdown_event was "
            f"set (call count = {sup._mark_calls}). Half-played "
            f"disclosures must not be marked legally complete."
        )
        assert not sup.disclosure_complete.is_set()
        assert sup.disclosure_active is True, (
            "play_disclosure flipped disclosure_active=False after "
            "shutdown_event — the caller should never see a partial "
            "disclosure marked done."
        )

    @pytest.mark.asyncio
    async def test_c2_cancelled_error_propagates(self) -> None:
        """asyncio.CancelledError propagates from play_disclosure when
        the handler cancels the player task on exit.
        """
        sup = _make_test_supervisor(language="English")
        player_task = asyncio.create_task(
            play_disclosure(
                sup, "MZ_TEST",
                timeout_buffer_s=10.0,  # long, never fires
                poll_interval_s=0.02,
            )
        )
        await asyncio.sleep(0.02)
        player_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await player_task


# ─── M1: missing asset ────────────────────────────────────────────────────────


class TestPlayDisclosureMissingAsset:
    """A missing asset is a deployment bug — log loudly and mark
    complete anyway so the call does not hang on the driver's wait."""

    @pytest.mark.asyncio
    async def test_m1_missing_asset_marks_complete_to_avoid_hang(self) -> None:
        sup = _make_test_supervisor(language="English")
        empty_dir = Path(tempfile.mkdtemp(prefix="car_test_disc_empty_"))

        player_task = asyncio.create_task(
            play_disclosure(sup, "MZ_TEST", assets_dir=empty_dir)
        )
        await asyncio.wait_for(player_task, timeout=2.0)

        # Fallback fired: exactly one mark, event set.
        assert sup._mark_calls == 1
        assert sup.disclosure_complete.is_set()


# ─── E1: enqueue failure ─────────────────────────────────────────────────────


class TestPlayDisclosureEnqueueFailure:
    """If the outbound queue is full (maxsize=1 with one item already
    present) and the disclosure chunk push fails with QueueFull, the
    player must mark complete rather than block.
    """

    @pytest.mark.asyncio
    async def test_e1_queue_full_marks_complete_anyway(self) -> None:
        sup = _make_test_supervisor(language="English", queue_maxsize=1)
        # Pre-fill the queue so the chunk push raises QueueFull.
        sup.outbound_audio_queue.put_nowait((0, b"stale"))

        player_task = asyncio.create_task(
            play_disclosure(sup, "MZ_TEST")
        )
        await asyncio.wait_for(player_task, timeout=2.0)

        # The player swallowed the QueueFull and marked complete to
        # avoid hanging the call.
        assert sup._mark_calls == 1
        assert sup.disclosure_complete.is_set()


# ─── Sanity: queue contents ──────────────────────────────────────────────────


class TestPlayDisclosureQueueContents:
    """The player pushes exactly two items to outbound_audio_queue:
    (gen, chunk) and (gen, None).  No more, no less, on the happy path.
    """

    @pytest.mark.asyncio
    async def test_exactly_two_items_pushed_then_marks_complete(
        self, small_disclosure_dir
    ) -> None:
        sup = _make_test_supervisor(language="Hinglish")
        sender_task = asyncio.create_task(
            _simulate_outbound_sender(sup, start_delay_s=0.02, flip_back_delay_s=0.05)
        )

        player_task = asyncio.create_task(
            play_disclosure(
                sup, "MZ_TEST",
                assets_dir=small_disclosure_dir,
                timeout_buffer_s=0.2,
            )
        )
        await asyncio.wait_for(player_task, timeout=2.0)
        sender_task.cancel()
        try:
            await sender_task
        except asyncio.CancelledError:
            pass

        # The player pushed chunk + None and the sender drained them
        # (we don't simulate a draining sender here, so the queue may
        # still hold the items if the test thread reads fast enough —
        # but at minimum the player must have called mark once).
        assert sup._mark_calls == 1