"""
backend/tests/test_concurrency.py — Unit tests for ConcurrencyManager (Issue 027).

Tests cover:
- Acquire-at-capacity rejection (returns False, never blocks).
- Release-then-reacquire works.
- Reset restores active count to zero.
- Capacity cannot be released below zero.
- Asyncio task interleaving safety (active never exceeds capacity under gather).
- Proof that no caller ever blocks waiting on acquire().
"""

from __future__ import annotations

import asyncio
import time

import pytest

from backend.concurrency import ConcurrencyManager


class TestConcurrencyManagerBasic:
    """Basic single-task acquire/release/reset behavior."""

    def test_default_capacity_is_one(self) -> None:
        """Manager initialized without args should have capacity 1."""
        manager = ConcurrencyManager()
        assert manager.capacity == 1

    def test_custom_capacity(self) -> None:
        """Manager should accept arbitrary capacity."""
        manager = ConcurrencyManager(capacity=5)
        assert manager.capacity == 5

    def test_negative_capacity_raises(self) -> None:
        """Negative capacity is invalid."""
        with pytest.raises(ValueError):
            ConcurrencyManager(capacity=-1)

    def test_initial_active_is_zero(self) -> None:
        """New manager should have zero active slots."""
        manager = ConcurrencyManager()
        assert manager.active == 0

    def test_acquire_returns_true_when_free(self) -> None:
        """acquire() should return True when capacity is available."""
        manager = ConcurrencyManager(capacity=1)
        assert manager.acquire() is True
        assert manager.active == 1

    def test_acquire_at_capacity_returns_false(self) -> None:
        """
        Acquire-at-capacity rejection: when full, acquire() returns False immediately.
        This is the core fail-fast requirement — no blocking, no queuing.
        """
        manager = ConcurrencyManager(capacity=1)
        assert manager.acquire() is True   # First slot acquired
        assert manager.acquire() is False  # Capacity exhausted
        assert manager.active == 1          # No change

    def test_release_returns_active_to_zero(self) -> None:
        """release() should decrement the active count."""
        manager = ConcurrencyManager()
        manager.acquire()
        assert manager.active == 1
        manager.release()
        assert manager.active == 0

    def test_release_then_reacquire(self) -> None:
        """
        Release-then-reacquire works: after releasing, a new acquire succeeds.
        """
        manager = ConcurrencyManager(capacity=1)
        assert manager.acquire() is True
        manager.release()
        assert manager.acquire() is True  # Can reacquire after release
        assert manager.active == 1

    def test_multiple_releases_rejected(self) -> None:
        """
        Capacity cannot be released below zero.
        Calling release() when no slots are active raises RuntimeError.
        """
        manager = ConcurrencyManager()
        with pytest.raises(RuntimeError, match="no slots are currently active"):
            manager.release()

    def test_reset_clears_active(self) -> None:
        """
        reset() should force active count to zero regardless of prior state.
        """
        manager = ConcurrencyManager(capacity=2)
        manager.acquire()
        manager.acquire()
        assert manager.active == 2
        manager.reset()
        assert manager.active == 0

    def test_available_property(self) -> None:
        """available reflects capacity - active correctly."""
        manager = ConcurrencyManager(capacity=3)
        assert manager.available == 3
        manager.acquire()
        assert manager.available == 2
        manager.acquire()
        assert manager.available == 1
        manager.release()
        assert manager.available == 2


class TestConcurrencyManagerHigherCapacity:
    """Test behavior with capacity > 1."""

    def test_fill_all_slots(self) -> None:
        """All slots can be acquired up to capacity."""
        manager = ConcurrencyManager(capacity=3)
        for _ in range(3):
            assert manager.acquire() is True
        assert manager.active == 3
        assert manager.acquire() is False  # Now full

    def test_release_frees_one_slot(self) -> None:
        """Releasing one slot allows one more acquire."""
        manager = ConcurrencyManager(capacity=2)
        manager.acquire()
        manager.acquire()
        manager.release()
        assert manager.acquire() is True


class TestConcurrencyManagerNoBlocking:
    """
    Proof that no caller ever blocks waiting on acquire().
    These tests measure the elapsed time of acquire() calls.
    """

    def test_acquire_is_instant_when_full(self) -> None:
        """
        acquire() returns False instantly when at capacity.
        The call must complete in well under 1 second (a blocking call would hang).
        """
        manager = ConcurrencyManager(capacity=1)
        manager.acquire()  # Fill the slot

        start = time.perf_counter()
        result = manager.acquire()  # Must return immediately
        elapsed = time.perf_counter() - start

        assert result is False
        assert elapsed < 0.1, f"acquire() took {elapsed:.4f}s — it may be blocking!"

    def test_acquire_is_instant_when_free(self) -> None:
        """acquire() on a free slot should also be instant."""
        manager = ConcurrencyManager(capacity=1)

        start = time.perf_counter()
        result = manager.acquire()
        elapsed = time.perf_counter() - start

        assert result is True
        assert elapsed < 0.1

    def test_release_is_instant(self) -> None:
        """release() must also be non-blocking."""
        manager = ConcurrencyManager()
        manager.acquire()

        start = time.perf_counter()
        manager.release()
        elapsed = time.perf_counter() - start

        assert elapsed < 0.1


class TestConcurrencyManagerAsyncioInterleaving:
    """
    Verify that self._active never exceeds capacity under asyncio task interleaving.

    Since acquire() and release() contain no await points, they are atomic with
    respect to the event loop. However, we still test the interleaving axis with
    asyncio.gather() to prove that concurrent coroutine attempts can't race past
    the capacity limit.
    """

    @pytest.mark.asyncio
    async def test_active_never_exceeds_capacity_under_gather(self) -> None:
        """
        When many coroutines all call acquire() simultaneously via asyncio.gather(),
        active must never exceed capacity. This is the hard invariant.
        """
        manager = ConcurrencyManager(capacity=3)
        max_observed = 0

        async def acquire_and_observe() -> bool:
            nonlocal max_observed
            acquired = manager.acquire()
            # Check max immediately after acquire (no await between)
            if acquired:
                max_observed = max(max_observed, manager.active)
                await asyncio.sleep(0)  # Yield to allow other tasks to run
                manager.release()
            return acquired

        # 30 concurrent tasks all trying to acquire a capacity-3 manager
        results = await asyncio.gather(*[acquire_and_observe() for _ in range(30)])

        assert max_observed <= manager.capacity, (
            f"Active count {max_observed} exceeded capacity {manager.capacity}!"
        )
        assert manager.active == 0  # All released by end

    @pytest.mark.asyncio
    async def test_all_tasks_complete_under_gather(self) -> None:
        """
        asyncio.gather() must complete for all tasks — prove acquire() never
        blocks and all coroutines return without hanging.
        """
        manager = ConcurrencyManager(capacity=5)

        async def try_acquire() -> bool:
            if manager.acquire():
                await asyncio.sleep(0.001)
                manager.release()
                return True
            return False

        start = time.perf_counter()
        results = await asyncio.gather(*[try_acquire() for _ in range(20)])
        elapsed = time.perf_counter() - start

        # With capacity=5 and 20 tasks, gather must finish quickly
        assert elapsed < 1.0, f"gather() took {elapsed:.2f}s — acquire() may be blocking!"
        assert manager.active == 0
        # Some succeeded (returned True), some failed (returned False)
        assert any(results)

    @pytest.mark.asyncio
    async def test_release_then_reacquire_under_gather(self) -> None:
        """
        Rapid release-then-reacquire under gather: each pair of acquire/release
        should cycle cleanly without exceeding capacity.
        """
        manager = ConcurrencyManager(capacity=2)
        max_observed = 0

        async def cycle() -> None:
            nonlocal max_observed
            if manager.acquire():
                max_observed = max(max_observed, manager.active)
                manager.release()

        await asyncio.gather(*[cycle() for _ in range(50)])

        assert max_observed <= manager.capacity
        assert manager.active == 0

    @pytest.mark.asyncio
    async def test_high_capacity_all_tasks_get_turn(self) -> None:
        """
        With high capacity, every coroutine should succeed under gather.
        """
        manager = ConcurrencyManager(capacity=100)

        async def try_acquire() -> bool:
            return manager.acquire()

        results = await asyncio.gather(*[try_acquire() for _ in range(100)])

        assert all(results)
        assert manager.active == 100

        # Clean up
        for _ in range(100):
            manager.release()
