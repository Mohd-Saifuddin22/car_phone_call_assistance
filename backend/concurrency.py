"""
backend/concurrency.py — Concurrency Slot Manager (Issue 027).

An in-process concurrency guard used by the call supervisor to limit active AI calls.
Strictly non-blocking: acquire() returns immediately with True/False, never awaits.

IMPORTANT: This manager must only be accessed from an async context on the event loop.
All methods are synchronous (no await) so they run atomically within a single task
switch — the event loop cannot interleave mid-method. Do NOT use threading.Lock or
any OS-thread primitives here.
"""

from __future__ import annotations


class ConcurrencyManager:
    """
    A non-blocking concurrency slot manager for asyncio applications.

    Ensures a hard limit on simultaneous active operations (e.g., AI calls).
    - acquire(): non-blocking check-and-take; returns True if slot reserved, False if full.
    - release(): restores one slot; cannot go below 0.
    - reset(): forcibly resets active count to zero (e.g., after a crash recovery).
    """

    def __init__(self, capacity: int = 1) -> None:
        """
        Initialize the concurrency manager.

        Args:
            capacity: Maximum number of concurrent operations allowed. Defaults to 1.

        Note:
            Must only be instantiated and used within a running asyncio event loop.
        """
        if capacity < 0:
            raise ValueError("Capacity must be non-negative.")
        self._capacity = capacity
        self._active = 0

    @property
    def capacity(self) -> int:
        """Return the maximum concurrency capacity."""
        return self._capacity

    @property
    def active(self) -> int:
        """Return the current number of active slots in use."""
        return self._active

    @property
    def available(self) -> int:
        """Return the number of slots currently available."""
        return self._capacity - self._active

    def acquire(self) -> bool:
        """
        Attempt to acquire a concurrency slot.

        Strictly non-blocking: returns immediately. Safe to call from any
        async task — the check-and-increment is atomic because there are
        no await points inside this method.

        Returns:
            True if a slot was successfully reserved (increments active count).
            False if already at capacity (no change made).
        """
        if self._active < self._capacity:
            self._active += 1
            return True
        return False

    def release(self) -> None:
        """
        Release one concurrency slot.

        The active count will not go below zero.

        Raises:
            RuntimeError: If called when no slots are active (defensive guard).
        """
        if self._active <= 0:
            raise RuntimeError("release() called but no slots are currently active.")
        self._active -= 1

    def reset(self) -> None:
        """
        Reset the active count to zero.

        Intended for recovery scenarios (e.g., process restart) where the
        active count should be restored to a known clean state.
        """
        self._active = 0
