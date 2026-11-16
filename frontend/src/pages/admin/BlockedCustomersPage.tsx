/**
 * frontend/src/pages/admin/BlockedCustomersPage.tsx — Issue 019: Blocked customers management.
 *
 * Key design decisions:
 * - Polling via `useEffect` + `setInterval` (consistent with LeadsListPage).
 * - Loading/error states handled locally.
 * - Unblock action with confirmation dialog.
 * - `formatIst` from Issue 015 is the single source of truth for IST display.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { api, ApiError, type BlockedCustomer } from '@/lib/apiClient';
import { formatIst } from '@/lib/datetime';

// ─── Polling interval ───────────────────────────────────────────────────────────

const POLL_INTERVAL_MS = 5_000;

// ─── Blocked customer row ──────────────────────────────────────────────────────

function BlockedCustomerRow({
  customer,
  onUnblock,
  isUnblocking,
}: {
  customer: BlockedCustomer;
  onUnblock: (phone: string) => void;
  isUnblocking?: boolean;
}) {
  return (
    <div className="rounded-lg border border-yellow-700 bg-yellow-950/30 p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        {/* Customer details */}
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-semibold text-yellow-200">
            {customer.name ?? '(no name snapshot)'}
          </p>
          <p className="mt-0.5 font-mono text-xs text-indigo-400">{customer.phone}</p>
          <p className="text-xs text-gray-500">{customer.email}</p>
        </div>

        {/* Block timestamp */}
        <div className="flex flex-col items-end gap-1.5">
          <span className="text-xs text-gray-400">
            Blocked: {formatIst(customer.created_at)}
          </span>
        </div>
      </div>

      {/* Action row */}
      <div className="mt-3 flex items-center justify-end border-t border-yellow-800/50 pt-2">
        <button
          type="button"
          onClick={() => onUnblock(customer.phone)}
          disabled={isUnblocking}
          className="flex items-center gap-1 rounded px-2 py-1 text-xs text-green-400 hover:bg-green-900/40 hover:text-green-300 disabled:pointer-events-none disabled:opacity-50 transition-colors"
          title="Unblock this customer"
        >
          {isUnblocking ? (
            <div className="h-3.5 w-3.5 animate-spin rounded-full border border-green-400 border-t-transparent" />
          ) : (
            /* Unlock icon */
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
              <path d="M7 11V7a5 5 0 0 1 9.9-1" />
            </svg>
          )}
          {isUnblocking ? 'Unblocking…' : 'Unblock'}
        </button>
      </div>
    </div>
  );
}

// ─── Empty / loading states ────────────────────────────────────────────────────

function EmptyState() {
  return (
    <div className="rounded-lg border border-dashed border-gray-700 py-16 text-center text-gray-500">
      No blocked customers — the block list is empty.
    </div>
  );
}

function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="rounded-lg border border-red-800 bg-red-950 py-6 text-center text-red-400">
      <p className="font-medium">Failed to load blocked customers: {message}</p>
      <button
        type="button"
        onClick={onRetry}
        className="mt-2 text-sm underline hover:no-underline"
      >
        Retry
      </button>
    </div>
  );
}

// ─── Main page ─────────────────────────────────────────────────────────────────

export function BlockedCustomersPage() {
  const [blocked, setBlocked] = useState<BlockedCustomer[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isInitial, setIsInitial] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [unblockingPhones, setUnblockingPhones] = useState<Set<string>>(new Set());
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const handleUnblock = useCallback(
    async (phone: string) => {
      if (!window.confirm(`Unblock ${phone}? They will be able to submit new requests again.`)) {
        return;
      }
      setUnblockingPhones((prev) => new Set(prev).add(phone));
      try {
        await api.unblockCustomer(phone);
        // Eagerly remove from the list so the UI feels responsive.
        setBlocked((prev) => prev.filter((c) => c.phone !== phone));
      } catch (err) {
        if (err instanceof ApiError) {
          if (err.status !== 401) {
            setError(`Failed to unblock customer: ${err.message}`);
          }
        } else {
          setError('Network error — please try again.');
        }
      } finally {
        setUnblockingPhones((prev) => {
          const next = new Set(prev);
          next.delete(phone);
          return next;
        });
      }
    },
    [],
  );

  const fetchBlocked = useCallback(async (isBackground = false) => {
    if (!isBackground) setIsInitial(true);
    if (isBackground) setIsRefreshing(true);
    try {
      const data = await api.getBlocked();
      setBlocked(data.blocked);
      setError(null);
    } catch (err) {
      if (err instanceof ApiError) {
        // 401 is handled by the global callback (redirect to login) — don't show it as a UI error.
        if (err.status !== 401) {
          setError(err.message);
        }
      } else {
        setError('Network error — please try again.');
      }
    } finally {
      setIsInitial(false);
      setIsRefreshing(false);
    }
  }, []);

  // ── Polling ──────────────────────────────────────────────────────────────────
  useEffect(() => {
    fetchBlocked(false);
    intervalRef.current = setInterval(() => fetchBlocked(true), POLL_INTERVAL_MS);

    return () => {
      if (intervalRef.current !== null) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [fetchBlocked]);

  return (
    <div className="space-y-4">
      {/* Page header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Blocked Customers</h1>
        <div className="flex items-center gap-3">
          {isRefreshing && (
            <div className="h-4 w-4 animate-spin rounded-full border-2 border-yellow-500 border-t-transparent" />
          )}
          <span className="text-xs text-gray-500">
            Auto-refreshes every {POLL_INTERVAL_MS / 1000}s
          </span>
          <span className="rounded-full bg-yellow-900/30 px-2 py-0.5 text-xs text-yellow-400">
            {blocked.length} {blocked.length === 1 ? 'blocked' : 'blocked'}
          </span>
        </div>
      </div>

      {/* Content */}
      {isInitial && blocked.length === 0 ? (
        <div className="py-16 text-center text-gray-500">
          <div className="mx-auto mb-3 h-6 w-6 animate-spin rounded-full border-2 border-yellow-500 border-t-transparent" />
          Loading blocked customers…
        </div>
      ) : error ? (
        <ErrorState message={error} onRetry={() => fetchBlocked(false)} />
      ) : blocked.length === 0 ? (
        <EmptyState />
      ) : (
        <div className="space-y-3">
          {blocked.map((customer) => (
            <BlockedCustomerRow
              key={customer.phone}
              customer={customer}
              onUnblock={handleUnblock}
              isUnblocking={unblockingPhones.has(customer.phone)}
            />
          ))}
        </div>
      )}
    </div>
  );
}
