/**
 * frontend/src/pages/admin/LeadsListPage.tsx — Issue 016: Dashboard leads list with polling.
 *
 * Key design decisions:
 * - Polling via `useEffect` + `setInterval` (no external deps — SWR/React Query not installed).
 * - Loading/error states handled locally so the SPA doesn't crash on transient failures.
 * - History rendered inline per lead — collapsed by default, expandable on click.
 * - `formatIst` from Issue 015 is the single source of truth for IST display.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { api, ApiError, type Lead } from '@/lib/apiClient';
import { formatIst } from '@/lib/datetime';

// ─── Polling interval ───────────────────────────────────────────────────────────

const POLL_INTERVAL_MS = 5_000;

// ─── Status badge ──────────────────────────────────────────────────────────────

const STATUS_LABELS: Record<string, { label: string; className: string }> = {
  New: { label: 'New', className: 'bg-blue-900 text-blue-300 border-blue-700' },
  Processing: { label: 'Processing', className: 'bg-yellow-900 text-yellow-300 border-yellow-700' },
  Completed: { label: 'Completed', className: 'bg-green-900 text-green-300 border-green-700' },
  Failed: { label: 'Failed', className: 'bg-red-900 text-red-300 border-red-700' },
};

function StatusBadge({ status }: { status: string }) {
  const style = STATUS_LABELS[status] ?? { label: status, className: 'bg-gray-800 text-gray-400 border-gray-600' };
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium ${style.className}`}
    >
      {style.label}
    </span>
  );
}

// ─── Lead history accordion ────────────────────────────────────────────────────

function LeadHistory({ history, currentId }: { history: Lead['history']; currentId: number }) {
  const [open, setOpen] = useState(false);

  if (history.length <= 1) return null;

  // The current lead is always first in the history (newest-first per phone).
  const otherEntries = history.filter((h) => h.id !== currentId);

  return (
    <div className="mt-2">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1 text-xs text-indigo-400 hover:text-indigo-300 transition-colors"
      >
        <span>{open ? '▲' : '▶'}</span>
        <span>
          {history.length - 1} prior {history.length - 1 === 1 ? 'lead' : 'leads'} for this phone
        </span>
      </button>
      {open && (
        <ul className="mt-1 space-y-1 rounded border border-gray-700 bg-gray-900 p-2">
          {otherEntries.map((h) => (
            <li key={h.id} className="flex items-center justify-between text-xs text-gray-400">
              <span className="flex items-center gap-2">
                <StatusBadge status={h.status} />
                <span>Lead #{h.id}</span>
              </span>
              <span>{formatIst(h.created_at)}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

// ─── Single lead row ───────────────────────────────────────────────────────────

function LeadRow({
  lead,
  onDelete,
  onBlock,
  isDeleting,
  isBlocking,
}: {
  lead: Lead;
  onDelete: (id: number) => void;
  onBlock: (id: number) => void;
  isDeleting?: boolean;
  isBlocking?: boolean;
}) {
  return (
    <div className="rounded-lg border border-gray-700 bg-gray-800 p-4">
      {/* Header row */}
      <div className="flex flex-wrap items-start justify-between gap-3">
        {/* Customer details */}
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-semibold text-white">{lead.name}</p>
          <p className="mt-0.5 font-mono text-xs text-indigo-400">{lead.phone}</p>
          <p className="text-xs text-gray-500">{lead.email}</p>
        </div>

        {/* Status & metrics */}
        <div className="flex flex-col items-end gap-1.5">
          <StatusBadge status={lead.status} />
          <span className="text-xs text-gray-400">
            {lead.attempt_count} {lead.attempt_count === 1 ? 'attempt' : 'attempts'}
          </span>
          {lead.failure_reason && (
            <span className="text-xs text-red-400">Reason: {lead.failure_reason}</span>
          )}
        </div>
      </div>

      {/* Metadata row */}
      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-gray-700 pt-2 text-xs text-gray-400">
        <span>
          <span className="text-gray-500">Submitted:</span>{' '}
          <span className="text-gray-200">{formatIst(lead.created_at)}</span>
        </span>
        <span>
          <span className="text-gray-500">City:</span> {lead.city}
        </span>
        <span>
          <span className="text-gray-500">Lang:</span> {lead.language}
        </span>
        {lead.vehicle_id && (
          <span>
            <span className="text-gray-500">Vehicle ID:</span> {lead.vehicle_id}
          </span>
        )}
      </div>

      {/* Action row */}
      <div className="mt-3 flex items-center justify-end gap-2 border-t border-gray-700 pt-2">
        <button
          type="button"
          onClick={() => onBlock(lead.id)}
          disabled={isBlocking}
          className="flex items-center gap-1 rounded px-2 py-1 text-xs text-yellow-400 hover:bg-yellow-900/40 hover:text-yellow-300 disabled:pointer-events-none disabled:opacity-50 transition-colors"
          title="Block this customer (prevents future submissions)"
        >
          {isBlocking ? (
            <div className="h-3.5 w-3.5 animate-spin rounded-full border border-yellow-400 border-t-transparent" />
          ) : (
            /* Shield icon */
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
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
            </svg>
          )}
          {isBlocking ? 'Blocking…' : 'Block'}
        </button>
        <button
          type="button"
          onClick={() => onDelete(lead.id)}
          disabled={isDeleting}
          className="flex items-center gap-1 rounded px-2 py-1 text-xs text-red-400 hover:bg-red-900/40 hover:text-red-300 disabled:pointer-events-none disabled:opacity-50 transition-colors"
          title="Delete lead (irreversible)"
        >
          {isDeleting ? (
            <div className="h-3.5 w-3.5 animate-spin rounded-full border border-red-400 border-t-transparent" />
          ) : (
            /* Trash icon */
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
              <polyline points="3 6 5 6 21 6" />
              <path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6" />
              <path d="M10 11v6M14 11v6" />
              <path d="M9 6V4a1 1 0 011-1h4a1 1 0 011 1v2" />
            </svg>
          )}
          {isDeleting ? 'Deleting…' : 'Delete'}
        </button>
      </div>

      {/* History */}
      <LeadHistory history={lead.history} currentId={lead.id} />
    </div>
  );
}

// ─── Empty / loading states ────────────────────────────────────────────────────

function EmptyState() {
  return (
    <div className="rounded-lg border border-dashed border-gray-700 py-16 text-center text-gray-500">
      No leads yet — they will appear here as submissions come in.
    </div>
  );
}

function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="rounded-lg border border-red-800 bg-red-950 py-6 text-center text-red-400">
      <p className="font-medium">Failed to load leads: {message}</p>
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

export function LeadsListPage() {
  const [leads, setLeads] = useState<Lead[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isInitial, setIsInitial] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [deletingIds, setDeletingIds] = useState<Set<number>>(new Set());
  const [blockingIds, setBlockingIds] = useState<Set<number>>(new Set());
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const handleBlockLead = useCallback(
    async (leadId: number) => {
      if (!window.confirm(`Block lead #${leadId}? They will not be able to submit new requests.`)) {
        return;
      }
      setBlockingIds((prev) => new Set(prev).add(leadId));
      try {
        await api.blockLead(leadId);
      } catch (err) {
        if (err instanceof ApiError) {
          if (err.status !== 401) {
            setError(`Failed to block lead: ${err.message}`);
          }
        } else {
          setError('Network error — please try again.');
        }
      } finally {
        setBlockingIds((prev) => {
          const next = new Set(prev);
          next.delete(leadId);
          return next;
        });
      }
    },
    [],
  );

  const handleDeleteLead = useCallback(
    async (leadId: number) => {
      if (!window.confirm(`Delete lead #${leadId}? This is permanent and cannot be undone.`)) {
        return;
      }
      setDeletingIds((prev) => new Set(prev).add(leadId));
      try {
        await api.deleteLead(leadId);
        // Eagerly remove from the list so the UI feels responsive.
        setLeads((prev) => prev.filter((l) => l.id !== leadId));
      } catch (err) {
        if (err instanceof ApiError) {
          // 401 is handled by the global callback (redirect to login) — don't show it as a UI error.
          if (err.status !== 401) {
            setError(`Failed to delete lead: ${err.message}`);
          }
        } else {
          setError('Network error — please try again.');
        }
      } finally {
        setDeletingIds((prev) => {
          const next = new Set(prev);
          next.delete(leadId);
          return next;
        });
      }
    },
    [],
  );

  const fetchLeads = useCallback(async (isBackground = false) => {
    if (!isBackground) setIsInitial(true);
    if (isBackground) setIsRefreshing(true);
    try {
      const data = await api.getLeads();
      setLeads(data.leads);
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
    fetchLeads(false);
    intervalRef.current = setInterval(() => fetchLeads(true), POLL_INTERVAL_MS);

    return () => {
      if (intervalRef.current !== null) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [fetchLeads]);

  return (
    <div className="space-y-4">
      {/* Page header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Leads</h1>
        <div className="flex items-center gap-3">
          {isRefreshing && (
            <div className="h-4 w-4 animate-spin rounded-full border-2 border-indigo-500 border-t-transparent" />
          )}
          <span className="text-xs text-gray-500">
            Auto-refreshes every {POLL_INTERVAL_MS / 1000}s
          </span>
          <span className="rounded-full bg-gray-800 px-2 py-0.5 text-xs text-gray-400">
            {leads.length} {leads.length === 1 ? 'lead' : 'leads'}
          </span>
        </div>
      </div>

      {/* Content */}
      {isInitial && leads.length === 0 ? (
        <div className="py-16 text-center text-gray-500">
          <div className="mx-auto mb-3 h-6 w-6 animate-spin rounded-full border-2 border-indigo-500 border-t-transparent" />
          Loading leads…
        </div>
      ) : error ? (
        <ErrorState message={error} onRetry={() => fetchLeads(false)} />
      ) : leads.length === 0 ? (
        <EmptyState />
      ) : (
        <div className="space-y-3">
          {leads.map((lead) => (
            <LeadRow
              key={lead.id}
              lead={lead}
              onDelete={handleDeleteLead}
              onBlock={handleBlockLead}
              isDeleting={deletingIds.has(lead.id)}
              isBlocking={blockingIds.has(lead.id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}
