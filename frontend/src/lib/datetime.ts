/**
 * frontend/src/lib/datetime.ts — IST timestamp formatting utilities (Issue 015).
 *
 * Problem
 * -------
 * The backend sends UTC timestamps with an explicit `Z` suffix, formatted as
 * ISO-8601: `2026-06-28T14:00:00Z`.  The browser's `Date` constructor parses
 * this correctly AS UTC, but `Intl.DateTimeFormat` defaults to the user's
 * local timezone — which may not be IST.
 *
 * Solution
 * --------
 * `formatIst` uses `Intl.DateTimeFormat` with `timeZone: 'Asia/Kolkata'`
 * as the single source of truth for displaying UTC timestamps in IST across
 * the entire application.  This ensures consistent display regardless of
 * the user's OS locale settings.
 *
 * Usage
 * -----
 * ```ts
 * import { formatIst } from './lib/datetime';
 *
 * // Backend sends: "2026-06-28T14:00:00Z" (14:00 UTC)
 * // Displayed in IST (UTC+5:30): "28 Jun 2026, 7:30 pm"
 * const istString = formatIst("2026-06-28T14:00:00Z");
 * ```
 */

/**
 * Format a UTC ISO-8601 timestamp for display in IST (Asia/Kolkata).
 *
 * @param isoUtc - A UTC timestamp string ending in `Z`, e.g. `"2026-06-28T14:00:00Z"`
 * @returns A human-readable string formatted for the IST locale, e.g. `"28 Jun 2026, 7:30 pm"`
 *          Returns `"—"`` if the input is null, undefined, or cannot be parsed.
 */
export function formatIst(isoUtc: string | null | undefined): string {
  if (!isoUtc) {
    return "—";
  }

  const date = new Date(isoUtc);
  if (isNaN(date.getTime())) {
    return "—";
  }

  return new Intl.DateTimeFormat("en-IN", {
    timeZone: "Asia/Kolkata",
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: true,
  }).format(date);
}
