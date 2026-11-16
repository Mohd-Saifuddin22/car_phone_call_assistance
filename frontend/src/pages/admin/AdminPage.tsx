/**
 * Admin area — the protected back-office shell.
 *
 * Auth is handled by RequireAuth at the router level.
 * The dashboard now embeds the live leads list (Issue 016).
 */
import { Link } from 'react-router-dom';

export function AdminPage() {
  return (
    <div className="space-y-4">
      <h1 className="text-3xl font-bold text-white">Admin Dashboard</h1>
      <p className="text-gray-400">
        You are securely authenticated.
      </p>

      {/* Quick-access cards */}
      <div className="grid gap-4 md:grid-cols-2">
        <Link
          to="/admin/leads"
          className="flex items-center justify-between rounded-lg border border-indigo-700 bg-indigo-950 p-6 transition-colors hover:border-indigo-500"
        >
          <div>
            <h2 className="text-lg font-semibold text-indigo-300">Leads</h2>
            <p className="mt-1 text-sm text-gray-400">
              Live feed of all submissions with per-phone history and IST timestamps.
            </p>
          </div>
          <span className="text-indigo-400">→</span>
        </Link>

        <Link
          to="/admin/blocks"
          className="flex items-center justify-between rounded-lg border border-yellow-700 bg-yellow-950/30 p-6 transition-colors hover:border-yellow-500"
        >
          <div>
            <h2 className="text-lg font-semibold text-yellow-300">Blocked Customers</h2>
            <p className="mt-1 text-sm text-gray-400">
              Manage blocked customers to prevent abuse and spam submissions.
            </p>
          </div>
          <span className="text-yellow-400">→</span>
        </Link>
      </div>
    </div>
  );
}
