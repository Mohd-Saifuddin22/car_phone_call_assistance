import { useEffect, useState } from 'react';
import { api, ApiError } from '@/lib/apiClient';

interface HealthResponse {
  status: string;
}

/**
 * Proof-of-life page — calls GET /api/health on mount and renders the response.
 *
 * This is the first visible end-to-end vertical: SPA → Vite proxy → FastAPI → JSON.
 * The backend must be running at http://localhost:8000 for this to succeed.
 */
export function HealthPage() {
  const [data, setData] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    let cancelled = false;

    api
      .get<HealthResponse>('/api/health')
      .then((response) => {
        if (!cancelled) {
          setData(response);
          setError(null);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          const message = err instanceof ApiError
            ? `${err.message} (HTTP ${err.status})`
            : String(err);
          setError(message);
          setData(null);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="space-y-4">
      <h1 className="text-3xl font-bold text-gray-900">Health Check</h1>
      <p className="text-gray-600">
        This page fetches <code className="rounded bg-gray-100 px-1 py-0.5 font-mono text-sm">GET /api/health</code>{' '}
        on mount, proving the end-to-end path from the React SPA through the Vite proxy
        to the FastAPI backend.
      </p>

      <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm">
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-gray-500">
          Response
        </h2>

        {loading && (
          <p className="text-gray-400 italic">Fetching…</p>
        )}

        {error && (
          <div className="space-y-1">
            <p className="font-medium text-red-600">Error</p>
            <p className="text-sm text-red-500">{error}</p>
            <p className="mt-2 text-xs text-gray-400">
              Make sure the backend is running:{' '}
              <code className="rounded bg-gray-100 px-1 font-mono">PYTHONNOUSERSITE=1 .venv/bin/python -m uvicorn backend.main:app --reload</code>
            </p>
          </div>
        )}

        {data && (
          <pre className="overflow-x-auto rounded bg-gray-900 p-4 text-sm text-green-400">
            {JSON.stringify(data, null, 2)}
          </pre>
        )}
      </div>
    </div>
  );
}
