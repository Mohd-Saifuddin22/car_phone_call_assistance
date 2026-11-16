/**
 * Centralized API client — the single place for all HTTP communication with the
 * backend.  Injects credentials on every request and provides a single hook point
 * for the X-CSRF-Token header populated by the auth system.
 *
 * Why this pattern:
 *   - Credentials are included automatically on every call (cookie-based auth/sessions).
 *   - The X-CSRF-Token is injected from the stored value — no hunting through components.
 *   - 401 responses trigger a configurable callback so the SPA can redirect to login.
 *   - The base URL resolves through Vite's proxy in dev; in production it is
 *     set to the deployed API origin via the VITE_API_BASE_URL env var.
 */

const BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? '';

// ─── CSRF token storage ────────────────────────────────────────────────────────

/** Current per-session CSRF token. Set by AuthProvider after login. */
let _csrfToken = '';

/**
 * Store the CSRF token returned by the login endpoint.
 * Call this once per session (after login / on app init).
 */
export const setCsrfToken = (token: string): void => {
  _csrfToken = token;
};

// ─── 401 redirect callback ─────────────────────────────────────────────────────

/**
 * Callback invoked whenever the API returns 401 Unauthorized.
 * The auth provider registers this to redirect the user to the login screen.
 */
let _onUnauthorized: (() => void) | null = null;

/**
 * Register the 401 handler.  Only one handler is active at a time.
 * Call with `null` to clear (e.g., on logout).
 */
export const setUnauthorizedHandler = (handler: (() => void) | null): void => {
  _onUnauthorized = handler;
};

interface RequestOptions extends RequestInit {
  /** Query params to append to the URL */
  params?: Record<string, string | number | boolean | null | undefined>;
}

/**
 * Make an authenticated HTTP request against the API.
 *
 * @param path     — absolute path e.g. `/api/health`; Vite proxy routes it to the FastAPI origin.
 * @param options  — standard fetch options + optional `params` object for query strings.
 * @returns        — the parsed JSON response.
 * @throws         — on network error, non-2xx responses, or JSON decode failure.
 */
export async function apiFetch<T>(
  path: string,
  { params, ...init }: RequestOptions = {},
): Promise<T> {
  let url = `${BASE_URL}${path}`;

  if (params !== undefined) {
    const searchParams = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) {
      if (v != null) {
        searchParams.set(k, String(v));
      }
    }
    const qs = searchParams.toString();
    if (qs) {
      url += `?${qs}`;
    }
  }

  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...(init.headers as Record<string, string> ?? {}),
  };

  // ── CSRF hook point ──────────────────────────────────────────────────────
  // When Issues 013/014 populate _csrfToken, this block will inject the header.
  // No-op until then (empty string is falsy and headers entry is skipped).
  if (_csrfToken) {
    headers['X-CSRF-Token'] = _csrfToken;
  }
  // ────────────────────────────────────────────────────────────────────────

  const response = await fetch(url, {
    ...init,
    headers,
    credentials: 'include',
  });

  if (!response.ok) {
    let message = `HTTP ${response.status} ${response.statusText}`;
    let detail: unknown = undefined;

    // Attempt to extract a structured error body for a better message.
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (body?.detail !== undefined) {
        // Only use detail as the message string when it's a primitive.
        // Objects / arrays are stored in .detail for programmatic consumption.
        if (typeof body.detail === 'string') {
          message = body.detail;
        } else {
          detail = body.detail;
        }
      }
    } catch {
      // Ignore JSON decode failures — surface the HTTP status instead.
    }

    // 401 → user session is invalid / expired → redirect to login.
    if (response.status === 401 && _onUnauthorized) {
      _onUnauthorized();
    }

    throw new ApiError(message, response.status, detail);
  }

  // Handle 204 No Content gracefully.
  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

// ─── Leads API types ─────────────────────────────────────────────────────────────

export interface LeadHistoryEntry {
  id: number;
  status: string;
  created_at: string; // ISO-8601 UTC with Z suffix
}

export interface Lead {
  id: number;
  name: string;
  phone: string;
  email: string;
  city: string;
  language: string;
  vehicle_id: number | null;
  status: string;
  failure_reason: string | null;
  attempt_count: number;
  created_at: string; // ISO-8601 UTC with Z suffix
  history: LeadHistoryEntry[];
}

export interface LeadsListResponse {
  leads: Lead[];
  total: number;
}

// ─── Blocked customers API types ──────────────────────────────────────────────

export interface BlockedCustomer {
  id: number;
  phone: string;
  email: string;
  name: string | null;
  created_at: string; // ISO-8601 UTC with Z suffix
}

export interface BlockedListResponse {
  blocked: BlockedCustomer[];
  total: number;
}

// ─── Typed convenience wrappers ──────────────────────────────────────────────

export const api = {
  get<T>(path: string, params?: RequestOptions['params']): Promise<T> {
    return apiFetch<T>(path, { method: 'GET', params });
  },

  /**
   * Fetch all leads for the admin dashboard (Issue 016).
   * Authenticated — requires a valid session cookie.
   * GET is CSRF-exempt.
   */
  getLeads(): Promise<LeadsListResponse> {
    return apiFetch<LeadsListResponse>('/api/leads', { method: 'GET' });
  },

  post<T>(path: string, body: unknown, params?: RequestOptions['params']): Promise<T> {
    return apiFetch<T>(path, {
      method: 'POST',
      params,
      body: JSON.stringify(body),
    });
  },

  put<T>(path: string, body: unknown, params?: RequestOptions['params']): Promise<T> {
    return apiFetch<T>(path, {
      method: 'PUT',
      params,
      body: JSON.stringify(body),
    });
  },

  patch<T>(path: string, body: unknown, params?: RequestOptions['params']): Promise<T> {
    return apiFetch<T>(path, {
      method: 'PATCH',
      params,
      body: JSON.stringify(body),
    });
  },

  delete<T>(path: string, params?: RequestOptions['params']): Promise<T> {
    return apiFetch<T>(path, { method: 'DELETE', params });
  },

  /**
   * Delete a lead by ID (Issue 018).
   * CSRF token is automatically injected by apiFetch.
   * Returns 204 No Content on success.
   * Throws ApiError on 401 / 403 / 404 / 409.
   */
  async deleteLead(leadId: number): Promise<void> {
    await apiFetch<void>(`/api/leads/${leadId}`, { method: 'DELETE' });
  },

  /**
   * Block a lead by ID (Issue 019).
   * CSRF token is automatically injected by apiFetch.
   * Returns 201 with confirmation message on success.
   * Throws ApiError on 401 / 403 / 404.
   */
  async blockLead(leadId: number): Promise<{ message: string }> {
    return apiFetch<{ message: string }>(`/api/leads/${leadId}/block`, { method: 'POST' });
  },

  /**
   * Fetch all blocked customers (Issue 019).
   * Authenticated — requires a valid session cookie.
   * GET is CSRF-exempt.
   */
  async getBlocked(): Promise<BlockedListResponse> {
    return apiFetch<BlockedListResponse>('/api/blocks', { method: 'GET' });
  },

  /**
   * Unblock a customer by exact E.164 phone (Issue 019).
   * CSRF token is automatically injected by apiFetch.
   * Returns 200 with confirmation message on success.
   * Throws ApiError on 401 / 403 / 404.
   */
  async unblockCustomer(phone: string): Promise<{ message: string }> {
    return apiFetch<{ message: string }>(`/api/blocks/${encodeURIComponent(phone)}`, { method: 'DELETE' });
  },
};

// ─── Error type ──────────────────────────────────────────────────────────────

export class ApiError extends Error {
  readonly status: number;
  /** Raw structured detail from the response body — objects or arrays when the backend
   *  returns field-level validation errors (e.g. Pydantic arrays or custom dicts). */
  readonly detail: unknown;

  constructor(message: string, status: number, detail: unknown = undefined) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}
