/**
 * Auth context — provides auth state and actions to the entire React tree.
 *
 * Centralizes:
 *   - isAuthenticated flag
 *   - isInitializing (true during the on-mount rehydration check)
 *   - login / logout actions
 *   - CSRF token management (delegated to apiClient)
 *   - 401 redirect wiring (via apiClient callback)
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from 'react';
import { useNavigate } from 'react-router-dom';
import { api, ApiError, setCsrfToken, setUnauthorizedHandler } from '@/lib/apiClient';

// ─── Types ────────────────────────────────────────────────────────────────────

interface LoginCredentials {
  email: string;
  password: string;
}

interface SessionResponse {
  authenticated: boolean;
  csrf_token: string | null;
  email: string | null;
}

interface AuthContextValue {
  /** True once a valid session has been established and rehydration is done. */
  isAuthenticated: boolean;
  /**
   * True while the on-mount rehydration check is in-flight.
   * Use this to show a loading spinner instead of flashing the login screen.
   */
  isInitializing: boolean;
  /** True while a login request is in-flight. */
  isLoading: boolean;
  /** Error message from the last failed login attempt, or null. */
  loginError: string | null;
  /** Attempt to log in with the given credentials. */
  login: (credentials: LoginCredentials) => Promise<void>;
  /** Clear the session and redirect to login. */
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

// ─── Provider ─────────────────────────────────────────────────────────────────

/**
 * Auth state machine backed by the /api/admin/login, /api/admin/logout, and
 * /api/admin/session endpoints.
 *
 * On mount, calls GET /api/admin/session to securely rehydrate the auth state
 * using the HttpOnly session cookie (server-side validation, no JS access needed).
 */
export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isInitializing, setIsInitializing] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [loginError, setLoginError] = useState<string | null>(null);

  const navigate = useNavigate();

  // Keep a ref so callbacks below don't need to be recreated on state change.
  const navigateRef = useRef(navigate);
  navigateRef.current = navigate;

  /** Wire the 401 callback so any API 401 auto-redirects to login. */
  const registerUnauthorizedHandler = useCallback(() => {
    setUnauthorizedHandler(() => {
      setCsrfToken('');
      setIsAuthenticated(false);
      navigateRef.current('/login');
    });
  }, []);

  /** Clear CSRF token, 401 handler, and update auth state. */
  const handleLogout = useCallback(async () => {
    // Clear the 401 redirect handler before the API call so we don't
    // accidentally redirect while logging out.
    setUnauthorizedHandler(null);

    try {
      await api.post<{ message: string }>('/api/admin/logout', {});
    } catch {
      // Logout is best-effort — always clear state regardless of the network result.
    } finally {
      setCsrfToken('');
      setIsAuthenticated(false);
    }
  }, []);

  /** Attempt to authenticate. On success, store the CSRF token and set auth=true. */
  const handleLogin = useCallback(async ({ email, password }: LoginCredentials) => {
    setIsLoading(true);
    setLoginError(null);

    try {
      const response = await api.post<{ csrf_token: string; message?: string }>(
        '/api/admin/login',
        { email, password },
      );

      setCsrfToken(response.csrf_token);

      // Wire up the 401 → redirect handler so any API call that returns 401
      // automatically bounces the user back to the login screen.
      registerUnauthorizedHandler();

      setIsAuthenticated(true);
      navigateRef.current('/admin');
    } catch (err) {
      setIsAuthenticated(false);
      if (err instanceof ApiError) {
        // Mirror the message the backend sends.
        setLoginError(err.message);
      } else {
        setLoginError('Something went wrong. Please try again.');
      }
      throw err; // Re-throw so the form can handle its own loading state.
    } finally {
      setIsLoading(false);
    }
  }, [registerUnauthorizedHandler]);

  // ── Secure rehydration on mount ───────────────────────────────────────────
  //
  // The session cookie is HttpOnly — JavaScript can never see it.
  // Instead, we call GET /api/admin/session which validates the cookie server-side
  // and returns the CSRF token + email if the session is still valid.
  //
  // This runs once on app mount and transitions isInitializing → false when done.
  useEffect(() => {
    let cancelled = false;

    async function rehydrate() {
      try {
        const session = await api.get<SessionResponse>('/api/admin/session');

        if (!cancelled) {
          if (session.authenticated && session.csrf_token) {
            setCsrfToken(session.csrf_token);
            registerUnauthorizedHandler();
            setIsAuthenticated(true);
          }
        }
      } catch {
        // Network error / server down — treat as unauthenticated; stay on login.
        if (!cancelled) {
          setIsAuthenticated(false);
        }
      } finally {
        if (!cancelled) {
          setIsInitializing(false);
        }
      }
    }

    rehydrate();
    return () => {
      cancelled = true;
    };
  }, [registerUnauthorizedHandler]);

  const value: AuthContextValue = {
    isAuthenticated,
    isInitializing,
    isLoading,
    loginError,
    login: handleLogin,
    logout: handleLogout,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

// ─── Hook ─────────────────────────────────────────────────────────────────────

/** Access auth state and actions from any component. */
export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error('useAuth must be used inside <AuthProvider>.');
  }
  return ctx;
}
