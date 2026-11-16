import { createBrowserRouter, RouterProvider, Outlet, Link, Navigate } from 'react-router-dom';
import { FormPage } from '@/pages/public/FormPage';
import { AdminPage } from '@/pages/admin/AdminPage';
import { LoginPage } from '@/pages/admin/LoginPage';
import { HealthPage } from '@/pages/public/HealthPage';
import { LeadsListPage } from '@/pages/admin/LeadsListPage';
import { BlockedCustomersPage } from '@/pages/admin/BlockedCustomersPage';
import { AuthProvider, useAuth } from '@/lib/authContext';

// ─── Auth guard ───────────────────────────────────────────────────────────────

/**
 * Route-level auth gate.
 *
 * During initial rehydration (isInitializing=true) we show a spinner so
 * the user doesn't see a flash of the login screen on page refresh.
 * Once initialised, unauthenticated visits redirect to /login.
 */
function RequireAuth({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isInitializing } = useAuth();

  if (isInitializing) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gray-950">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-indigo-500 border-t-transparent" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
}

// ─── Public layout ────────────────────────────────────────────────────────────

/**
 * Shared layout for all public-facing pages (customer form area).
 */
function PublicLayout() {
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <nav className="flex gap-6 text-sm font-medium">
          <Link to="/form" className="text-gray-600 hover:text-gray-900 transition-colors">
            Customer Form
          </Link>
          <Link to="/health" className="text-gray-600 hover:text-gray-900 transition-colors">
            Health Check
          </Link>
          <Link to="/admin" className="text-gray-600 hover:text-gray-900 transition-colors">
            Admin
          </Link>
        </nav>
      </header>
      <main className="mx-auto max-w-3xl px-6 py-10">
        <Outlet />
      </main>
    </div>
  );
}

// ─── Admin layout ─────────────────────────────────────────────────────────────

/**
 * Shared layout for protected admin pages.
 * Includes the logout control in the header.
 */
function AdminLayout() {
  const { logout } = useAuth();

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <header className="flex items-center justify-between bg-gray-900 border-b border-gray-700 px-6 py-4">
        <nav className="flex gap-6 text-sm font-medium">
          <Link to="/admin" className="text-gray-400 hover:text-white transition-colors">
            Dashboard
          </Link>
          <Link to="/admin/leads" className="text-gray-400 hover:text-white transition-colors">
            Leads
          </Link>
          <Link to="/admin/blocks" className="text-gray-400 hover:text-white transition-colors">
            Blocked
          </Link>
          <Link to="/form" className="text-gray-400 hover:text-white transition-colors">
            ← Back to Form
          </Link>
        </nav>
        <button
          type="button"
          onClick={logout}
          className="rounded-md px-3 py-1.5 text-sm font-medium text-gray-400 hover:bg-gray-800 hover:text-white transition-colors"
        >
          Sign out
        </button>
      </header>
      <main className="mx-auto max-w-3xl px-6 py-10">
        <Outlet />
      </main>
    </div>
  );
}

// ─── Router ───────────────────────────────────────────────────────────────────

/**
 * AuthProvider is the root layout so it sits inside RouterProvider's context.
 * All child routes — public, login, and admin — are descendants of AuthProvider,
 * allowing any page to call useAuth().
 */
const router = createBrowserRouter([
  {
    element: (
      <AuthProvider>
        <Outlet />
      </AuthProvider>
    ),
    children: [
      {
        // Public area — customer-facing form and demo pages.
        element: <PublicLayout />,
        children: [
          { index: true, element: <FormPage /> },
          { path: '/form', element: <FormPage /> },
          { path: '/health', element: <HealthPage /> },
        ],
      },
      {
        // Login page — publicly accessible but still inside AuthProvider
        // so the form can call useAuth().login() on success.
        path: '/login',
        element: <LoginPage />,
      },
      {
        // Protected admin area — wrapped in RequireAuth.
        element: (
          <RequireAuth>
            <AdminLayout />
          </RequireAuth>
        ),
        children: [
          { path: '/admin', element: <AdminPage /> },
          {
            path: '/admin/leads',
            element: <LeadsListPage />,
          },
          {
            path: '/admin/blocks',
            element: <BlockedCustomersPage />,
          },
        ],
      },
    ],
  },
]);

// ─── Root export ──────────────────────────────────────────────────────────────

export function AppRouter() {
  return <RouterProvider router={router} />;
}
