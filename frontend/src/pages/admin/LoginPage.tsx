/**
 * Admin login page — the entry gate for the protected admin area.
 *
 * Authenticates via POST /api/admin/login. On success the AuthProvider
 * stores the CSRF token and redirects to /admin.
 */

import { useState } from 'react';
import { useAuth } from '@/lib/authContext';
import { cn } from '@/lib/utils';

export function LoginPage() {
  const { login, loginError, isLoading } = useAuth();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fieldErrors, setFieldErrors] = useState<{ email?: string; password?: string }>({});

  // Clear field errors as the user types.
  function handleEmailChange(value: string) {
    setEmail(value);
    if (fieldErrors.email) setFieldErrors((e) => ({ ...e, email: undefined }));
  }

  function handlePasswordChange(value: string) {
    setPassword(value);
    if (fieldErrors.password) setFieldErrors((e) => ({ ...e, password: undefined }));
  }

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();

    const errors: typeof fieldErrors = {};

    if (!email.trim()) {
      errors.email = 'Email address is required.';
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim())) {
      errors.email = 'Enter a valid email address.';
    }

    if (!password) {
      errors.password = 'Password is required.';
    }

    if (Object.keys(errors).length > 0) {
      setFieldErrors(errors);
      return;
    }

    try {
      await login({ email: email.trim(), password });
    } catch {
      // AuthProvider.handleLogin already set loginError; no extra action needed.
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-950 px-4">
      <div className="w-full max-w-sm space-y-8">
        {/* Logo / brand */}
        <div className="text-center">
          <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-xl bg-indigo-600">
            <svg
              className="h-8 w-8 text-white"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
              aria-hidden="true"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M8.25 18.75a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m3 0h6m-9 0H3.375a1.125 1.125 0 01-1.125-1.125V14.25m17.25 4.5a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m3 0h1.125c.621 0 1.129-.504 1.09-1.124a17.902 17.902 0 00-3.213-9.193 2.056 2.056 0 00-1.58-.86H14.25M16.5 18.75h-2.25m0-11.177v-.958c0-.568-.422-1.048-.987-1.106a48.554 48.554 0 00-10.026 0 1.106 1.106 0 00-.987 1.106v7.635m12-6.677v6.677m0 4.5v-4.5m0 0h-12"
              />
            </svg>
          </div>
          <h1 className="text-2xl font-bold text-white">Admin Portal</h1>
          <p className="mt-2 text-sm text-gray-400">Sign in to access the dashboard</p>
        </div>

        {/* Login form */}
        <form
          onSubmit={handleSubmit}
          noValidate
          className="space-y-5 rounded-2xl border border-gray-800 bg-gray-900 p-8 shadow-2xl"
        >
          {/* Email */}
          <div>
            <label htmlFor="email" className="block text-sm font-medium text-gray-300">
              Email address
            </label>
            <input
              type="email"
              id="email"
              name="email"
              autoComplete="username"
              value={email}
              onChange={(e) => handleEmailChange(e.target.value)}
              disabled={isLoading}
              aria-required="true"
              aria-describedby={fieldErrors.email ? 'email-error' : undefined}
              className={cn(
                'mt-1 block w-full rounded-lg border bg-gray-800 px-3 py-2.5 text-sm text-white shadow-sm',
                'placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500',
                'transition-colors duration-150',
                fieldErrors.email
                  ? 'border-red-500 bg-red-950/30 text-red-200 focus:border-red-500 focus:ring-red-500'
                  : 'border-gray-700',
                isLoading && 'cursor-not-allowed opacity-60',
              )}
              placeholder="admin@example.com"
            />
            {fieldErrors.email && (
              <p id="email-error" role="alert" className="mt-1.5 text-xs text-red-400">
                {fieldErrors.email}
              </p>
            )}
          </div>

          {/* Password */}
          <div>
            <label htmlFor="password" className="block text-sm font-medium text-gray-300">
              Password
            </label>
            <input
              type="password"
              id="password"
              name="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => handlePasswordChange(e.target.value)}
              disabled={isLoading}
              aria-required="true"
              aria-describedby={fieldErrors.password ? 'password-error' : undefined}
              className={cn(
                'mt-1 block w-full rounded-lg border bg-gray-800 px-3 py-2.5 text-sm text-white shadow-sm',
                'placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500',
                'transition-colors duration-150',
                fieldErrors.password
                  ? 'border-red-500 bg-red-950/30 text-red-200 focus:border-red-500 focus:ring-red-500'
                  : 'border-gray-700',
                isLoading && 'cursor-not-allowed opacity-60',
              )}
              placeholder="••••••••"
            />
            {fieldErrors.password && (
              <p id="password-error" role="alert" className="mt-1.5 text-xs text-red-400">
                {fieldErrors.password}
              </p>
            )}
          </div>

          {/* API-level login error (wrong credentials etc.) */}
          {loginError && (
            <div role="alert" className="rounded-lg border border-red-800/50 bg-red-950/50 px-4 py-3 text-sm text-red-300">
              {loginError}
            </div>
          )}

          {/* Submit */}
          <button
            type="submit"
            disabled={isLoading}
            className={cn(
              'w-full rounded-lg px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition-all duration-150',
              'focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 focus:ring-offset-gray-900',
              isLoading
                ? 'bg-indigo-500 cursor-not-allowed opacity-70'
                : 'bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700',
            )}
          >
            {isLoading ? 'Signing in…' : 'Sign in'}
          </button>
        </form>

        <p className="text-center text-xs text-gray-600">
          Car Phone Call Assistance — Admin Panel
        </p>
      </div>
    </div>
  );
}
