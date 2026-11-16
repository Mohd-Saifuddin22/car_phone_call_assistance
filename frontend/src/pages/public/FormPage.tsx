import { useState, useEffect, useRef } from 'react';
import { api, ApiError } from '@/lib/apiClient';
import { CityCombobox, type CityComboboxHandle } from '@/components/CityCombobox';
import { cn } from '@/lib/utils';

// ─── Types ────────────────────────────────────────────────────────────────────

interface Car {
  id: number;
  name: string;
}

interface LeadPayload {
  name: string;
  phone: string;
  email: string;
  city: string;
  language: string;
  vehicle_id: number | null;
}

interface FieldErrors {
  [field: string]: string;
}

// ─── Validation helpers ───────────────────────────────────────────────────────

const PHONE_RE = /^[6-9]\d{9}$/;

function validatePhone(raw: string): string | null {
  const digits = raw.replace(/\D/g, '');
  if (!digits) return 'Phone number is required.';
  if (!PHONE_RE.test(digits)) {
    return 'Enter a valid 10-digit Indian mobile number (starts with 6–9).';
  }
  return null;
}

function validateRequired(value: string, label: string): string | null {
  return value.trim() ? null : `${label} is required.`;
}

function validateEmail(value: string): string | null {
  if (!value.trim()) return 'Email is required.';
  // RFC-5322 simplified — catches the common mistakes without being overly strict.
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(value.trim())) {
    return 'Enter a valid email address.';
  }
  return null;
}

// ─── Component ────────────────────────────────────────────────────────────────

type SubmitState =
  | { phase: 'idle' }
  | { phase: 'loading' }
  | { phase: 'success'; message: string }
  | { phase: 'error'; message: string };

export function FormPage() {
  // ── Field values ─────────────────────────────────────────────────────────
  const [name, setName] = useState('');
  const [phone, setPhone] = useState('');
  const [phoneTouched, setPhoneTouched] = useState(false);
  const [email, setEmail] = useState('');
  const [city, setCity] = useState('');
  const [language, setLanguage] = useState('');
  const [vehicleId, setVehicleId] = useState<string>(''); // '' = not selected

  // ── Server-sourced data ───────────────────────────────────────────────────
  const [cars, setCars] = useState<Car[]>([]);
  const [carsError, setCarsError] = useState(false);

  // ── Inline field errors (live validation) ─────────────────────────────────
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({});

  // ── Submission state ──────────────────────────────────────────────────────
  const [submitState, setSubmitState] = useState<SubmitState>({ phase: 'idle' });

  const cityRef = useRef<CityComboboxHandle>(null);

  // ── Fetch car catalogue on mount ──────────────────────────────────────────
  useEffect(() => {
    api
      .get<Car[]>('/api/cars')
      .then(setCars)
      .catch(() => setCarsError(true));
  }, []);

  // ── Inline phone validation (triggers after first blur) ───────────────────
  const phoneError = phoneTouched ? validatePhone(phone) : null;
  /** Unified phone error — prioritises server-side submit error over live blur error. */
  const displayedPhoneError = fieldErrors.phone ?? phoneError ?? undefined;

  // ── Form validity (used to gate submission) ───────────────────────────────
  function computeFieldErrors(): FieldErrors {
    const errors: FieldErrors = {};
    const nameErr = validateRequired(name, 'Name');
    if (nameErr) errors.name = nameErr;

    const phoneErr = validatePhone(phone);
    if (phoneErr) errors.phone = phoneErr;

    const emailErr = validateEmail(email);
    if (emailErr) errors.email = emailErr;

    const cityErr = validateRequired(city, 'City');
    if (cityErr) errors.city = cityErr;

    if (!language) errors.language = 'Please select a language.';

    return errors;
  }

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();

    // Mark all fields touched before validating
    setPhoneTouched(true);
    const errors = computeFieldErrors();
    setFieldErrors(errors);

    if (Object.keys(errors).length > 0) return;

    const payload: LeadPayload = {
      name: name.trim(),
      phone: phone.replace(/\D/g, ''),
      email: email.trim().toLowerCase(),
      city: city.trim(),
      language,
      vehicle_id: vehicleId ? Number(vehicleId) : null,
    };

    setSubmitState({ phase: 'loading' });

    try {
      const response = await api.post<{ message: string }>('/api/leads', payload);
      setSubmitState({ phase: 'success', message: response.message });
    } catch (err) {
      if (err instanceof ApiError) {
        if (err.status === 429) {
          setSubmitState({
            phase: 'error',
            message: 'Too many requests. Please try again later.',
          });
        } else if (err.status === 422) {
          const detail = err.detail;

          if (Array.isArray(detail)) {
            // Pydantic default validation array shape:
            // [{ loc: ["body", "field_name"], msg: "..." }]
            const errors: FieldErrors = {};
            for (const item of detail) {
              if (
                item &&
                typeof item === 'object' &&
                'loc' in item &&
                'msg' in item
              ) {
                const loc = (item as { loc: string[] }).loc;
                const field = loc[loc.length - 1]; // last segment is the field name
                errors[field] = (item as { msg: string }).msg;
              }
            }
            setFieldErrors(errors);
            setSubmitState({
              phase: 'error',
              message: 'Please correct the errors below.',
            });
          } else if (detail && typeof detail === 'object') {
            // Custom dict shape: { "field_name": "Error message" }
            setFieldErrors(detail as FieldErrors);
            setSubmitState({
              phase: 'error',
              message: 'Please correct the errors below.',
            });
          } else {
            // Fallback: plain string or undefined
            setSubmitState({
              phase: 'error',
              message:
                typeof detail === 'string' && detail
                  ? detail
                  : 'Validation failed. Please check your input.',
            });
          }
        } else {
          setSubmitState({
            phase: 'error',
            message: err.message,
          });
        }
      } else {
        setSubmitState({
          phase: 'error',
          message: 'Something went wrong. Please try again.',
        });
      }
    }
  }

  const isLoading = submitState.phase === 'loading';
  const isSuccess = submitState.phase === 'success';

  // ── Success state ─────────────────────────────────────────────────────────
  if (isSuccess) {
    return (
      <div className="mx-auto max-w-xl space-y-6 rounded-xl border border-green-200 bg-green-50 p-8 text-center shadow-sm">
        <div className="flex justify-center">
          <svg
            className="h-12 w-12 text-green-500"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={1.5}
            aria-hidden="true"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
        </div>
        <h2 className="text-2xl font-bold text-green-800">Thank you!</h2>
        <p className="text-green-700">{submitState.message}</p>
        <button
          type="button"
          onClick={() => {
            setSubmitState({ phase: 'idle' });
            setName('');
            setPhone('');
            setEmail('');
            setCity('');
            setLanguage('');
            setVehicleId('');
            setPhoneTouched(false);
            setFieldErrors({});
            cityRef.current?.focus();
          }}
          className="rounded-lg border border-green-300 bg-white px-5 py-2.5 text-sm font-medium text-green-700 shadow-sm hover:bg-green-100 focus:outline-none focus:ring-2 focus:ring-green-500 transition-colors"
        >
          Submit another request
        </button>
      </div>
    );
  }

  // ── Form ─────────────────────────────────────────────────────────────────
  return (
    <div className="mx-auto max-w-xl space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Request a Callback</h1>
        <p className="mt-2 text-gray-600">
          Fill in your details and our team will call you shortly.
        </p>
      </div>

      <form
        onSubmit={handleSubmit}
        noValidate
        className="space-y-5 rounded-xl border border-gray-200 bg-white p-6 shadow-sm"
      >
        {/* ── Name ────────────────────────────────────────────────────────── */}
        <div>
          <label
            htmlFor="name"
            className="block text-sm font-medium text-gray-700"
          >
            Name <span className="ml-1 text-red-500" aria-hidden="true">*</span>
          </label>
          <input
            type="text"
            id="name"
            name="name"
            value={name}
            onChange={(e) => {
              setName(e.target.value);
              if (fieldErrors.name) {
                setFieldErrors((prev) => ({ ...prev, name: '' }));
              }
            }}
            disabled={isLoading}
            autoComplete="name"
            aria-required="true"
            aria-describedby={fieldErrors.name ? 'name-error' : undefined}
            className={cn(
              'mt-1 block w-full rounded-md border px-3 py-2 text-sm shadow-sm',
              'focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500',
              'transition-colors duration-150',
              fieldErrors.name
                ? 'border-red-400 bg-red-50 text-red-900 focus:border-red-500 focus:ring-red-500'
                : 'border-gray-300 bg-white text-gray-900 placeholder-gray-400',
              isLoading && 'cursor-not-allowed bg-gray-100',
            )}
            placeholder="Rahul Sharma"
          />
          {fieldErrors.name && (
            <p id="name-error" role="alert" className="mt-1 text-sm text-red-600">
              {fieldErrors.name}
            </p>
          )}
        </div>

        {/* ── Phone ───────────────────────────────────────────────────────── */}
        <div>
          <label
            htmlFor="phone"
            className="block text-sm font-medium text-gray-700"
          >
            Phone <span className="ml-1 text-red-500" aria-hidden="true">*</span>
          </label>
          <div className="relative">
            <span className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3 text-sm text-gray-400">
              +91
            </span>
            <input
              type="tel"
              id="phone"
              name="phone"
              value={phone}
              onChange={(e) => {
                // Strip non-digits and cap at 10
                const digits = e.target.value.replace(/\D/g, '').slice(0, 10);
                setPhone(digits);
                if (fieldErrors.phone) {
                  setFieldErrors((prev) => ({ ...prev, phone: '' }));
                }
              }}
              onBlur={() => {
                setPhoneTouched(true);
                setFieldErrors((prev) => ({ ...prev, phone: '' }));
              }}
              disabled={isLoading}
              autoComplete="tel"
              aria-required="true"
              aria-describedby={displayedPhoneError ? 'phone-error' : undefined}
              maxLength={10}
              placeholder="9876543210"
              className={cn(
                'mt-1 block w-full rounded-md border py-2 pl-10 pr-3 text-sm shadow-sm',
                'focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500',
                'transition-colors duration-150',
                displayedPhoneError
                  ? 'border-red-400 bg-red-50 text-red-900 placeholder-red-300 focus:border-red-500 focus:ring-red-500'
                  : 'border-gray-300 bg-white text-gray-900 placeholder-gray-400',
                isLoading && 'cursor-not-allowed bg-gray-100',
              )}
            />
          </div>
          {displayedPhoneError && (
            <p id="phone-error" role="alert" className="mt-1 text-sm text-red-600">
              {displayedPhoneError}
            </p>
          )}
        </div>

        {/* ── Email ───────────────────────────────────────────────────────── */}
        <div>
          <label
            htmlFor="email"
            className="block text-sm font-medium text-gray-700"
          >
            Email <span className="ml-1 text-red-500" aria-hidden="true">*</span>
          </label>
          <input
            type="email"
            id="email"
            name="email"
            value={email}
            onChange={(e) => {
              setEmail(e.target.value);
              if (fieldErrors.email) {
                setFieldErrors((prev) => ({ ...prev, email: '' }));
              }
            }}
            disabled={isLoading}
            autoComplete="email"
            aria-required="true"
            aria-describedby={fieldErrors.email ? 'email-error' : undefined}
            className={cn(
              'mt-1 block w-full rounded-md border px-3 py-2 text-sm shadow-sm',
              'focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500',
              'transition-colors duration-150',
              fieldErrors.email
                ? 'border-red-400 bg-red-50 text-red-900 focus:border-red-500 focus:ring-red-500'
                : 'border-gray-300 bg-white text-gray-900 placeholder-gray-400',
              isLoading && 'cursor-not-allowed bg-gray-100',
            )}
            placeholder="rahul@example.com"
          />
          {fieldErrors.email && (
            <p id="email-error" role="alert" className="mt-1 text-sm text-red-600">
              {fieldErrors.email}
            </p>
          )}
        </div>

        {/* ── City ────────────────────────────────────────────────────────── */}
        <CityCombobox
          ref={cityRef}
          id="city"
          name="city"
          label="Current City"
          required
          value={city}
          onChange={(val) => {
            setCity(val);
            if (fieldErrors.city) {
              setFieldErrors((prev) => ({ ...prev, city: '' }));
            }
          }}
          error={fieldErrors.city}
          disabled={isLoading}
        />

        {/* ── Language ────────────────────────────────────────────────────── */}
        <fieldset>
          <legend className="block text-sm font-medium text-gray-700">
            Preferred Language
            <span className="ml-1 text-red-500" aria-hidden="true">*</span>
          </legend>
          <div className="mt-2 flex gap-4">
            {(['English', 'Hinglish'] as const).map((lang) => (
              <label
                key={lang}
                htmlFor={`lang-${lang.toLowerCase()}`}
                className="flex cursor-pointer items-center gap-2 text-sm text-gray-700"
              >
                <input
                  type="radio"
                  id={`lang-${lang.toLowerCase()}`}
                  name="language"
                  value={lang}
                  checked={language === lang}
                  onChange={() => {
                    setLanguage(lang);
                    if (fieldErrors.language) {
                      setFieldErrors((prev) => ({ ...prev, language: '' }));
                    }
                  }}
                  disabled={isLoading}
                  aria-required="true"
                  className="h-4 w-4 border-gray-300 text-indigo-600 focus:ring-indigo-500"
                />
                {lang}
              </label>
            ))}
          </div>
          {fieldErrors.language && (
            <p role="alert" className="mt-1 text-sm text-red-600">
              {fieldErrors.language}
            </p>
          )}
        </fieldset>

        {/* ── Car of Interest ─────────────────────────────────────────────── */}
        <div>
          <label
            htmlFor="vehicle_id"
            className="block text-sm font-medium text-gray-700"
          >
            Car of Interest
            <span className="ml-1 text-xs font-normal text-gray-400">
              (optional)
            </span>
          </label>
          {carsError ? (
            <p className="mt-1 text-sm text-gray-500">
              Could not load vehicle list. You may continue without selecting.
            </p>
          ) : (
            <select
              id="vehicle_id"
              name="vehicle_id"
              value={vehicleId}
              onChange={(e) => setVehicleId(e.target.value)}
              disabled={isLoading}
              className={cn(
                'mt-1 block w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm shadow-sm',
                'focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500',
                'transition-colors duration-150',
                isLoading && 'cursor-not-allowed bg-gray-100 text-gray-500',
              )}
            >
              <option value="">— Not specified —</option>
              {cars.map((car) => (
                <option key={car.id} value={String(car.id)}>
                  {car.name}
                </option>
              ))}
            </select>
          )}
        </div>

        {/* ── Global submit error ─────────────────────────────────────────── */}
        {submitState.phase === 'error' && (
          <div
            role="alert"
            className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700"
          >
            {submitState.message}
          </div>
        )}

        {/* ── Submit ──────────────────────────────────────────────────────── */}
        <button
          type="submit"
          disabled={isLoading}
          className={cn(
            'w-full rounded-lg px-6 py-3 text-sm font-semibold text-white shadow-sm transition-all duration-150',
            'focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2',
            isLoading
              ? 'bg-indigo-400 cursor-not-allowed'
              : 'bg-indigo-600 hover:bg-indigo-700 active:bg-indigo-800',
          )}
        >
          {isLoading ? 'Submitting…' : 'Request Callback'}
        </button>
      </form>
    </div>
  );
}
