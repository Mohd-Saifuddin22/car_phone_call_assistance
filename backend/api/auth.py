"""
backend/api/auth.py — Admin authentication endpoints (Issues 012, 013).

Login  — POST /api/admin/login
Logout — POST /api/admin/logout

Route Gating (Issue 013)
────────────────────────
``get_current_admin`` is a FastAPI dependency that:
  1. Extracts the ``session_token`` cookie.
  2. Calls ``refresh_session()`` to validate and rolling-refresh the session.
  3. Enforces CSRF on all state-changing methods (POST, PUT, DELETE, PATCH) via
     the ``X-CSRF-Token`` request header.
  4. Returns a 401 Unauthorized on any failure.

Pool injection
──────────────
The pool is injected via FastAPI's dependency system (``Depends``).  A factory
``get_pool`` returns the pool stored in ``app.state`` (set during lifespan startup).
Tests override ``get_pool`` via ``app.dependency_overrides[get_pool] = fake_pool``.
"""

from __future__ import annotations

import secrets

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from backend.database._auth import verify_password
from backend.database._pool import ConnectionPool
from backend.database._sessions import Session, refresh_session

# ─── Constants ────────────────────────────────────────────────────────────────

_SESSION_COOKIE_NAME = "session_token"
_SESSION_COOKIE_PATH = "/"
_BUSY_TIMEOUT = 30.0

# HTTP methods considered state-changing (require CSRF validation).
_STATE_CHANGING_METHODS = frozenset({"POST", "PUT", "DELETE", "PATCH"})
_X_CSRF_HEADER = "x-csrf-token"

# ─── Pool dependency ──────────────────────────────────────────────────────────


def get_pool(request: Request) -> ConnectionPool:
    """
    Resolve the DB connection pool from app state.

    Injected into route handlers via ``Depends(get_pool)``.
    Tests override via ``app.dependency_overrides[get_pool] = fake_pool``.
    """
    return request.app.state.pool


# ─── Route-gating dependency ──────────────────────────────────────────────────


async def get_current_admin(
    request: Request,
    pool: ConnectionPool = Depends(get_pool),
) -> Session:
    """
    FastAPI dependency — enforces session auth and CSRF on state-changing requests.

    Behaviour
    ─────────
    1. Reads the ``session_token`` cookie.
    2. Calls ``refresh_session()`` to validate the token and roll its expiry forward.
    3. For ``POST`` / ``PUT`` / ``DELETE`` / ``PATCH`` requests, validates the
       ``X-CSRF-Token`` request header against the stored ``csrf_token``.
    4. Returns the ``Session`` object on success.
    5. Raises ``HTTPException(401)`` on any failure.

    Note: GET, HEAD, OPTIONS are exempt from CSRF checks.
    """
    session_token: str | None = request.cookies.get(_SESSION_COOKIE_NAME)
    if not session_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
        )

    session = await refresh_session(pool, session_token)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session.",
        )

    # CSRF enforcement — only for state-changing methods
    method = request.method.upper()
    if method in _STATE_CHANGING_METHODS:
        # Read the X-CSRF-Token header; case-sensitive exact match required
        csrf_header_value = request.headers.get(_X_CSRF_HEADER, "").strip()
        if not csrf_header_value:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="X-CSRF-Token header is required.",
            )
        # Use secrets.compare_digest to prevent timing attacks
        if not secrets.compare_digest(csrf_header_value, session.csrf_token):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid CSRF token.",
            )

    return session


# ─── Router ──────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/api/admin", tags=["admin"])

# ─── Request / Response models ────────────────────────────────────────────────


class LoginRequest(BaseModel):
    """Credentials body for POST /api/admin/login."""

    email: str = Field(..., min_length=1, description="Admin email address")
    password: str = Field(..., min_length=1, description="Admin password")


class LoginSuccessResponse(BaseModel):
    """Response body on successful login."""

    message: str = Field(default="Login successful.")
    csrf_token: str = Field(
        ..., description="Per-session CSRF token — attach as X-CSRF-Token header."
    )


class SessionStatusResponse(BaseModel):
    """Response body for GET /api/admin/session."""

    authenticated: bool = Field(..., description="Whether a valid session cookie is present.")
    csrf_token: str | None = Field(
        None,
        description=(
            "Per-session CSRF token. Present only when authenticated is true."
        ),
    )
    email: str | None = Field(
        None,
        description="Admin email address. Present only when authenticated is true.",
    )


# ─── Internal helpers ────────────────────────────────────────────────────────


async def _get_admin_by_email(pool: ConnectionPool, email: str) -> tuple[int, str] | None:
    """
    Look up an admin row by email and return (admin_id, password_hash).

    Returns None if no row matches.
    """
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT id, password_hash FROM admin WHERE email = ?;",
            (email,),
        )
        row = await cursor.fetchone()
        return (row[0], row[1]) if row else None


async def _get_admin_email_by_id(pool: ConnectionPool, admin_id: int) -> str | None:
    """
    Look up an admin row by id and return the email.

    Returns None if no row matches.
    """
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT email FROM admin WHERE id = ?;",
            (admin_id,),
        )
        row = await cursor.fetchone()
        return row[0] if row else None


# ─── Endpoints ──────────────────────────────────────────────────────────────


@router.post(
    "/login",
    response_model=LoginSuccessResponse,
    responses={
        status.HTTP_401_UNAUTHORIZED: {
            "description": "Invalid credentials"
        },
    },
    summary="Admin login",
    description="Authenticate with admin credentials. On success, issues a secure HttpOnly session cookie.",
)
async def login(
    payload: LoginRequest,
    response: Response,
    pool: ConnectionPool = Depends(get_pool),
) -> LoginSuccessResponse:
    """
    Authenticate an admin and establish a server-side session.

    Steps
    ─────
    1. Look up the admin row by email.
    2. Verify the plaintext password against the stored bcrypt hash.
    3. On success, create a server-side session.
    4. Issue the raw token as a fully-hardened HttpOnly cookie:
       - HttpOnly=True  → prevents JavaScript / XSS access
       - Secure=True    → HTTPS-only transit
       - SameSite=None  → required for cross-origin use with allow_credentials
    5. Return the CSRF token in the JSON body.

    Security note
    ─────────────
    Both "admin not found" and "wrong password" return the same 401 message
    to prevent user-enumeration attacks.
    """
    # Import inside handler to avoid circular imports.
    from backend.database._sessions import create_session
    from backend.config import load_settings

    admin = await _get_admin_by_email(pool, payload.email)
    if admin is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    admin_id, stored_hash = admin

    if not verify_password(payload.password, stored_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    settings = load_settings()
    session, raw_token = await create_session(
        pool,
        admin_id,
        expiry_days=settings.session_expiry_days,
    )

    response.set_cookie(
        key=_SESSION_COOKIE_NAME,
        value=raw_token,
        httponly=True,
        secure=True,
        samesite="none",
        path=_SESSION_COOKIE_PATH,
    )

    return LoginSuccessResponse(csrf_token=session.csrf_token)


@router.post(
    "/logout",
    summary="Admin logout",
    description=(
        "Revoke the current session and clear the session cookie. "
        "Always returns 200 so a logout request cannot be distinguished from "
        "a logout of an already-revoked/expired session."
    ),
)
async def logout(
    response: Response,
    pool: ConnectionPool = Depends(get_pool),
    session_token: str | None = Cookie(default=None, alias=_SESSION_COOKIE_NAME),
) -> dict[str, str]:
    """
    Revoke the session identified by the incoming cookie and clear the cookie.

    Idempotent: if no cookie is present, no session is revoked, but the
    cleared-cookie response is still returned (prevents state leakage).
    """
    from backend.database._sessions import revoke_session

    if session_token:
        await revoke_session(pool, session_token)

    response.delete_cookie(
        key=_SESSION_COOKIE_NAME,
        path=_SESSION_COOKIE_PATH,
        httponly=True,
        secure=True,
        samesite="none",
    )
    return {"message": "Logged out."}


# ─── Internal session-probe helper ─────────────────────────────────────────────
# Defined BEFORE get_session so FastAPI can resolve Depends(_try_get_session)
# at function-signature parse time.


async def _try_get_session(
    request: Request,
    pool: ConnectionPool = Depends(get_pool),
) -> Session | None:
    """
    Lightweight session probe — reads the cookie and validates it without
    enforcing CSRF or raising 401.  Used exclusively by GET /session.
    """
    session_token: str | None = request.cookies.get(_SESSION_COOKIE_NAME)
    if not session_token:
        return None
    return await refresh_session(pool, session_token)


@router.get(
    "/session",
    response_model=SessionStatusResponse,
    summary="Current session status",
    description=(
        "CSRF-exempt GET endpoint for the SPA to securely rehydrate auth state on mount. "
        "Validates the session cookie server-side (HttpOnly, invisible to JS) and "
        "returns the CSRF token and admin email so the SPA can restore the session "
        "without re-prompting for credentials."
    ),
)
async def get_session(
    session: Session | None = Depends(_try_get_session),
    pool: ConnectionPool = Depends(get_pool),
) -> SessionStatusResponse:
    """
    Return the current auth state.

    Unlike ``get_current_admin``, this endpoint is **never** protected — a missing
    or expired session simply returns ``{authenticated: false}``.  This allows the
    SPA to probe the session without needing a CSRF token.

    The endpoint also never enforces CSRF (it bypasses ``get_current_admin`` and uses
    the lighter ``_try_get_session`` helper directly).
    """
    if session is None:
        return SessionStatusResponse(authenticated=False)

    # Look up the admin email from the session's admin_id.
    email = await _get_admin_email_by_id(pool, session.admin_id)

    return SessionStatusResponse(
        authenticated=True,
        csrf_token=session.csrf_token,
        email=email,
    )
