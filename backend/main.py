"""
backend/main.py — FastAPI application entrypoint.

Houses:
  - Async lifespan (startup / shutdown hooks)
  - `GET /api/health` endpoint
  - App factory used by tests and the uvicorn entry point
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import Depends, FastAPI, Request
from pydantic import ValidationError
from starlette.types import ASGIApp, Receive, Scope, Send

from backend.api.arrange import router as arrange_router, set_concurrency_manager
from backend.api.arrange import _status_callback_router
from backend.api.auth import router as auth_router
from backend.api.leads import router as leads_router
from backend.api.media_stream import router as media_stream_router
from backend.catalogue import get_vehicle_by_id, load_catalogue, vehicles
from backend.concurrency import ConcurrencyManager
from backend.config import load_settings

log = logging.getLogger(__name__)

# Module-level reference populated during lifespan startup.
# Available to route handlers and other modules that run inside the event loop.
settings = None  # type: ignore[assignment]

# Module-level pool — populated during lifespan startup.
# Route handlers and other components inside the event loop can import and use it.
_db_pool = None  # type: ignore[assignment]

# Global shutdown event — signalled when the lifespan exits so background tasks exit cleanly.
_sweep_shutdown = asyncio.Event()


def _project_root() -> Path:
    """Absolute path to the project root (one level up from this file)."""
    return Path(__file__).parent.parent.resolve()


# ─── Logging configuration ────────────────────────────────────────────────────


def _configure_logging() -> None:
    """
    Surface INFO-level logs from ``backend.*`` loggers for HITL debugging.

    Why
    ───
    uvicorn's ``log_level="info"`` only configures its own loggers
    (``uvicorn``, ``uvicorn.error``, ``uvicorn.access``).  Our
    ``backend.*`` loggers propagate to the *root* logger, which defaults
    to WARNING — so the Deepgram transcripts, Mercury replies, and
    bridge events emitted at INFO level never reached the console during
    recent HITL calls.  We were debugging half-blind.

    Called from the lifespan so it runs regardless of how the app is
    launched (``uvicorn backend.main:app``, ``run_backend.py``, or
    ``python -m backend.main``).  Idempotent and conservative: only bumps
    the root level if it isn't already lower, and only attaches a handler
    if none exists (so uvicorn's own handler setup on its loggers is
    left alone).
    """
    root = logging.getLogger()
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s")
        )
        root.addHandler(handler)


# ─── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Startup → load settings; bootstrap DB; run migrations; create pool; start sweeper.
    Shutdown → signal background tasks to stop; close pool.
    """
    global settings, _db_pool, _sweep_shutdown
    # Run BEFORE any other startup work so every subsequent log line is
    # visible.  No-op if the root logger is already at INFO with a handler.
    _configure_logging()
    log.info("Starting up car-phone-call-assistance backend …")
    _sweep_shutdown.clear()

    try:
        settings = load_settings()
    except ValidationError as exc:
        errors = "; ".join(
            f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}"
            for e in exc.errors()
        )
        raise ValueError(
            f"Configuration validation failed: {errors}\n"
            "Set the required values in .env or your shell environment."
        ) from exc

    _validate_required_secrets(settings)

    # ── Issue 037: Disclosure asset validation (fail-fast compliance gate) ──
    # The legal-recording disclosure is a legally-required preamble that
    # must play before any customer audio.  If the pre-rendered μ-law
    # asset is missing or empty, the backend MUST refuse to boot rather
    # than fail-open silently with no legal disclosure.  This check runs
    # before any DB work so a misconfigured deployment fails fast with a
    # clear, actionable error message (re-run the disclosure generator).
    from backend.startup import validate_disclosure_assets

    validate_disclosure_assets()
    log.info("Disclosure assets validated — compliance gate passed.")

    # ── Database bootstrap + migrations ─────────────────────────────────────
    from backend.database import bootstrap_admin, bootstrap_db, create_pool, run_migrations

    schema_path = _project_root() / "database" / "app_schema.sql"
    migrations_dir = _project_root() / "database" / "migrations"

    log.info("Checking database at %s …", settings.app_db_path)
    created = await bootstrap_db(
        settings.app_db_path,
        schema_path,
        busy_timeout_seconds=settings.busy_timeout_seconds,
    )
    if created:
        log.info("app.db created from schema — applying migrations next.")
    else:
        log.info("app.db already exists — applying any pending migrations.")

    await run_migrations(
        settings.app_db_path,
        migrations_dir,
        busy_timeout_seconds=settings.busy_timeout_seconds,
    )

    # ── Admin account bootstrap ───────────────────────────────────────────────
    # Seed exactly one admin row from env vars on first boot; no-op on
    # subsequent boots (idempotent — existing row is never overwritten).
    await bootstrap_admin(
        settings.app_db_path,
        settings.admin_email,
        settings.admin_password.get_secret_value(),
        busy_timeout_seconds=settings.busy_timeout_seconds,
    )

    # ── Connection pool ──────────────────────────────────────────────────────
    _db_pool = create_pool(
        db_path=settings.app_db_path,
        busy_timeout_seconds=settings.busy_timeout_seconds,
        pool_size=settings.db_pool_size,
    )
    app.state.pool = _db_pool
    log.info(
        "DB connection pool ready (pool_size=%d, busy_timeout=%.1fs).",
        settings.db_pool_size,
        settings.busy_timeout_seconds,
    )

    # ── Session sweeper — startup purge ──────────────────────────────────────
    from backend.database import sweep_expired

    log.info("Running startup session sweep …")
    try:
        count = await sweep_expired(_db_pool)
        log.info("Startup sweep removed %d expired session(s).", count)
    except Exception as exc:
        log.warning("Startup sweep failed (non-fatal): %s", exc)

    # ── Background sweeper task (runs every hour while the server is alive) ───
    _sweep_task = asyncio.create_task(_run_background_sweeper(_db_pool))

    # ── In-memory vehicle catalogue ─────────────────────────────────────────────
    log.info("Loading vehicle catalogue from %s …", settings.vehicles_db_path)
    load_catalogue(settings.vehicles_db_path)
    log.info(
        "Vehicle catalogue loaded: %d records in memory.",
        len(vehicles),
    )

    # ── In-memory per-IP rate-limit store (Issue 010) ────────────────────────────
    # Dict mapping IP address -> list of request timestamps (seconds since epoch).
    # Used by the rate-limit dependency in backend/api/leads.py to enforce
    # rate_limit_per_ip_per_hour before any DB work.
    app.state.rate_limit_store: dict[str, list[float]] = {}
    # Store the limit value directly so the dependency can read it from app.state
    # without needing to import load_settings (which triggers I/O at import time).
    app.state.rate_limit_limit: int = settings.rate_limit_per_ip_per_hour
    log.info(
        "Rate-limit store initialised (limit=%d req/IP/hour).",
        settings.rate_limit_per_ip_per_hour,
    )

    # ── Issue 013: CORS allow-list ─────────────────────────────────────────────
    # Stored on app.state so LazyCORSMiddleware can read it per-request.
    # If operators forget to set CORS_ALLOWED_ORIGINS, requests from non-local
    # origins will receive no CORS headers and will be blocked by the browser.
    app.state.cors_allowed_origins: list[str] = settings.cors_allowed_origins
    log.info(
        "CORS allow-list initialised (%d origin(s): %s).",
        len(settings.cors_allowed_origins),
        settings.cors_allowed_origins,
    )

    # ── Issue 027/030: Concurrency slot manager ────────────────────────────────
    # Wired into the arrange module so the endpoint can acquire/release slots.
    concurrency_manager = ConcurrencyManager(capacity=settings.max_concurrent_calls)
    set_concurrency_manager(concurrency_manager)
    log.info(
        "ConcurrencyManager initialised (capacity=%d).",
        settings.max_concurrent_calls,
    )

    log.info("All required secrets present — startup complete.")
    yield
    # ── Shutdown ─────────────────────────────────────────────────────────────
    log.info("Shutting down — signalling background tasks …")
    _sweep_shutdown.set()
    if _sweep_task.done():
        log.info("Background sweeper had already exited.")
    else:
        try:
            await asyncio.wait_for(_sweep_task, timeout=5.0)
            log.info("Background sweeper shut down cleanly.")
        except asyncio.TimeoutError:
            log.warning("Background sweeper did not exit in time — cancelling.")
            _sweep_task.cancel()
        except Exception as exc:
            log.warning("Background sweeper raised during shutdown: %s", exc)

    log.info("Shutting down — closing DB pool …")
    await _db_pool.close()
    log.info("Shutdown complete.")


def _validate_required_secrets(cfg) -> None:
    """
    Raise `ValueError` listing every required secret that is blank.
    """
    required = [
        ("DEEPGRAM_API_KEY",   cfg.deepgram_api_key.get_secret_value()  if cfg.deepgram_api_key         else ""),
        ("MERCURY_TWO_API_KEY",cfg.mercury_two_api_key.get_secret_value() if cfg.mercury_two_api_key  else ""),
        ("GEMINI_API_KEY",     cfg.gemini_api_key.get_secret_value()   if cfg.gemini_api_key          else ""),
        ("SARVAM_API_KEY",     cfg.sarvam_api_key.get_secret_value()   if cfg.sarvam_api_key           else ""),
        ("ADMIN_EMAIL",        cfg.admin_email                            if cfg.admin_email              else ""),
        ("ADMIN_PASSWORD",     cfg.admin_password.get_secret_value()   if cfg.admin_password          else ""),
    ]

    missing = [name for name, value in required if not value]
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}.  "
            "Set them in .env or your shell environment before starting the server."
        )


# ─── Background sweeper ──────────────────────────────────────────────────────

_SWEEP_INTERVAL_SECS = 3600  # 1 hour


async def _run_background_sweeper(pool) -> None:
    """
    Low-frequency background task that sweeps expired sessions every hour.

    Runs until ``_sweep_shutdown`` is set or the event loop is cancelled.
    Errors are logged but do not crash the task — the sweeper must be resilient.
    """
    from backend.database import sweep_expired

    log.info("Background session sweeper started (interval=%ds).", _SWEEP_INTERVAL_SECS)
    try:
        while not _sweep_shutdown.is_set():
            try:
                count = await sweep_expired(pool)
                if count > 0:
                    log.info("Background sweep removed %d expired session(s).", count)
                else:
                    log.debug("Background sweep: no expired sessions found.")
            except asyncio.CancelledError:
                raise  # propagate — we are being cancelled on shutdown
            except Exception as exc:
                log.warning("Background sweep error (will retry): %s", exc)

            # Wait for the next interval, but exit early if shutdown is signalled.
            try:
                await asyncio.wait_for(
                    _sweep_shutdown.wait(),
                    timeout=_SWEEP_INTERVAL_SECS,
                )
                # Shutdown was signalled — exit the loop cleanly.
                break
            except asyncio.TimeoutError:
                pass  # timeout expired → next sweep iteration
    finally:
        log.info("Background session sweeper stopped.")


# ─── Custom CORS middleware (pure ASGI) ──────────────────────────────────────


class LazyCORSMiddleware:
    """
    Pure-ASGI CORS middleware — SSE / streaming-safe.

    Operates directly on ``scope``, ``receive``, and ``send`` with zero response
    buffering, so it does not interfere with SSE streams or chunked transfers.

    Reads the allow-list from ``app.state.cors_allowed_origins`` on every request,
    so the allow-list can be changed at runtime without restarting the server.

    Behaviour
    ─────────
    • Preflight OPTIONS (with ``Origin`` header):
        - Origin in allow-list → 200 with full CORS response headers
        - Origin not in allow-list → 400 "Disallowed CORS origin"
    • Non-preflight OPTIONS (no ``Origin`` header):
        - Passthrough to the ASGI app (standard CORS spec — no CORS involvement)
    • Regular requests with ``Origin`` header:
        - Inject ``Access-Control-Allow-Origin`` and
          ``Access-Control-Allow-Credentials`` on the response
        - Add ``Vary: Origin`` to prevent cross-origin cache poisoning
    • Requests without ``Origin`` header:
        - Passthrough without any CORS headers
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            # Pass non-HTTP scopes (websocket, etc.) through unchanged
            await self.app(scope, receive, send)
            return

        method: str = scope["method"]
        headers: list[tuple[bytes, bytes]] = scope["headers"]
        origin: bytes | None = _header_value(headers, b"origin")
        origin_str: str | None = origin.decode() if origin else None

        # Read the runtime allow-list from app state
        allowed_origins: list[str] = []
        if scope.get("app") and hasattr(scope["app"], "state"):
            allowed_origins = getattr(scope["app"].state, "cors_allowed_origins", [])

        # ── Preflight OPTIONS with an Origin header ──────────────────────────
        if method == "OPTIONS" and origin_str:
            if origin_str in allowed_origins:
                await self._send_preflight_response(send, origin_str, method)
            else:
                await self._send_cors_rejection(send)
            return

        # ── All other requests → pass through and inject CORS headers ───────
        await self.app(scope, receive, self._cors_injecting_send(send, origin_str, allowed_origins))

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    async def _send_preflight_response(
        send: Send,
        origin: str,
        method: str,
    ) -> None:
        """Send a bare 200 preflight response with CORS headers."""
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", b"text/plain"),
                (b"access-control-allow-origin", origin.encode()),
                (b"access-control-allow-credentials", b"true"),
                (b"access-control-allow-methods", b"GET, POST, PUT, DELETE, PATCH, OPTIONS"),
                (b"access-control-allow-headers", b"Content-Type, X-CSRF-Token"),
                (b"access-control-max-age", b"3600"),
                (b"vary", b"Origin"),
            ],
        })
        await send({"type": "http.response.body", "body": b""})

    @staticmethod
    async def _send_cors_rejection(send: Send) -> None:
        """Send a 400 response for disallowed CORS origin."""
        body = b"Disallowed CORS origin"
        await send({
            "type": "http.response.start",
            "status": 400,
            "headers": [
                (b"content-type", b"text/plain; charset=utf-8"),
                (b"vary", b"Origin"),
            ],
        })
        await send({"type": "http.response.body", "body": body})

    @staticmethod
    def _cors_injecting_send(
        send: Send,
        origin_str: str | None,
        allowed_origins: list[str],
    ) -> Send:
        """
        Wrap ``send`` to inject CORS headers into the response start event.

        Returns a coroutine-compatible ``send`` callable.
        """
        injected = False

        async def wrapped_send(message: dict[str, Any]) -> None:
            nonlocal injected
            if message["type"] == "http.response.start" and not injected:
                injected = True
                extra_headers: list[tuple[bytes, bytes]] = []
                if origin_str and origin_str in allowed_origins:
                    extra_headers += [
                        (b"access-control-allow-origin", origin_str.encode()),
                        (b"access-control-allow-credentials", b"true"),
                        (b"vary", b"Origin"),
                    ]
                await send({
                    **message,
                    "headers": list(message.get("headers", [])) + extra_headers,
                })
            else:
                await send(message)

        return wrapped_send


# ─── Internal helpers ──────────────────────────────────────────────────────────


def _header_value(headers: list[tuple[bytes, bytes]], key: bytes) -> bytes | None:
    """Case-insensitive lookup of a header value from the ASGI headers list."""
    for k, v in headers:
        if k.lower() == key.lower():
            return v
    return None


# ─── App factory ─────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Car Phone Call Assistance API",
        version="0.1.0",
        lifespan=lifespan,
    )

    # ── Issue 013: Lazy CORS middleware ─────────────────────────────────────
    # Reads allowed_origins from app.state on every request — no import-time
    # side-effects.  The lifespan populates app.state.cors_allowed_origins.
    app.add_middleware(LazyCORSMiddleware)

    @app.get("/api/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/cars", tags=["catalogue"])
    async def list_cars() -> list[dict[str, int | str]]:
        """
        Return the full vehicle catalogue as a list of {id, name} objects.

        This endpoint serves directly from the in-memory copy loaded at startup —
        zero per-request database queries.  Public; no auth required.
        """
        return [{"id": v.id, "name": v.name} for v in vehicles]

    # ── Issue 007: Lead submission endpoint ─────────────────────────────────
    app.include_router(leads_router)

    # ── Issue 012: Admin auth endpoints ─────────────────────────────────────
    app.include_router(auth_router)

    # ── Issue 030: Arrange AI Call endpoint ────────────────────────────────
    app.include_router(arrange_router)
    app.include_router(_status_callback_router)

    # ── Issue 034: Twilio Media Streams inbound reader ─────────────────────
    # Single WebSocket endpoint at /api/twilio/media-stream.  Twilio opens
    # this when the call is answered and TwiML <Connect><Stream> redirects
    # the audio plane here.  This router MUST be included on the same app
    # so its WebSocket shares the same lifecycle / middleware stack as the
    # HTTP routes (notably LazyCORSMiddleware's non-HTTP passthrough for
    # WS scopes, configured in the lifespan above).
    app.include_router(media_stream_router)

    return app


# ─── Shared FastAPI dependencies ──────────────────────────────────────────────

def get_pool(request: Request):
    """
    FastAPI dependency that resolves the DB connection pool from app state.

    Injected into route handlers via ``Depends(get_pool)`` — pattern will be
    replicated across all 12 endpoints.  Tests can override via
    ``app.dependency_overrides[get_pool] = test_pool``.
    """
    return request.app.state.pool


# ─── Module-level app (used by `uvicorn` CLI) ─────────────────────────────────

app = create_app()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    cfg = load_settings()
    uvicorn.run(
        "backend.main:app",
        host=cfg.host,
        port=cfg.port,
        reload=cfg.reload,
    )
