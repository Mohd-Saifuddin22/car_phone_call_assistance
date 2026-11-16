"""
Smoke tests for the FastAPI application lifecycle.

Run with:
    pytest backend/tests/test_health.py -v

Or from the project root:
    pytest -v

Tests use `httpx.AsyncClient` wrapped in `asgi_lifespan.LifespanManager` to
correctly drive the FastAPI lifespan without the TestClient subprocess issues
seen with older httpx versions.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from backend.config import load_settings
from backend.main import _validate_required_secrets, create_app


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    """
    Always start with a fresh settings cache so tests don't bleed into each other.
    No module may call load_settings() at import time; this fixture ensures every
    test starts with a clean slate.
    """
    load_settings.cache_clear()
    yield
    load_settings.cache_clear()


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """
    Async HTTP client wired to the full FastAPI lifespan via LifespanManager.

    APP_DB_PATH is redirected to a temp file so the real database/ directory is
    never touched.  The admin table is seeded from env vars in the temp DB.

    Usage:
        async def test_something(client: AsyncClient) -> None:
            response = await client.get("/api/health")
    """
    import shutil

    tmpdir = Path(tempfile.mkdtemp(prefix="car_test_health_"))
    app_db = tmpdir / "app.db"

    load_settings.cache_clear()
    old_env = os.environ.get("APP_DB_PATH")
    os.environ["APP_DB_PATH"] = str(app_db)

    app = create_app()
    try:
        async with LifespanManager(app):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as ac:
                yield ac
    finally:
        if old_env is None:
            os.environ.pop("APP_DB_PATH", None)
        else:
            os.environ["APP_DB_PATH"] = old_env
        load_settings.cache_clear()
        shutil.rmtree(tmpdir, ignore_errors=True)


# ─── Tests ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_returns_200(client: AsyncClient) -> None:
    response = await client.get("/api/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_payload_is_json_ok(client: AsyncClient) -> None:
    response = await client.get("/api/health")
    assert response.json() == {"status": "ok"}


# ─── Secrets validation — unit ───────────────────────────────────────────────

def test_validate_required_secrets_raises_when_secrets_missing() -> None:
    """
    Verify that _validate_required_secrets raises ValueError with a readable
    message listing every missing key, not an opaque pydantic traceback.
    """
    blank_settings = MagicMock()
    blank_settings.deepgram_api_key.get_secret_value.return_value = ""
    blank_settings.mercury_two_api_key.get_secret_value.return_value = ""
    blank_settings.gemini_api_key.get_secret_value.return_value = ""
    blank_settings.sarvam_api_key.get_secret_value.return_value = ""
    blank_settings.admin_email = ""
    blank_settings.admin_password.get_secret_value.return_value = ""

    with pytest.raises(ValueError) as exc_info:
        _validate_required_secrets(blank_settings)

    msg = str(exc_info.value)
    assert "Missing required environment variables" in msg
    for key in (
        "DEEPGRAM_API_KEY",
        "MERCURY_TWO_API_KEY",
        "GEMINI_API_KEY",
        "SARVAM_API_KEY",
        "ADMIN_EMAIL",
        "ADMIN_PASSWORD",
    ):
        assert key in msg


def test_validate_required_secrets_succeeds_when_secrets_present() -> None:
    """No exception when every required secret is non-empty."""
    populated_settings = MagicMock()
    populated_settings.deepgram_api_key.get_secret_value.return_value = "key"
    populated_settings.mercury_two_api_key.get_secret_value.return_value = "key"
    populated_settings.gemini_api_key.get_secret_value.return_value = "key"
    populated_settings.sarvam_api_key.get_secret_value.return_value = "key"
    populated_settings.admin_email = "admin@example.com"
    populated_settings.admin_password.get_secret_value.return_value = "password"

    # Should not raise
    _validate_required_secrets(populated_settings)


# ─── Lifespan ValidationError → ValueError — end-to-end ──────────────────────

@pytest.mark.asyncio
async def test_lifespan_converts_pydantic_validation_error_to_friendly_valueerror() -> None:
    """
    When pydantic.ValidationError fires during startup (e.g. required secrets
    absent with _env_file=None), the lifespan handler must catch it and re-raise
    a ValueError with a human-readable message — not let the raw pydantic
    traceback bubble up to the caller.
    """
    import unittest.mock as mock
    from backend import main as main_module

    load_settings.cache_clear()

    fake_validation_error = ValidationError.from_exception_data(
        title="Settings",
        line_errors=[
            {
                "type": "missing",
                "loc": ("deepgram_api_key",),
                "msg": "Field required",
                "input": {},
            },
        ],
    )

    with mock.patch.object(
        main_module, "load_settings", side_effect=fake_validation_error
    ):
        app = create_app()
        with pytest.raises(ValueError) as exc_info:
            async with LifespanManager(app):
                pass  # lifespan runs on context manager entry

    msg = str(exc_info.value)
    assert "Configuration validation failed" in msg


# ─── CORS_ALLOWED_ORIGINS parsing ─────────────────────────────────────────────

def test_cors_allowed_origins_accepts_comma_separated_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify that the field_validator splits a comma-separated CORS_ALLOWED_ORIGINS
    string into a list[str] without crashing.

    Operators following the .env.example documentation (comma-separated format)
    must be able to boot the server without a SettingsError.
    """
    load_settings.cache_clear()

    # Simulate the documented .env format: CORS_ALLOWED_ORIGINS=a.com,b.com
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://admin.dealership.local,http://localhost:5173,  https://spa.example.com  ")

    from backend.config import Settings

    settings = Settings()
    assert settings.cors_allowed_origins == [
        "https://admin.dealership.local",
        "http://localhost:5173",
        "https://spa.example.com",
    ]


def test_cors_allowed_origins_accepts_json_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """JSON-style list (pydantic default) must still work."""
    load_settings.cache_clear()
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", '["https://a.com","https://b.com"]')

    from backend.config import Settings

    settings = Settings()
    assert settings.cors_allowed_origins == ["https://a.com", "https://b.com"]


def test_cors_allowed_origins_empty_string_becomes_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blank / empty env var must resolve to an empty list, not crash."""
    load_settings.cache_clear()
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "")

    from backend.config import Settings

    settings = Settings()
    assert settings.cors_allowed_origins == []


def test_cors_allowed_origins_omitted_becomes_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """When CORS_ALLOWED_ORIGINS is not set at all, default to an empty list."""
    load_settings.cache_clear()
    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)

    from backend.config import Settings

    settings = Settings()
    assert settings.cors_allowed_origins == []


def test_other_env_vars_not_ignored_by_cors_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Regression guard: overriding CORS_ALLOWED_ORIGINS must not silently drop other
    environment variables (e.g. PORT, APP_DB_PATH).

    The previous CORSEnvSource.__call__ implementation returned only
    {'cors_allowed_origins': ...}, causing all other env vars to fall through to
    their defaults.  This test would have caught that.
    """
    load_settings.cache_clear()
    monkeypatch.setenv("PORT", "9001")
    monkeypatch.setenv("RATE_LIMIT_PER_IP_PER_HOUR", "99")

    from backend.config import Settings

    settings = Settings()
    assert settings.port == 9001, "PORT env var must be respected"
    assert settings.rate_limit_per_ip_per_hour == 99, "RATE_LIMIT_PER_IP_PER_HOUR env var must be respected"
