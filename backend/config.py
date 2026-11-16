"""
Configuration loader — single source of truth for all settings.

Importing this module does NOT trigger .env validation.  Settings are loaded
lazily inside the FastAPI lifespan startup event via `load_settings()`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from typing import Annotated

from pydantic import BeforeValidator, Field, SecretStr, model_validator
import pydantic_settings
from pydantic_settings import BaseSettings, SettingsConfigDict


def _comma_separated_to_list(value: str | list[str] | None) -> list[str]:
    """
    Convert a raw CORS_ALLOWED_ORIGINS value to a list of origin strings.

    Accepts:
      - comma-separated str  : "https://a.com, http://b.com"
      - JSON-array str        : '["https://a.com","https://b.com"]'
      - plain list            : already-parsed Python list
      - None / empty str      : []
    """
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        # Try JSON first (allows explicit JSON-array in env var)
        if stripped.startswith("["):
            import json

            return json.loads(stripped)
        # Fall back to comma-separated
        return [item.strip() for item in stripped.split(",") if item.strip()]
    return list(value)

# ─── Types ────────────────────────────────────────────────────────────────────

Language = Literal["English", "Hinglish"]


# ─── Path helpers ─────────────────────────────────────────────────────────────

def _project_root() -> Path:
    """Absolute path to the project root (one level up from this file)."""
    return Path(__file__).parent.parent.resolve()


# ─── Settings ─────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """
    All configuration tunables for the car phone-call assistant backend.

    Defaults are documented inline.  Missing *required* secrets (API keys,
    admin credentials) raise a clear error at startup.
    """

    model_config = SettingsConfigDict(
        env_file=str(_project_root() / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,  # allow raw env-string assignment before type coercion
    )

    # ── Required secrets ───────────────────────────────────────────────────────

    deepgram_api_key: SecretStr = Field(
        description="Deepgram API key for Flux STT"
    )
    mercury_two_api_key: SecretStr = Field(
        description="Inception Labs Mercury-2 API key"
    )
    gemini_api_key: SecretStr = Field(
        description="Google Gemini API key (Matcher / preference extractor)"
    )
    gemini_model: str = Field(
        default="gemini-3.1-flash-lite",
        description=(
            "Gemini model ID used for preference extraction and blurb formatting. "
            "Must support generateContent. Verify via: "
            "GET https://generativelanguage.googleapis.com/v1beta/models"
        ),
    )
    sarvam_api_key: SecretStr = Field(
        description="Sarvam AI API key for Bulbul v3 TTS"
    )
    sarvam_tts_speaker: str = Field(
        default="priya",
        description=(
            "Sarvam Bulbul v3 speaker voice. Configurable to A/B test voices "
            "without redeploying. Speaker must be valid for the bulbul:v3 model. "
            "Default is lowercase ``priya`` to match the API's accepted casing "
            "(Sarvam returns HTTP 422 if the speaker is not lowercase)."
        ),
    )

    admin_email: str = Field(
        description="Admin login email address (also used for seeding)"
    )
    admin_password: SecretStr = Field(
        description="Admin login password (hashed on first boot; never stored in plain text)"
    )

    # ── Twilio ────────────────────────────────────────────────────────────────

    twilio_account_sid: str = Field(
        default="",
        description="Twilio Account SID"
    )
    twilio_auth_token: SecretStr = Field(
        default=SecretStr(""),
        description="Twilio Auth Token"
    )
    twilio_phone_number: str = Field(
        default="",
        description="Twilio outbound phone number (E.164)"
    )

    # ── Call-control tunables ─────────────────────────────────────────────────

    max_concurrent_calls: int = Field(
        default=1,
        description="Maximum simultaneous live calls (default 1, matching trial account)"
    )
    max_attempts: int = Field(
        default=3,
        description="Max retry attempts per lead before Retry button disables"
    )
    max_call_duration_seconds: int = Field(
        default=720,
        description="~12-minute max call duration (seconds); triggers wrap-up backstop"
    )
    poll_interval_seconds: float = Field(
        default=5.0,
        description="Twilio status-poll interval (seconds)"
    )

    # ── Resilience / watchdog tunables ────────────────────────────────────────

    stt_no_first_response_timeout: float = Field(
        default=3.0,
        description="STT no-first-response timeout (seconds); triggers one retry"
    )
    stt_midstream_stall_timeout: float = Field(
        default=1.5,
        description="STT mid-stream stall timeout (seconds); triggers one retry"
    )
    stt_reconnect_attempts: int = Field(
        default=2,
        description="Max STT socket reconnect attempts before falling through to apology"
    )
    busy_timeout_seconds: float = Field(
        default=5.0,
        description="SQLite busy_timeout PRAGMA (seconds); how long to wait on lock"
    )

    # ── Deepgram Flux STT thresholds ───────────────────────────────────────────

    deepgram_eager_eot_threshold: float = Field(
        default=0.4,
        description="eager_eot_threshold — enables eager pair (v2 only)"
    )
    deepgram_eot_threshold: float = Field(
        default=0.7,
        description="eot_threshold — politeness / allowed-to-speak dial"
    )
    deepgram_eot_timeout_ms: int = Field(
        default=5500,
        description="eot_timeout_ms — silence backstop (milliseconds)"
    )

    # ── Rate limiting ─────────────────────────────────────────────────────────

    rate_limit_per_ip_per_hour: int = Field(
        default=5,
        description="Max form submissions per IP per hour"
    )

    # ── Database ──────────────────────────────────────────────────────────────

    app_db_path: Path | None = Field(
        default=None,
        description="Path to the operational SQLite database (absolute or relative to project root)"
    )
    vehicles_db_path: Path | None = Field(
        default=None,
        description="Path to the read-only car catalogue SQLite database (absolute or relative to project root)"
    )
    db_pool_size: int = Field(
        default=5,
        description="SQLite connection pool size (connections are borrowed per operation)"
    )

    # ── Sessions / auth ───────────────────────────────────────────────────────

    session_expiry_days: int = Field(
        default=7,
        description="Rolling session expiry window (days); refreshed on activity"
    )
    session_token_hash_algorithm: str = Field(
        default="sha256",
        description="Hash algorithm for session tokens"
    )

    # ── CORS ─────────────────────────────────────────────────────────────────

    cors_allowed_origins: Annotated[list[str], BeforeValidator(_comma_separated_to_list)] = Field(
        default=[],
        description=(
            "Comma-separated list of allowed origin URLs for CORS. "
            "Requests from origins not in this list are rejected. "
            "Example: https://admin.dealership.local,http://localhost:5173"
        ),
    )

    # ── Recordings ───────────────────────────────────────────────────────────

    recordings_root: Path | None = Field(
        default=None,
        description="Root directory for call recordings (absolute or relative to project root)"
    )

    # ── Server ────────────────────────────────────────────────────────────────

    host: str = Field(default="0.0.0.0", description="Bind host")
    port: int = Field(default=8000, description="Bind port")
    reload: bool = Field(default=False, description="Enable FastAPI auto-reload")

    # ── Telephony tunnel (MVP only — ngrok or equivalent) ────────────────────

    public_hostname: str = Field(
        default="",
        description="Public hostname for Twilio webhooks (e.g. ngrok URL); empty = dev mode"
    )

    # ── Custom env source that never JSON-decodes CORS_ALLOWED_ORIGINS ─────────────
    # pydantic_settings 2.x EnvSettingsSource calls json.loads() for list/dict
    # fields unconditionally.  We create our own source that:
    #   1. Reads the raw env value for cors_allowed_origins
    #   2. Splits comma-separated strings into a list
    #   3. Never attempts JSON decode on that field
    # The source is plugged in via settings_customise_sources below.

    @classmethod
    def _build_cors_env_source(cls, **kwargs: Any) -> type:  # type: ignore[type-arg]
        """
        Return a subclass of EnvSettingsSource whose __call__ intercepts
        cors_allowed_origins before pydantic_settings' hard-coded json.loads()
        crashes on a comma-separated string.
        """
        from pydantic_settings.sources import EnvSettingsSource

        # ── inner subclass ────────────────────────────────────────────────────
        class CORSEnvSource(EnvSettingsSource):  # type: ignore[type-arg]
            """
            EnvSettingsSource subclass that handles CORS_ALLOWED_ORIGINS directly.

            We override prepare_field_value to intercept ONLY the cors_allowed_origins
            field: return the raw string so the BeforeValidator can split it.
            All other fields pass through to the parent unchanged.
            """

            __name__ = "CORSEnvSource"

            def prepare_field_value(self, field_name, field, value, value_is_complex):  # type: ignore[type-arg]
                if field_name == "cors_allowed_origins":
                    # Return the raw string so the Annotated BeforeValidator handles it.
                    # Never pass it to the parent's json.loads() path.
                    return value
                return super().prepare_field_value(field_name, field, value, value_is_complex)

        return CORSEnvSource(
            cls,
            case_sensitive=kwargs.get("_case_sensitive"),
            env_prefix=kwargs.get("_env_prefix"),
            env_nested_delimiter=kwargs.get("_env_nested_delimiter"),
            env_ignore_empty=kwargs.get("_env_ignore_empty"),
            env_parse_none_str=kwargs.get("_env_parse_none_str"),
            env_parse_enums=kwargs.get("_env_parse_enums"),
        )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type,  # type: ignore[type-arg]
        init_settings: pydantic_settings.sources.InitSettingsSource,  # type: ignore[type-arg]
        env_settings: pydantic_settings.sources.EnvSettingsSource,  # type: ignore[type-arg]
        dotenv_settings: pydantic_settings.sources.DotEnvSettingsSource,  # type: ignore[type-arg]
        file_secret_settings: pydantic_settings.sources.SecretsSettingsSource,  # type: ignore[type-arg]
    ) -> tuple[pydantic_settings.sources.PydanticBaseSettingsSource, ...]:  # type: ignore[type-arg]
        # Replace the standard EnvSettingsSource with our CORS-aware one
        import inspect

        kwargs: dict[str, Any] = {}
        for name in inspect.signature(pydantic_settings.sources.EnvSettingsSource.__init__).parameters:  # type: ignore[arg-type]
            if hasattr(env_settings, name):
                kwargs[name] = getattr(env_settings, name)
        wrapped_env = cls._build_cors_env_source(**kwargs)
        return init_settings, wrapped_env, dotenv_settings, file_secret_settings

    # ── Post-initialization normalisation ────────────────────────────────────────

    @model_validator(mode="after")
    def _resolve_relative_paths(self) -> "Settings":
        """
        Resolve relative paths to project root; leave absolute paths untouched.
        Runs after all field validation so defaults are already populated.
        """
        root = _project_root()

        def resolve(path: Path | None, default: Path) -> Path:
            if path is None:
                return root / default
            return path if path.is_absolute() else root / path

        self.app_db_path = resolve(self.app_db_path, Path("database") / "app.db")
        self.vehicles_db_path = resolve(self.vehicles_db_path, Path("database") / "vehicles.db")
        self.recordings_root = resolve(self.recordings_root, Path("recordings"))
        return self


# ─── Lazy accessor (no I/O on import) ────────────────────────────────────────

@lru_cache(maxsize=1)
def load_settings() -> Settings:
    """
    Lazily load and cache settings once per process.

    Must only be called from inside the FastAPI lifespan startup event so that
    startup errors surface clearly instead of silently at import time.
    """
    return Settings()
