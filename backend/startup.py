"""
backend/startup.py — Crash/restart recovery sweep (Issue 029) +
                    boot-time disclosure asset validation (Issue 037).

On every boot the orchestrator owns no live calls because all in-memory state is
fresh.  Any lead still marked ``Processing`` is a zombie left over from a prior
crashed run.  This module provides the sole automatic path out of a wedged
``Processing`` state on restart, allowing the admin to redial those leads via the
Retry feature later.

The sweep is idempotent — it is safe to call multiple times.

Issue 037 also adds :func:`validate_disclosure_assets` — a fail-fast boot
check that the pre-rendered μ-law disclosure files exist and are non-empty.
Missing or empty assets are a compliance failure (no legal disclosure can be
played on a call) so the backend refuses to boot rather than fail-open silently.

No module calls ``load_settings()`` at import time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from backend.database._txn import transaction

if TYPE_CHECKING:
    from backend.database._pool import ConnectionPool

log = logging.getLogger(__name__)

# The failure reason assigned to zombie Processing leads during sweep.
# "dropped" signals the backend died mid-call.
_DROPPED_REASON = "dropped"

#: Default location of the disclosure assets directory (relative to this
#: module — ``backend/startup.py`` → ``backend/audio/assets/``).
#: Tests and operators may override via the ``assets_dir`` parameter.
_DEFAULT_DISCLOSURE_ASSETS_DIR: Path = (
    Path(__file__).resolve().parent / "audio" / "assets"
)

#: Languages the disclosure MUST be available for at boot.  Mirrors
#: ``backend.api.media_stream._disclosure_asset_path`` — the player
#: picks the file via the same convention so adding a language here
#: without producing its asset is a deliberate operator error.
_DISCLOSURE_LANGUAGES: tuple[str, ...] = ("en", "hi")


def validate_disclosure_assets(
    assets_dir: Path | None = None,
    *,
    languages: tuple[str, ...] = _DISCLOSURE_LANGUAGES,
) -> None:
    """
    Boot-time check: every required disclosure ``.ulaw`` asset must exist
    AND be strictly greater than zero bytes.

    Issue 037 (compliance): the legal-recording disclosure is a
    legally-required preamble that plays before any customer audio.  If
    the asset is missing or empty, ``play_disclosure`` falls back to
    marking the disclosure complete anyway (so the call doesn't hang
    on the driver's wait) — which is the correct runtime behaviour
    but a compliance failure in production.  This check fails the
    boot LOUDLY instead of failing open silently.

    Behaviour
    ─────────
    * Missing file → raises ``FileNotFoundError`` (loud, specific).
    * Empty file (0 bytes) → raises ``RuntimeError`` (loud, specific).
    * All assets present and non-empty → logs an info line + returns.

    Parameters
    ----------
    assets_dir
        Directory containing ``disclosure_<lang>.ulaw`` files.
        Defaults to ``backend/audio/assets/``.  Tests pass a temp dir
        populated with fixtures.
    languages
        Language codes (each maps to ``disclosure_<lang>.ulaw``).
        Defaults to ``("en", "hi")`` — the languages the media-stream
        player actually plays for.  Extend this tuple if a third
        language is added upstream.

    Raises
    ------
    FileNotFoundError
        A required disclosure asset is absent from ``assets_dir``.
    RuntimeError
        A required disclosure asset is present but empty (zero bytes).
    """
    base = assets_dir if assets_dir is not None else _DEFAULT_DISCLOSURE_ASSETS_DIR
    for lang in languages:
        path = base / f"disclosure_{lang}.ulaw"
        if not path.is_file():
            raise FileNotFoundError(
                f"Disclosure asset missing: {path}. The backend cannot "
                f"boot without pre-rendered disclosure audio (Issue 037 "
                f"compliance gate). Re-run backend/scripts/"
                f"generate_disclosures.py to produce the missing file(s)."
            )
        size = path.stat().st_size
        if size <= 0:
            raise RuntimeError(
                f"Disclosure asset empty: {path} ({size} bytes). The "
                f"backend cannot boot without a non-empty disclosure "
                f"(Issue 037 compliance gate). Re-run backend/scripts/"
                f"generate_disclosures.py to regenerate."
            )

    log.info(
        "validate_disclosure_assets: %d file(s) present and non-empty in %s "
        "(languages=%s).",
        len(languages),
        base,
        languages,
    )


async def reconcile_startup(pool: "ConnectionPool") -> int:
    """
    Run the startup reconciliation sweep on *pool*.

    Within a single database transaction this function:
    1. Resolves all orphaned in-flight ``call_attempts`` rows (those with
       ``outcome IS NULL``) belonging to zombie Processing leads by setting
       ``outcome = 'dropped'`` and ``ended_at = datetime('now')``.
    2. Transitions every zombie ``Processing`` lead to ``Failed`` with
       ``failure_reason = 'dropped'`` and bumps ``updated_at``.
       All other leads (New, Completed, Failed) are left untouched.

    Parameters
    ----------
    pool : ConnectionPool
        The application connection pool.

    Returns
    -------
    int — the number of leads that were transitioned from Processing → Failed.

    Idempotency
    ───────────
    Calling this function twice in a row on the same pool is safe.  The second
    call finds zero Processing leads, finds zero orphaned attempts to resolve,
    and returns 0, leaving the database unchanged.
    """
    async with pool.connection() as conn:
        async with transaction(conn):
            # Step 1 — collect the IDs of zombie leads so we can safely
            # reference them in the subsequent UPDATE without a self-join.
            cursor = await conn.execute(
                "SELECT id FROM leads WHERE status = 'Processing';",
            )
            zombie_ids: tuple[int, ...] = tuple(row[0] for row in await cursor.fetchall())

            # Step 2 — resolve orphaned in-flight call_attempts rows.
            # Only touch attempts where outcome IS NULL (genuinely live calls).
            # Terminal attempts (outcome set) are left alone.
            if zombie_ids:
                await conn.execute(
                    """
                    UPDATE call_attempts
                    SET outcome = ?, ended_at = datetime('now')
                    WHERE lead_id IN (%s)
                      AND outcome IS NULL;
                    """
                    % ",".join("?" * len(zombie_ids)),
                    (_DROPPED_REASON, *zombie_ids),
                )

            # Step 3 — flip zombie leads to Failed.
            # updated_at is bumped so the dashboard reflects the correct update time.
            cursor = await conn.execute(
                """
                UPDATE leads
                SET status = 'Failed',
                    failure_reason = ?,
                    updated_at = datetime('now')
                WHERE status = 'Processing';
                """,
                (_DROPPED_REASON,),
            )
            updated = cursor.rowcount

    log.info(
        "Startup sweep: %d Processing lead(s) transitioned to Failed (%s); "
        "%d orphaned attempt(s) resolved.",
        updated,
        _DROPPED_REASON,
        len(zombie_ids),
    )
    return updated
