"""
backend/database/_leads_deletion.py — Hard-delete cascade for a lead (Issue 017/018).

Provides a single repository function that:
  1. Drops the entire per-lead recording folder (recordings_root/<lead_id>/)
     using shutil.rmtree with onexc= — only real OS errors (locks/permissions)
     trigger logging; a missing folder is treated as silent success.
  2. Executes DELETE FROM leads WHERE id = ? — SQLite's ON DELETE CASCADE
     wipes all associated call_attempts, transcript_turns, and recommendations.

The block_list table is intentionally left untouched; blocks are independent of
lead rows and must survive a hard delete.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from backend.database._pool import ConnectionPool

log = logging.getLogger(__name__)


async def delete_lead_cascade(
    pool: ConnectionPool,
    lead_id: int,
    recordings_root: Path,
) -> bool:
    """
    Hard-delete a lead and all its associated data.

    Parameters
    ----------
    pool : ConnectionPool
        Database connection pool injected by the caller.
    lead_id : int
        Primary key of the lead whose folder and rows should be deleted.
    recordings_root : Path
        Root directory where per-lead recording folders are stored.
        The folder ``<recordings_root>/<lead_id>/`` (and everything inside it)
        is dropped before the row is deleted.  Injected from
        ``settings.recordings_root`` by the caller — never resolved via
        ``load_settings()`` at import time.

    Returns
    -------
    bool
        ``True`` if the lead row was deleted (exactly one row affected).
        ``False`` if no lead with that id existed.

    File deletion policy
    -------------------
    The entire ``<lead_id>`` subfolder is removed in one call:

    - **Folder already absent**: ``lead_folder.is_dir()`` guards the ``rmtree``
      call so a missing folder results in silent success with no filesystem
      call at all.
    - **Permission / OS error on any file inside the folder**: ``onexc``
      logs the path and the error, but ``rmtree`` continues.  A stuck or
      locked file never blocks the cascade.  (In Python 3.12+ this also
      silences the deprecation warning for the legacy ``onerror`` name.)

      The ``onexc`` callback receives ``(func, path, exc)`` where ``exc`` is a
      bare exception instance — **not** the ``(type, value, traceback)`` 3-tuple
      that the legacy ``onerror`` callback received.

    Database deletion policy
    -------------------------
    A single ``DELETE FROM leads WHERE id = ?`` is issued.  SQLite's
    ``ON DELETE CASCADE`` takes care of ``call_attempts``,
    ``transcript_turns``, and ``recommendations``.  The ``block_list`` table
    is untouched.
    """
    # ── Step 1: drop the lead's recording folder ──────────────────────────────
    lead_folder = recordings_root / str(lead_id)

    def _on_rmtree_error(func, path, exc: BaseException):
        """
        Log errors from shutil.rmtree but never raise — non-fatal by design.

        ``onexc=`` passes a bare exception instance, NOT a (type, value,
        traceback) tuple.  This contrasts with the legacy ``onerror=`` callback
        which received ``(exc_type, exc_value, exc_tb)``.
        """
        log.warning(
            "Failed to delete recording file %r during rmtree of %r: %s %s. "
            "Continuing with database cleanup.",
            path,
            str(lead_folder),
            type(exc).__name__,
            exc,
        )

    # Only call rmtree when the folder exists — absent folder = silent success.
    if lead_folder.is_dir():
        shutil.rmtree(lead_folder, onexc=_on_rmtree_error)

    # ── Step 2: delete the lead row ─────────────────────────────────────────
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "DELETE FROM leads WHERE id = ?",
            (lead_id,),
        )
        return cursor.rowcount == 1
