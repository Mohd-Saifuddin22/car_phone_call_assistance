"""
backend/tests/test_leads_deletion.py — Unit tests for delete_lead_cascade (Issue 017).

Covers three core scenarios:
  1. Full cascade  — recording folder exists, gets deleted via shutil.rmtree,
     all child rows are wiped.
  2. Missing folder — lead_id subfolder absent on disk; rmtree succeeds anyway,
     all rows are still wiped.
  3. Genuine failure — rmtree raises OSError on a locked file; the delete
     still succeeds and all rows are wiped.

Recording paths stored in the DB are relative (e.g. "1/call_001.wav") and are
resolved against the injected ``recordings_root`` by shutil.rmtree operating on
the ``<recordings_root>/<lead_id>/`` subfolder.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import patch

import pytest
import pytest_asyncio

from backend.database import create_pool
from backend.database._leads_deletion import delete_lead_cascade


# ─── Test-database bootstrap ───────────────────────────────────────────────────


def _schema_path() -> Path:
    """Absolute path to the application schema file."""
    return (
        Path(__file__).parent.parent.parent / "database" / "app_schema.sql"
    ).resolve()


def _create_temp_db() -> Path:
    """
    Create a temporary app.db from the application schema.

    Uses ``PRAGMA foreign_keys = ON`` so the schema's CASCADE constraints
    are enforced immediately.
    """
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".db", prefix="car_test_"
    ) as fh:
        tmp = Path(fh.name)
    conn = sqlite3.connect(str(tmp))
    conn.execute("PRAGMA foreign_keys = ON;")
    with open(_schema_path(), encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.close()
    return tmp


# ─── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    """Fresh settings cache for every test — no bleed between test runs."""
    from backend.config import load_settings

    load_settings.cache_clear()
    yield
    load_settings.cache_clear()


@pytest_asyncio.fixture
async def pool(temp_db: Path) -> AsyncGenerator:
    """Per-test isolated pool backed by a temp database."""
    db_path = _create_temp_db()
    p = create_pool(
        db_path=db_path,
        busy_timeout_seconds=5.0,
        pool_size=5,
    )
    # Warm the pool.
    conns = []
    for _ in range(p.max_size):
        c = await p.acquire()
        conns.append(c)
    for c in conns:
        await p.release(c)
    yield p
    await p.close()
    try:
        db_path.unlink()
    except OSError:
        pass


@pytest_asyncio.fixture
async def temp_db() -> Path:
    """Per-test isolated database backed by a temp file."""
    db_path = _create_temp_db()
    yield db_path
    try:
        db_path.unlink()
    except OSError:
        pass


# ─── Helpers ───────────────────────────────────────────────────────────────────


async def _insert_lead_with_attempt(
    pool,
    recordings_root: Path,
    relative_recording_path: str,
) -> int:
    """
    Insert a lead, one call_attempt, one transcript_turn, and one recommendation.

    Parameters
    ----------
    recordings_root : Path
        Root directory (from the tmp_path fixture) used to construct the real
        on-disk path for the recording file.
    relative_recording_path : str
        Relative path stored in the DB (e.g. "1/call_001.wav").  The file is
        created at ``recordings_root / relative_recording_path`` to mirror the
        real production layout.

    Returns the lead_id.
    """
    lead_folder = recordings_root / str(1)  # lead_id hardcoded to 1 in these tests
    lead_folder.mkdir(parents=True, exist_ok=True)

    real_path = recordings_root / relative_recording_path
    real_path.parent.mkdir(parents=True, exist_ok=True)
    real_path.write_text("fake wav data", encoding="utf-8")

    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            INSERT INTO leads
                (name, phone, email, city, language, status, attempt_count)
            VALUES ('Delete Me', '+919900000001', 'delete@example.com', 'Mumbai', 'English', 'Processing', 1)
            """
        )
        lead_id = cursor.lastrowid

        await conn.execute(
            """
            INSERT INTO call_attempts
                (lead_id, attempt_number, outcome, recording_path)
            VALUES (?, 1, 'completed', ?)
            """,
            (lead_id, relative_recording_path),
        )

        await conn.execute(
            """
            INSERT INTO transcript_turns
                (call_attempt_id, turn_index, speaker, text)
            VALUES (1, 0, 'customer', 'I want a car.')
            """
        )

        await conn.execute(
            """
            INSERT INTO recommendations
                (lead_id, items, preference_snapshot)
            VALUES (?, '[{"vehicle_id":1,"rank":1,"score":0.9}]', '{}')
            """,
            (lead_id,),
        )

    return lead_id


async def _count_rows(pool, table: str) -> int:
    """Return the row count for a given table."""
    async with pool.connection() as conn:
        cursor = await conn.execute(f"SELECT COUNT(*) FROM {table}")
        row = await cursor.fetchone()
        return row[0] if row else 0


# ─── Tests ─────────────────────────────────────────────────────────────────────


class TestDeleteLeadCascadeFull:
    """
    Full cascade: the lead's recording folder exists on disk, shutil.rmtree
    removes it, and all child rows are wiped.
    """

    @pytest.mark.asyncio
    async def test_recording_folder_is_deleted_and_all_rows_are_wiped(
        self, pool, tmp_path
    ) -> None:
        """
        Given a lead with a real recording file inside its lead_id subfolder,
        delete_lead_cascade must remove the entire subfolder and wipe all
        rows.  The block_list table must be untouched.
        """
        # Store a relative path; file lives at tmp_path / "1" / "call_001.wav"
        lead_id = await _insert_lead_with_attempt(
            pool,
            recordings_root=tmp_path,
            relative_recording_path="1/call_001.wav",
        )

        # Verify the folder exists before the call.
        assert (tmp_path / "1").exists()
        assert (tmp_path / "1" / "call_001.wav").exists()

        # Add a block_list entry — it must survive the cascade.
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO block_list (phone, email, name) VALUES (?, ?, ?)",
                ("+919900000001", "delete@example.com", "Block Survivor"),
            )

        # Act.
        result = await delete_lead_cascade(pool, lead_id, recordings_root=tmp_path)

        # Assert return value.
        assert result is True

        # Assert the folder is gone.
        assert not (tmp_path / "1").exists(), (
            f"Recording folder {tmp_path / '1'!r} still exists after delete"
        )

        # Assert all child rows are wiped.
        assert await _count_rows(pool, "leads") == 0
        assert await _count_rows(pool, "call_attempts") == 0
        assert await _count_rows(pool, "transcript_turns") == 0
        assert await _count_rows(pool, "recommendations") == 0

        # block_list must survive.
        assert await _count_rows(pool, "block_list") == 1

    @pytest.mark.asyncio
    async def test_returns_false_when_lead_does_not_exist(
        self, pool, tmp_path
    ) -> None:
        """Deleting a non-existent lead returns False; no rows are touched."""
        result = await delete_lead_cascade(pool, lead_id=99999, recordings_root=tmp_path)
        assert result is False
        assert await _count_rows(pool, "leads") == 0

    @pytest.mark.asyncio
    async def test_subfolder_with_multiple_files_is_fully_removed(
        self, pool, tmp_path
    ) -> None:
        """rmtree removes all files and subdirectories inside the lead folder."""
        lead_id = await _insert_lead_with_attempt(
            pool,
            recordings_root=tmp_path,
            relative_recording_path="1/call_001.wav",
        )
        # Add a second file directly in the lead folder.
        (tmp_path / "1" / "call_002.wav").write_text("second wav", encoding="utf-8")

        assert (tmp_path / "1").exists()
        assert (tmp_path / "1" / "call_002.wav").exists()

        result = await delete_lead_cascade(pool, lead_id, recordings_root=tmp_path)

        assert result is True
        assert not (tmp_path / "1").exists()


class TestDeleteLeadCascadeMissingFolder:
    """
    Missing folder: the lead_id subfolder is absent on disk (no recordings were
    stored, or they were manually removed).  rmtree succeeds silently.
    """

    @pytest.mark.asyncio
    async def test_delete_succeeds_when_folder_is_already_absent(
        self, pool, tmp_path
    ) -> None:
        """
        When the recording folder does not exist, shutil.rmtree must not raise.
        delete_lead_cascade returns True and wipes all rows.
        """
        # Insert a lead without ever creating its recording folder.
        async with pool.connection() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO leads
                    (name, phone, email, city, language, status, attempt_count)
                VALUES ('No Folder', '+919900000002', 'nofolder@example.com',
                        'Delhi', 'Hinglish', 'New', 0)
                """
            )
            lead_id = cursor.lastrowid

        assert not (tmp_path / str(lead_id)).exists()

        # Act — must not raise.
        result = await delete_lead_cascade(pool, lead_id, recordings_root=tmp_path)

        assert result is True
        assert await _count_rows(pool, "leads") == 0
        assert await _count_rows(pool, "call_attempts") == 0

    @pytest.mark.asyncio
    async def test_delete_succeeds_when_attempt_exists_but_file_is_untracked(
        self, pool, tmp_path
    ) -> None:
        """
        The DB has a call_attempt with a relative recording_path, but the file
        on disk is already gone.  This must not raise; rows are still wiped.
        """
        async with pool.connection() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO leads
                    (name, phone, email, city, language, status, attempt_count)
                VALUES ('Orphaned Path', '+919900000003', 'orphan@example.com',
                        'Mumbai', 'English', 'Processing', 1)
                """
            )
            lead_id = cursor.lastrowid
            await conn.execute(
                """
                INSERT INTO call_attempts
                    (lead_id, attempt_number, outcome, recording_path)
                VALUES (?, 1, 'completed', '3/call_001.wav')
                """,
                (lead_id,),
            )

        # Folder never created — file is untracked.
        assert not (tmp_path / "3").exists()

        result = await delete_lead_cascade(pool, lead_id, recordings_root=tmp_path)
        assert result is True
        assert await _count_rows(pool, "leads") == 0


class TestDeleteLeadCascadeGenuineFailure:
    """
    Genuine OS error: rmtree encounters a locked / permission-denied file.

    The error must be logged (via the onexc callback) but the function must
    return True and wipe the rows.

    Approach: Patch ``os.unlink`` at the shutil level so that when
    ``shutil.rmtree`` calls it on the recording file, it raises a real
    ``PermissionError``.  ``shutil.rmtree`` catches it internally and invokes
    our ``onexc`` handler with the correct native shape: ``(func, path, exc)``
    where ``exc`` is a bare exception instance.

    This exercises the full real call chain — no hand-crafted callback
    arguments.
    """

    @pytest.mark.asyncio
    async def test_delete_still_succeeds_when_rmtree_reports_permission_error(
        self, pool, tmp_path
    ) -> None:
        """
        When os.unlink raises PermissionError (locked file), shutil.rmtree invokes
        the onexc handler.  The handler must accept a bare exception instance
        (not a 3-tuple) and log it.  The database rows must still be wiped.
        """
        lead_id = await _insert_lead_with_attempt(
            pool,
            recordings_root=tmp_path,
            relative_recording_path="1/call_001.wav",
        )

        assert (tmp_path / "1").exists()

        # Patch os.unlink *inside* shutil so that rmtree's call hits the mock.
        # This is a real OS-level fault — not a fabricated callback argument.
        #
        # We accept ``**kwargs`` because shutil.rmtree invokes os.unlink
        # differently per platform: on POSIX / Linux it uses the fd-based
        # fast path (``os.unlink(name, dir_fd=...)``); on Windows it calls
        # ``os.unlink(path)``. Production behavior is unchanged — we only
        # widen the mock's signature so the same handler works in both
        # environments.
        def _unlink_raises(path, **_kwargs):
            raise PermissionError(
                "Access denied: file is locked by another process"
            )

        with patch("shutil.os.unlink", side_effect=_unlink_raises):
            result = await delete_lead_cascade(pool, lead_id, recordings_root=tmp_path)

        # Function must return True — the row was still deleted despite the OS error.
        assert result is True

        # All child rows must be gone despite the file error.
        assert await _count_rows(pool, "leads") == 0
        assert await _count_rows(pool, "call_attempts") == 0
        assert await _count_rows(pool, "transcript_turns") == 0
        assert await _count_rows(pool, "recommendations") == 0

    @pytest.mark.asyncio
    async def test_other_os_errors_are_also_non_fatal(self, pool, tmp_path) -> None:
        """Any OSError from the filesystem must not block the row delete."""
        lead_id = await _insert_lead_with_attempt(
            pool,
            recordings_root=tmp_path,
            relative_recording_path="1/call_001.wav",
        )

        class CustomOSError(OSError):
            pass

        # See note in the sibling test above: ``os.unlink`` is invoked with
        # an extra ``dir_fd=`` kwarg on POSIX/Linux but plain ``path`` on
        # Windows. Accepting ``**kwargs`` keeps the mock platform-neutral.
        def _unlink_raises(path, **_kwargs):
            raise CustomOSError("Disk busy or resource unavailable")

        with patch("shutil.os.unlink", side_effect=_unlink_raises):
            result = await delete_lead_cascade(pool, lead_id, recordings_root=tmp_path)

        assert result is True
        assert await _count_rows(pool, "leads") == 0


class TestDeleteLeadCascadeNoRecordings:
    """Lead exists but has no call_attempts or no recording_path at all."""

    @pytest.mark.asyncio
    async def test_lead_with_no_call_attempts_is_deleted(self, pool, tmp_path) -> None:
        """A lead with zero call_attempts must still be deleted cleanly."""
        async with pool.connection() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO leads
                    (name, phone, email, city, language, status, attempt_count)
                VALUES ('No Attempts', '+919900000004', 'noattempts@example.com',
                        'Chennai', 'English', 'New', 0)
                """
            )
            lead_id = cursor.lastrowid

        result = await delete_lead_cascade(pool, lead_id, recordings_root=tmp_path)
        assert result is True
        assert await _count_rows(pool, "leads") == 0

    @pytest.mark.asyncio
    async def test_lead_with_attempt_but_null_recording_path(self, pool, tmp_path) -> None:
        """A call_attempt with recording_path=NULL must not cause rmtree errors."""
        async with pool.connection() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO leads
                    (name, phone, email, city, language, status, attempt_count)
                VALUES ('No Recording', '+919900000005', 'norec@example.com',
                        'Bangalore', 'Hinglish', 'Processing', 1)
                """
            )
            lead_id = cursor.lastrowid
            await conn.execute(
                """
                INSERT INTO call_attempts
                    (lead_id, attempt_number, outcome, recording_path)
                VALUES (?, 1, 'no_answer', NULL)
                """,
                (lead_id,),
            )

        result = await delete_lead_cascade(pool, lead_id, recordings_root=tmp_path)
        assert result is True
        assert await _count_rows(pool, "leads") == 0
        assert await _count_rows(pool, "call_attempts") == 0
