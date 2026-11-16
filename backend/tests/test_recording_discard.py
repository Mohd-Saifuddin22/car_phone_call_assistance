"""
backend/tests/test_recording_discard.py — Unit tests for Issue 037
discard-on-voicemail branching logic.

Test plan
─────────

Supervisor teardown branching (Step 4b)
  T1  Failed/no_answer → ``_delete_call_recordings_async`` IS called
      with the supervisor's CallSid.
  T2  Failed/dropped → delete IS NOT called (only no_answer
      triggers the discard — other Failed reasons are consent-passing
      customer-side outcomes we keep).
  T3  Failed/error → delete IS NOT called (same rationale).
  T4  Completed (no failure_reason) → delete IS NOT called
      (consultation succeeded — recording is kept per Issue 049).

Recording-callback webhook branching (POST /api/twilio/recording-callback)
  W1  Lead is Failed/no_answer → delete IS called.
  W2  Lead is Completed → delete IS NOT called.
  W3  CallSid not in DB → delete IS NOT called (stale callback).

These tests prove the privacy guarantee: recordings of voicemails are
NEVER retained, recordings of successful consultations ARE retained.

The tests mock the Twilio client (via ``sys.modules`` patch on
``twilio.rest.Client``) and the DB pool (via patching the lifespan
state and the connection's ``execute`` method).  No live Twilio or DB
is touched.
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from backend.call_supervisor import (
    CallSupervisor,
    _delete_call_recordings_async,
    _twilio_delete_recordings,
)
from backend.concurrency import ConcurrencyManager


# ─── Twilio client fixture ──────────────────────────────────────────────────


@pytest.fixture
def fake_twilio_client():
    """
    Build a MagicMock that mimics ``twilio.rest.Client`` for
    ``calls(call_sid).recordings.list()`` and
    ``client.recordings(rec_sid).delete()``.

    The recordings list is configurable per-test via
    ``client._recordings_to_return`` (defaults to two recordings).
    Tracks every list/delete call so tests can assert on them.
    """
    client = MagicMock(name="twilio_rest_client")

    # Two recordings by default; tests can mutate to [] or to a custom list.
    rec1 = SimpleNamespace(sid="REaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1")
    rec2 = SimpleNamespace(sid="REbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb2")
    client._recordings_to_return: list = [rec1, rec2]

    # Test introspection: collected lists/deletes.
    client._list_calls: list[str] = []
    client._delete_calls: list[str] = []

    # ``client.calls(call_sid).recordings.list()`` — return the configurable list.
    recordings_list_resource = MagicMock(name="recordings_list")

    def _list_call(*args, **kwargs):
        last_call_sid = client.calls.call_args[0][0]
        client._list_calls.append(last_call_sid)
        return list(client._recordings_to_return)

    recordings_list_resource.list = MagicMock(side_effect=_list_call)
    calls_resource = MagicMock()
    calls_resource.return_value.recordings = recordings_list_resource
    client.calls = calls_resource

    # ``client.recordings(sid).delete()`` — track per-sid, allow per-recording override.
    def _get_recording(sid: str):
        r = MagicMock()
        r._sid = sid
        r.delete = MagicMock(
            side_effect=lambda *a, **k: _track_delete(sid)
        )
        return r

    client.recordings = MagicMock(side_effect=_get_recording)

    def _track_delete(sid: str):
        client._delete_calls.append(sid)
        return None

    return client


# ─── Shared DB helpers ──────────────────────────────────────────────────────


def _schema_path() -> Path:
    return (
        Path(__file__).parent.parent.parent / "database" / "app_schema.sql"
    ).resolve()


def _create_temp_db() -> Path:
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".db", prefix="car_test_disc_"
    ) as fh:
        tmp = Path(fh.name)
    conn = sqlite3.connect(str(tmp))
    conn.execute("PRAGMA foreign_keys = ON;")
    with open(_schema_path(), encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.close()
    return tmp


@pytest_asyncio.fixture
async def temp_db():
    db_path = _create_temp_db()
    yield db_path
    try:
        db_path.unlink()
    except OSError:
        pass


@pytest_asyncio.fixture
async def pool(temp_db):
    from backend.database import create_pool

    p = create_pool(db_path=temp_db, busy_timeout_seconds=5.0, pool_size=5)
    yield p
    await p.close()


async def _seed_attempt(
    pool,
    *,
    call_sid: str,
    status: str,
    failure_reason: str | None,
) -> int:
    """Insert a Processing lead + a call_attempt with the given outcome."""
    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            INSERT INTO leads
                (name, phone, email, city, language, status, attempt_count)
            VALUES ('Discard Test', '+919999999999', 'd@x.com',
                    'Mumbai', 'English', ?, 1)
            """,
            (status,),
        )
        lead_id = cursor.lastrowid
        cursor = await conn.execute(
            """
            INSERT INTO call_attempts (lead_id, attempt_number, twilio_call_sid)
            VALUES (?, 1, ?)
            """,
            (lead_id, call_sid),
        )
        attempt_id = cursor.lastrowid

        # Update the attempt with outcome + reason.  Status may already
        # be the desired value on the leads row (above), but the
        # call_attempts row needs an outcome set for the webhook JOIN.
        await conn.execute(
            """
            UPDATE call_attempts
            SET outcome = ?,
                ended_at = datetime('now')
            WHERE id = ?;
            """,
            (failure_reason if failure_reason else "completed", attempt_id),
        )
        # failure_reason lives on the leads row.
        await conn.execute(
            "UPDATE leads SET failure_reason = ? WHERE id = ?;",
            (failure_reason, lead_id),
        )
        return lead_id


# ─── T1–T4: supervisor teardown branching ────────────────────────────────────


class TestSupervisorTeardownDiscardBranching:
    """
    ``CallSupervisor.trigger_teardown`` Step 4b calls
    ``_delete_call_recordings_async`` ONLY when
    ``failure_reason == "no_answer"``.  These tests pin that branching
    with a real DB and a mocked delete function — the Twilio client is
    not invoked because we patch the async wrapper.
    """

    @pytest.mark.asyncio
    async def test_t1_failed_no_answer_calls_delete(
        self,
        pool,
        monkeypatch,
    ) -> None:
        """Failed/no_answer → delete IS called with the CallSid."""
        await _seed_attempt(
            pool,
            call_sid="CA_NO_ANSWER_001",
            status="Failed",
            failure_reason="no_answer",
        )

        manager = ConcurrencyManager(capacity=2)
        manager.acquire()
        supervisor = CallSupervisor(
            lead_id=1,
            attempt_id=1,
            call_sid="CA_NO_ANSWER_001",
            language="English",
            concurrency_manager=manager,
            pool=pool,
        )

        # Track whether the delete was called and with what CallSid.
        called_with: list[str] = []

        async def _spy_delete(call_sid: str) -> int:
            called_with.append(call_sid)
            return 0

        # Patch the async wrapper imported into the supervisor module.
        monkeypatch.setattr(
            "backend.call_supervisor._delete_call_recordings_async",
            _spy_delete,
        )

        # Avoid the real Twilio hangup call.
        monkeypatch.setattr(
            "backend.call_supervisor._twilio_hangup_call",
            lambda *a, **k: True,
        )

        await supervisor.trigger_teardown("Failed", "no_answer")

        assert called_with == ["CA_NO_ANSWER_001"], (
            f"_delete_call_recordings_async should have been called "
            f"with ['CA_NO_ANSWER_001']; got {called_with}"
        )

    @pytest.mark.asyncio
    async def test_t2_failed_dropped_does_not_call_delete(
        self,
        pool,
        monkeypatch,
    ) -> None:
        """Failed/dropped → delete is NOT called (consent-passing customer outcome)."""
        await _seed_attempt(
            pool,
            call_sid="CA_DROPPED_001",
            status="Failed",
            failure_reason="dropped",
        )

        manager = ConcurrencyManager(capacity=2)
        manager.acquire()
        supervisor = CallSupervisor(
            lead_id=1,
            attempt_id=1,
            call_sid="CA_DROPPED_001",
            language="English",
            concurrency_manager=manager,
            pool=pool,
        )

        called_with: list[str] = []

        async def _spy_delete(call_sid: str) -> int:
            called_with.append(call_sid)
            return 0

        monkeypatch.setattr(
            "backend.call_supervisor._delete_call_recordings_async",
            _spy_delete,
        )
        monkeypatch.setattr(
            "backend.call_supervisor._twilio_hangup_call",
            lambda *a, **k: True,
        )

        await supervisor.trigger_teardown("Failed", "dropped")

        assert called_with == [], (
            f"_delete_call_recordings_async must NOT be called for "
            f"Failed/dropped (only no_answer triggers the voicemail "
            f"discard); got {called_with}"
        )

    @pytest.mark.asyncio
    async def test_t3_failed_error_does_not_call_delete(
        self,
        pool,
        monkeypatch,
    ) -> None:
        """Failed/error → delete is NOT called."""
        await _seed_attempt(
            pool,
            call_sid="CA_ERROR_001",
            status="Failed",
            failure_reason="error",
        )

        manager = ConcurrencyManager(capacity=2)
        manager.acquire()
        supervisor = CallSupervisor(
            lead_id=1,
            attempt_id=1,
            call_sid="CA_ERROR_001",
            language="English",
            concurrency_manager=manager,
            pool=pool,
        )

        called_with: list[str] = []

        async def _spy_delete(call_sid: str) -> int:
            called_with.append(call_sid)
            return 0

        monkeypatch.setattr(
            "backend.call_supervisor._delete_call_recordings_async",
            _spy_delete,
        )
        monkeypatch.setattr(
            "backend.call_supervisor._twilio_hangup_call",
            lambda *a, **k: True,
        )

        await supervisor.trigger_teardown("Failed", "error")

        assert called_with == []

    @pytest.mark.asyncio
    async def test_t4_completed_does_not_call_delete(
        self,
        pool,
        monkeypatch,
    ) -> None:
        """Completed (no failure_reason) → delete is NOT called
        (consultation succeeded — recording is kept per Issue 049)."""
        await _seed_attempt(
            pool,
            call_sid="CA_COMPLETED_001",
            status="Completed",
            failure_reason=None,
        )

        manager = ConcurrencyManager(capacity=2)
        manager.acquire()
        supervisor = CallSupervisor(
            lead_id=1,
            attempt_id=1,
            call_sid="CA_COMPLETED_001",
            language="English",
            concurrency_manager=manager,
            pool=pool,
        )

        called_with: list[str] = []

        async def _spy_delete(call_sid: str) -> int:
            called_with.append(call_sid)
            return 0

        monkeypatch.setattr(
            "backend.call_supervisor._delete_call_recordings_async",
            _spy_delete,
        )
        monkeypatch.setattr(
            "backend.call_supervisor._twilio_hangup_call",
            lambda *a, **k: True,
        )

        await supervisor.trigger_teardown("Completed", completed_timeout=False)

        assert called_with == [], (
            f"_delete_call_recordings_async must NOT be called on "
            f"Completed; got {called_with}"
        )


# ─── W1–W3: recording-callback webhook branching ───────────────────────────


class TestRecordingCallbackDiscardBranching:
    """
    ``POST /api/twilio/recording-callback`` looks up the lead's outcome
    from the DB and only calls delete when status=Failed AND
    failure_reason=no_answer.  These tests pin that branching with a
    real DB and a mocked delete function.
    """

    @pytest.mark.asyncio
    async def test_w1_no_answer_calls_delete(
        self,
        pool,
        monkeypatch,
    ) -> None:
        """Failed/no_answer recording → delete IS called."""
        await _seed_attempt(
            pool,
            call_sid="CA_CB_NO_ANSWER",
            status="Failed",
            failure_reason="no_answer",
        )

        called_with: list[str] = []

        async def _spy_delete(call_sid: str) -> int:
            called_with.append(call_sid)
            return 0

        monkeypatch.setattr(
            "backend.call_supervisor._delete_call_recordings_async", _spy_delete
        )

        # Import inside the test so monkeypatch is applied first.
        from backend.api.arrange import twilio_recording_callback

        # Build a minimal Request-like object with form data + app.state.pool.
        request = MagicMock()
        request.app.state.pool = pool
        form_data = {
            "CallSid": "CA_CB_NO_ANSWER",
            "RecordingSid": "RE_CB_TEST",
            "RecordingStatus": "completed",
        }

        async def _fake_form():
            return form_data

        request.form = _fake_form

        response = await twilio_recording_callback(request)
        # PlainTextResponse("") with status 200 — we don't need the
        # exact body, just the call side-effects.
        assert called_with == ["CA_CB_NO_ANSWER"], (
            f"recording-callback should have called delete for "
            f"no_answer; got {called_with}"
        )

    @pytest.mark.asyncio
    async def test_w2_completed_does_not_call_delete(
        self,
        pool,
        monkeypatch,
    ) -> None:
        """Completed recording → delete is NOT called (kept per Issue 049)."""
        await _seed_attempt(
            pool,
            call_sid="CA_CB_COMPLETED",
            status="Completed",
            failure_reason=None,
        )

        called_with: list[str] = []

        async def _spy_delete(call_sid: str) -> int:
            called_with.append(call_sid)
            return 0

        monkeypatch.setattr(
            "backend.call_supervisor._delete_call_recordings_async", _spy_delete
        )

        from backend.api.arrange import twilio_recording_callback

        request = MagicMock()
        request.app.state.pool = pool
        form_data = {
            "CallSid": "CA_CB_COMPLETED",
            "RecordingSid": "RE_CB_TEST",
            "RecordingStatus": "completed",
        }

        async def _fake_form():
            return form_data

        request.form = _fake_form

        await twilio_recording_callback(request)

        assert called_with == [], (
            f"recording-callback must NOT call delete on Completed; "
            f"got {called_with}"
        )

    @pytest.mark.asyncio
    async def test_w3_unknown_callsid_does_not_call_delete(
        self,
        pool,
        monkeypatch,
    ) -> None:
        """Stale callback for a CallSid not in the DB → delete is NOT called."""
        called_with: list[str] = []

        async def _spy_delete(call_sid: str) -> int:
            called_with.append(call_sid)
            return 0

        monkeypatch.setattr(
            "backend.call_supervisor._delete_call_recordings_async", _spy_delete
        )

        from backend.api.arrange import twilio_recording_callback

        request = MagicMock()
        request.app.state.pool = pool
        form_data = {
            "CallSid": "CA_NEVER_EXISTED",
            "RecordingSid": "RE_GHOST",
            "RecordingStatus": "completed",
        }

        async def _fake_form():
            return form_data

        request.form = _fake_form

        await twilio_recording_callback(request)

        assert called_with == []


# ─── Sync helper: real Twilio client mock + introspection ───────────────────


class TestTwilioDeleteRecordingsSync:
    """
    The sync helper ``_twilio_delete_recordings`` is the lowest-level
    building block.  These tests verify it talks to the Twilio SDK
    correctly (list + delete per recording).
    """

    def test_lists_then_deletes_each_recording(
        self, monkeypatch, fake_twilio_client
    ) -> None:
        """The helper enumerates recordings.list() and calls .delete() on each."""

        # Monkey-patch the TwilioClient class so the helper picks up our mock.
        def _fake_client_constructor(sid, token):
            return fake_twilio_client

        monkeypatch.setattr(
            "twilio.rest.Client", _fake_client_constructor
        )

        deleted = _twilio_delete_recordings(
            call_sid="CA_LIST_DELETE_TEST",
            twilio_account_sid="AC_TEST",
            twilio_auth_token="token",
        )

        assert deleted == 2, f"Expected 2 recordings deleted, got {deleted}"
        assert fake_twilio_client._list_calls == ["CA_LIST_DELETE_TEST"]
        assert sorted(fake_twilio_client._delete_calls) == sorted(
            [
                "REaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1",
                "REbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb2",
            ]
        )

    def test_empty_recordings_returns_zero(
        self, monkeypatch, fake_twilio_client
    ) -> None:
        """No recordings on the call → list returns [], delete never called."""

        # Override the fixture's configurable list (the side_effect
        # reads from this on every call).
        fake_twilio_client._recordings_to_return = []
        monkeypatch.setattr(
            "twilio.rest.Client", lambda sid, token: fake_twilio_client
        )

        deleted = _twilio_delete_recordings(
            call_sid="CA_EMPTY",
            twilio_account_sid="AC_TEST",
            twilio_auth_token="token",
        )

        assert deleted == 0
        assert fake_twilio_client._delete_calls == []

    def test_missing_credentials_returns_zero(
        self, monkeypatch, fake_twilio_client
    ) -> None:
        """No Twilio creds → returns 0 without touching the SDK."""
        # The Client constructor should NOT be called.
        def _explode(sid, token):
            raise AssertionError(
                "Client() must not be invoked when credentials are missing"
            )

        monkeypatch.setattr("twilio.rest.Client", _explode)

        deleted = _twilio_delete_recordings(
            call_sid="CA_NO_CREDS",
            twilio_account_sid="",  # empty
            twilio_auth_token="",
        )

        assert deleted == 0

    def test_partial_delete_failure_still_logs(
        self, monkeypatch, fake_twilio_client
    ) -> None:
        """If one recording's delete raises, the other is still deleted
        and the helper returns the partial count."""

        # Make rec1's delete raise, rec2's succeed.
        def _maybe_delete(sid):
            if sid == "REaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1":
                raise RuntimeError("simulated Twilio glitch")
            return None

        # Replace the by-sid mapping with one whose delete can raise.
        recordings_resource_1 = MagicMock()
        recordings_resource_1.delete = MagicMock(side_effect=_maybe_delete)
        recordings_resource_2 = MagicMock()
        recordings_resource_2.delete = MagicMock(return_value=None)
        fake_twilio_client.recordings = MagicMock(
            side_effect=lambda sid: (
                recordings_resource_1
                if sid == "REaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1"
                else recordings_resource_2
            )
        )
        monkeypatch.setattr(
            "twilio.rest.Client", lambda sid, token: fake_twilio_client
        )

        deleted = _twilio_delete_recordings(
            call_sid="CA_PARTIAL",
            twilio_account_sid="AC_TEST",
            twilio_auth_token="token",
        )

        assert deleted == 1, (
            f"Partial delete should return 1 (rec2 succeeded); got {deleted}"
        )