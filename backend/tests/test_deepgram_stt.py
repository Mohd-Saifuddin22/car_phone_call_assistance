"""
backend/tests/test_deepgram_stt.py — Unit tests for the Deepgram Flux reader (Issue 035).

Test plan
─────────
Pure-function coverage (no async, no asyncio):
    L1  DEEPGRAM_KEYTERMS has ≤100 entries (Deepgram's hard cap).
    L2  DEEPGRAM_KEYTERMS contains the required domain terms.
    L3  DEEPGRAM_KEYTERMS includes every brand+model name from the catalogue.

Connection-param serialization:
    S1  URL base is wss://api.deepgram.com/v2/listen (Flux requires v2).
    S2  Required scalar params present: model, encoding, sample_rate,
        eot_threshold, eot_timeout_ms.
    S3  language_hint is serialized as a repeated param (hi&en), not a single
        comma-joined token.
    S4  keyterm is serialized as repeated params (one per entry), not a
        comma-joined token.
    S5  Authorization header is "Token <key>".
    S6  eot_threshold and eot_timeout_ms come from caller-supplied values
        (not hardcoded) — verified by passing distinct values.

Event dispatch (pure, supervisor is a stub):
    D1  StartOfTurn is logged at debug; no queue push.
    D2  EagerEndOfTurn is logged at debug; no queue push.
    D3  TurnResumed is logged at debug; no queue push.
    D4  EndOfTurn with non-empty transcript pushes onto completed_turn_queue.
    D5  EndOfTurn with empty transcript is dropped (no push).
    D6  Update / unknown events are debug-logged; no push.
    D7  dispatch_event is tolerant of events missing expected fields.

Inbound queue drain:
    Q1  drain_inbound_queue returns 0 on an empty queue.
    Q2  drain_inbound_queue drops all items and returns the count.
    Q3  drain_inbound_queue does not block (uses get_nowait).

Reconnect loop (async, mocked websockets):
    R1  A single session failure → reconnect with the expected backoff.
    R2  Reconnect succeeds when the next session runs cleanly.
    R3  After _MAX_RECONNECT_ATTEMPTS consecutive failures, the supervisor's
        trigger_teardown is called with provider_fault=True.
    R4  Each reconnect drains the inbound_audio_queue before retrying.

Graceful shutdown (async, mocked websockets):
    G1  When shutdown_event is set during the reader loop, CloseStream is sent.
    G2  After CloseStream, the reader drains any final EndOfTurn into
        completed_turn_queue.
    G3  The run_deepgram_reader task exits cleanly when shutdown_event is set
        without triggering the reconnect loop.

Supervisor integration (Issue 032 teardown + Issue 035 STT task):
    T1  CallSupervisor.__init__ creates completed_turn_queue, shutdown_event,
        stt_task (None).
    T2  start_background_tasks() spawns the STT task (non-blocking).
    T3  trigger_teardown sets shutdown_event and awaits stt_task's graceful
        exit.

Implementation note on mocking
──────────────────────────────
The session-level mocks build a fake ``websockets.connect`` that returns an
async context manager wrapping a FakeWebSocket. The FakeWebSocket records
every ``send()`` call, exposes a programmatic ``push_event`` and
``push_json`` to feed the reader, and can simulate a drop via ``simulate_drop``.
Tests never open a real socket.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import patch

import pytest
import pytest_asyncio

import backend.audio.deepgram_stt as deepgram_stt_mod
from backend.audio.deepgram_stt import (
    build_connection_params,
    dispatch_event,
    drain_inbound_queue,
)
from backend.audio.lexicon import DEEPGRAM_KEYTERMS, DEEPGRAM_KEYTERMS_HINTS
from backend.call_supervisor import CallSupervisor, _active_calls
from backend.concurrency import ConcurrencyManager


# ─── L1/L2/L3 — lexicon contents ──────────────────────────────────────────────


class TestLexiconContents:
    """The keyterms seed list must satisfy Deepgram's hard cap and our domain."""

    def test_keyterms_count_under_hard_cap(self) -> None:
        """Deepgram rejects sessions with more than 100 keyterms."""
        assert len(DEEPGRAM_KEYTERMS) <= 100, (
            f"DEEPGRAM_KEYTERMS has {len(DEEPGRAM_KEYTERMS)} entries — "
            "Deepgram's hard cap is 100. Remove entries or split sessions."
        )

    def test_keyterms_count_comfortably_under_cap(self) -> None:
        """Sanity check — should leave room to add more terms later."""
        # Cap is 100; we want to leave at least a few entries of headroom
        # for future catalogue additions or feature terms without a refactor.
        assert len(DEEPGRAM_KEYTERMS) <= 99

    def test_keyterms_contains_core_domain_terms(self) -> None:
        """The PRD-mandated terms must all be present."""
        for term in ("SUV", "EMI", "sunroof", "petrol", "diesel", "EV", "lakh"):
            assert term in DEEPGRAM_KEYTERMS, (
                f"required domain term {term!r} missing from DEEPGRAM_KEYTERMS"
            )

    def test_keyterms_contains_body_styles(self) -> None:
        for body in ("SUV", "MUV", "Hatchback", "Sedan"):
            assert body in DEEPGRAM_KEYTERMS

    def test_keyterms_contains_all_catalogue_brands(self) -> None:
        """Every brand name from vehicles.sql must be in the lexicon."""
        required_brands = {
            "Tata", "Maruti", "Suzuki", "Mahindra", "Kia", "MG",
            "Toyota", "Hyundai", "Honda", "Skoda", "Volkswagen",
            "Renault", "Nissan", "Citroen", "BYD",
        }
        missing = required_brands - set(DEEPGRAM_KEYTERMS)
        assert not missing, f"missing brand terms: {sorted(missing)}"

    def test_keyterms_contains_all_catalogue_models(self) -> None:
        """Every model token from vehicles.sql must be in the lexicon.

        We test against the unique *tokens* in the model names — multi-word
        models like "Grand i10 Nios" contribute ``Grand``, ``i10``, ``Nios``.
        That's the actual STT-unit granularity; Deepgram weights one token at
        a time.
        """
        catalogue_models = {
            "Nexon", "Swift", "Thar", "Seltos", "Punch", "Creta",
            "Baleno", "Tiago", "XUV300", "Venue", "Sonet", "Dzire",
            "City", "Kushaq", "Virtus", "Harrier", "Fortuner", "Ertiga",
            "Scorpio", "Safari", "Nios", "Kiger", "Magnite", "Hector",
            "Tigor", "Jimny", "Elevate", "Verna", "Slavia", "XUV700",
            "Innova", "Crysta", "Seal",
        }
        missing = catalogue_models - set(DEEPGRAM_KEYTERMS)
        assert not missing, f"missing model terms: {sorted(missing)}"

    def test_keyterms_hints_documents_acronyms(self) -> None:
        """Every acronym in the keyterm list has an expansion hint."""
        acronyms_in_use = {"SUV", "MUV", "EV", "CNG", "AMT", "CVT",
                           "DCT", "EMI", "ABS", "ESP", "ADAS", "NCAP"}
        for acronym in acronyms_in_use:
            assert acronym in DEEPGRAM_KEYTERMS_HINTS, (
                f"acronym {acronym!r} has no expansion hint"
            )

    def test_keyterms_is_immutable_tuple(self) -> None:
        """The lexicon is a tuple — callers cannot mutate it at runtime."""
        assert isinstance(DEEPGRAM_KEYTERMS, tuple)


# ─── S1–S6 — connection-param serialization ──────────────────────────────────


class TestBuildConnectionParams:
    """build_connection_params produces the wire format Deepgram expects."""

    def test_url_base_is_v2_listen(self) -> None:
        """Flux requires /v2/listen — /v1/listen would silently fail."""
        url, _ = build_connection_params("k", eot_threshold=0.7, eot_timeout_ms=5500, language="English")
        assert url.startswith("wss://api.deepgram.com/v2/listen?")

    def test_required_scalar_params_present(self) -> None:
        # Hinglish path exercises the multilingual model name.  The
        # English model name is asserted in TestBuildConnectionParamsLanguageBranching.
        url, _ = build_connection_params("k", eot_threshold=0.7, eot_timeout_ms=5500, language="Hinglish")
        for needle in (
            "model=flux-general-multi",
            "encoding=mulaw",
            "sample_rate=8000",
            "eot_threshold=0.7",
            "eot_timeout_ms=5500",
        ):
            assert needle in url, f"missing required scalar param: {needle}"

    def test_language_hint_serialized_as_repeated_param(self) -> None:
        """language_hint must be repeated, not comma-joined (Hinglish only)."""
        url, _ = build_connection_params("k", eot_threshold=0.7, eot_timeout_ms=5500, language="Hinglish")
        # Repeated form (correct):
        assert "language_hint=hi&language_hint=en" in url
        # Comma-joined form (wrong) must NOT appear:
        assert "language_hint=hi%2Cen" not in url
        assert "language_hint=hi,en" not in url

    def test_keyterm_serialized_as_repeated_param(self) -> None:
        """keyterm must be one URL param per entry."""
        url, _ = build_connection_params("k", eot_threshold=0.7, eot_timeout_ms=5500, language="English")
        # Each lexicon entry appears as its own `keyterm=...` param:
        for term in ("SUV", "MUV", "Hatchback", "lakh", "EMI"):
            assert f"keyterm={term}" in url, f"missing keyterm: {term}"
        # First two keyterms (SUV, MUV) must be consecutive &-joined:
        assert "keyterm=SUV&keyterm=MUV" in url
        # Comma-joined form must NOT appear:
        assert "keyterm=SUV%2CMUV" not in url

    def test_authorization_header_format(self) -> None:
        _, headers = build_connection_params("XYZ123", eot_threshold=0.7, eot_timeout_ms=5500, language="English")
        assert headers == [("Authorization", "Token XYZ123")]

    def test_custom_eot_values_pass_through(self) -> None:
        """Caller-supplied thresholds are reflected verbatim in the URL."""
        url, _ = build_connection_params("k", eot_threshold=0.85, eot_timeout_ms=7500, language="English")
        assert "eot_threshold=0.85" in url
        assert "eot_timeout_ms=7500" in url

    def test_no_double_url_encoding_of_lexicon_terms(self) -> None:
        """Spaces in terms (none currently, but defensive) should not be %20."""
        url, _ = build_connection_params("k", eot_threshold=0.7, eot_timeout_ms=5500, language="English")
        # Every keyterm= value should appear verbatim — no percent-encoding
        # on simple alphanumeric tokens.
        for term in ("Sunroof", "Bluetooth", "Petrol", "Nexon"):
            assert f"keyterm={term}" in url
            assert f"keyterm={term.replace('o', '%6F')}" not in url  # silly defensive check

    def test_keyterm_count_in_url_matches_lexicon(self) -> None:
        """Number of ``keyterm=...`` occurrences in URL == len(lexicon)."""
        url, _ = build_connection_params("k", eot_threshold=0.7, eot_timeout_ms=5500, language="English")
        occurrences = url.count("keyterm=")
        assert occurrences == len(DEEPGRAM_KEYTERMS), (
            f"URL has {occurrences} keyterm params, "
            f"lexicon has {len(DEEPGRAM_KEYTERMS)} entries"
        )


class TestWebsocketsHeaderCompatibility:
    """Deepgram auth works with both WebSocket APIs in the declared range."""

    def test_legacy_api_receives_extra_headers(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        expected_result = object()
        captured: dict[str, Any] = {}

        def _legacy_connect(url: str, *, extra_headers=None):
            captured.update(url=url, headers=extra_headers)
            return expected_result

        monkeypatch.setattr(
            deepgram_stt_mod.websockets,
            "connect",
            _legacy_connect,
        )
        headers = [("Authorization", "Token test-key")]

        result = deepgram_stt_mod._connect_with_headers(
            "wss://api.deepgram.com/v2/listen",
            headers,
        )

        assert result is expected_result
        assert captured == {
            "url": "wss://api.deepgram.com/v2/listen",
            "headers": headers,
        }

    def test_modern_api_receives_additional_headers(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        expected_result = object()
        captured: dict[str, Any] = {}

        def _modern_connect(url: str, *, additional_headers=None):
            captured.update(url=url, headers=additional_headers)
            return expected_result

        monkeypatch.setattr(
            deepgram_stt_mod.websockets,
            "connect",
            _modern_connect,
        )
        headers = [("Authorization", "Token test-key")]

        result = deepgram_stt_mod._connect_with_headers(
            "wss://api.deepgram.com/v2/listen",
            headers,
        )

        assert result is expected_result
        assert captured == {
            "url": "wss://api.deepgram.com/v2/listen",
            "headers": headers,
        }


# ─── Language-branching (Issue 035 follow-up) ────────────────────────────────


class TestBuildConnectionParamsLanguageBranching:
    """
    build_connection_params must select the Flux model and the language-hint
    list based on the lead's preferred language.  English calls use the
    monolingual model and emit no language hints at all; Hinglish calls use
    the multilingual model with repeated ``language_hint=hi`` and
    ``language_hint=en`` params.  Mis-routing a language here causes
    measurable accuracy loss (see Issue 035 follow-up).
    """

    def test_english_uses_english_model(self) -> None:
        """English calls route to flux-general-en for higher accuracy."""
        url, _ = build_connection_params(
            "k", eot_threshold=0.7, eot_timeout_ms=5500, language="English",
        )
        assert "model=flux-general-en" in url
        # And the multilingual model must NOT be present.
        assert "model=flux-general-multi" not in url

    def test_english_emits_no_language_hint_params(self) -> None:
        """
        English calls must NOT include any ``language_hint=`` query
        parameters — biasing Flux toward Hindi on a monolingual English
        call degrades transcription.  We assert absence rather than the
        empty form so a regression that emits a stale ``language_hint=``
        with an empty value is caught.
        """
        url, _ = build_connection_params(
            "k", eot_threshold=0.7, eot_timeout_ms=5500, language="English",
        )
        assert "language_hint=" not in url, (
            f"English call leaked a language_hint param. URL: {url}"
        )

    def test_hinglish_uses_multilingual_model(self) -> None:
        """Hinglish calls route to flux-general-multi."""
        url, _ = build_connection_params(
            "k", eot_threshold=0.7, eot_timeout_ms=5500, language="Hinglish",
        )
        assert "model=flux-general-multi" in url
        # And the English-only model must NOT be present.
        assert "model=flux-general-en" not in url

    def test_hinglish_emits_repeated_hi_and_en_hints(self) -> None:
        """Hinglish sends repeated language_hint=hi and language_hint=en."""
        url, _ = build_connection_params(
            "k", eot_threshold=0.7, eot_timeout_ms=5500, language="Hinglish",
        )
        assert "language_hint=hi&language_hint=en" in url

    def test_invalid_language_raises_value_error(self) -> None:
        """
        Passing any value other than 'English' / 'Hinglish' must raise
        ValueError.  A silent fallback (e.g. defaulting to multilingual)
        would hide caller bugs and corrupt transcripts.
        """
        with pytest.raises(ValueError) as exc_info:
            build_connection_params(
                "k", eot_threshold=0.7, eot_timeout_ms=5500, language="French",
            )
        # Error message must name the bad value for diagnosability.
        assert "French" in str(exc_info.value)
        # And must mention the valid choices so the caller can fix it.
        assert "English" in str(exc_info.value)
        assert "Hinglish" in str(exc_info.value)


# ─── D1–D7 — event dispatch ───────────────────────────────────────────────────


class _StubSupervisor:
    """Minimal supervisor stand-in — fields touched by both dispatch and drain."""

    def __init__(self) -> None:
        self.call_sid = "CA_DISPATCH_TEST"
        self.inbound_audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.completed_turn_queue: asyncio.Queue[str] = asyncio.Queue()


class TestDispatchEvent:
    """Pure routing of Flux events to logs / queue."""

    def _setup(self) -> tuple[_StubSupervisor, list[str]]:
        sup = _StubSupervisor()
        pushed: list[str] = []
        return sup, pushed

    @pytest.mark.asyncio
    async def test_start_of_turn_no_queue_push(self) -> None:
        sup, _ = self._setup()
        dispatch_event({"event": "StartOfTurn", "turn_index": 0,
                        "transcript": "Hi"}, sup)
        assert sup.completed_turn_queue.empty()

    @pytest.mark.asyncio
    async def test_eager_end_of_turn_no_queue_push(self) -> None:
        """Eager signals are observable only — never enqueue (v1 has no
        speculation)."""
        sup, _ = self._setup()
        dispatch_event({"event": "EagerEndOfTurn", "turn_index": 0,
                        "transcript": "Hi I"}, sup)
        assert sup.completed_turn_queue.empty()

    @pytest.mark.asyncio
    async def test_turn_resumed_no_queue_push(self) -> None:
        sup, _ = self._setup()
        dispatch_event({"event": "TurnResumed", "turn_index": 0,
                        "transcript": "Hi I need"}, sup)
        assert sup.completed_turn_queue.empty()

    @pytest.mark.asyncio
    async def test_end_of_turn_pushes_nonempty_transcript(self) -> None:
        sup, _ = self._setup()
        dispatch_event({"event": "EndOfTurn", "turn_index": 0,
                        "transcript": "I want an SUV under fifteen lakh"}, sup)
        assert sup.completed_turn_queue.get_nowait() == \
            "I want an SUV under fifteen lakh"
        assert sup.completed_turn_queue.empty()

    @pytest.mark.asyncio
    async def test_end_of_turn_with_empty_transcript_is_dropped(self) -> None:
        sup, _ = self._setup()
        dispatch_event({"event": "EndOfTurn", "turn_index": 0, "transcript": ""}, sup)
        dispatch_event({"event": "EndOfTurn", "turn_index": 0}, sup)  # missing
        dispatch_event({"event": "EndOfTurn", "turn_index": 0,
                        "transcript": "   "}, sup)  # whitespace-only
        assert sup.completed_turn_queue.empty()

    @pytest.mark.asyncio
    async def test_update_event_no_queue_push(self) -> None:
        sup, _ = self._setup()
        dispatch_event({"event": "Update", "turn_index": 0,
                        "transcript": "partial"}, sup)
        assert sup.completed_turn_queue.empty()

    @pytest.mark.asyncio
    async def test_unknown_event_no_queue_push(self) -> None:
        sup, _ = self._setup()
        dispatch_event({"event": "Connected"}, sup)
        dispatch_event({"event": "SomethingFromTheFuture"}, sup)
        assert sup.completed_turn_queue.empty()

    @pytest.mark.asyncio
    async def test_event_missing_event_field_does_not_crash(self) -> None:
        sup, _ = self._setup()
        # No ``event`` key at all — dispatch_event must be tolerant.
        dispatch_event({"turn_index": 0, "transcript": "huh"}, sup)
        assert sup.completed_turn_queue.empty()

    @pytest.mark.asyncio
    async def test_multiple_end_of_turns_in_order(self) -> None:
        """Each EndOfTurn pushes one transcript; FIFO order preserved."""
        sup, _ = self._setup()
        for i, t in enumerate(("first turn", "second turn", "third turn")):
            dispatch_event({"event": "EndOfTurn", "turn_index": i, "transcript": t}, sup)
        seen = [
            sup.completed_turn_queue.get_nowait(),
            sup.completed_turn_queue.get_nowait(),
            sup.completed_turn_queue.get_nowait(),
        ]
        assert seen == ["first turn", "second turn", "third turn"]


# ─── Q1–Q3 — inbound queue drain ──────────────────────────────────────────────


class TestDrainInboundQueue:
    """Drain is non-blocking; returns the count; tolerates empty queues."""

    def _sup_with_queue(self, items: list[bytes]) -> _StubSupervisor:
        sup = _StubSupervisor()
        for item in items:
            sup.inbound_audio_queue.put_nowait(item)  # type: ignore[attr-defined]
        return sup

    def test_drain_empty_queue_returns_zero(self) -> None:
        sup = _StubSupervisor()
        assert drain_inbound_queue(sup) == 0  # type: ignore[arg-type]

    def test_drain_returns_count_and_empties(self) -> None:
        sup = self._sup_with_queue([b"a", b"b", b"c", b"d", b"e"])
        assert drain_inbound_queue(sup) == 5  # type: ignore[arg-type]
        assert sup.inbound_audio_queue.empty()  # type: ignore[attr-defined]

    def test_drain_does_not_block(self) -> None:
        """Drain must be safe to call from sync code (e.g., reconnect loop
        before its next await)."""
        sup = self._sup_with_queue([bytes([i]) for i in range(50)])
        # If this blocked, the test would hang and pytest would time out.
        n = drain_inbound_queue(sup)  # type: ignore[arg-type]
        assert n == 50


# ─── Async test scaffolding — fake websocket + supervisor ─────────────────────


class FakeWebSocket:
    """
    In-memory WebSocket stand-in for tests.

    Implements the subset of the ``websockets.ClientConnection`` surface that
    ``backend.audio.deepgram_stt`` uses: ``send``, ``recv``, and the
    ``async for msg in ws`` protocol (``__aiter__`` / ``__anext__``).

    Records every ``send()`` and exposes ``push_event`` / ``push_json`` to
    feed the reader's recv stream.  ``simulate_drop`` raises
    ``ConnectionClosed`` on the next recv / iteration so the reader sees a
    clean network error.
    """

    def __init__(self) -> None:
        self.sent: list[Any] = []
        self._incoming: asyncio.Queue[Any] = asyncio.Queue()
        self._drop_requested: bool = False
        self.closed = False

    async def send(self, data: Any) -> None:
        if self.closed:
            raise RuntimeError("send on closed socket")
        self.sent.append(data)

    async def recv(self) -> Any:
        if self._drop_requested:
            self.closed = True
            self._drop_requested = False
            from websockets.exceptions import ConnectionClosed
            raise ConnectionClosed(None, None)  # type: ignore[arg-type]
        return await self._incoming.get()

    def __aiter__(self) -> "FakeWebSocket":
        return self

    async def __anext__(self) -> Any:
        if self._drop_requested:
            self.closed = True
            self._drop_requested = False
            from websockets.exceptions import ConnectionClosed
            raise ConnectionClosed(None, None)  # type: ignore[arg-type]
        try:
            return await self._incoming.get()
        except asyncio.CancelledError:
            raise
        # Note: an empty queue + asyncio.Queue.get() would block forever here.
        # Tests that drive ``async for`` on this fake must push events first.

    def push_json(self, obj: dict) -> None:
        """Push a dict, encoded as JSON, into the reader's queue."""
        self._incoming.put_nowait(json.dumps(obj))

    def push_event(self, event_type: str, **fields: Any) -> None:
        """Convenience: push a Flux-style event."""
        self.push_json({"event": event_type, **fields})

    def push_raw(self, raw: bytes | str) -> None:
        """Push a raw non-JSON message (to test malformed-input tolerance)."""
        self._incoming.put_nowait(raw)

    def simulate_drop(self) -> None:
        """Make the next recv / iteration raise ``ConnectionClosed``."""
        self._drop_requested = True

    async def aclose(self) -> None:
        self.closed = True

    async def close(self) -> None:
        self.closed = True


@asynccontextmanager
async def _fake_ws_context(ws: FakeWebSocket):
    """``async with websockets.connect(...) as ws:`` shim."""
    yield ws


def _stub_websockets_connect(ws: FakeWebSocket):
    """Return a sync callable that returns the async-context-manager shim."""

    def _connect(url, *, additional_headers=None, **kwargs):
        return _fake_ws_context(ws)

    return _connect


class _StubSupervisorForSession:
    """Supervisor stand-in for the session-loop tests.

    Mirrors the parts of CallSupervisor the reader/writer loops touch.
    """

    def __init__(self) -> None:
        self.call_sid = "CA_SESSION_TEST"
        # ``language`` is read by ``build_connection_params`` to pick the
        # Flux model + language hints.  Default to English in session tests
        # since they don't exercise the language branch (covered by the
        # dedicated ``TestBuildConnectionParamsLanguageBranching`` class).
        self.language = "English"
        self.inbound_audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.completed_turn_queue: asyncio.Queue[str] = asyncio.Queue()
        self.shutdown_event: asyncio.Event = asyncio.Event()
        self._teardown_run = False

    async def trigger_teardown(
        self,
        outcome_status: str,
        failure_reason: str | None = None,
        provider_fault: bool = False,
        completed_timeout: bool = False,
        call_already_terminal: bool = False,
    ) -> None:
        """Mirror CallSupervisor's latch + trigger side-effects."""
        if self._teardown_run:
            return
        self._teardown_run = True
        self.last_outcome_status = outcome_status
        self.last_failure_reason = failure_reason
        self.last_provider_fault = provider_fault
        # Mirror the supervisor's shutdown_event set + task cancel, so the
        # reader loop sees the signal and exits the same way it would in
        # production.
        self.shutdown_event.set()


# ─── R1–R4 — reconnect loop ───────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_active_calls_registry():
    _active_calls.clear()
    yield
    _active_calls.clear()


class TestReconnectLoop:
    """run_deepgram_reader reconnects with bounded attempts on session errors."""

    @pytest.mark.asyncio
    async def test_session_failure_triggers_reconnect(self) -> None:
        """First session errors, second session runs cleanly — exactly one reconnect."""
        # Build a list of FakeWebSocket instances the connect() shim will hand out.
        ws_fail = FakeWebSocket()
        ws_fail.simulate_drop()  # recv() will raise ConnectionClosed immediately
        ws_ok = FakeWebSocket()
        ws_ok.push_event("EndOfTurn", turn_index=0,
                         transcript="hello after reconnect")

        call_count = {"n": 0}
        sockets = [ws_fail, ws_ok]

        def connect(url, *, additional_headers=None, **kwargs):
            ws = sockets[call_count["n"]]
            call_count["n"] += 1

            @asynccontextmanager
            async def ctx():
                yield ws

            return ctx()

        sup = _StubSupervisorForSession()

        with patch("backend.audio.deepgram_stt.websockets") as ws_mod:
            ws_mod.connect = connect
            # Speed up reconnect so the test isn't slow.
            with patch("backend.audio.deepgram_stt._INITIAL_RECONNECT_BACKOFF_S", 0.0), \
                 patch("backend.audio.deepgram_stt._BACKOFF_GROWTH", 0.0):
                # Drive shutdown_event so the reader exits cleanly after one cycle.
                async def shutdown_after_a_beat():
                    await asyncio.sleep(0.05)
                    sup.shutdown_event.set()

                asyncio.create_task(shutdown_after_a_beat())

                from backend.audio.deepgram_stt import run_deepgram_reader
                # Bound the run so a regression doesn't hang the suite:
                await asyncio.wait_for(run_deepgram_reader(sup), timeout=2.0)

        # Two connect() calls: one failure, one success.
        assert call_count["n"] == 2
        # The successful session pushed its transcript:
        assert sup.completed_turn_queue.get_nowait() == "hello after reconnect"
        assert sup.completed_turn_queue.empty()
        # No teardown trigger fired (success path).
        assert not sup._teardown_run

    @pytest.mark.asyncio
    async def test_reconnect_exhaustion_triggers_provider_fault_teardown(self) -> None:
        """After _MAX_RECONNECT_ATTEMPTS consecutive failures, trigger_teardown is called."""
        # Every connect returns a ws that immediately closes.
        attempts: list[int] = []

        def connect(url, *, additional_headers=None, **kwargs):
            attempts.append(1)
            ws = FakeWebSocket()
            ws.simulate_drop()

            @asynccontextmanager
            async def ctx():
                yield ws

            return ctx()

        sup = _StubSupervisorForSession()

        with patch("backend.audio.deepgram_stt.websockets") as ws_mod:
            ws_mod.connect = connect
            with patch("backend.audio.deepgram_stt._INITIAL_RECONNECT_BACKOFF_S", 0.0), \
                 patch("backend.audio.deepgram_stt._BACKOFF_GROWTH", 0.0), \
                 patch("backend.audio.deepgram_stt._MAX_RECONNECT_ATTEMPTS", 3):
                from backend.audio.deepgram_stt import run_deepgram_reader
                await asyncio.wait_for(run_deepgram_reader(sup), timeout=2.0)

        # 3 connect attempts, no more — backstop enforced.
        assert len(attempts) == 3
        # Teardown fired with provider_fault=True.
        assert sup._teardown_run is True
        assert sup.last_outcome_status == "Failed"
        assert sup.last_failure_reason == "error"
        assert sup.last_provider_fault is True

    @pytest.mark.asyncio
    async def test_each_reconnect_drains_inbound_audio_queue(self) -> None:
        """Stale audio must be dropped before each reconnect attempt."""
        # First connect succeeds but the session fails on the second event.
        # Pre-load stale audio so we can verify the drain.
        ws_first = FakeWebSocket()
        ws_first.push_event("StartOfTurn", turn_index=0, transcript="hi")
        # Make the second ws call raise so we reconnect.
        ws_first.simulate_drop()
        # Second session runs cleanly.
        ws_second = FakeWebSocket()
        ws_second.push_event("EndOfTurn", turn_index=0, transcript="hi again")

        ws_pool = [ws_first, ws_second]
        idx = {"n": 0}

        def connect(url, *, additional_headers=None, **kwargs):
            ws = ws_pool[idx["n"]]
            idx["n"] += 1

            @asynccontextmanager
            async def ctx():
                yield ws

            return ctx()

        sup = _StubSupervisorForSession()
        # Stale audio waiting on the queue:
        sup.inbound_audio_queue.put_nowait(b"\x00" * 640)
        sup.inbound_audio_queue.put_nowait(b"\x00" * 640)
        sup.inbound_audio_queue.put_nowait(b"\x00" * 640)

        with patch("backend.audio.deepgram_stt.websockets") as ws_mod:
            ws_mod.connect = connect
            with patch("backend.audio.deepgram_stt._INITIAL_RECONNECT_BACKOFF_S", 0.0), \
                 patch("backend.audio.deepgram_stt._BACKOFF_GROWTH", 0.0):

                async def shutdown_after_a_beat():
                    await asyncio.sleep(0.05)
                    sup.shutdown_event.set()

                asyncio.create_task(shutdown_after_a_beat())

                from backend.audio.deepgram_stt import run_deepgram_reader
                await asyncio.wait_for(run_deepgram_reader(sup), timeout=2.0)

        # Two reconnect cycles, queue empty (drained between them).
        assert idx["n"] == 2
        assert sup.inbound_audio_queue.empty()
        # The successful session's transcript was preserved:
        assert sup.completed_turn_queue.get_nowait() == "hi again"


# ─── G1–G3 — graceful shutdown ────────────────────────────────────────────────


class TestGracefulShutdown:
    """When shutdown_event is set, the reader sends CloseStream and exits."""

    @pytest.mark.asyncio
    async def test_close_stream_sent_when_shutdown_event_fires(self) -> None:
        """shutdown_event triggers a {"type": "CloseStream"} JSON control message."""
        ws = FakeWebSocket()
        ws.push_event("StartOfTurn", turn_index=0, transcript="hi")

        # Schedule shutdown after a short delay so the reader sees the start
        # event first.
        sup = _StubSupervisorForSession()

        async def shutdown_later():
            await asyncio.sleep(0.02)
            sup.shutdown_event.set()

        asyncio.create_task(shutdown_later())

        def connect(url, *, additional_headers=None, **kwargs):
            @asynccontextmanager
            async def ctx():
                yield ws
            return ctx()

        with patch("backend.audio.deepgram_stt.websockets") as ws_mod:
            ws_mod.connect = connect
            with patch("backend.audio.deepgram_stt._INITIAL_RECONNECT_BACKOFF_S", 0.0):
                from backend.audio.deepgram_stt import run_deepgram_reader
                await asyncio.wait_for(run_deepgram_reader(sup), timeout=2.0)

        # Among the messages sent, exactly one must be a CloseStream JSON.
        close_msgs = [
            s for s in ws.sent
            if isinstance(s, str) and json.loads(s).get("type") == "CloseStream"
        ]
        assert len(close_msgs) == 1, (
            f"expected exactly one CloseStream; got sent list = {ws.sent!r}"
        )

    @pytest.mark.asyncio
    async def test_final_transcript_drained_after_close_stream(self) -> None:
        """EndOfTurn events arriving after CloseStream are pushed to completed_turn_queue."""
        ws = FakeWebSocket()
        # Queue the final transcript BEFORE shutdown so it arrives during the
        # CloseStream drain phase.
        ws.push_event("EndOfTurn", turn_index=0,
                      transcript="this is the final turn")
        # After CloseStream, send another transcript to confirm drain semantics.
        # (In production, Flux sends the final TurnInfo then closes.)

        sup = _StubSupervisorForSession()
        sup.shutdown_event.set()  # already set — reader should still drain.

        def connect(url, *, additional_headers=None, **kwargs):
            @asynccontextmanager
            async def ctx():
                yield ws
            return ctx()

        with patch("backend.audio.deepgram_stt.websockets") as ws_mod:
            ws_mod.connect = connect
            from backend.audio.deepgram_stt import run_deepgram_reader
            await asyncio.wait_for(run_deepgram_reader(sup), timeout=2.0)

        # The EndOfTurn was pushed:
        assert sup.completed_turn_queue.get_nowait() == \
            "this is the final turn"
        assert sup.completed_turn_queue.empty()
        # And CloseStream was sent.
        assert any(
            isinstance(s, str) and json.loads(s).get("type") == "CloseStream"
            for s in ws.sent
        )

    @pytest.mark.asyncio
    async def test_clean_exit_does_not_trigger_reconnect_loop(self) -> None:
        """A session that ends cleanly (shutdown_event set) must NOT reconnect."""
        ws = FakeWebSocket()
        # No events — reader loops on recv().  shutdown_event forces exit.
        sup = _StubSupervisorForSession()
        sup.shutdown_event.set()  # pre-set: reader sees it before first event.

        connect_calls = {"n": 0}

        def connect(url, *, additional_headers=None, **kwargs):
            connect_calls["n"] += 1
            @asynccontextmanager
            async def ctx():
                yield ws
            return ctx()

        with patch("backend.audio.deepgram_stt.websockets") as ws_mod:
            ws_mod.connect = connect
            from backend.audio.deepgram_stt import run_deepgram_reader
            await asyncio.wait_for(run_deepgram_reader(sup), timeout=2.0)

        # Exactly ONE connect — no reconnect attempt.
        assert connect_calls["n"] == 1
        # No teardown triggered.
        assert not sup._teardown_run


# ─── T1–T3 — CallSupervisor integration ───────────────────────────────────────


class TestCallSupervisorSTTIntegration:
    """The supervisor owns the STT task and shuts it down during teardown.

    The ``supervisor`` fixture is async: it constructs a fresh supervisor,
    yields it, and — via try/finally — cleans up every task the tests
    spawn through ``start_background_tasks()`` (Issue 041: STT, Mercury
    driver, Matcher).  Cleanup is per-test, owner-aware, and never
    depends on a global task-drain.
    """

    @pytest_asyncio.fixture
    async def supervisor(self):
        """
        Per-test, owner-aware task lifecycle.

        Steps in order:

          1. Construct a fresh ``CallSupervisor``.
          2. Yield to the test.
          3. Cleanup — terminate every task the test may have spawned:

             a. ``MatcherConsumer.close()`` (graceful drain via sentinel).
             b. ``task.cancel()`` on each of STT / Mercury driver / reader /
                sender; ``asyncio.wait_for(gather(..., return_exceptions=True),
                timeout=5.0)``.

          4. If matcher graceful close FAILS, fall back to cancelling the
             matcher's internal task directly so the test never leaks a
             running queue-reader.  The test fails on any other error.

          5. Surface a *single* deduplicated set of non-``CancelledError``
             exceptions from BOTH the in-flight gather AND from any
             already-completed task handles — the same exception must not
             be logged twice.
        """
        import logging

        log = logging.getLogger(__name__)

        cm = ConcurrencyManager()
        cm.acquire()
        sup = CallSupervisor(
            lead_id=1,
            attempt_id=1,
            call_sid="CA_SUP_STT_TEST",
            language="English",
            concurrency_manager=cm,
            pool=None,
        )

        # Deduplicated set of surfaced exceptions.  Keyed by id() so the
        # same exception instance is never reported twice.
        surfaced: dict[int, BaseException] = {}

        def _record(exc: BaseException) -> None:
            surfaced.setdefault(id(exc), exc)

        try:
            yield sup
        finally:
            # Step 3a: Matcher's graceful shutdown.  CancelledError is
            # the expected outcome; any other error falls back to
            # cancelling the internal task and recording the original
            # exception for surfacing.
            matcher = getattr(sup, "matcher_task", None)
            if matcher is not None and hasattr(matcher, "close"):
                try:
                    await matcher.close()
                except asyncio.CancelledError:
                    pass  # expected: queue.get cancelled on shutdown
                except BaseException as exc:  # noqa: BLE001
                    # Graceful path failed → fall back to cancelling the
                    # matcher's internal task so we do not leak a running
                    # queue-reader.  Then surface the original error.
                    internal = getattr(sup, "_matcher_internal_task", None)
                    if (
                        internal is not None
                        and not internal.done()
                    ):
                        internal.cancel()
                        try:
                            await internal
                        except BaseException as cancel_exc:  # noqa: BLE001
                            # The cancel-induced exception is the same one
                            # we are surfacing — don't double-log.
                            if id(cancel_exc) != id(exc):
                                _record(cancel_exc)
                    _record(exc)

            # Step 3b: STT / Mercury / reader / sender cancel + await.
            owned_tasks = [
                t
                for t in (
                    getattr(sup, "stt_task", None),
                    getattr(sup, "mercury_task", None),
                    getattr(sup, "reader_task", None),
                    getattr(sup, "sender_task", None),
                )
                if t is not None and not t.done()
            ]
            for task in owned_tasks:
                task.cancel()
            if owned_tasks:
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*owned_tasks, return_exceptions=True),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    _record(
                        TimeoutError(
                            f"supervisor fixture cleanup timed out after 5s "
                            f"waiting on {len(owned_tasks)} task(s)"
                        )
                    )
                else:
                    for task, result in zip(owned_tasks, results):
                        if isinstance(result, BaseException) and not isinstance(
                            result, asyncio.CancelledError
                        ):
                            _record(result)

            # After the in-flight gather, also walk the supervisor's
            # *already-completed* task handles and surface anything that
            # raised but was not yet seen in the gather results.  This
            # catches tasks that completed before the gather started.
            for attr in ("stt_task", "mercury_task", "reader_task", "sender_task"):
                task = getattr(sup, attr, None)
                if task is None or task.cancelled():
                    continue
                if not task.done():
                    # Still running — should not happen since we cancelled
                    # it above, but defensive.
                    continue
                if task in owned_tasks:
                    # Already inspected via gather results.
                    continue
                exc = task.exception()
                if exc is not None and not isinstance(exc, asyncio.CancelledError):
                    _record(exc)

            if surfaced:
                # Log each surfaced exception ONCE, with the actual
                # exception context.  We are inside an except block-free
                # path, so use ``log.error(..., exc_info=...)`` to attach
                # the original traceback to the log record (NOT
                # ``log.exception``, which is only valid inside an
                # except block).
                lines: list[str] = []
                for exc in surfaced.values():
                    log.error(
                        "test_deepgram_stt supervisor fixture cleanup error: %s",
                        exc,
                        exc_info=exc,
                    )
                    lines.append(repr(exc))
                raise RuntimeError(
                    "test_deepgram_stt supervisor fixture: cleanup surfaced "
                    f"{len(surfaced)} error(s): " + "; ".join(lines)
                )

    def test_init_creates_stt_state(self, supervisor):
        """__init__ exposes the new pipeline-task fields."""
        from backend.audio.deepgram_stt import build_connection_params  # noqa: F401
        assert isinstance(supervisor.completed_turn_queue, asyncio.Queue)
        assert isinstance(supervisor.shutdown_event, asyncio.Event)
        assert supervisor.stt_task is None

    @pytest.mark.asyncio
    async def test_start_background_tasks_spawns_stt_task(self, supervisor):
        """start_background_tasks creates the asyncio.Task handle."""
        # Patch run_deepgram_reader to immediately set shutdown_event and exit,
        # so the task finishes quickly and we can observe the handle.
        async def fake_run_deepgram_reader(sup):
            sup.shutdown_event.set()
            return

        # Stub every other provider entry point start_background_tasks() spawns.
        async def fake_no_op(_sup):
            try:
                await asyncio.Event().wait()  # park; fixture cleanup cancels us
            except asyncio.CancelledError:
                raise

        with patch(
            "backend.audio.deepgram_stt.run_deepgram_reader",
            side_effect=fake_run_deepgram_reader,
        ), patch(
            "backend.conversation_driver.run_mercury_conversation_driver",
            side_effect=fake_no_op,
        ):
            supervisor.start_background_tasks()
            # Let the fake STT task finish on its own.
            assert supervisor.stt_task is not None
            assert not supervisor.stt_task.done()
            await supervisor.stt_task
            assert supervisor.stt_task.done()

    @pytest.mark.asyncio
    async def test_start_background_tasks_idempotent(self, supervisor):
        """Calling start_background_tasks twice does not spawn a second task."""
        async def fake_run(sup):
            sup.shutdown_event.set()
            return

        async def fake_no_op(_sup):
            try:
                await asyncio.Event().wait()  # parked; fixture cleanup cancels
            except asyncio.CancelledError:
                raise

        with patch(
            "backend.audio.deepgram_stt.run_deepgram_reader",
            side_effect=fake_run,
        ), patch(
            "backend.conversation_driver.run_mercury_conversation_driver",
            side_effect=fake_no_op,
        ):
            supervisor.start_background_tasks()
            first = supervisor.stt_task
            supervisor.start_background_tasks()  # second call — should no-op
            assert first is supervisor.stt_task
            await first  # drain

    @pytest.mark.asyncio
    async def test_trigger_teardown_sets_shutdown_and_awaits_stt_task(
        self, supervisor,
    ):
        """The teardown sequence must:

          1. set ``shutdown_event`` so the reader's next iteration sees it;
          2. ``await`` the STT task to completion (or cancel+await it);
          3. clear the ``stt_task`` attribute so no strong reference to a
             finished Task lingers on the supervisor (Issue 032/035/041
             teardown contract — release the handle to guarantee no
             dangling Task objects block GC or pytest teardown);
          4. flip the ``_teardown_run`` latch exactly once.

        The test mocks every outbound provider call so no real Twilio /
        Mercury / Deepgram network traffic occurs.  ``_delete_call_recordings_async``
        is stubbed because ``no_answer`` failures trigger Twilio's recording
        cleanup path, which must NOT reach the live Twilio servers.

        We capture a local reference to ``stt_task`` BEFORE teardown so the
        assertion that the task is ``done()`` is not defeated by the same
        attribute-clear that the production code performs at line ~1092 of
        call_supervisor.py.
        """
        observed_shutdown = {"at_end": False}

        async def fake_run(sup):
            try:
                await sup.shutdown_event.wait()
                observed_shutdown["at_end"] = True
            except asyncio.CancelledError:
                observed_shutdown["at_end"] = sup.shutdown_event.is_set()
                raise

        # Matcher's internal task — park until fixture cleanup cancels us.
        async def fake_no_op(_sup):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise

        async def fake_resolve_failed(*_a: Any, **_kw: Any) -> None:
            return None

        async def fake_delete_recordings(_call_sid: str) -> int:
            return 0

        with patch(
            "backend.audio.deepgram_stt.run_deepgram_reader",
            side_effect=fake_run,
        ), patch(
            "backend.conversation_driver.run_mercury_conversation_driver",
            side_effect=fake_no_op,
        ), patch(
            "backend.call_supervisor._resolve_failed_attempt",
            side_effect=fake_resolve_failed,
        ), patch(
            "backend.call_supervisor._twilio_hangup_call",
            return_value=True,
        ), patch(
            "backend.call_supervisor._delete_call_recordings_async",
            side_effect=fake_delete_recordings,
        ):
            supervisor.start_background_tasks()
            await asyncio.sleep(0)  # yield so tasks start
            # Capture the handle BEFORE teardown; teardown will null it.
            captured_stt_task = supervisor.stt_task
            assert captured_stt_task is not None
            assert not captured_stt_task.done()

            await supervisor.trigger_teardown("Failed", "no_answer")

        # Contract: shutdown_event is set.
        assert supervisor.shutdown_event.is_set()
        # Contract: the captured task was awaited to completion (not leaked).
        assert observed_shutdown["at_end"] is True
        assert captured_stt_task.done(), (
            "STT task was not awaited by trigger_teardown — it must complete "
            "(or be cancelled+awaited) before the supervisor releases its "
            "handle; otherwise a finished coroutine leaks across the test."
        )
        # Contract: handle cleared per Issue 032/035/041 — no dangling Task.
        assert supervisor.stt_task is None
        # Contract: one-shot latch was flipped exactly once.
        assert supervisor._teardown_run is True

    @pytest.mark.asyncio
    async def test_trigger_teardown_with_no_stt_task_does_not_raise(
        self, supervisor,
    ):
        """If the STT task was never started, teardown still completes cleanly."""
        assert supervisor.stt_task is None

        async def fake_resolve_failed(*a, **kw):
            return None

        with patch(
            "backend.call_supervisor._resolve_failed_attempt",
            side_effect=fake_resolve_failed,
        ), patch(
            "backend.call_supervisor._twilio_hangup_call",
            return_value=True,
        ):
            # Must NOT raise even though there's no STT task to cancel.
            await supervisor.trigger_teardown("Failed", "dropped")

        assert supervisor._teardown_run is True
        # shutdown_event is still set so any future task spawn sees the signal.
        assert supervisor.shutdown_event.is_set()
