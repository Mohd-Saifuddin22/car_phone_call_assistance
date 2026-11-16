"""
backend/tests/test_network_guard.py — focused behavioral tests for the
test-suite network guard.

These tests exercise the **real** ``_scoped_network_guard`` autouse
fixture (installed by ``conftest.py``) — they do not reimplement the
guard.  The conftest patches ``socket.socket.connect`` and the httpx
senders; these tests make actual ``socket.connect`` / ``httpx.Client.send``
calls and assert that the real guard rejects (or allows) them by
catching the canonical ``backend.tests._network_guard.ExternalTrafficError``
class.

Contract being proved:

  1. Public IPv4 destinations are BLOCKED at ``socket.connect``.
  2. Public IPv6 destinations are BLOCKED at ``socket.connect``.
  3. ``127.0.0.1`` (IPv4 loopback) is ALLOWED.
  4. ``::1`` (IPv6 loopback) is ALLOWED.
  5. Unix-domain sockets are ALLOWED.
  6. RFC1918 private addresses are BLOCKED.
  7. httpx requests to non-loopback URLs are BLOCKED.
  8. In-process ASGI tests still work (synthetic test hostnames).
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from httpx import ASGITransport, AsyncClient

from backend.tests._network_guard import (
    ExternalTrafficError,
    extract_connect_target,
    host_is_loopback,
    host_is_synthetic_test,
)
from backend.tests.conftest import _scoped_network_guard  # noqa: F401 — pulls fixture

if TYPE_CHECKING:
    pass


# ─── 1. ``socket.connect`` blocks public IPv4 ──────────────────────────────


def test_socket_connect_blocks_public_ipv4() -> None:
    """
    ``socket.socket.connect((public_ipv4, 443))`` raises the canonical
    ``ExternalTrafficError`` before any real TCP attempt is made.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.close()
    with pytest.raises(ExternalTrafficError, match="8\\.8\\.8\\.8"):
        s.connect(("8.8.8.8", 443))


# ─── 2. ``socket.connect`` blocks public IPv6 ──────────────────────────────


def test_socket_connect_blocks_public_ipv6() -> None:
    """Public IPv6 is non-loopback → blocked."""
    s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    s.close()
    with pytest.raises(ExternalTrafficError, match="2606:4700"):
        s.connect(("2606:4700:4700::1111", 443, 0, 0))


# ─── 3. ``socket.connect`` allows 127.0.0.1 ────────────────────────────────


def test_socket_connect_allows_ipv4_loopback() -> None:
    """Connecting to ``127.0.0.1`` must succeed (no exception from guard)."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("127.0.0.1", port))  # must not raise ExternalTrafficError
        client.close()
    finally:
        server.close()


# ─── 4. ``socket.connect`` allows ::1 ──────────────────────────────────────


def test_socket_connect_allows_ipv6_loopback() -> None:
    """Connecting to ``::1`` must succeed (no exception from guard)."""
    server = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    try:
        server.bind(("::1", 0))
    except OSError:
        server.close()
        pytest.skip("IPv6 loopback bind unavailable on this host")
        return
    server.listen(1)
    port = server.getsockname()[1]
    try:
        client = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        try:
            client.connect(("::1", port))
            client.close()
        except ExternalTrafficError:
            client.close()
            pytest.fail("Guard wrongly rejected ::1")
    finally:
        server.close()


# ─── 5. ``socket.connect`` allows Unix-domain sockets ──────────────────────


def test_socket_connect_allows_unix_socket(tmp_path: Path) -> None:
    """A string ``address`` is a Unix-domain path → allowed."""
    sock_path = tmp_path / "test.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))
    server.listen(1)
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(sock_path))  # must not raise ExternalTrafficError
        client.close()
    finally:
        server.close()
        sock_path.unlink(missing_ok=True)


# ─── 6. ``socket.connect`` blocks RFC1918 / link-local ───────────────────


@pytest.mark.parametrize(
    "private_ip",
    ["10.0.0.1", "172.16.5.5", "192.168.0.1", "169.254.1.1"],
)
def test_socket_connect_blocks_rfc1918_and_link_local(private_ip: str) -> None:
    """RFC1918 + link-local are still external — must be rejected."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.close()
    with pytest.raises(ExternalTrafficError):
        s.connect((private_ip, 443))


# ─── 7. Unrecognised AF_INET/AF_INET6 address shape is rejected ──────────


def test_socket_connect_rejects_unsupported_address_shape() -> None:
    """
    An address shape the guard does not recognise (e.g. a 3-tuple IPv4
    form) is REJECTED rather than silently delegated to the real connect.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.close()
    # Pass an integer (clearly not a valid host representation).
    with pytest.raises(ExternalTrafficError):
        s.connect(12345)


# ─── 8. httpx blocks non-loopback URLs ────────────────────────────────────


def test_httpx_sender_blocks_public_ipv4_url() -> None:
    """``httpx.Client.send`` to ``https://api.twilio.com/health`` is rejected."""
    import httpx

    req = httpx.Request("GET", "https://api.twilio.com/health")
    with pytest.raises(ExternalTrafficError, match="api\\.twilio\\.com"):
        with httpx.Client() as client:
            client.send(req)


# ─── 9. In-process ASGI still works under the guard ─────────────────────


@pytest.mark.asyncio
async def test_in_process_asgi_still_works_under_guard() -> None:
    """``ASGITransport`` is in-process; the guard does not break it.

    The autouse ``_scoped_network_guard`` fixture is active for every
    test (pulled in by ``from backend.tests.conftest import
    _scoped_network_guard`` at module import).  ``ASGITransport`` does
    not open a real socket, so the guard has nothing to intercept.
    """
    from fastapi import FastAPI

    app = FastAPI()

    @app.get("/ping")
    async def _ping() -> dict[str, str]:
        return {"pong": "ok"}

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",  # synthetic httpx host — allowed
    ) as ac:
        response = await ac.get("/ping")
        assert response.status_code == 200
        assert response.json() == {"pong": "ok"}


# ─── 10. ``testserver`` synthetic host also works for in-process ASGI ─


@pytest.mark.asyncio
async def test_testserver_synthetic_host_works_under_guard() -> None:
    """``http://testserver/`` is the Starlette default — also allowed."""
    from fastapi import FastAPI

    app = FastAPI()

    @app.get("/ping")
    async def _ping() -> dict[str, object]:
        return {"ok": True}

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        response = await ac.get("/ping")
        assert response.status_code == 200


# ─── 11. Direct predicate checks ──────────────────────────────────────


@pytest.mark.parametrize(
    "host,expected",
    [
        ("127.0.0.1", True),
        ("127.10.20.30", True),
        ("::1", True),
        ("127.0.0.2", True),
        # Public IPv4 / IPv6 — must be blocked
        ("8.8.8.8", False),
        ("1.1.1.1", False),
        ("2606:4700:4700::1111", False),
        # RFC1918 private ranges must be blocked too
        ("10.0.0.1", False),
        ("172.16.5.5", False),
        ("192.168.0.1", False),
        # IPv6 link-local is *not* loopback
        ("fe80::1", False),
        # IPv4 link-local is *not* loopback
        ("169.254.1.1", False),
        # None / empty
        (None, False),
        ("", False),
    ],
)
def test_host_is_loopback_classification(host: str | None, expected: bool) -> None:
    """``host_is_loopback`` only treats IPv4 127/8 and IPv6 ::1 as loopback.

    Synthetic httpx/Starlette hostnames (``test`` / ``testserver``) are
    NOT classified as loopback here — they are allowed only at the
    httpx URL layer.
    """
    assert host_is_loopback(host) is expected


@pytest.mark.parametrize(
    "host,expected",
    [
        ("test", True),
        ("testserver", True),
        ("127.0.0.1", False),
        ("8.8.8.8", False),
        ("", False),
        (None, False),
    ],
)
def test_host_is_synthetic_test_classification(
    host: str | None, expected: bool
) -> None:
    """``host_is_synthetic_test`` matches only the well-known synthetic ASGI hostnames."""
    assert host_is_synthetic_test(host) is expected


# ─── 12. ``extract_connect_target`` classification ────────────────────


def test_extract_connect_target_unix() -> None:
    kind, payload = extract_connect_target("/var/run/something.sock")
    assert kind == "unix"
    assert payload == "/var/run/something.sock"


def test_extract_connect_target_inet() -> None:
    kind, payload = extract_connect_target(("127.0.0.1", 80))
    assert kind == "inet"
    assert payload == "127.0.0.1"


def test_extract_connect_target_rejects_bad_shape() -> None:
    """Integer (or other unsupported shape) raises ExternalTrafficError."""
    with pytest.raises(ExternalTrafficError):
        extract_connect_target(12345)
