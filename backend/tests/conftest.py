"""
backend/tests/conftest.py — shared pytest configuration.

Autouse fixture:

  ``_scoped_network_guard`` — for every test, monkeypatch (auto-reverted on
  teardown):

    * ``socket.socket.connect`` accepts only actual loopback addresses
      (IPv4 ``127.0.0.0/8`` and IPv6 ``::1``) and Unix-domain sockets.
      Every other TCP destination — including RFC1918, link-local, public
      IPs, and the synthetic ``test``/``testserver`` hostnames — is
      REJECTED with ``ExternalTrafficError``.  Address shapes the
      guard does not recognise are rejected immediately rather than
      delegated to the real connect.
    * ``httpx.Client.send`` / ``httpx.AsyncClient.send`` refuse outbound
      URLs whose hostname is not loopback OR one of the in-process
      synthetic ASGI hostnames (``test``, ``testserver``).  Same
      ``ExternalTrafficError``.

The guard never installs anything at conftest import time — every
patch is scoped via ``monkeypatch.setattr`` and reverted at test
teardown.  Tests must fail loudly when a production endpoint is hit
by accident; no silent fallback to a fake success.
"""

from __future__ import annotations

import socket
import urllib.parse
from typing import Any

import pytest

from backend.tests._network_guard import (
    ExternalTrafficError,
    extract_connect_target,
    host_is_loopback,
    host_is_synthetic_test,
)


@pytest.fixture(autouse=True)
def _scoped_network_guard(monkeypatch: pytest.MonkeyPatch) -> Any:
    """
    Per-test network guard.  All patches are scoped via ``monkeypatch`` so
    they auto-revert; nothing is installed at conftest import time.

    Rules:

      * Unix-domain sockets (string ``address``) → allowed.
      * IPv4 ``127.0.0.0/8`` and IPv6 ``::1`` loopback → allowed.
      * httpx outbound URLs whose hostname is loopback or one of the
        synthetic ASGI test hostnames (``test`` / ``testserver``) →
        allowed.  All other hostnames → REJECTED.
      * Anything else (public IPv4/IPv6, RFC1918, link-local) →
        REJECTED with ``ExternalTrafficError``.
    """
    real_socket_connect = socket.socket.connect

    def _guarded_connect(self: socket.socket, address: Any, /) -> None:
        kind, payload = extract_connect_target(address)
        if kind == "unix":
            return real_socket_connect(self, address)
        # kind == "inet"
        if host_is_loopback(payload):
            return real_socket_connect(self, address)
        # NB: synthetic ASGI hostnames ("test" / "testserver") are NOT
        # allowed at the raw socket layer — a real connect to "test"
        # would have to leave the test process.  In-process ASGI fixtures
        # never open a real socket.
        raise ExternalTrafficError(
            f"_scoped_network_guard blocked socket.connect to non-loopback "
            f"host {payload!r}. Loopback + Unix sockets only."
        )

    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)

    try:
        import httpx
    except Exception:  # pragma: no cover
        httpx = None  # type: ignore

    if httpx is not None:
        real_sync_send = httpx.Client.send
        real_async_send = httpx.AsyncClient.send

        def _allow_url(url: str) -> None:
            host = urllib.parse.urlparse(url).hostname
            if host is None:
                # No host (file://, etc.) — let the real call proceed.
                return
            if host_is_loopback(host) or host_is_synthetic_test(host):
                return
            raise ExternalTrafficError(
                f"_scoped_network_guard blocked httpx request to non-loopback "
                f"host {host!r}. Loopback + synthetic test hosts only."
            )

        def _sync_send(
            self: Any, request: Any, *args: Any, **kwargs: Any
        ) -> Any:
            _allow_url(str(request.url))
            return real_sync_send(self, request, *args, **kwargs)

        async def _async_send(
            self: Any, request: Any, *args: Any, **kwargs: Any
        ) -> Any:
            _allow_url(str(request.url))
            return await real_async_send(self, request, *args, **kwargs)

        monkeypatch.setattr(httpx.Client, "send", _sync_send)
        monkeypatch.setattr(httpx.AsyncClient, "send", _async_send)
