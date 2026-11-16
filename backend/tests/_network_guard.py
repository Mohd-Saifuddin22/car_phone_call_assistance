"""
backend/tests/_network_guard.py — shared, importable helpers for the
test-suite network guard.

Lives outside ``conftest.py`` so the helper logic (an ``ExternalTrafficError``
exception plus a few predicates) is importable from a plain test module
and from the conftest itself, ensuring a single shared class identity.
``conftest.py`` imports the predicates from here and installs the guard;
``test_network_guard.py`` exercises the real guard under the conftest's
autouse fixture.
"""

from __future__ import annotations

import socket
from typing import Any


class ExternalTrafficError(RuntimeError):
    """Raised by the network guard when a test attempts to open a TCP
    socket to a non-loopback address, or to issue an httpx request to a
    non-loopback URL.  Tests must fail loudly when this happens — never
    silently return a fake success.

    This is the **single canonical exception class**.  Conftest and test
    modules both import it via ``from backend.tests._network_guard import
    ExternalTrafficError`` so the same class is in scope everywhere —
    ``isinstance(exc, ExternalTrafficError)`` is reliable across files.
    """


# Synthetic hostnames httpx / Starlette use for in-process ASGI transports.
# These are NOT real loopback addresses — they are simply the well-known
# hostnames of the default httpx ``base_url`` for ASGITransport.  The
# network guard treats them as legitimate only at the httpx layer (where
# the ASGI app runs in-process and never opens a TCP socket), never at
# the raw ``socket.connect`` layer (where a connection to ``test`` would
# have to leave the test process).
_SYNTHETIC_TEST_HOSTS: frozenset[str] = frozenset({"test", "testserver"})


def host_is_loopback(host: str | None) -> bool:
    """
    True only for actual loopback addresses:

      * IPv4 ``127.0.0.0/8`` (commonly ``127.0.0.1``)
      * IPv6 ``::1``

    Returns ``False`` for None / empty / private-RFC1918 / link-local /
    public / synthetic-test / unparseable inputs.  Use
    ``host_is_synthetic_test`` separately to allow the httpx-layer
    exemptions for in-process ASGI fixtures.
    """
    if not host:
        return False
    candidate = host.strip("[]").split("%", 1)[0].lower()
    try:
        packed = socket.inet_pton(socket.AF_INET, candidate)
    except OSError:
        try:
            packed6 = socket.inet_pton(socket.AF_INET6, candidate)
        except OSError:
            return False
        else:
            return packed6 == (
                b"\x00\x00\x00\x00\x00\x00\x00\x00"
                b"\x00\x00\x00\x00\x00\x00\x00\x01"
            )
    return packed[0] == 127


def host_is_synthetic_test(host: str | None) -> bool:
    """
    True only for the well-known httpx/Starlette synthetic hostnames
    (``"test"``, ``"testserver"``).  These never appear in real DNS and
    are used solely as ``base_url`` markers for in-process ASGITransport.
    """
    if not host:
        return False
    return host.strip("[]").split("%", 1)[0].lower() in _SYNTHETIC_TEST_HOSTS


def extract_connect_target(address: Any) -> tuple[str, str]:
    """
    Classify a ``socket.connect`` ``address`` argument.

    Returns one of:

      * ``("unix", path)`` — string path for an AF_UNIX socket.
      * ``("inet", host)`` — IPv4/IPv6 host for a (host, port) tuple.

    Raises ``ExternalTrafficError`` for any unsupported address shape —
    the conftest's guard must NEVER delegate to the real connect when the
    address is not a recognised AF_UNIX / AF_INET / AF_INET6 form.  That
    is what makes the guard fail loudly on unexpected call sites.
    """
    if isinstance(address, str):
        return ("unix", address)
    if isinstance(address, tuple) and address:
        host = address[0]
        if isinstance(host, bytes):
            try:
                host = host.decode("ascii")
            except UnicodeDecodeError as exc:
                raise ExternalTrafficError(
                    f"socket.connect: non-ASCII bytes host is not a recognised "
                    f"AF_UNIX / AF_INET / AF_INET6 address: {address!r}"
                ) from exc
        if isinstance(host, str):
            return ("inet", host)
    raise ExternalTrafficError(
        f"socket.connect: unrecognised address shape {address!r}; only "
        f"AF_UNIX paths and AF_INET/AF_INET6 (host, port) tuples are allowed."
    )
