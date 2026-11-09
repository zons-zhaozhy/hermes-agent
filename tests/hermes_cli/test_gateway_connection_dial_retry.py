"""A dial that fails on the transport is redialled inside the caller's deadline.

A CLI launch dials the gateway before its first request (``GatewayConnection._send`` -> ``start``).
The owner answers a handshake only while its event loop runs, so during a graceful restart
(``hermes gateway restart`` waits up to 120 s for in-flight cron jobs) or on a host loaded enough to
miss ``connect_gateway``'s 10 s open timeout the launch's one dial failed and the viewer went
``final``: every launch in that window died with "Gateway connection failed; no local fallback".

The retry is bounded, and a dial failure that is a verdict about the gateway — it could not start,
an update holds the install — is reported on the first attempt: waiting cannot change it.
"""
import threading
from contextlib import asynccontextmanager

import pytest

from hermes_cli import gateway_connection
from hermes_cli.gateway_client import GatewayClientError, GatewayUnavailableError
from hermes_cli.gateway_connection import GatewayConnection


class Dial:
    """A gateway that fails its first ``failures`` dials on the transport and answers the rest."""

    def __init__(self, failures, error=TimeoutError):
        self.failures = failures
        self.error = error
        self.dials = 0
        self.first = threading.Event()  # a caller that gives up may return before its dial runs

    @asynccontextmanager
    async def dial(self):
        self.dials += 1
        self.first.set()
        if self.dials <= self.failures:
            raise self.error("timed out during opening handshake")
        yield Client()


class Client:
    async def rpc(self, method, *, _timeout=None, **params):
        return {"status": "ok"}


class Starting:
    """``ensure_gateway_runtime`` has no ready owner to dial: a verdict, not a transport blip."""

    def __init__(self):
        self.dials = 0

    @asynccontextmanager
    async def dial(self):
        self.dials += 1
        raise GatewayUnavailableError("Gateway starting: deadline.", "starting", "starting")
        yield  # pragma: no cover - unreachable: the verdict is the whole answer


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    """The pause between dials is the source's business; the runner's clock is not."""
    monkeypatch.setattr(gateway_connection, "_DIAL_RETRY_DELAY", 0.0, raising=False)
    monkeypatch.setattr(gateway_connection, "_DIAL_RETRY_DELAY_CAP", 0.0, raising=False)


@pytest.fixture
def connections():
    opened = []

    def open(connect):
        connection = GatewayConnection(connect=connect)
        opened.append(connection)
        return connection

    yield open
    for connection in opened:
        connection.close()


def test_a_dial_that_times_out_is_redialled_and_the_launch_goes_on(connections):
    dial = Dial(failures=2)
    connection = connections(dial.dial)

    connection.start(5)

    assert dial.dials == 3
    assert connection.epoch == 1 and connection.call("status") == {"status": "ok"}


def test_a_refused_dial_is_redialled_too(connections):
    """A refused connect is the same class as a timed-out one: the listener is not open yet."""
    dial = Dial(failures=1, error=ConnectionRefusedError)
    connection = connections(dial.dial)

    connection.start(5)

    assert dial.dials == 2 and connection.epoch == 1


def test_a_verdict_about_the_gateway_is_reported_on_the_first_dial(connections):
    dial = Starting()
    connection = connections(dial.dial)

    with pytest.raises(GatewayUnavailableError, match="Gateway starting"):
        connection.start(5)

    assert dial.dials == 1


def test_a_gateway_that_never_answers_is_final_after_the_attempt_cap(connections):
    """The cap keeps a wayward dial from spending the whole deadline; what is left is today's
    behaviour — a first-connect failure, and no later call dials again."""
    dial = Dial(failures=99)
    connection = connections(dial.dial)

    with pytest.raises(GatewayClientError, match="no local fallback"):
        connection.start(30)
    assert dial.dials == gateway_connection._DIAL_RETRY_ATTEMPTS + 1

    with pytest.raises(GatewayClientError, match="no local fallback"):
        connection.call("status")
    assert dial.dials == gateway_connection._DIAL_RETRY_ATTEMPTS + 1  # final: never dialled again


def test_the_retry_budget_does_not_outlive_its_deadline(connections):
    """A caller that gave ``start`` a short timeout is not held past it by the retry."""
    dial = Dial(failures=99)
    connection = connections(dial.dial)

    with pytest.raises(GatewayClientError, match="no local fallback"):
        connection.start(0.001)

    assert dial.first.wait(5)  # it dialled once, then the deadline ended the wait
    assert dial.dials == 1


def test_a_connect_wrapper_with_no_visible_cause_still_reads_as_transport():
    """``connect_gateway`` names its cause in the message, so the marker outlives a re-raise that
    dropped ``__cause__``: without it a refused dial became a verdict and was never redialled."""
    assert gateway_connection._transport_failure(
        GatewayClientError("Gateway connection failed; no local fallback (TimeoutError)"))


def test_a_refused_ws_dial_reports_the_transport_error(monkeypatch):
    """The one line a viewer shows has to carry WHY: a refused connect (nothing listening) and a
    missed open handshake (the owner's loop is stalled) are different problems with the same
    remedy-free message before this."""
    import asyncio

    from hermes_cli.gateway_client import connect_gateway

    monkeypatch.setenv("HERMES_TUI_GATEWAY_URL", "ws://127.0.0.1:1/tui-gateway")

    async def _dial():
        async with connect_gateway():
            raise AssertionError("a closed loopback port cannot answer a handshake")

    with pytest.raises(GatewayClientError) as caught:
        asyncio.run(_dial())

    assert "no local fallback" in str(caught.value)
    assert "ConnectionRefusedError" in str(caught.value)
