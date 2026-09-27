"""Stage 4 — ZeroMQ Signal Publisher (Intelligence Process Side).

The intelligence process runs this publisher. It computes signals from
indicators/AI and broadcasts them on a ZMQ PUB socket.

Every message includes a UTC timestamp so the subscriber can detect stale
signals.

Fail-safe contract
------------------
If the intelligence process crashes, hangs, or is slow:
- Messages simply stop arriving on the socket.
- The subscriber side (ExecutionSubscriber) ages out any cached signal after
  ``max_signal_age_seconds`` and returns HOLD.
- The execution core NEVER acts on a signal older than that threshold.
- Recovery: when the intelligence process restarts, it publishes a new signal
  and the execution core resumes normally — no manual intervention required.

Assumption: both processes run on the same machine (uses tcp://127.0.0.1).
For multi-machine deployments replace the address with the actual host IP.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import zmq


DEFAULT_ADDRESS = "tcp://127.0.0.1:5555"

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _require_loopback(address: str) -> None:
    """Refuse to bind the signal channel on a non-loopback interface.

    The PUB/SUB channel carries UNAUTHENTICATED trade commands: any
    process that can reach the socket can publish SELL/BUY signals and
    the executor will act on them (flatten positions, open shorts).
    That is safe on loopback, and an account-takeover primitive on
    0.0.0.0. ipc:// and inproc:// transports are filesystem/process
    scoped and equally fine. Set ZMQ_ALLOW_NONLOCAL=true only after
    adding transport auth (CURVE) or network-level isolation.
    """
    if address.startswith(("ipc://", "inproc://")):
        return
    host = address.split("//", 1)[-1].rsplit(":", 1)[0].strip("[]")
    if host in _LOOPBACK_HOSTS:
        return
    from config.settings import settings

    if getattr(settings, "zmq_allow_nonlocal", False):
        return
    raise ValueError(
        f"Refusing to bind unauthenticated signal channel on {address!r} — "
        "non-loopback exposure lets any network peer inject trade signals. "
        "Set ZMQ_ALLOW_NONLOCAL=true only with CURVE auth or an isolated network."
    )
TOPIC = b"signal"


class SignalPublisher:
    """Publishes trading signals from the intelligence process.

    Usage::

        pub = SignalPublisher()
        pub.publish(signal="BUY", pair="BTC/USDT", score=0.72, confidence=0.81)
        pub.close()

    Or as a context manager::

        with SignalPublisher() as pub:
            pub.publish(signal="HOLD", pair="BTC/USDT", score=0.0, confidence=0.5)
    """

    def __init__(self, address: str = DEFAULT_ADDRESS) -> None:
        self._address = address
        self._ctx = zmq.Context.instance()
        self._socket: zmq.Socket | None = None
        self._connect()

    def _connect(self) -> None:
        _require_loopback(self._address)
        self._socket = self._ctx.socket(zmq.PUB)
        self._socket.bind(self._address)

    def publish(
        self,
        *,
        signal: str,
        pair: str,
        score: float,
        confidence: float,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Publish one signal message.

        Args:
            signal:     "BUY", "SELL", or "HOLD"
            pair:       Trading pair, e.g. "BTC/USDT"
            score:      Combined AI score in [-1, +1]
            confidence: Confidence in [0, 1]
            extra:      Optional extra fields (regime name, HMM state, etc.)
        """
        if self._socket is None:
            raise RuntimeError("Publisher is closed")

        payload: dict[str, Any] = {
            "signal": signal,
            "pair": pair,
            "score": round(float(score), 6),
            "confidence": round(float(confidence), 6),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            # Unique per message: lets the execution core act on each
            # signal exactly once instead of re-acting on the cached
            # last message every poll until it goes stale.
            "message_id": uuid.uuid4().hex,
        }
        if extra:
            payload.update(extra)

        message = json.dumps(payload, separators=(",", ":")).encode()
        self._socket.send_multipart([TOPIC, message])

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None

    def __enter__(self) -> "SignalPublisher":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
