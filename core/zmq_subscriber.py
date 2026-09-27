"""Stage 4 — ZeroMQ Signal Subscriber (Execution Process Side).

The execution core uses this subscriber to receive signals from the
intelligence process. It enforces a strict stale-signal rule:

    If the most recently received signal is older than ``max_signal_age_seconds``,
    the execution core MUST treat it as HOLD regardless of its content.

This means the execution core **fails safe** (does nothing) when:
- The intelligence process has crashed.
- The intelligence process is hung / slow.
- The network between the two processes is temporarily broken.
- No signal has ever arrived yet (cold start).

How it works
------------
``poll_signal()`` is non-blocking. It drains all pending messages from the
socket (keeping only the latest) and returns a ``ReceivedSignal``. The caller
checks ``signal.is_fresh`` before acting:

    sub = SignalSubscriber()
    while True:
        sig = sub.poll_signal()
        if sig.is_fresh:
            execute(sig.signal)
        else:
            pass  # do nothing — fail safe

Stale-signal threshold
----------------------
The default is 30 seconds. For 1H candle strategies this is very conservative
(candles close only every 3600 s). For tick-level strategies you would lower
it to 1–5 seconds.

Assumption: publisher and subscriber clocks are synchronized to UTC.
For multi-machine setups, clock drift > ``max_signal_age_seconds`` would cause
spurious HOLD outputs — use NTP.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import zmq
from loguru import logger


DEFAULT_ADDRESS = "tcp://127.0.0.1:5555"
TOPIC = b"signal"
DEFAULT_MAX_AGE_SECONDS = 30


@dataclass(frozen=True)
class ReceivedSignal:
    """Immutable signal value object returned by SignalSubscriber.poll_signal()."""

    signal: str           # "BUY", "SELL", or "HOLD"
    pair: str
    score: float
    confidence: float
    timestamp_utc: datetime
    extra: dict[str, Any] = field(default_factory=dict)
    stale: bool = False   # True when age > max_signal_age_seconds
    message_id: str = ""  # Unique per published message (dedup key)

    @property
    def is_fresh(self) -> bool:
        """True only when signal is NOT stale and IS actionable."""
        return not self.stale and self.signal != "HOLD"

    @property
    def age_seconds(self) -> float:
        return (datetime.now(UTC) - self.timestamp_utc).total_seconds()


# Sentinel returned when no signal has ever arrived or when stale
_HOLD_SIGNAL = ReceivedSignal(
    signal="HOLD",
    pair="",
    score=0.0,
    confidence=0.0,
    timestamp_utc=datetime(2000, 1, 1, tzinfo=UTC),
    stale=True,
)


class SignalSubscriber:
    """Subscribes to intelligence process signals over ZMQ SUB socket.

    Thread-safety: not safe for concurrent calls. Use one subscriber per thread.

    Usage::

        sub = SignalSubscriber(max_signal_age_seconds=30)
        sig = sub.poll_signal()
        if sig.is_fresh:
            place_order(sig.signal, sig.pair)
    """

    def __init__(
        self,
        address: str = DEFAULT_ADDRESS,
        max_signal_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
    ) -> None:
        self._address = address
        self._max_age = timedelta(seconds=max_signal_age_seconds)
        self._ctx = zmq.Context.instance()
        self._socket: zmq.Socket | None = None
        self._last: ReceivedSignal | None = None
        # Monotonic receive time of the last message of ANY kind (including
        # heartbeats) — lets the execution core distinguish "intelligence
        # alive, market says HOLD" from "intelligence dead". Monotonic so an
        # NTP step can't fake liveness.
        self._last_recv_monotonic: float | None = None
        self._connect()

    def _connect(self) -> None:
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.connect(self._address)
        self._socket.setsockopt(zmq.SUBSCRIBE, TOPIC)
        self._socket.setsockopt(zmq.RCVTIMEO, 0)   # non-blocking

    def poll_signal(self) -> ReceivedSignal:
        """Drain all pending messages, keep only the latest, enforce stale rule.

        Returns:
            ReceivedSignal — always non-None.
            ``signal.stale = True`` when no fresh signal is available.
            ``signal.signal = "HOLD"`` in that case.
        """
        if self._socket is None:
            return _HOLD_SIGNAL

        # Drain socket. Each message is parsed individually so (a) one
        # malformed frame is discarded instead of poisoning the whole
        # drained batch, and (b) a heartbeat arriving after a trade signal
        # refreshes liveness without overwriting the actionable signal.
        while True:
            try:
                _topic, raw = self._socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            self._last_recv_monotonic = time.monotonic()
            if _topic != TOPIC:
                # ZMQ SUBSCRIBE is prefix-matching: a future "signal_v2"
                # topic would otherwise be parsed as a trade signal.
                logger.warning("Ignoring message on unexpected topic {!r}", _topic)
                continue
            try:
                parsed = self._parse(raw)
            except Exception as e:
                logger.warning("Discarding malformed signal message: {}", e)
                continue
            if not parsed.extra.get("heartbeat"):
                self._last = parsed

        if self._last is None:
            return _HOLD_SIGNAL  # Nothing ever received

        age = datetime.now(UTC) - self._last.timestamp_utc
        if age > self._max_age:
            # Return a stale-flagged copy of the last known signal
            return ReceivedSignal(
                signal="HOLD",
                pair=self._last.pair,
                score=0.0,
                confidence=0.0,
                timestamp_utc=self._last.timestamp_utc,
                stale=True,
            )

        return self._last

    @property
    def seconds_since_last_message(self) -> float | None:
        """Seconds since ANY message (heartbeats included) arrived, or None
        if nothing has ever been received. Monotonic-clock based."""
        if self._last_recv_monotonic is None:
            return None
        return time.monotonic() - self._last_recv_monotonic

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None

    def __enter__(self) -> "SignalSubscriber":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _parse(raw: bytes) -> ReceivedSignal:
        data: dict[str, Any] = json.loads(raw.decode())
        ts_raw = data.get("timestamp_utc", "")
        try:
            ts = datetime.fromisoformat(ts_raw)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
        except (ValueError, TypeError):
            ts = datetime(2000, 1, 1, tzinfo=UTC)  # treat as ancient / stale

        return ReceivedSignal(
            signal=str(data.get("signal", "HOLD")),
            pair=str(data.get("pair", "")),
            score=float(data.get("score", 0.0)),
            confidence=float(data.get("confidence", 0.0)),
            timestamp_utc=ts,
            extra={k: v for k, v in data.items()
                   if k not in {"signal", "pair", "score", "confidence",
                                "timestamp_utc", "message_id"}},
            stale=False,
            message_id=str(data.get("message_id", "")),
        )
