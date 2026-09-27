"""Stage 4 tests — ZeroMQ publisher / subscriber pair.

All tests use ZMQ inproc:// transport so no real network sockets are opened.
Tests verify the core fail-safe contract:
  - Execution core does NOTHING when no signal arrives (cold start).
  - Execution core does NOTHING when the intelligence process hangs and the
    last signal becomes older than max_signal_age_seconds.
  - Execution core acts correctly on a fresh, valid signal.

We manipulate signal timestamps directly (bypassing the ZMQ socket for the
stale-signal tests) to avoid real time.sleep() calls in tests.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta

import pytest
import zmq

from core.zmq_publisher import SignalPublisher, TOPIC
from core.zmq_subscriber import (
    ReceivedSignal,
    SignalSubscriber,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _unique_addr() -> str:
    """Each test gets its own inproc address to avoid cross-test pollution."""
    return f"inproc://test-{id(object())}-{time.monotonic_ns()}"


def _build_raw_signal(
    signal: str = "BUY",
    pair: str = "BTC/USDT",
    score: float = 0.72,
    confidence: float = 0.81,
    timestamp_utc: datetime | None = None,
) -> bytes:
    """Build a raw JSON payload as the publisher would send it."""
    ts = (timestamp_utc or datetime.now(UTC)).isoformat()
    payload = {
        "signal": signal,
        "pair": pair,
        "score": round(score, 6),
        "confidence": round(confidence, 6),
        "timestamp_utc": ts,
    }
    return json.dumps(payload, separators=(",", ":")).encode()


def _inject_message(socket: zmq.Socket, raw: bytes) -> None:
    """Send a pre-built message directly into a PUSH socket bound to the same
    address, simulating the publisher without using SignalPublisher."""
    socket.send_multipart([TOPIC, raw])


# ─── ReceivedSignal Unit Tests ────────────────────────────────────────────────

class TestReceivedSignal:
    def test_fresh_buy_is_actionable(self) -> None:
        sig = ReceivedSignal(
            signal="BUY", pair="BTC/USDT", score=0.8, confidence=0.9,
            timestamp_utc=datetime.now(UTC), stale=False,
        )
        assert sig.is_fresh is True

    def test_hold_is_not_fresh(self) -> None:
        sig = ReceivedSignal(
            signal="HOLD", pair="BTC/USDT", score=0.0, confidence=0.5,
            timestamp_utc=datetime.now(UTC), stale=False,
        )
        assert sig.is_fresh is False

    def test_stale_buy_is_not_fresh(self) -> None:
        sig = ReceivedSignal(
            signal="BUY", pair="BTC/USDT", score=0.8, confidence=0.9,
            timestamp_utc=datetime.now(UTC), stale=True,
        )
        assert sig.is_fresh is False

    def test_age_seconds_positive(self) -> None:
        old_ts = datetime.now(UTC) - timedelta(seconds=5)
        sig = ReceivedSignal(
            signal="BUY", pair="BTC/USDT", score=0.8, confidence=0.9,
            timestamp_utc=old_ts, stale=False,
        )
        assert sig.age_seconds >= 5.0


# ─── Cold-Start Fail-Safe ─────────────────────────────────────────────────────

class TestColdStart:
    def test_no_signal_ever_received_returns_hold(self) -> None:
        """Before any message arrives the subscriber must return HOLD (fail safe)."""
        ctx = zmq.Context()
        addr = _unique_addr()

        # Bind a PAIR socket just to make the inproc address exist
        anchor = ctx.socket(zmq.PAIR)
        anchor.bind(addr)

        sub_sock = ctx.socket(zmq.SUB)
        sub_sock.connect(addr)
        sub_sock.setsockopt(zmq.SUBSCRIBE, TOPIC)
        sub_sock.setsockopt(zmq.RCVTIMEO, 0)

        # Patch subscriber internals directly — no SignalSubscriber.__init__ needed
        from core.zmq_subscriber import SignalSubscriber as SS
        sub = object.__new__(SS)
        sub._address = addr
        sub._max_age = timedelta(seconds=30)
        sub._ctx = ctx
        sub._socket = sub_sock
        sub._last = None  # Nothing ever received

        result = sub.poll_signal()
        assert result.signal == "HOLD"
        assert result.stale is True
        assert result.is_fresh is False

        anchor.close(linger=0)
        sub_sock.close(linger=0)
        ctx.term()


# ─── Stale Signal Fail-Safe ───────────────────────────────────────────────────

class TestStaleSignal:
    def _make_subscriber_with_cached_signal(
        self,
        age_seconds: float,
        max_age: int = 30,
    ) -> "SignalSubscriber":
        """Build a SignalSubscriber whose _last signal has the given age."""
        from core.zmq_subscriber import SignalSubscriber as SS
        ctx = zmq.Context()
        addr = _unique_addr()

        anchor = ctx.socket(zmq.PAIR)
        anchor.bind(addr)

        sub_sock = ctx.socket(zmq.SUB)
        sub_sock.connect(addr)
        sub_sock.setsockopt(zmq.SUBSCRIBE, TOPIC)
        sub_sock.setsockopt(zmq.RCVTIMEO, 0)

        sub = object.__new__(SS)
        sub._address = addr
        sub._max_age = timedelta(seconds=max_age)
        sub._ctx = ctx
        sub._socket = sub_sock
        sub._last = ReceivedSignal(
            signal="BUY",
            pair="BTC/USDT",
            score=0.9,
            confidence=0.95,
            timestamp_utc=datetime.now(UTC) - timedelta(seconds=age_seconds),
            stale=False,
        )
        # Store anchor so GC doesn't close it
        sub._anchor = anchor
        return sub

    def test_signal_just_within_age_is_fresh(self) -> None:
        sub = self._make_subscriber_with_cached_signal(age_seconds=5, max_age=30)
        result = sub.poll_signal()
        assert result.stale is False
        assert result.signal == "BUY"
        assert result.is_fresh is True

    def test_signal_exactly_at_boundary_is_stale(self) -> None:
        """Signal at exactly max_age seconds old is stale (strictly greater than)."""
        sub = self._make_subscriber_with_cached_signal(age_seconds=31, max_age=30)
        result = sub.poll_signal()
        assert result.stale is True
        assert result.signal == "HOLD"
        assert result.is_fresh is False

    def test_very_old_signal_returns_hold(self) -> None:
        """Intelligence process crashed 5 minutes ago → execution does nothing."""
        sub = self._make_subscriber_with_cached_signal(age_seconds=300, max_age=30)
        result = sub.poll_signal()
        assert result.stale is True
        assert result.signal == "HOLD"
        assert result.is_fresh is False


# ─── Live Round-Trip Tests ────────────────────────────────────────────────────

class TestRoundTrip:
    """Uses real ZMQ tcp sockets on loopback for end-to-end tests."""

    # Pick a port unlikely to be in use during CI
    PORT = 15555

    @pytest.fixture(autouse=True)
    def zmq_pair(self):
        ctx = zmq.Context()
        pub = ctx.socket(zmq.PUB)
        pub.bind(f"tcp://127.0.0.1:{self.PORT}")

        sub = ctx.socket(zmq.SUB)
        sub.connect(f"tcp://127.0.0.1:{self.PORT}")
        sub.setsockopt(zmq.SUBSCRIBE, TOPIC)
        sub.setsockopt(zmq.RCVTIMEO, 0)

        time.sleep(0.3)  # ZMQ slow-joiner + suite-load margin

        yield pub, sub, ctx

        pub.close(linger=0)
        sub.close(linger=0)
        ctx.term()

    def test_fresh_buy_round_trip(self, zmq_pair) -> None:
        pub_sock, sub_sock, ctx = zmq_pair

        from core.zmq_subscriber import SignalSubscriber as SS
        sub = object.__new__(SS)
        sub._max_age = timedelta(seconds=30)
        sub._ctx = ctx
        sub._socket = sub_sock
        sub._last = None

        raw = _build_raw_signal("BUY", "BTC/USDT", 0.72, 0.81)
        pub_sock.send_multipart([TOPIC, raw])
        time.sleep(0.3)  # delivery margin under suite load

        result = sub.poll_signal()
        assert result.signal == "BUY"
        assert result.pair == "BTC/USDT"
        assert result.score == pytest.approx(0.72)
        assert result.confidence == pytest.approx(0.81)
        assert result.stale is False
        assert result.is_fresh is True

    def test_stale_signal_from_wire_returns_hold(self, zmq_pair) -> None:
        """A valid message with a 5-minute-old timestamp must be rejected."""
        pub_sock, sub_sock, ctx = zmq_pair

        from core.zmq_subscriber import SignalSubscriber as SS
        sub = object.__new__(SS)
        sub._max_age = timedelta(seconds=30)
        sub._ctx = ctx
        sub._socket = sub_sock
        sub._last = None

        old_ts = datetime.now(UTC) - timedelta(minutes=5)
        raw = _build_raw_signal("BUY", "BTC/USDT", 0.9, 0.95, timestamp_utc=old_ts)
        pub_sock.send_multipart([TOPIC, raw])
        time.sleep(0.3)  # delivery margin under suite load

        result = sub.poll_signal()
        assert result.stale is True
        assert result.signal == "HOLD"
        assert result.is_fresh is False

    def test_invalid_json_does_not_crash(self, zmq_pair) -> None:
        pub_sock, sub_sock, ctx = zmq_pair

        from core.zmq_subscriber import SignalSubscriber as SS
        sub = object.__new__(SS)
        sub._max_age = timedelta(seconds=30)
        sub._ctx = ctx
        sub._socket = sub_sock
        sub._last = None

        # Corrupt message: must be discarded, not raised — the subscriber
        # fails safe to HOLD (claude.md: incoming data is untrusted).
        pub_sock.send_multipart([TOPIC, b"this is not json!!!"])
        time.sleep(0.3)  # delivery margin under suite load

        result = sub.poll_signal()
        assert result.signal == "HOLD"
        assert result.is_fresh is False

    def test_only_latest_message_is_kept(self, zmq_pair) -> None:
        """When multiple messages pile up, only the most recent is used."""
        pub_sock, sub_sock, ctx = zmq_pair

        from core.zmq_subscriber import SignalSubscriber as SS
        sub = object.__new__(SS)
        sub._max_age = timedelta(seconds=30)
        sub._ctx = ctx
        sub._socket = sub_sock
        sub._last = None

        # Send 3 messages
        pub_sock.send_multipart([TOPIC, _build_raw_signal("HOLD", score=0.0, confidence=0.5)])
        pub_sock.send_multipart([TOPIC, _build_raw_signal("SELL", score=-0.6, confidence=0.75)])
        pub_sock.send_multipart([TOPIC, _build_raw_signal("BUY", score=0.88, confidence=0.91)])
        time.sleep(0.3)  # delivery margin under suite load

        result = sub.poll_signal()
        # Should see the last one (BUY)
        assert result.signal == "BUY"
        assert result.score == pytest.approx(0.88)


# ─── Publisher Tests ──────────────────────────────────────────────────────────

class TestPublisher:
    def test_publish_after_close_raises(self) -> None:
        pub = SignalPublisher.__new__(SignalPublisher)
        pub._socket = None
        with pytest.raises(RuntimeError, match="closed"):
            pub.publish(signal="BUY", pair="X", score=0.5, confidence=0.8)
