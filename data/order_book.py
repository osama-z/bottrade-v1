"""Order Book Manager — Stage 2.

A pure-Python, pandas-free order book that processes WebSocket diff messages
from Binance and maintains a consistent local state.

Design decisions
----------------
* Storage: ``dict[float, float]`` — price → quantity. Price is the natural key;
  quantity=0 signals a level removal (standard Binance diff format).
* Best bid / ask: cached after every apply call so reads are O(1). The full
  book is never re-sorted on reads.
* Imbalance: computed on demand within a configurable price band.
* Thread-safety: a single ``threading.RLock`` guards all state mutations.
  Callers from the WS thread (writer) and the trading loop (reader) both
  acquire the same lock, so reads always see a consistent snapshot.

Assumptions flagged
-------------------
* Prices from the exchange arrive as floats (already parsed by ccxt/json).
* Quantity = 0.0 means "remove this level" (Binance diff protocol).
* A snapshot must be applied before diffs; applying a diff to an empty book
  raises ``OrderBookError``.
* All timestamps are UTC milliseconds (int) as received from the WS feed.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Iterable

from loguru import logger


class OrderBookError(Exception):
    """Raised when an operation is invalid given current book state."""


class OrderBookDesyncError(OrderBookError):
    """Raised when a sequence gap is detected — the caller MUST discard the
    book and resync from a fresh snapshot (claude.md Data Validation)."""


@dataclass
class BookSnapshot:
    """Immutable read-only view of the order book at a point in time."""
    best_bid: float | None
    best_ask: float | None
    spread: float | None
    mid_price: float | None
    bid_depth_usdt: float          # total USDT liquidity on bid side
    ask_depth_usdt: float          # total USDT liquidity on ask side
    imbalance_ratio: float | None  # ask_vol / bid_vol within price_band_pct
    last_update_id: int
    timestamp_ms: int


# Type alias for a list of [price, qty] pairs (as from ccxt / raw Binance WS)
Level = tuple[float, float]


class OrderBookManager:
    """Maintains a local L2 order book by applying snapshots and diffs.

    Usage::

        book = OrderBookManager(symbol="BTC/USDT")
        book.apply_snapshot(bids=snapshot["bids"], asks=snapshot["asks"],
                            last_update_id=snapshot["lastUpdateId"],
                            timestamp_ms=snapshot["timestamp"])
        for message in ws_stream:
            book.apply_diff(bids=message["b"], asks=message["a"],
                            last_update_id=message["u"],
                            timestamp_ms=message["T"])
        snap = book.snapshot()
    """

    def __init__(
        self,
        symbol: str,
        imbalance_band_pct: float = 0.01,   # 1% above/below mid price
        max_depth: int = 500,               # prune levels beyond this depth
        stale_after_seconds: float | None = 30.0,
    ) -> None:
        if imbalance_band_pct <= 0 or imbalance_band_pct >= 1:
            raise ValueError("imbalance_band_pct must be in (0, 1)")
        if max_depth <= 0:
            raise ValueError("max_depth must be positive")

        self.symbol = symbol
        self._band_pct = imbalance_band_pct
        self._max_depth = max_depth
        self._stale_after = stale_after_seconds
        self._lock = threading.RLock()

        self._bids: dict[float, float] = {}   # price → qty
        self._asks: dict[float, float] = {}   # price → qty
        self._best_bid: float | None = None
        self._best_ask: float | None = None
        self._last_update_id: int = 0
        self._timestamp_ms: int = 0
        self._initialised: bool = False
        # Monotonic receive time of the last applied update: a frozen feed
        # must flip is_ready to False instead of serving hours-old numbers
        # as fresh (claude.md stale-data rule).
        self._last_apply_monotonic: float | None = None

    # ── Public write API ──────────────────────────────────────────────────────

    def apply_snapshot(
        self,
        *,
        bids: Iterable[Level],
        asks: Iterable[Level],
        last_update_id: int,
        timestamp_ms: int,
    ) -> None:
        """Replace the entire book with a fresh REST/WS snapshot.

        Levels with NaN/zero/negative prices or NaN/non-positive
        quantities are rejected at ingest (untrusted WS data).
        """
        with self._lock:
            self._bids = {
                float(p): float(q) for p, q in bids
                if self._valid_level(float(p), float(q)) and float(q) > 0
            }
            self._asks = {
                float(p): float(q) for p, q in asks
                if self._valid_level(float(p), float(q)) and float(q) > 0
            }
            self._last_update_id = last_update_id
            self._timestamp_ms = timestamp_ms
            self._initialised = True
            self._last_apply_monotonic = time.monotonic()
            self._refresh_best()
            self._prune()

    def apply_diff(
        self,
        *,
        bids: Iterable[Level],
        asks: Iterable[Level],
        last_update_id: int,
        timestamp_ms: int,
        first_update_id: int | None = None,
    ) -> None:
        """Apply an incremental diff message to the book.

        Sequence continuity (Binance ``U``/``u`` semantics) is enforced
        when ``first_update_id`` (the message's ``U``) is supplied:
        - events entirely at/before the current book state are ignored;
        - a gap (``U > current + 1``) marks the book desynced and raises
          ``OrderBookDesyncError`` — the caller must resync from a fresh
          snapshot. Without ``first_update_id`` gaps are undetectable, so
          protocol-aware feeds must always pass it.

        Raises ``OrderBookError`` if ``apply_snapshot`` has not been called yet.
        """
        if not self._initialised:
            raise OrderBookError(
                "apply_snapshot() must be called before apply_diff()"
            )
        with self._lock:
            if first_update_id is not None:
                if last_update_id <= self._last_update_id:
                    return  # stale event, already reflected in the book
                if first_update_id > self._last_update_id + 1:
                    self._initialised = False
                    raise OrderBookDesyncError(
                        f"sequence gap: expected U <= {self._last_update_id + 1}, "
                        f"got U={first_update_id} — book must resync"
                    )
            self._apply_side(self._bids, bids)
            self._apply_side(self._asks, asks)
            self._last_update_id = last_update_id
            self._timestamp_ms = timestamp_ms
            self._last_apply_monotonic = time.monotonic()
            self._refresh_best()
            self._prune()

    def reset(self) -> None:
        """Clear all book state (e.g. after a WS reconnect)."""
        with self._lock:
            self._bids.clear()
            self._asks.clear()
            self._best_bid = None
            self._best_ask = None
            self._last_update_id = 0
            self._timestamp_ms = 0
            self._initialised = False

    # ── Public read API ───────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """Initialised AND fresh: a frozen feed must not serve old numbers."""
        if not self._initialised:
            return False
        if self._stale_after is not None and self._last_apply_monotonic is not None:
            if time.monotonic() - self._last_apply_monotonic > self._stale_after:
                return False
        return True

    @property
    def seconds_since_update(self) -> float | None:
        """Monotonic seconds since the last applied update, None if never."""
        if self._last_apply_monotonic is None:
            return None
        return time.monotonic() - self._last_apply_monotonic

    def snapshot(self) -> BookSnapshot:
        """Return an immutable snapshot of the current book state."""
        with self._lock:
            best_bid = self._best_bid
            best_ask = self._best_ask
            spread = (best_ask - best_bid) if (best_bid and best_ask) else None
            mid = ((best_bid + best_ask) / 2) if (best_bid and best_ask) else None

            bid_depth = sum(p * q for p, q in self._bids.items())
            ask_depth = sum(p * q for p, q in self._asks.items())

            imbalance = self._compute_imbalance(mid) if mid else None

            return BookSnapshot(
                best_bid=best_bid,
                best_ask=best_ask,
                spread=spread,
                mid_price=mid,
                bid_depth_usdt=bid_depth,
                ask_depth_usdt=ask_depth,
                imbalance_ratio=imbalance,
                last_update_id=self._last_update_id,
                timestamp_ms=self._timestamp_ms,
            )

    def best_bid(self) -> float | None:
        with self._lock:
            return self._best_bid

    def best_ask(self) -> float | None:
        with self._lock:
            return self._best_ask

    def top_bids(self, n: int = 5) -> list[Level]:
        """Return the top-n bid levels sorted descending by price."""
        with self._lock:
            return sorted(self._bids.items(), reverse=True)[:n]

    def top_asks(self, n: int = 5) -> list[Level]:
        """Return the top-n ask levels sorted ascending by price."""
        with self._lock:
            return sorted(self._asks.items())[:n]

    def __len__(self) -> int:
        with self._lock:
            return len(self._bids) + len(self._asks)

    def __repr__(self) -> str:
        snap = self.snapshot()
        # NB: everything after ':' in an f-string is a literal format spec —
        # the old `{x:.4f if x else 'N/A'}` raised ValueError whenever a
        # spread existed, crashing any debug log that printed the book.
        spread = f"{snap.spread:.4f}" if snap.spread is not None else "N/A"
        return (
            f"OrderBookManager({self.symbol!r} "
            f"bid={snap.best_bid} ask={snap.best_ask} "
            f"spread={spread} "
            f"update_id={snap.last_update_id})"
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _valid_level(price: float, qty: float) -> bool:
        """Reject NaN/zero/negative prices and NaN/negative quantities —
        incoming WS data is untrusted (claude.md). A NaN price would
        become a dict key and could poison best-bid via max()."""
        if math.isnan(price) or price <= 0:
            return False
        if math.isnan(qty) or qty < 0:
            return False
        return True

    @classmethod
    def _apply_side(cls, store: dict[float, float], levels: Iterable[Level]) -> None:
        for p, q in levels:
            price, qty = float(p), float(q)
            if not cls._valid_level(price, qty):
                logger.warning(
                    "Rejected invalid book level: price={} qty={}", price, qty
                )
                continue
            if qty == 0.0:
                store.pop(price, None)
            else:
                store[price] = qty

    def _refresh_best(self) -> None:
        """Recompute cached best bid / ask from the full dict."""
        self._best_bid = max(self._bids, default=None)
        self._best_ask = min(self._asks, default=None)

    def _prune(self) -> None:
        """Remove levels beyond max_depth to bound memory usage."""
        if len(self._bids) > self._max_depth:
            # Keep only the top max_depth bids (highest prices)
            overflow = sorted(self._bids)[:len(self._bids) - self._max_depth]
            for p in overflow:
                del self._bids[p]
        if len(self._asks) > self._max_depth:
            # Keep only the top max_depth asks (lowest prices)
            overflow = sorted(self._asks, reverse=True)[:len(self._asks) - self._max_depth]
            for p in overflow:
                del self._asks[p]

    def _compute_imbalance(self, mid: float) -> float | None:
        """Ask volume / Bid volume within ±band_pct of mid price.

        Volumes are BASE-QUANTITY sums (not notional): this matches the
        definition used by the live filter (`fetch_order_book_imbalance`),
        so this book can replace the REST source without silently changing
        what the strategy's 1.5 threshold means (audit V-33).

        Returns float('inf') if bid volume is zero.
        """
        lower = mid * (1.0 - self._band_pct)
        upper = mid * (1.0 + self._band_pct)
        bid_vol = sum(q for p, q in self._bids.items() if p >= lower)
        ask_vol = sum(q for p, q in self._asks.items() if p <= upper)
        if bid_vol == 0.0:
            return float("inf")
        return ask_vol / bid_vol
