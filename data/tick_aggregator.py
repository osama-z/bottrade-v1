"""Tick Aggregator — Stage 2.

Accumulates raw trade ticks (from a WebSocket trade stream) into OHLCV candles
without performing a full REST re-fetch on every candle close.

How it works
------------
1. Trades arrive as ``push_trade(price, qty, ts_ms)`` calls.
2. The aggregator accumulates them into the *current open candle*.
3. When ``close_candle()`` is called (at candle-close time, driven by
   APScheduler), it finalises the candle and returns a single-row dict that
   can be appended to the historical DataFrame.
4. ``merge_candle_into(candle, df)`` appends the completed row and trims
   the DataFrame to a rolling window so memory stays bounded.

Assumptions flagged
-------------------
* Candle open time is the timestamp of the first trade in the candle.
  If no trades arrive, the candle is synthesised from the last known close
  price (open=high=low=close=last_close, volume=0).
* Timestamps are UTC milliseconds (int).
* Price and qty are plain Python floats.
* The caller is responsible for aligning close_candle() calls with actual
  exchange candle boundaries — the aggregator does not self-schedule.
* Thread-safety: a single RLock guards all state; the WS thread writes via
  push_trade() while the scheduler thread reads via close_candle().
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Deque

import pandas as pd


@dataclass
class _OpenCandle:
    """Mutable accumulator for the current incomplete candle."""
    open_ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int = 0

    def push(self, price: float, qty: float) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += qty
        self.trade_count += 1

    def to_dict(self) -> dict:
        return {
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


@dataclass
class CompletedCandle:
    """Immutable completed candle ready to append to a DataFrame."""
    timestamp_utc: datetime   # candle open time (UTC, timezone-aware)
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int
    synthesised: bool         # True if no trades arrived this candle


class TickAggregator:
    """Accumulates trade ticks into OHLCV candles.

    Example::

        agg = TickAggregator(timeframe_seconds=3600)
        agg.seed(last_close=50_000.0)

        # In WS callback:
        agg.push_trade(price=50_100.0, qty=0.01, ts_ms=1_700_000_000_000)

        # At candle close (called by scheduler):
        candle = agg.close_candle()
        df = merge_candle_into(candle, df, window=500)
    """

    def __init__(
        self,
        timeframe_seconds: int = 3600,
        max_recent_trades: int = 10_000,
    ) -> None:
        if timeframe_seconds <= 0:
            raise ValueError("timeframe_seconds must be positive")
        self._tf_seconds = timeframe_seconds
        self._lock = threading.RLock()
        self._candle: _OpenCandle | None = None
        self._last_close: float | None = None
        # Rolling buffer of recent raw trades for debugging/auditing
        self._recent: Deque[tuple[int, float, float]] = deque(
            maxlen=max_recent_trades
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def seed(self, last_close: float) -> None:
        """Provide the last known close price so the first synthesised
        candle has a valid OHLC even if no trades arrive."""
        with self._lock:
            self._last_close = float(last_close)

    # ── Write path (WS thread) ────────────────────────────────────────────────

    def push_trade(self, price: float, qty: float, ts_ms: int) -> None:
        """Record a single trade tick into the current open candle.

        If no candle is open yet, a new one is started with this trade as
        the opening tick.
        """
        price, qty = float(price), float(qty)
        with self._lock:
            self._recent.append((ts_ms, price, qty))
            if self._candle is None:
                self._candle = _OpenCandle(
                    open_ts_ms=ts_ms,
                    open=price, high=price, low=price, close=price,
                    volume=qty, trade_count=1,
                )
                self._last_close = price
            else:
                self._candle.push(price, qty)
                self._last_close = price

    # ── Read / close path (scheduler thread) ─────────────────────────────────

    def close_candle(self, open_ts_ms: int | None = None) -> CompletedCandle:
        """Finalise and return the current candle, then open a new accumulator.

        Args:
            open_ts_ms: The exchange-defined open timestamp for this candle
                (ms since epoch UTC). If None, uses the timestamp of the
                first trade in the candle, or ``now`` if the candle was
                synthesised.

        Returns:
            ``CompletedCandle`` — always non-None, synthesised if no trades.
        """
        with self._lock:
            now_ms = int(datetime.now(UTC).timestamp() * 1000)
            ts = open_ts_ms or now_ms

            if self._candle is not None:
                c = self._candle
                completed = CompletedCandle(
                    timestamp_utc=_ms_to_utc(ts),
                    open=c.open,
                    high=c.high,
                    low=c.low,
                    close=c.close,
                    volume=c.volume,
                    trade_count=c.trade_count,
                    synthesised=False,
                )
            else:
                # No trades — synthesise a flat candle from last known close
                price = self._last_close or 0.0
                completed = CompletedCandle(
                    timestamp_utc=_ms_to_utc(ts),
                    open=price, high=price, low=price, close=price,
                    volume=0.0,
                    trade_count=0,
                    synthesised=True,
                )
            # Reset the accumulator
            self._candle = None
            return completed

def merge_candle_into(
    candle: CompletedCandle,
    df: pd.DataFrame,
    window: int = 500,
) -> pd.DataFrame:
    """Append a ``CompletedCandle`` to a historical OHLCV DataFrame.

    Args:
        candle: The completed candle from ``TickAggregator.close_candle()``.
        df:     Historical OHLCV DataFrame with UTC ``DatetimeIndex`` and
                columns ``open, high, low, close, volume``.
        window: Rolling window — trim ``df`` to this many rows after append.

    Returns:
        New DataFrame with the candle appended (deduplicated, sorted).

    Raises:
        ValueError: If ``df`` is missing required columns.
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    new_row = pd.DataFrame(
        [{
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
        }],
        index=pd.DatetimeIndex([candle.timestamp_utc], name=df.index.name or "timestamp"),
    )

    combined = pd.concat([df, new_row])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)

    if len(combined) > window:
        combined = combined.iloc[-window:]

    return combined


def _ms_to_utc(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC)
