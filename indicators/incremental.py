"""Stateful O(1) incremental indicators with periodic resync.

The classes here update from candle-close ticks without pandas. They retain a
bounded rolling window only for validation/resync. Resync is useful because live
systems can accumulate tiny floating-point differences, and more importantly can
receive corrected candles after reconnects; periodically replaying the retained
window brings the state back toward the batch reference implementation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable


@dataclass(frozen=True)
class OHLCTick:
    high: float
    low: float
    close: float


@dataclass
class IncrementalEMA:
    period: int
    resync_interval: int = 1_000
    window_size: int | None = None
    value: float | None = None
    _ticks: int = 0
    _seed: list[float] = field(default_factory=list)
    _window: Deque[float] = field(init=False)

    def __post_init__(self) -> None:
        self._validate_periods()
        self.window_size = self.window_size or max(self.period * 100, self.period + 1)
        self._window = deque(maxlen=self.window_size)

    def update(self, close: float) -> float | None:
        self._window.append(float(close))
        output = self._update_core(float(close))
        self._maybe_resync()
        return output

    def resync(self) -> float | None:
        values = list(self._window)
        self.reset(clear_window=False)
        for close in values:
            self._update_core(close)
        return self.value

    def reset(self, *, clear_window: bool = True) -> None:
        self.value = None
        self._ticks = 0
        self._seed = []
        if clear_window:
            self._window.clear()

    @property
    def ready(self) -> bool:
        return self.value is not None

    def _update_core(self, close: float) -> float | None:
        self._ticks += 1
        if self.value is None:
            self._seed.append(close)
            if len(self._seed) < self.period:
                return None
            self.value = sum(self._seed) / self.period
            return self.value

        alpha = 2.0 / (self.period + 1.0)
        self.value = alpha * close + (1.0 - alpha) * self.value
        return self.value

    def _maybe_resync(self) -> None:
        if self.resync_interval > 0 and self._ticks % self.resync_interval == 0:
            self.resync()

    def _validate_periods(self) -> None:
        if self.period <= 0:
            raise ValueError("period must be positive")
        if self.resync_interval < 0:
            raise ValueError("resync_interval cannot be negative")


@dataclass
class IncrementalRSI:
    period: int = 14
    resync_interval: int = 1_000
    window_size: int | None = None
    value: float | None = None
    _ticks: int = 0
    _previous_close: float | None = None
    _avg_gain: float | None = None
    _avg_loss: float | None = None
    _window: Deque[float] = field(init=False)

    def __post_init__(self) -> None:
        self._validate_periods()
        self.window_size = self.window_size or max(self.period * 100, self.period + 2)
        self._window = deque(maxlen=self.window_size)

    def update(self, close: float) -> float | None:
        self._window.append(float(close))
        output = self._update_core(float(close))
        self._maybe_resync()
        return output

    def resync(self) -> float | None:
        values = list(self._window)
        self.reset(clear_window=False)
        for close in values:
            self._update_core(close)
        return self.value

    def reset(self, *, clear_window: bool = True) -> None:
        self.value = None
        self._ticks = 0
        self._previous_close = None
        self._avg_gain = None
        self._avg_loss = None
        if clear_window:
            self._window.clear()

    @property
    def ready(self) -> bool:
        return self.value is not None

    def _update_core(self, close: float) -> float | None:
        self._ticks += 1
        if self._previous_close is None:
            self._previous_close = close
            return None

        delta = close - self._previous_close
        self._previous_close = close
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)

        if self._avg_gain is None or self._avg_loss is None:
            # pandas-ta RSI uses RMA via ewm(alpha=1/period, adjust=False),
            # which seeds the average from the first non-NaN delta rather than
            # from an initial SMA window.
            self._avg_gain = gain
            self._avg_loss = loss
        else:
            self._avg_gain = (self._avg_gain * (self.period - 1) + gain) / self.period
            self._avg_loss = (self._avg_loss * (self.period - 1) + loss) / self.period

        self.value = self._rsi(self._avg_gain, self._avg_loss)
        return self.value

    @staticmethod
    def _rsi(avg_gain: float, avg_loss: float) -> float:
        if avg_loss == 0.0:
            return 100.0 if avg_gain > 0.0 else 50.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _maybe_resync(self) -> None:
        if self.resync_interval > 0 and self._ticks % self.resync_interval == 0:
            self.resync()

    def _validate_periods(self) -> None:
        if self.period <= 0:
            raise ValueError("period must be positive")
        if self.resync_interval < 0:
            raise ValueError("resync_interval cannot be negative")


@dataclass
class IncrementalATR:
    period: int = 14
    resync_interval: int = 1_000
    window_size: int | None = None
    value: float | None = None
    _ticks: int = 0
    _previous_close: float | None = None
    _true_ranges: list[float] = field(default_factory=list)
    _window: Deque[OHLCTick] = field(init=False)

    def __post_init__(self) -> None:
        self._validate_periods()
        self.window_size = self.window_size or max(self.period * 100, self.period + 2)
        self._window = deque(maxlen=self.window_size)

    def update(self, high: float, low: float, close: float) -> float | None:
        tick = OHLCTick(float(high), float(low), float(close))
        self._window.append(tick)
        output = self._update_core(tick)
        self._maybe_resync()
        return output

    def resync(self) -> float | None:
        ticks = list(self._window)
        self.reset(clear_window=False)
        for tick in ticks:
            self._update_core(tick)
        return self.value

    def reset(self, *, clear_window: bool = True) -> None:
        self.value = None
        self._ticks = 0
        self._previous_close = None
        self._true_ranges = []
        if clear_window:
            self._window.clear()

    @property
    def ready(self) -> bool:
        return self.value is not None

    def _update_core(self, tick: OHLCTick) -> float | None:
        self._ticks += 1
        true_range = self._true_range(tick)
        self._previous_close = tick.close

        if self.value is None:
            self._true_ranges.append(true_range)
            if len(self._true_ranges) < self.period:
                return None
            self.value = sum(self._true_ranges) / self.period
            return self.value

        self.value = (self.value * (self.period - 1) + true_range) / self.period
        return self.value

    def _true_range(self, tick: OHLCTick) -> float:
        if self._previous_close is None:
            return tick.high - tick.low
        return max(
            tick.high - tick.low,
            abs(tick.high - self._previous_close),
            abs(tick.low - self._previous_close),
        )

    def _maybe_resync(self) -> None:
        if self.resync_interval > 0 and self._ticks % self.resync_interval == 0:
            self.resync()

    def _validate_periods(self) -> None:
        if self.period <= 0:
            raise ValueError("period must be positive")
        if self.resync_interval < 0:
            raise ValueError("resync_interval cannot be negative")


def replay_ema(values: Iterable[float], period: int) -> list[float | None]:
    indicator = IncrementalEMA(period=period, resync_interval=0)
    return [indicator.update(value) for value in values]


def replay_rsi(values: Iterable[float], period: int) -> list[float | None]:
    indicator = IncrementalRSI(period=period, resync_interval=0)
    return [indicator.update(value) for value in values]


def replay_atr(rows: Iterable[tuple[float, float, float]], period: int) -> list[float | None]:
    indicator = IncrementalATR(period=period, resync_interval=0)
    return [indicator.update(high, low, close) for high, low, close in rows]


@dataclass
class IncrementalADX:
    """Stateful O(1) Average Directional Index (Wilder's smoothing).

    Seeding requirement:
    - First ``period`` candles build the initial smoothed +DM, -DM, and ATR.
    - The next ``period`` DX values are averaged to seed ADX itself.
    - Total warmup: ``2 * period`` candles before the first ADX value is
      emitted. This matches the pandas-ta / Wilder reference behaviour.

    Known failure mode:
    - ADX lags by design — it smooths DX over ``period`` bars, so it reacts
      slowly to regime changes. Do not use as a fast-entry signal.
    """

    period: int = 14
    resync_interval: int = 1_000
    window_size: int | None = None
    value: float | None = None          # ADX
    plus_di: float | None = None        # +DI
    minus_di: float | None = None       # -DI
    _ticks: int = 0
    _previous_close: float | None = None
    _previous_high: float | None = None
    _previous_low: float | None = None
    # Phase 1 accumulators (first period candles)
    _dm_plus_list: list[float] = field(default_factory=list)
    _dm_minus_list: list[float] = field(default_factory=list)
    _tr_list: list[float] = field(default_factory=list)
    # Wilder smoothed values
    _sm_dm_plus: float | None = None
    _sm_dm_minus: float | None = None
    _sm_tr: float | None = None
    # Phase 2 accumulators (next period DX values → seed ADX)
    _dx_list: list[float] = field(default_factory=list)
    _window: Deque[OHLCTick] = field(init=False)

    def __post_init__(self) -> None:
        if self.period <= 0:
            raise ValueError("period must be positive")
        if self.resync_interval < 0:
            raise ValueError("resync_interval cannot be negative")
        self.window_size = self.window_size or max(self.period * 100, self.period * 2 + 1)
        self._window = deque(maxlen=self.window_size)

    def update(self, high: float, low: float, close: float) -> float | None:
        tick = OHLCTick(float(high), float(low), float(close))
        self._window.append(tick)
        output = self._update_core(tick)
        self._maybe_resync()
        return output

    def resync(self) -> float | None:
        ticks = list(self._window)
        self.reset(clear_window=False)
        for tick in ticks:
            self._update_core(tick)
        return self.value

    def reset(self, *, clear_window: bool = True) -> None:
        self.value = None
        self.plus_di = None
        self.minus_di = None
        self._ticks = 0
        self._previous_close = None
        self._previous_high = None
        self._previous_low = None
        self._dm_plus_list = []
        self._dm_minus_list = []
        self._tr_list = []
        self._sm_dm_plus = None
        self._sm_dm_minus = None
        self._sm_tr = None
        self._dx_list = []
        if clear_window:
            self._window.clear()

    @property
    def ready(self) -> bool:
        return self.value is not None

    def _update_core(self, tick: OHLCTick) -> float | None:  # noqa: PLR0912
        self._ticks += 1

        if self._previous_high is None:
            # First candle — store reference values only
            self._previous_high = tick.high
            self._previous_low = tick.low
            self._previous_close = tick.close
            return None

        # ── Directional Movement & True Range ──────────────────────────────
        up_move = tick.high - self._previous_high
        down_move = self._previous_low - tick.low

        dm_plus = up_move if (up_move > down_move and up_move > 0) else 0.0
        dm_minus = down_move if (down_move > up_move and down_move > 0) else 0.0

        tr = max(
            tick.high - tick.low,
            abs(tick.high - self._previous_close),
            abs(tick.low - self._previous_close),
        )

        self._previous_high = tick.high
        self._previous_low = tick.low
        self._previous_close = tick.close

        # ── Phase 1: accumulate first ``period`` DM / TR values ────────────
        if self._sm_tr is None:
            self._dm_plus_list.append(dm_plus)
            self._dm_minus_list.append(dm_minus)
            self._tr_list.append(tr)
            if len(self._tr_list) < self.period:
                return None
            # Seed Wilder smoothed values with simple sum
            self._sm_dm_plus = sum(self._dm_plus_list)
            self._sm_dm_minus = sum(self._dm_minus_list)
            self._sm_tr = sum(self._tr_list)
            # ── Fall-through to compute first DX ───────────────────────
        else:
            # ── Phase 2+: Wilder smoothing (subtract 1/period, add new) ────
            self._sm_dm_plus = self._sm_dm_plus - (self._sm_dm_plus / self.period) + dm_plus
            self._sm_dm_minus = self._sm_dm_minus - (self._sm_dm_minus / self.period) + dm_minus
            self._sm_tr = self._sm_tr - (self._sm_tr / self.period) + tr

        # ── DI calculation ─────────────────────────────────────────────
        if self._sm_tr == 0.0:
            return None  # Guard: flat price session, no range

        self.plus_di = 100.0 * self._sm_dm_plus / self._sm_tr
        self.minus_di = 100.0 * self._sm_dm_minus / self._sm_tr

        di_sum = self.plus_di + self.minus_di
        dx = 100.0 * abs(self.plus_di - self.minus_di) / di_sum if di_sum != 0.0 else 0.0

        # ── Phase 2: accumulate ``period`` DX values to seed ADX ───────────
        if self.value is None:
            self._dx_list.append(dx)
            if len(self._dx_list) < self.period:
                return None
            self.value = sum(self._dx_list) / self.period
            return self.value

        # ── Phase 3: Wilder-smooth ADX ────────────────────────────────
        self.value = (self.value * (self.period - 1) + dx) / self.period
        return self.value

    def _maybe_resync(self) -> None:
        if self.resync_interval > 0 and self._ticks % self.resync_interval == 0:
            self.resync()


def replay_adx(
    rows: Iterable[tuple[float, float, float]], period: int
) -> list[float | None]:
    """Convenience: replay a sequence of (high, low, close) tuples through ADX."""
    indicator = IncrementalADX(period=period, resync_interval=0)
    return [indicator.update(high, low, close) for high, low, close in rows]
