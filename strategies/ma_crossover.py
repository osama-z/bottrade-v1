"""
Moving Average Crossover Strategy.

Logic:
- BUY  when fast SMA crosses ABOVE slow SMA (golden cross)
- SELL when fast SMA crosses BELOW slow SMA (death cross)

Simple, classic, surprisingly effective with proper risk management.
"""

import pandas as pd
from loguru import logger

from strategies.base import BaseStrategy, TradeSignal
from config.constants import Signal, DEFAULT_SMA_FAST, DEFAULT_SMA_SLOW


class MACrossoverStrategy(BaseStrategy):
    """Moving Average Crossover — the classic trend-following strategy."""

    name = "ma_crossover"

    def __init__(
        self,
        fast: int = DEFAULT_SMA_FAST,
        slow: int = DEFAULT_SMA_SLOW,
        use_ema: bool = False,       # EMA reacts faster than SMA
    ) -> None:
        self.fast = fast
        self.slow = slow
        self.use_ema = use_ema
        self._ma_type = "EMA" if use_ema else "SMA"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate +1/-1/0 signals for backtesting."""
        import pandas_ta as ta

        df = df.copy()

        if self.use_ema:
            fast_ma = ta.ema(df["close"], length=self.fast)
            slow_ma = ta.ema(df["close"], length=self.slow)
        else:
            fast_ma = ta.sma(df["close"], length=self.fast)
            slow_ma = ta.sma(df["close"], length=self.slow)

        # Crossover detection
        above = fast_ma > slow_ma          # Fast is above slow
        # astype(bool) matters: shift+fillna yields an OBJECT-dtype series,
        # where `~` is Python's bitwise inversion (~True == -2), not logical
        # negation — silently corrupting the crossover mask.
        above_prev = above.shift(1).fillna(False).astype(bool)
        cross_up = above & ~above_prev      # Just crossed up
        cross_down = ~above & above_prev    # Just crossed down

        signals = pd.Series(0, index=df.index)
        signals[cross_up] = 1
        signals[cross_down] = -1

        logger.debug(
            "MA Crossover signals: {} buys, {} sells",
            (signals == 1).sum(), (signals == -1).sum()
        )

        return signals

    def get_signal(self, df: pd.DataFrame, pair: str) -> TradeSignal:
        """Get latest signal for live trading."""
        signals = self.generate_signals(df)
        latest_signal = signals.iloc[-1]
        price = df["close"].iloc[-1]

        # Calculate confidence based on distance between MAs
        fast_col = f"{self._ma_type}_{self.fast}"
        slow_col = f"{self._ma_type}_{self.slow}"

        if fast_col in df.columns and slow_col in df.columns:
            fast_val = df[fast_col].iloc[-1]
            slow_val = df[slow_col].iloc[-1]
            spread_pct = abs(fast_val - slow_val) / slow_val
            # Wider spread = stronger trend = higher confidence
            confidence = min(0.50 + spread_pct * 10, 0.90)
        else:
            confidence = 0.60

        if latest_signal == 1:
            signal = Signal.BUY
            reason = f"{self._ma_type}{self.fast} crossed above {self._ma_type}{self.slow}"
        elif latest_signal == -1:
            signal = Signal.SELL
            reason = f"{self._ma_type}{self.fast} crossed below {self._ma_type}{self.slow}"
        else:
            signal = Signal.HOLD
            reason = "No crossover — waiting"
            confidence = 0.0

        return TradeSignal(
            signal=signal,
            confidence=confidence,
            pair=pair,
            price=price,
            reason=reason,
        )

    def get_params(self) -> dict:
        return {"fast": self.fast, "slow": self.slow, "use_ema": self.use_ema}
