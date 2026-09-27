"""
RSI Mean Reversion Strategy.

Logic:
- BUY  when RSI drops below oversold level (default 30) — price is cheap
- SELL when RSI rises above overbought level (default 70) — price is expensive

Best used in ranging/sideways markets. Combine with trend filter for better results.
"""

import pandas as pd
from loguru import logger

from strategies.base import BaseStrategy, TradeSignal
from config.constants import Signal, DEFAULT_RSI_PERIOD, DEFAULT_RSI_OVERSOLD, DEFAULT_RSI_OVERBOUGHT


class RSIReversalStrategy(BaseStrategy):
    """RSI Mean Reversion — buy oversold, sell overbought."""

    name = "rsi_reversal"

    def __init__(
        self,
        period: int = DEFAULT_RSI_PERIOD,
        oversold: float = DEFAULT_RSI_OVERSOLD,
        overbought: float = DEFAULT_RSI_OVERBOUGHT,
        use_trend_filter: bool = True,   # Only buy in uptrend
    ) -> None:
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.use_trend_filter = use_trend_filter

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate +1/-1/0 signals for backtesting."""
        import pandas_ta as ta

        df = df.copy()
        rsi = ta.rsi(df["close"], length=self.period)
        signals = pd.Series(0, index=df.index)

        if self.use_trend_filter:
            # Only buy in uptrend (price above 200 SMA)
            sma_200 = ta.sma(df["close"], length=200)
            uptrend = df["close"] > sma_200
        else:
            uptrend = pd.Series(True, index=df.index)

        # BUY: RSI crosses back above oversold from below
        rsi_was_oversold = rsi.shift(1) < self.oversold
        rsi_recovering = rsi >= self.oversold
        buy_condition = rsi_was_oversold & rsi_recovering & uptrend
        signals[buy_condition] = 1

        # SELL: RSI crosses below overbought from above
        rsi_was_overbought = rsi.shift(1) > self.overbought
        rsi_cooling = rsi <= self.overbought
        sell_condition = rsi_was_overbought & rsi_cooling
        signals[sell_condition] = -1

        logger.debug(
            "RSI Reversal signals: {} buys, {} sells",
            (signals == 1).sum(), (signals == -1).sum()
        )

        return signals

    def get_signal(self, df: pd.DataFrame, pair: str) -> TradeSignal:
        """Get latest signal for live trading."""
        signals = self.generate_signals(df)
        latest = signals.iloc[-1]
        price = df["close"].iloc[-1]

        # Use RSI column if already computed, else compute fresh
        if "RSI" in df.columns:
            rsi_val = df["RSI"].iloc[-1]
        else:
            import pandas_ta as ta
            rsi_val = ta.rsi(df["close"], length=self.period).iloc[-1]

        if latest == 1:
            # Confidence: deeper the oversold, stronger the signal
            confidence = min(0.50 + (self.oversold - rsi_val) / self.oversold * 0.5, 0.90)
            return TradeSignal(
                signal=Signal.BUY,
                confidence=confidence,
                pair=pair,
                price=price,
                reason=f"RSI={rsi_val:.1f} recovering from oversold (<{self.oversold})",
            )
        elif latest == -1:
            confidence = min(0.50 + (rsi_val - self.overbought) / (100 - self.overbought) * 0.5, 0.90)
            return TradeSignal(
                signal=Signal.SELL,
                confidence=confidence,
                pair=pair,
                price=price,
                reason=f"RSI={rsi_val:.1f} retreating from overbought (>{self.overbought})",
            )
        else:
            return TradeSignal(
                signal=Signal.HOLD,
                confidence=0.0,
                pair=pair,
                price=price,
                reason=f"RSI={rsi_val:.1f} — neutral zone",
            )

    def get_params(self) -> dict:
        return {
            "period": self.period,
            "oversold": self.oversold,
            "overbought": self.overbought,
            "use_trend_filter": self.use_trend_filter,
        }
