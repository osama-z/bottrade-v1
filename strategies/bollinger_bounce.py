"""
Bollinger Band Bounce Strategy.

Logic:
- BUY  when price touches/crosses lower band (oversold, expect bounce up)
- SELL when price touches/crosses upper band (overbought, expect reversal)

Best in sideways/mean-reverting markets.
Also detects BB Squeeze (low volatility = breakout incoming).
"""

import pandas as pd
from loguru import logger

from strategies.base import BaseStrategy, TradeSignal
from config.constants import Signal, DEFAULT_BB_PERIOD, DEFAULT_BB_STD


class BollingerBounceStrategy(BaseStrategy):
    """Bollinger Band Bounce — fade extremes, trade the mean."""

    name = "bollinger_bounce"

    def __init__(
        self,
        period: int = DEFAULT_BB_PERIOD,
        std: float = DEFAULT_BB_STD,
        use_squeeze_filter: bool = True,   # Skip trades during low volatility
    ) -> None:
        self.period = period
        self.std = std
        self.use_squeeze_filter = use_squeeze_filter

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate +1/-1/0 signals for backtesting."""
        import pandas_ta as ta

        df = df.copy()

        bb = ta.bbands(df["close"], length=self.period, std=self.std)
        if bb is None or bb.empty:
            return pd.Series(0, index=df.index)

        bb_pct = bb.iloc[:, 4]    # %B: 0=lower band, 0.5=mid, 1=upper band

        signals = pd.Series(0, index=df.index)

        # BUY: price touches lower band (%B < 0.05)
        buy_condition = bb_pct < 0.05

        # SELL: price touches upper band (%B > 0.95)
        sell_condition = bb_pct > 0.95

        # Optional: skip during BB squeeze (low volatility)
        if self.use_squeeze_filter:
            bb_width = bb.iloc[:, 3]
            not_squeezing = bb_width > bb_width.rolling(20).mean() * 0.8
            buy_condition = buy_condition & not_squeezing
            sell_condition = sell_condition & not_squeezing

        signals[buy_condition] = 1
        signals[sell_condition] = -1

        # Only enter once per band touch (avoid repeated signals)
        # Transition signals only (new touches, not sustained)
        signals = self._deduplicate_signals(signals)

        logger.debug(
            "BB Bounce signals: {} buys, {} sells",
            (signals == 1).sum(), (signals == -1).sum()
        )

        return signals

    @staticmethod
    def _deduplicate_signals(signals: pd.Series) -> pd.Series:
        """Keep only the first signal in a consecutive sequence."""
        result = pd.Series(0, index=signals.index)
        prev = 0
        for i, val in enumerate(signals):
            if val != prev and val != 0:
                result.iloc[i] = val
            prev = val if val != 0 else prev
        return result

    def get_signal(self, df: pd.DataFrame, pair: str) -> TradeSignal:
        """Get latest signal for live trading."""
        signals = self.generate_signals(df)
        latest = signals.iloc[-1]
        price = df["close"].iloc[-1]

        bb_pct = df.get("BB_pct", pd.Series([0.5])).iloc[-1]
        bb_squeeze = df.get("BB_squeeze", pd.Series([False])).iloc[-1]

        if latest == 1:
            # Clamp to the documented [0, 1] contract: with a missing
            # BB_pct column the 0.5 default used to yield confidence -4.0
            confidence = max(0.0, min(0.50 + (0.05 - bb_pct) * 10, 0.88))
            return TradeSignal(
                signal=Signal.BUY,
                confidence=confidence,
                pair=pair,
                price=price,
                reason=f"Price at lower BB (%B={bb_pct:.2f}) — bounce expected",
            )
        elif latest == -1:
            confidence = min(0.50 + (bb_pct - 0.95) * 10, 0.88)
            return TradeSignal(
                signal=Signal.SELL,
                confidence=confidence,
                pair=pair,
                price=price,
                reason=f"Price at upper BB (%B={bb_pct:.2f}) — reversal expected",
            )
        else:
            squeeze_note = " (BB squeeze — breakout incoming!)" if bb_squeeze else ""
            return TradeSignal(
                signal=Signal.HOLD,
                confidence=0.0,
                pair=pair,
                price=price,
                reason=f"Inside bands (%B={bb_pct:.2f}){squeeze_note}",
            )

    def get_params(self) -> dict:
        return {
            "period": self.period,
            "std": self.std,
            "use_squeeze_filter": self.use_squeeze_filter,
        }
