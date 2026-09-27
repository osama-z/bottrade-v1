"""
Trend Following — single-idea strategy: trade Supertrend flips, but only
when the flip agrees with the trend side (EMA 50 — the slowest MA the
indicator layer guarantees on short frames) and the trend is actually
strong (ADX).

Why this exists (measured, 2026-07): the AI-combined pipeline — three
blended model scores gated by five filters — produced ~2 trades/year in
walk-forward. Statistically untestable. This strategy is its structural
opposite: ONE idea a human can fully reason about, symmetric long/short,
no trained components (nothing to overfit, no retraining windows), and
enough signals to accumulate a real sample. It is the experiment control
that tells us whether the AI ensemble adds anything over "follow the
trend with disciplined exits" — exits, sizing, and stops all come from
the SAME risk engine either way.

Signal semantics (state-change, not state): +1/-1 only on the candle
where the entry condition BECOMES true — the engine opens there and
manages the position via ATR stops/targets; an opposite signal closes
and may reverse.
"""

import pandas as pd
from loguru import logger

from config.constants import Signal
from strategies.base import BaseStrategy, TradeSignal


class TrendFollowingStrategy(BaseStrategy):
    """Supertrend flip + EMA-50 side + ADX strength, long and short."""

    name = "trend_following"

    def __init__(self, adx_min: float = 20.0, use_trend_filter: bool = False) -> None:
        self.adx_min = adx_min
        # Optional 200-SMA trend-alignment gate. OFF by default so the base
        # strategy is unchanged (byte-identical signals). Measured (2026-07,
        # docs/STRATEGY_NOTES.md): turning it ON improves full-period Sharpe on
        # the aggregate and rescues LINK/BTC — it drops counter-trend flips in
        # non-trending markets. Enabled by the trend_following_filtered variant.
        self.use_trend_filter = use_trend_filter

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Vectorized signals: ±1 on qualifying Supertrend flips, else 0.

        Uses only same-candle and prior-candle values — no lookahead by
        construction (verified by the tier-16 shift test).
        """
        required = {"Supertrend_dir", "EMA_50", "ADX", "close"}
        missing = required - set(df.columns)
        if missing:
            logger.warning("trend_following: missing columns {}", missing)
            return pd.Series(0, index=df.index)

        direction = df["Supertrend_dir"]
        flipped_up = (direction == 1) & (direction.shift(1) == -1)
        flipped_down = (direction == -1) & (direction.shift(1) == 1)

        # Long-term side: EMA_50 is the slowest MA the indicator layer
        # guarantees on short frames; the Supertrend flip supplies timing,
        # the EMA supplies the side, ADX supplies "is there a trend at all".
        above_ema = df["close"] > df["EMA_50"]
        strong = df["ADX"] >= self.adx_min

        signals = pd.Series(0, index=df.index)
        signals[flipped_up & above_ema & strong] = 1
        signals[flipped_down & ~above_ema & strong] = -1

        # Optional long-term trend-alignment gate: only take LONGS in a
        # confirmed uptrend (close > SMA_200) and SHORTS in a confirmed
        # downtrend — "don't trade against the big trend". Missing SMA_200
        # (short frame) → gate skipped, never crashes.
        if self.use_trend_filter and "SMA_200" in df.columns:
            above_200 = df["close"] > df["SMA_200"]
            signals[(signals == 1) & ~above_200] = 0
            signals[(signals == -1) & above_200] = 0

        n_buy, n_sell = int((signals == 1).sum()), int((signals == -1).sum())
        logger.debug("Trend-following signals: {} buys, {} sells", n_buy, n_sell)
        return signals

    def get_signal(self, df: pd.DataFrame, pair: str) -> TradeSignal:
        """Latest-candle signal for live use — same code path as backtest."""
        signals = self.generate_signals(df)
        latest = int(signals.iloc[-1]) if len(signals) else 0
        mapping = {1: Signal.BUY, -1: Signal.SELL, 0: Signal.HOLD}
        adx = float(df["ADX"].iloc[-1]) if "ADX" in df.columns else 0.0
        return TradeSignal(
            signal=mapping[latest],
            confidence=0.7 if latest != 0 else 0.0,
            pair=pair,
            price=float(df["close"].iloc[-1]),
            reason=(
                f"Supertrend flip with EMA side + ADX {adx:.0f}"
                if latest != 0 else "No qualifying trend flip"
            ),
        )

    def get_params(self) -> dict:
        return {"adx_min": self.adx_min, "use_trend_filter": self.use_trend_filter}


class TrendFollowingFilteredStrategy(TrendFollowingStrategy):
    """trend_following + 200-SMA long-term trend-alignment gate.

    A more regime-robust variant: it drops Supertrend flips that fight the
    long-term trend, which measured better on the full-period aggregate and
    on BTC/LINK across BOTH halves of the sample (docs/STRATEGY_NOTES.md).
    Not yet the live default — a candidate for the next run / an A/B against
    the base, not a mid-run swap.
    """

    name = "trend_following_filtered"

    def __init__(self, adx_min: float = 20.0) -> None:
        super().__init__(adx_min=adx_min, use_trend_filter=True)
