"""Funding-Rate Carry — a delta-neutral structural-edge strategy (Roadmap Task 1.4).

Idea (market-neutral, not directional): perpetual futures charge a periodic
*funding* payment between longs and shorts. When funding is strongly POSITIVE,
longs pay shorts. You can harvest that by holding a delta-neutral pair:

    LONG spot  +  SHORT the equivalent perp

The two legs cancel price exposure (Δ ≈ 0), so the P&L is (approximately) the
funding collected by the short-perp leg minus trading costs — an edge that comes
from market *structure*, not from predicting direction. This is the structural
opposite of the price-prediction strategies: no view on where the market goes.

Rules
-----
- Enter when annualised funding APR exceeds ``entry_apr`` (e.g. > 30%).
- Hold through the funding epoch(s); exit when funding NORMALISES below
  ``exit_apr`` (hysteresis: exit_apr < entry_apr avoids churning around the
  threshold).
- Positive-funding only ("long spot, short perp"); the symmetric negative-funding
  carry would need to short spot, which spot trading can't do.

Signal semantics match the other rule-based strategies (state-CHANGE): +1 on the
bar the carry is ENTERED, -1 on the bar it is EXITED, 0 otherwise. The engine
trades the spot leg; the short-perp hedge is modelled by ``carry_returns()``,
which is the delta-neutral return series the CPCV rig (Task 1.3) should evaluate
— running it through the *price*-based backtest alone would ignore the hedge.

Annualisation: Binance settles funding every 8h → 3 epochs/day →
APR = funding_rate * 3 * 365.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from config.constants import Signal
from strategies.base import BaseStrategy, TradeSignal

EPOCHS_PER_DAY = 3          # Binance perp funding settles every 8 hours
DAYS_PER_YEAR = 365


class FundingCarryStrategy(BaseStrategy):
    """Delta-neutral funding-rate carry: long spot + short perp when funding is rich."""

    name = "funding_carry"

    def __init__(
        self,
        entry_apr: float = 0.30,       # enter when annualised funding > 30%
        exit_apr: float = 0.10,        # exit when it normalises below 10%
        funding_col: str = "funding_rate",
        epochs_per_day: int = EPOCHS_PER_DAY,
    ) -> None:
        if exit_apr > entry_apr:
            raise ValueError(
                f"exit_apr ({exit_apr}) must be <= entry_apr ({entry_apr}) for hysteresis"
            )
        self.entry_apr = entry_apr
        self.exit_apr = exit_apr
        self.funding_col = funding_col
        self.epochs_per_day = epochs_per_day

    # ── Funding math ──────────────────────────────────────────────────────────
    def annualized_apr(self, funding_rate: float) -> float:
        """Per-epoch funding rate → annualised APR (3 epochs/day * 365)."""
        return funding_rate * self.epochs_per_day * DAYS_PER_YEAR

    def _apr_series(self, df: pd.DataFrame) -> pd.Series:
        return df[self.funding_col].astype(float) * self.epochs_per_day * DAYS_PER_YEAR

    # ── Carry state (hysteresis) ──────────────────────────────────────────────
    def in_carry_state(self, df: pd.DataFrame) -> pd.Series:
        """Boolean series: True on bars the delta-neutral carry is held.

        Enter when APR > entry_apr; stay held until APR < exit_apr. The gap
        between the two thresholds prevents flip-flopping around the boundary.
        """
        apr = self._apr_series(df)
        state = pd.Series(False, index=df.index)
        holding = False
        for ts, a in apr.items():
            if not holding and a > self.entry_apr:
                holding = True
            elif holding and a < self.exit_apr:
                holding = False
            state.at[ts] = holding
        return state

    # ── BaseStrategy contract ─────────────────────────────────────────────────
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """State-change signals: +1 on carry entry, -1 on carry exit, else 0."""
        if self.funding_col not in df.columns:
            logger.warning("funding_carry: missing '{}' column", self.funding_col)
            return pd.Series(0, index=df.index)

        state = self.in_carry_state(df)
        prev = state.shift(1, fill_value=False)
        signals = pd.Series(0, index=df.index)
        signals[state & ~prev] = 1        # rising edge → enter carry
        signals[~state & prev] = -1       # falling edge → exit carry
        return signals

    def get_signal(self, df: pd.DataFrame, pair: str) -> TradeSignal:
        """Latest-bar signal for live use. BUY = open the delta-neutral carry."""
        price = float(df["close"].iloc[-1]) if "close" in df.columns and len(df) else 0.0
        if self.funding_col not in df.columns or len(df) == 0:
            return TradeSignal(
                signal=Signal.HOLD, confidence=0.0, pair=pair, price=price,
                reason="No funding-rate data",
            )

        apr = float(self._apr_series(df).iloc[-1])
        if apr > self.entry_apr:
            confidence = min(1.0, 0.70 + (apr - self.entry_apr))
            return TradeSignal(
                signal=Signal.BUY, confidence=confidence, pair=pair, price=price,
                reason=(
                    f"Funding carry: annualised {apr:.1%} APR > {self.entry_apr:.0%} "
                    f"— LONG spot + SHORT perp (delta-neutral)"
                ),
            )
        return TradeSignal(
            signal=Signal.HOLD, confidence=0.0, pair=pair, price=price,
            reason=f"Funding {apr:.1%} APR below entry {self.entry_apr:.0%}",
        )

    # ── Delta-neutral return series (for the CPCV rig) ────────────────────────
    def carry_returns(self, df: pd.DataFrame, cost_per_leg: float = 0.0004) -> pd.Series:
        """Per-bar delta-neutral carry return = funding collected while held.

        The long-spot / short-perp legs cancel price P&L, so the return is the
        funding rate captured by the short leg on each held epoch, minus a
        round-trip cost on both legs at entry and exit (4 taker fills total).
        This — not the price-based backtest — is what CPCV should evaluate.
        """
        if self.funding_col not in df.columns:
            return pd.Series(0.0, index=df.index)
        state = self.in_carry_state(df)
        funding = df[self.funding_col].astype(float).fillna(0.0)
        ret = state.astype(float) * funding            # short perp receives funding
        prev = state.shift(1, fill_value=False)
        turns = (state & ~prev).astype(float) + (~state & prev).astype(float)
        return ret - turns * (2.0 * cost_per_leg)      # 2 legs per entry / exit

    def get_params(self) -> dict:
        return {
            "entry_apr": self.entry_apr,
            "exit_apr": self.exit_apr,
            "epochs_per_day": self.epochs_per_day,
            "funding_col": self.funding_col,
        }
