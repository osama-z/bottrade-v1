"""Realistic transaction-cost & slippage model for the backtest simulator.

Roadmap Phase 1, Task 1.2 — penalise every simulated trade with real-world
friction so the backtest reflects harsh market reality instead of frictionless
fills at the close:

- Exchange fees: Binance VIP0 taker (0.04%) / maker (0.02%), by order type.
- Slippage: depth-based. fill = signal_price ± (order_qty / liquidity) *
  impact_factor, where `liquidity` is the top-N-levels depth. Live, that depth
  comes from data/order_book.py; a historical OHLCV backtest has no stored
  book, so depth is proxied from candle volume (depth_fraction * volume).
- Latency: a mandatory 200–2000 ms delay between signal and fill, converted to
  a small adverse price move (adverse selection during the delay).
- Funding: for perpetuals, funding accrues every `funding_interval_hours`
  (8h) — longs pay when funding is positive, shorts receive (and vice-versa).

The model is deterministic given `seed` (latency draws use a seeded RNG), so
backtests remain reproducible.

`CostModel.flat(commission_pct, slippage_pct)` reproduces the engine's legacy
flat-percentage behaviour exactly (used for frictionless / fee-isolation tests
and backward compatibility).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class CostModel:
    # ── Exchange fees (Binance VIP0) ──────────────────────────────────────────
    taker_fee_pct: float = 0.0004     # 0.04% — market / crossing orders
    maker_fee_pct: float = 0.0002     # 0.02% — resting limit (e.g. take-profit)

    # ── Depth-based slippage ──────────────────────────────────────────────────
    impact_factor: float = 0.5        # coefficient on (order_qty / liquidity)
    depth_fraction: float = 0.10      # share of candle volume treated as top-N depth
    max_slippage_pct: float = 0.01    # cap total adverse slippage at 1%

    # ── Latency penalty ───────────────────────────────────────────────────────
    latency_min_ms: float = 200.0
    latency_max_ms: float = 2000.0
    latency_bps_per_s: float = 0.5    # adverse basis points per second of latency
    enable_latency: bool = True

    # ── Funding (perpetuals) ──────────────────────────────────────────────────
    funding_interval_hours: float = 8.0

    # ── Legacy flat mode ──────────────────────────────────────────────────────
    # When set, slippage is a flat percentage and latency is disabled — this
    # reproduces the pre-Task-1.2 engine exactly. None → depth+latency model.
    flat_slippage_pct: Optional[float] = None

    seed: int = 0

    # ─── Construction helpers ─────────────────────────────────────────────────
    @classmethod
    def flat(cls, commission_pct: float, slippage_pct: float) -> "CostModel":
        """Legacy flat-percentage model: taker == maker == commission_pct,
        constant slippage_pct, no latency. Reproduces old engine behaviour."""
        return cls(
            taker_fee_pct=commission_pct,
            maker_fee_pct=commission_pct,
            flat_slippage_pct=slippage_pct,
            enable_latency=False,
        )

    @property
    def is_flat(self) -> bool:
        return self.flat_slippage_pct is not None

    # ─── Fees ─────────────────────────────────────────────────────────────────
    def fee(self, notional: float, order_type: str = "taker") -> float:
        """Fee in quote currency for a fill of `notional`, by order type."""
        rate = self.maker_fee_pct if order_type == "maker" else self.taker_fee_pct
        return abs(notional) * rate

    # ─── Slippage / fill price ────────────────────────────────────────────────
    def slippage_pct_for(
        self,
        order_qty: float,
        candle_volume: float,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Total adverse slippage fraction (depth impact + latency), capped."""
        if self.is_flat:
            return float(self.flat_slippage_pct)

        liquidity = max(candle_volume * self.depth_fraction, _EPS)
        depth_slip = (abs(order_qty) / liquidity) * self.impact_factor

        latency_slip = 0.0
        if self.enable_latency:
            latency_ms = self.draw_latency_ms(rng)
            latency_slip = (latency_ms / 1000.0) * (self.latency_bps_per_s / 10_000.0)

        return min(depth_slip + latency_slip, self.max_slippage_pct)

    def fill_price(
        self,
        signal_price: float,
        order_qty: float,
        candle_volume: float,
        is_buy: bool,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Adverse fill price: buys fill higher, sells/exits fill lower."""
        slip = self.slippage_pct_for(order_qty, candle_volume, rng)
        direction = 1.0 if is_buy else -1.0
        return signal_price * (1.0 + direction * slip)

    def draw_latency_ms(self, rng: Optional[np.random.Generator] = None) -> float:
        """A mandatory signal→fill latency in [latency_min_ms, latency_max_ms]."""
        if rng is None:
            rng = np.random.default_rng(self.seed)
        return float(rng.uniform(self.latency_min_ms, self.latency_max_ms))

    # ─── Funding (perpetuals) ─────────────────────────────────────────────────
    def funding_cost(
        self,
        notional: float,
        hours_held: float,
        funding_rate: float,
        is_long: bool,
    ) -> float:
        """Funding paid (positive) or received (negative) over the hold.

        A perp position crosses one funding epoch every
        ``funding_interval_hours``. With positive funding, longs PAY and shorts
        RECEIVE; the sign flips for negative funding.
        """
        if hours_held <= 0 or funding_rate == 0:
            return 0.0
        epochs = int(hours_held // self.funding_interval_hours)
        if epochs <= 0:
            return 0.0
        direction = 1.0 if is_long else -1.0
        return epochs * funding_rate * abs(notional) * direction
