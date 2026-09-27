"""Shadow-mode decision recording (Roadmap Task 5.1).

Shadow mode runs the live decision path but executes NOTHING — not even a paper
trade. For each decided candle it records the signal + the *expected* fill price
(signal price adjusted by the modelled slippage from backtesting/costs.py), so
live signals can later be compared against what the backtest would have produced
on the exact same data (parity gate: signal divergence < 5%).

``build_shadow_record`` is a pure function of (signal, price, cost model, candle
volume) so the expected-fill math is unit-tested with no live loop.
"""

from __future__ import annotations

from dataclasses import dataclass

from config.constants import Signal


@dataclass(frozen=True)
class ShadowRecord:
    decision: str            # BUY | SELL | HOLD
    price: float             # signal price (last close)
    expected_slippage: float # modelled adverse slippage fraction
    expected_fill: float     # price adjusted for slippage (== price for HOLD)


def build_shadow_record(
    signal: Signal,
    price: float,
    cost_model,
    *,
    candle_volume: float = 0.0,
    notional_usd: float = 1000.0,
) -> ShadowRecord:
    """Expected fill for a would-be order at ``price`` (no order placed).

    A BUY fills higher, a SELL lower, by the cost model's depth+latency slippage
    for a ``notional_usd`` order against ``candle_volume``. HOLD has no fill.
    """
    if signal not in (Signal.BUY, Signal.SELL) or price <= 0:
        return ShadowRecord(decision=signal.value, price=price,
                            expected_slippage=0.0, expected_fill=price)

    side_sign = 1.0 if signal == Signal.BUY else -1.0
    qty = notional_usd / price
    slippage = float(cost_model.slippage_pct_for(order_qty=qty, candle_volume=candle_volume))
    return ShadowRecord(
        decision=signal.value,
        price=price,
        expected_slippage=slippage,
        expected_fill=price * (1.0 + side_sign * slippage),
    )
