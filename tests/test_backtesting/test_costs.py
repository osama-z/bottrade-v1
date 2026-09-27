"""Task 1.2 — realistic transaction-cost & slippage model.

Verifies fee-by-order-type, depth-based slippage, the latency penalty, perp
funding, legacy-flat compatibility, and that the realistic default actually
penalises backtested returns (and stays deterministic).
"""
import numpy as np
import pandas as pd
import pytest

from backtesting.costs import CostModel
from backtesting.engine import BacktestEngine


class TestFees:
    def test_vip0_taker_and_maker_by_order_type(self):
        cm = CostModel()
        assert cm.taker_fee_pct == 0.0004 and cm.maker_fee_pct == 0.0002
        assert cm.fee(1000.0, "taker") == pytest.approx(0.40)
        assert cm.fee(1000.0, "maker") == pytest.approx(0.20)
        # Unknown type defaults to the (more expensive) taker rate.
        assert cm.fee(1000.0, "whatever") == pytest.approx(0.40)


class TestDepthSlippage:
    def test_bigger_order_relative_to_depth_slips_more(self):
        cm = CostModel(enable_latency=False)
        small = cm.slippage_pct_for(order_qty=0.1, candle_volume=1000.0)
        big = cm.slippage_pct_for(order_qty=5.0, candle_volume=1000.0)
        assert big > small > 0

    def test_slippage_is_capped(self):
        cm = CostModel(enable_latency=False, max_slippage_pct=0.01)
        # Order dwarfs available depth → would be huge, but capped at 1%.
        assert cm.slippage_pct_for(order_qty=1e6, candle_volume=1.0) == pytest.approx(0.01)

    def test_fill_price_is_adverse_both_directions(self):
        cm = CostModel(enable_latency=False)
        buy = cm.fill_price(100.0, 1.0, 1000.0, is_buy=True)
        sell = cm.fill_price(100.0, 1.0, 1000.0, is_buy=False)
        assert buy > 100.0 > sell        # buys fill higher, sells lower


class TestLatency:
    def test_latency_draw_within_mandatory_window(self):
        cm = CostModel()
        rng = np.random.default_rng(0)
        for _ in range(50):
            assert 200.0 <= cm.draw_latency_ms(rng) <= 2000.0

    def test_latency_adds_adverse_slippage(self):
        with_lat = CostModel(enable_latency=True)
        without = CostModel(enable_latency=False)
        rng = np.random.default_rng(0)
        s_with = with_lat.slippage_pct_for(0.1, 1000.0, rng)
        s_without = without.slippage_pct_for(0.1, 1000.0)
        assert s_with > s_without


class TestFunding:
    def test_long_pays_positive_funding_per_8h_epoch(self):
        cm = CostModel()
        # 24h hold = 3 epochs; long pays 3 * 0.0001 * 1000 = 0.30
        assert cm.funding_cost(1000.0, 24.0, 0.0001, is_long=True) == pytest.approx(0.30)

    def test_short_receives_positive_funding(self):
        cm = CostModel()
        assert cm.funding_cost(1000.0, 24.0, 0.0001, is_long=False) == pytest.approx(-0.30)

    def test_sub_epoch_hold_pays_nothing(self):
        cm = CostModel()
        assert cm.funding_cost(1000.0, 4.0, 0.0001, is_long=True) == 0.0


class TestLegacyFlatCompatibility:
    def test_flat_reproduces_single_rate_and_constant_slippage(self):
        cm = CostModel.flat(commission_pct=0.001, slippage_pct=0.0005)
        assert cm.is_flat is True
        assert cm.fee(1000.0, "taker") == cm.fee(1000.0, "maker") == pytest.approx(1.0)
        assert cm.slippage_pct_for(order_qty=999.0, candle_volume=1.0) == pytest.approx(0.0005)
        assert cm.fill_price(100.0, 1.0, 1.0, is_buy=True) == pytest.approx(100.05)

    def test_zero_flat_is_frictionless(self):
        cm = CostModel.flat(0.0, 0.0)
        assert cm.fee(1000.0, "taker") == 0.0
        assert cm.fill_price(100.0, 5.0, 1.0, is_buy=True) == 100.0


def _profitable_df(n=20):
    dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    prices = np.linspace(100, 104, n)
    df = pd.DataFrame(
        {"open": prices, "high": prices * 1.001, "low": prices * 0.999,
         "close": prices, "volume": np.full(n, 5000.0)},
        index=dates,
    )
    df["ATR"] = 2.0
    return df


class TestEngineIntegration:
    def _signals(self, df):
        s = pd.Series(0, index=df.index)
        s.iloc[1] = 1
        s.iloc[-1] = -1
        return s

    def test_realistic_costs_reduce_returns_vs_frictionless(self):
        df = _profitable_df()
        sig = self._signals(df)
        frictionless = BacktestEngine(1000.0, commission_pct=0, slippage_pct=0).run(
            df, sig, verbose=False)
        realistic = BacktestEngine(1000.0).run(df, sig, verbose=False)   # default CostModel
        assert realistic.total_return_pct < frictionless.total_return_pct

    def test_default_engine_uses_realistic_cost_model(self):
        eng = BacktestEngine(1000.0)
        assert eng.cost_model.taker_fee_pct == 0.0004
        assert eng.cost_model.is_flat is False

    def test_realistic_backtest_is_deterministic(self):
        df = _profitable_df()
        sig = self._signals(df)
        r1 = BacktestEngine(1000.0).run(df, sig, verbose=False)
        r2 = BacktestEngine(1000.0).run(df, sig, verbose=False)
        assert r1.total_return_pct == r2.total_return_pct   # seeded latency
