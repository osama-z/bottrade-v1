"""Tier-16 tests: trend_following strategy contract + strategy lab.

The strategy exists as the experiment CONTROL: one human-auditable idea,
symmetric long/short, nothing trained. These tests pin its contract —
especially no-lookahead, the property that makes lab results meaningful.
"""

import numpy as np
import pandas as pd
import pytest

from config.constants import Signal
from strategies.registry import get_strategy, list_strategies
from strategies.trend_following import TrendFollowingStrategy


def make_df(n: int = 120, seed: int = 7) -> pd.DataFrame:
    """Frame with the columns the strategy needs, deterministic."""
    rng = np.random.RandomState(seed)
    prices = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    df = pd.DataFrame(
        {"close": prices},
        index=pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
    )
    # Alternate trend regimes so flips exist in both directions
    direction = np.where(np.sin(np.arange(n) / 10) > 0, 1, -1)
    df["Supertrend_dir"] = direction
    df["EMA_50"] = pd.Series(prices).rolling(20, min_periods=1).mean().values
    df["ADX"] = 25.0
    return df


class TestSignalContract:
    def test_registered(self):
        assert "trend_following" in list_strategies()
        assert isinstance(get_strategy("trend_following"), TrendFollowingStrategy)

    def test_signals_are_ternary_and_aligned(self):
        df = make_df()
        s = TrendFollowingStrategy().generate_signals(df)
        assert set(s.unique()) <= {-1, 0, 1}
        assert s.index.equals(df.index)

    def test_signals_fire_on_flips_only(self):
        """State-change semantics: a +1 requires the PREVIOUS candle to be
        in the opposite Supertrend state."""
        df = make_df()
        s = TrendFollowingStrategy().generate_signals(df)
        for ts in s[s == 1].index:
            i = df.index.get_loc(ts)
            assert df["Supertrend_dir"].iloc[i] == 1
            assert df["Supertrend_dir"].iloc[i - 1] == -1

    def test_adx_gate_blocks_weak_trends(self):
        df = make_df()
        df["ADX"] = 5.0   # no trend anywhere
        s = TrendFollowingStrategy(adx_min=20.0).generate_signals(df)
        assert (s == 0).all()

    def test_missing_columns_yield_all_hold(self):
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        s = TrendFollowingStrategy().generate_signals(df)
        assert (s == 0).all()

    def test_no_lookahead(self):
        """Signal at candle t must not change when future candles change.

        Truncate the frame at every flip point and require the same signal
        — the property that makes backtest results transferable to live.
        """
        df = make_df()
        strat = TrendFollowingStrategy()
        full = strat.generate_signals(df)
        for ts in full[full != 0].index:
            i = df.index.get_loc(ts)
            truncated = strat.generate_signals(df.iloc[: i + 1])
            assert truncated.iloc[-1] == full.iloc[i], (
                f"signal at {ts} depends on future data"
            )

    def test_get_signal_matches_series_tail(self):
        df = make_df()
        strat = TrendFollowingStrategy()
        series = strat.generate_signals(df)
        live = strat.get_signal(df, "BTC/USDT")
        expected = {1: Signal.BUY, -1: Signal.SELL, 0: Signal.HOLD}[int(series.iloc[-1])]
        assert live.signal == expected
        assert live.pair == "BTC/USDT"
        assert live.price == pytest.approx(float(df["close"].iloc[-1]))


class TestTrendFilter:
    """The optional 200-SMA trend-alignment gate (trend_following_filtered)."""

    def _flip_df(self, sma200: float) -> pd.DataFrame:
        # Supertrend flips up at index 1; close>EMA_50, ADX>=20 → base long.
        return pd.DataFrame(
            {
                "Supertrend_dir": [-1, 1],
                "close": [100.0, 110.0],
                "EMA_50": [95.0, 95.0],
                "ADX": [25.0, 25.0],
                "SMA_200": [sma200, sma200],
            },
            index=pd.date_range("2024-01-01", periods=2, freq="4h", tz="UTC"),
        )

    def test_filtered_is_registered(self):
        from strategies.trend_following import TrendFollowingFilteredStrategy
        assert "trend_following_filtered" in list_strategies()
        assert isinstance(
            get_strategy("trend_following_filtered"), TrendFollowingFilteredStrategy
        )

    def test_base_default_is_unfiltered(self):
        # Base must be byte-identical whether or not the filter exists.
        df = self._flip_df(sma200=200.0)  # close 110 < SMA_200 200
        assert TrendFollowingStrategy().generate_signals(df).iloc[1] == 1

    def test_filter_drops_counter_trend_long(self):
        df = self._flip_df(sma200=200.0)  # close 110 < 200 → below 200-SMA
        assert get_strategy("trend_following_filtered").generate_signals(df).iloc[1] == 0

    def test_filter_keeps_aligned_long(self):
        df = self._flip_df(sma200=90.0)  # close 110 > 90 → above 200-SMA
        assert get_strategy("trend_following_filtered").generate_signals(df).iloc[1] == 1

    def test_filtered_signals_are_subset_of_base(self):
        df = make_df()
        base = TrendFollowingStrategy().generate_signals(df)
        filt = get_strategy("trend_following_filtered").generate_signals(df)
        # The gate only removes signals, never adds or flips them.
        assert (filt != 0).sum() <= (base != 0).sum()
        assert ((filt != 0) & (filt != base)).sum() == 0


class TestLabHarness:
    def test_lab_runs_end_to_end_offline(self, tmp_path, monkeypatch):
        """Full lab pass on synthetic data — no network."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        import strategy_lab

        df_raw = make_df(300)[["close"]].copy()
        df_raw["open"] = df_raw["close"]
        df_raw["high"] = df_raw["close"] * 1.002
        df_raw["low"] = df_raw["close"] * 0.998
        df_raw["volume"] = 100.0

        class FakeFetcher:
            def get_historical_data(self, pair, timeframe, days):
                return df_raw.copy()

        monkeypatch.setattr(strategy_lab, "DataFetcher", lambda: FakeFetcher())
        monkeypatch.setattr(strategy_lab, "CACHE_DIR", tmp_path)

        report = strategy_lab.run_lab(
            pairs=["BTC/USDT"], timeframes=["1h"],
            strategies=["trend_following", "ma_crossover"], days=30,
        )
        assert len(report) == 2
        assert {"strategy", "n_trades", "sharpe", "return_pct"} <= set(report.columns)

    def test_breaker_deadlock_is_broken_by_simulated_operator(self):
        """A 3-loss streak trips the breaker (parity with live via
        register_closed_trade). Without the simulated operator the rest of
        the backtest is dead; with review_hours it resumes and counts."""
        from backtesting.engine import BacktestEngine

        n = 400
        # Sawtooth: every long entered at a local top gets stopped out
        prices = []
        for k in range(n):
            base = 100 - (k // 40)  # slow decline
            prices.append(base * (1.0 + (0.03 if k % 4 == 0 else -0.01)))
        df = pd.DataFrame({"open": prices, "high": [p * 1.001 for p in prices],
                           "low": [p * 0.94 for p in prices],  # deep lows → stops hit
                           "close": prices, "volume": [100.0] * n},
                          index=pd.date_range("2024-01-01", periods=n,
                                              freq="1h", tz="UTC"))
        df["ATR"] = 2.0
        sig = pd.Series(0, index=df.index)
        for i in range(2, n, 8):
            sig.iloc[i] = 1

        strict = BacktestEngine(initial_capital=10_000.0).run(
            df, sig, verbose=False, breaker_review_hours=None)
        reviewed = BacktestEngine(initial_capital=10_000.0).run(
            df, sig, verbose=False, breaker_review_hours=24.0)

        assert strict.n_breaker_trips >= 1, "streak must trip the breaker"
        assert reviewed.n_breaker_trips >= strict.n_breaker_trips
        assert reviewed.total_trades > strict.total_trades, (
            "operator reset must let trading resume after review window"
        )

    def test_engine_calls_register_closed_trade(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "backtesting" / "engine.py").read_text()
        assert src.count("register_closed_trade(") >= 2, (
            "both close paths must feed the edge-triggered loss breaker"
        )

    def test_engine_verbose_flag_suppresses_banner(self, capsys):
        from backtesting.engine import BacktestEngine
        df = make_df(60)[["close"]].copy()
        df["open"] = df["high"] = df["low"] = df["close"]
        df["volume"] = 100.0
        df["ATR"] = 2.0
        sig = pd.Series(0, index=df.index)
        BacktestEngine().run(df, sig, verbose=False)
        assert "BACKTEST RESULTS" not in capsys.readouterr().out
