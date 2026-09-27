"""Tier-8 regression tests: backtest/live parity redesign (audit V-43/45/46/52).

Covers: golden-file behavior preservation of the decide() extraction,
live-vs-replay identity, no-lookahead context slicing, engine parity
(no fallback sizing, affordability invariant, gap-through fills).
"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtesting.engine import BacktestEngine
from data.preprocessor import DataPreprocessor
from indicators.technical import TechnicalIndicators
from risk.manager import RiskManager
from strategies.ai_combined import AICombinedStrategy
from strategies.context import (
    MarketContext,
    build_replay_context,
    closed_candles_as_of,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Captured from the PRE-refactor get_signal (mocked fetchers, no disk
# models, mock LLM, empty headlines) — the decide() extraction must
# reproduce these exactly.
GOLDENS = {
    "up": {
        "signal": "HOLD",
        "confidence": 0.4688,
        "reason": "AI HOLD — score=+0.32, confidence=47%",
        "ai_score": 0.3199,
        "ai_signal": "BUY",
    },
    "down": {
        "signal": "HOLD",
        "confidence": 0.3513,
        "reason": "AI HOLD — score=-0.16, confidence=35%",
        "ai_score": -0.1638,
        "ai_signal": "HOLD",
    },
}


def make_frame(trend: str, n: int, freq: str) -> pd.DataFrame:
    rng = np.random.RandomState(7)
    dates = pd.date_range(
        start=datetime(2024, 1, 1, tzinfo=timezone.utc), periods=n, freq=freq
    )
    base = np.linspace(100, 160, n) if trend == "up" else np.linspace(160, 100, n)
    prices = base * (1 + rng.normal(0, 0.002, n))
    return pd.DataFrame({
        "open": prices * 0.999, "high": prices * 1.006, "low": prices * 0.994,
        "close": prices, "volume": rng.uniform(80, 120, n),
    }, index=dates)


class MockFetcher:
    def __init__(self, trend): self.trend = trend
    def fetch_ohlcv(self, pair, timeframe="1h", limit=500, since=None):
        return make_frame(self.trend, 100, "1d" if timeframe == "1d" else "4h")
    def fetch_funding_rate(self, pair): return 0.0001
    def fetch_order_book_imbalance(self, pair, percentage=0.01): return 1.2


def make_strategy(trend: str) -> AICombinedStrategy:
    # Production default: math-only execution (Roadmap Task 1.1) — LLM & VADER
    # sentiment decoupled from the decision path.
    strat = AICombinedStrategy(pair="BTC/USDT", use_llm=False)
    strat._model_loaded = False   # portability: no disk models in CI
    strat._regime_loaded = False
    strat._fetcher = MockFetcher(trend)
    return strat


def make_showcase_strategy(trend: str) -> AICombinedStrategy:
    """Full-fusion (LLM + ML + sentiment) strategy — the retained showcase path.

    The pre-refactor goldens captured this fusion behavior. Task 1.1 moved it
    off the execution path but kept the code, so we still regression-pin it here
    by injecting a full-fusion (math_only=False) combiner.
    """
    from ai.signal_combiner import SignalCombiner
    strat = make_strategy(trend)
    strat._combiner = SignalCombiner(buy_threshold=0.25, sell_threshold=-0.25, math_only=False)
    return strat


@pytest.fixture(scope="module")
def frames():
    pre, ti = DataPreprocessor(), TechnicalIndicators()
    out = {}
    for trend in ("up", "down"):
        out[trend] = ti.compute_all(pre.process(make_frame(trend, 300, "1h")))
    return out


# ─── Golden-file: decide() extraction preserved behavior exactly ───────────────

class TestGoldenParity:
    @pytest.mark.parametrize("trend", ["up", "down"])
    def test_get_signal_matches_pre_refactor_goldens(self, frames, trend):
        # Goldens capture the full-fusion showcase path (retained under Task 1.1).
        strat = make_showcase_strategy(trend)
        sig = strat.get_signal(frames[trend], "BTC/USDT")
        ai = strat.last_ai_signal
        g = GOLDENS[trend]
        assert sig.signal.value == g["signal"]
        assert round(sig.confidence, 10) == g["confidence"]
        assert sig.reason == g["reason"]
        assert round(ai.score, 10) == g["ai_score"]
        assert ai.signal.value == g["ai_signal"]

    @pytest.mark.parametrize("trend", ["up", "down"])
    def test_live_and_replay_paths_agree(self, frames, trend):
        """The acceptance criterion: same data through get_signal (live,
        mocked fetchers) and through decide() on a replay-style context
        must produce the same decision."""
        strat = make_strategy(trend)
        live = strat.get_signal(frames[trend], "BTC/USDT")

        pre, ti = DataPreprocessor(), TechnicalIndicators()
        ctx = MarketContext(
            df=frames[trend],
            df_1d=ti.compute_all(pre.process(make_frame(trend, 100, "1d"))),
            df_4h=ti.compute_all(pre.process(make_frame(trend, 100, "4h"))),
            funding_rate=0.0001,
            imbalance=1.2,
            news_headlines=(),
        )
        replay = strat.decide(ctx, "BTC/USDT", verbose=False)
        assert replay.signal == live.signal
        assert replay.confidence == pytest.approx(live.confidence)
        assert replay.reason == live.reason


# ─── No-lookahead slicing ──────────────────────────────────────────────────────

class TestNoLookahead:
    def test_replay_context_never_sees_the_future(self, frames):
        df = frames["up"]
        as_of = df.index[100]
        ctx = build_replay_context(history_1h=df, as_of=as_of)
        assert ctx.df.index[-1] == as_of
        assert (ctx.df.index <= as_of).all()

    def test_forming_higher_tf_candle_is_excluded(self):
        df_1d = make_frame("up", 30, "1d")
        # as-of the close of an 1h candle 5 hours into day 10: the day-10
        # daily candle is still forming and must be excluded.
        day10_open = df_1d.index[10]
        as_of_close = day10_open + pd.Timedelta(hours=5)
        sliced = closed_candles_as_of(df_1d, "1d", as_of_close)
        assert sliced.index[-1] == df_1d.index[9]

    def test_fully_closed_candle_is_included(self):
        df_1d = make_frame("up", 30, "1d")
        as_of_close = df_1d.index[10] + pd.Timedelta(days=1)
        sliced = closed_candles_as_of(df_1d, "1d", as_of_close)
        assert sliced.index[-1] == df_1d.index[10]

    def test_replay_generator_returns_aligned_series(self, frames):
        strat = make_strategy("up")
        df = frames["up"]
        idx = df.index[-5:]
        signals = strat.generate_signals_via_decide(df, decide_index=idx)
        assert list(signals.index) == list(idx)
        assert set(signals.unique()) <= {-1, 0, 1}


# ─── Engine parity ─────────────────────────────────────────────────────────────

def engine_df(prices: np.ndarray, atr: float | None = 2.0) -> pd.DataFrame:
    n = len(prices)
    dates = pd.date_range(start="2024-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "open": prices, "high": prices * 1.001, "low": prices * 0.999,
        "close": prices, "volume": np.full(n, 100.0),
    }, index=dates)
    if atr is not None:
        df["ATR"] = atr
    return df


class TestEngineParity:
    def test_no_atr_means_no_trades(self):
        """The static fallback sizing is gone: without ATR the engine must
        not invent a position (parity with live, which cannot size)."""
        df = engine_df(np.linspace(100, 110, 30), atr=None)
        signals = pd.Series(0, index=df.index)
        signals.iloc[1] = 1
        result = BacktestEngine(initial_capital=1000.0).run(df, signals)
        assert result.total_trades == 0

    def test_affordability_capital_never_negative(self):
        """Tiny ATR → notional capped at max (== initial capital); with
        commission the all-in cost exceeds cash → entry must be skipped."""
        df = engine_df(np.linspace(100, 110, 30), atr=0.05)
        signals = pd.Series(0, index=df.index)
        signals.iloc[1] = 1
        result = BacktestEngine(
            initial_capital=1000.0, commission_pct=0.01, slippage_pct=0.0
        ).run(df, signals)
        assert result.total_trades == 0
        assert result.total_return_pct == 0.0

    def test_gap_through_stop_fills_at_open(self):
        """A candle gapping below the stop must fill at the open — the
        gapped run loses more than the non-gapped run."""
        base = [100.0] * 6
        gapped = np.array(base + [90.0] * 6)      # opens far below the stop
        touched = np.array(base + [96.9] * 6)     # touches stop, no gap

        results = {}
        for name, prices in (("gap", gapped), ("touch", touched)):
            df = engine_df(prices)
            df.loc[df.index[6:], "low"] = prices[6:] * 0.99
            signals = pd.Series(0, index=df.index)
            signals.iloc[1] = 1  # entry ~100; ATR 2 → stop ≈ 97
            results[name] = BacktestEngine(
                initial_capital=1000.0, commission_pct=0, slippage_pct=0
            ).run(df, signals)

        assert results["gap"].total_trades == 1
        assert results["gap"].total_return_pct < results["touch"].total_return_pct

    def test_check_position_exits_gap_fills(self):
        rm = RiskManager(initial_balance=10_000.0)
        long_trade = {"side": "buy", "stop_loss": 97.0, "take_profit": 106.0}
        # gap below stop → fill at open
        assert rm.check_position_exits(
            long_trade, candle_high=95.0, candle_low=90.0, candle_open=91.0
        ) == ("stop_loss", 91.0)
        # no gap → fill at stop
        assert rm.check_position_exits(
            long_trade, candle_high=99.0, candle_low=96.5, candle_open=99.0
        ) == ("stop_loss", 97.0)
        # gap above TP → fill at (better) open
        assert rm.check_position_exits(
            long_trade, candle_high=112.0, candle_low=107.0, candle_open=108.0
        ) == ("take_profit", 108.0)
        # short: gap above stop → fill at open
        short_trade = {"side": "sell", "stop_loss": 103.0, "take_profit": 94.0}
        assert rm.check_position_exits(
            short_trade, candle_high=110.0, candle_low=104.0, candle_open=105.0
        ) == ("stop_loss", 105.0)


# ─── Walk-forward runner uses the parity path ──────────────────────────────────

class TestWalkForwardRunner:
    def test_runner_replays_through_decide(self):
        src = (PROJECT_ROOT / "scripts" / "walk_forward.py").read_text()
        assert "generate_signals_via_decide" in src
        assert "predict_series" not in src  # hand-replicated HMM filter gone
        assert 'timeframe="1d"' in src and 'timeframe="4h"' in src
