"""Focused tests for AICombinedStrategy.decide() — the live trading brain.

decide() turns an AI signal into a TradeSignal through a stack of filter
gates (macro trend, micro timing, volume, funding, order-book imbalance).
These were only covered indirectly by the golden-file parity test. Here we
drive a *controlled actionable BUY* through decide() and assert each gate
independently converts it to HOLD — so a regression that silently opens the
gates (or jams one shut) is caught.

The AI stack is bypassed by stubbing the combiner: decide() still runs the
real indicator/sentiment/LLM calls, but the decision it acts on is fixed,
isolating the gate logic under test.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from ai.signal_combiner import AISignal
from config.constants import Signal
from data.preprocessor import DataPreprocessor
from indicators.technical import TechnicalIndicators
from strategies.ai_combined import AICombinedStrategy
from strategies.context import MarketContext


def _computed(trend: str, n: int, freq: str) -> pd.DataFrame:
    rng = np.random.RandomState(7)
    dates = pd.date_range(datetime(2024, 1, 1, tzinfo=timezone.utc), periods=n, freq=freq)
    base = np.linspace(100, 160, n) if trend == "up" else np.linspace(160, 100, n)
    px = base * (1 + rng.normal(0, 0.002, n))
    df = pd.DataFrame(
        {"open": px * 0.999, "high": px * 1.006, "low": px * 0.994,
         "close": px, "volume": rng.uniform(80, 120, n)},
        index=dates,
    )
    return TechnicalIndicators().compute_all(DataPreprocessor().process(df))


@pytest.fixture(scope="module")
def frames():
    return {
        "h1": _computed("up", 300, "1h"),
        "d1": _computed("up", 120, "1D"),
        "h4": _computed("up", 200, "4h"),
    }


def _actionable_buy(**_):
    return AISignal(signal=Signal.BUY, score=0.9, confidence=0.9, is_actionable=True,
                    llm_score=0.0, ml_score=0.0, sentiment_score=0.0)


def _strategy():
    strat = AICombinedStrategy(pair="BTC/USDT", use_llm=False)
    strat._model_loaded = False
    strat._regime_loaded = False
    strat._combiner.combine = _actionable_buy   # fix the decision; test the gates
    return strat


def _macro(df: pd.DataFrame) -> pd.DataFrame:
    """Force the last row to satisfy the macro-trend gate (close>EMA_50, up)."""
    d = df.copy()
    d.loc[d.index[-1], "Supertrend_dir"] = 1
    d.loc[d.index[-1], "EMA_50"] = d["close"].iloc[-1] * 0.9
    return d


def _passing_ctx(frames) -> MarketContext:
    """A context in which a BUY clears every gate."""
    df = frames["h1"].copy()
    df.loc[df.index[-1], "RSI"] = 25.0                                  # oversold → timing ok
    df.loc[df.index[-1], "volume"] = float(df["volume"].iloc[:-1].mean() * 20)  # high vol
    return MarketContext(
        df=df,
        df_1d=_macro(frames["d1"]),
        df_4h=_macro(frames["h4"]),
        funding_rate=None,
        imbalance=None,
    )


class TestDecideGates:
    def test_all_filters_pass_yields_buy(self, frames):
        sig = _strategy().decide(_passing_ctx(frames), "BTC/USDT", verbose=False)
        assert sig.signal == Signal.BUY

    def test_missing_macro_frames_block_buy(self, frames):
        ctx = _passing_ctx(frames)
        ctx = MarketContext(df=ctx.df, df_1d=None, df_4h=None,
                            funding_rate=None, imbalance=None)
        sig = _strategy().decide(ctx, "BTC/USDT", verbose=False)
        assert sig.signal == Signal.HOLD and "Macro Trend" in sig.reason

    def test_low_volume_blocks_buy(self, frames):
        ctx = _passing_ctx(frames)
        ctx.df.loc[ctx.df.index[-1], "volume"] = 0.001   # far below SMA
        sig = _strategy().decide(ctx, "BTC/USDT", verbose=False)
        assert sig.signal == Signal.HOLD and "Volume" in sig.reason

    def test_high_funding_blocks_buy(self, frames):
        ctx = _passing_ctx(frames)
        ctx = MarketContext(df=ctx.df, df_1d=ctx.df_1d, df_4h=ctx.df_4h,
                            funding_rate=0.001, imbalance=None)   # > 0.0005
        sig = _strategy().decide(ctx, "BTC/USDT", verbose=False)
        assert sig.signal == Signal.HOLD and "Funding" in sig.reason

    def test_high_imbalance_blocks_buy(self, frames):
        ctx = _passing_ctx(frames)
        ctx = MarketContext(df=ctx.df, df_1d=ctx.df_1d, df_4h=ctx.df_4h,
                            funding_rate=None, imbalance=2.0)      # > 1.5
        sig = _strategy().decide(ctx, "BTC/USDT", verbose=False)
        assert sig.signal == Signal.HOLD and "imbalance" in sig.reason

    def test_bad_timing_blocks_buy(self, frames):
        ctx = _passing_ctx(frames)
        last = ctx.df.index[-1]
        ctx.df.loc[last, "RSI"] = 70.0            # not oversold
        ctx.df.loc[last, "MACD"] = -5.0           # below signal → no bullish cross
        ctx.df.loc[last, "MACD_signal"] = 5.0
        sig = _strategy().decide(ctx, "BTC/USDT", verbose=False)
        assert sig.signal == Signal.HOLD and "Micro Timing" in sig.reason

    def test_volume_reason_quotes_the_live_threshold(self, frames):
        # The volume-fail message must reflect settings.buy_volume_multiple,
        # not a hardcoded "1.5x" literal that goes stale when the config changes.
        from config.settings import settings
        ctx = _passing_ctx(frames)
        ctx.df.loc[ctx.df.index[-1], "volume"] = 0.001
        sig = _strategy().decide(ctx, "BTC/USDT", verbose=False)
        assert f"{settings.buy_volume_multiple:.1f}x" in sig.reason
