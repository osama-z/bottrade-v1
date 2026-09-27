"""
Tests for backtesting strategies.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from strategies.ma_crossover import MACrossoverStrategy
from strategies.rsi_reversal import RSIReversalStrategy
from strategies.bollinger_bounce import BollingerBounceStrategy
from strategies.registry import get_strategy, list_strategies
from strategies.base import TradeSignal
from config.constants import Signal


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """300 rows of synthetic price data."""
    np.random.seed(99)
    n = 300
    dates = pd.date_range(
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        periods=n, freq="1h"
    )
    price = 50000.0
    prices = [price]
    for _ in range(n - 1):
        price *= (1 + np.random.normal(0, 0.006))
        prices.append(price)

    prices = np.array(prices)
    return pd.DataFrame({
        "open": prices * 0.999,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": np.random.uniform(50, 500, n),
    }, index=dates)


class TestMACrossover:
    def test_signals_are_valid_values(self, sample_ohlcv):
        strategy = MACrossoverStrategy()
        signals = strategy.generate_signals(sample_ohlcv)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_signals_length_matches_df(self, sample_ohlcv):
        strategy = MACrossoverStrategy()
        signals = strategy.generate_signals(sample_ohlcv)
        assert len(signals) == len(sample_ohlcv)

    def test_get_signal_returns_trade_signal(self, sample_ohlcv):
        strategy = MACrossoverStrategy()
        result = strategy.get_signal(sample_ohlcv, "BTC/USDT")
        assert isinstance(result, TradeSignal)
        assert result.signal in list(Signal)
        assert 0.0 <= result.confidence <= 1.0

    def test_params_structure(self):
        strategy = MACrossoverStrategy(fast=10, slow=30)
        params = strategy.get_params()
        assert params["fast"] == 10
        assert params["slow"] == 30


class TestRSIReversal:
    def test_signals_are_valid(self, sample_ohlcv):
        strategy = RSIReversalStrategy()
        signals = strategy.generate_signals(sample_ohlcv)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_get_signal_returns_trade_signal(self, sample_ohlcv):
        strategy = RSIReversalStrategy()
        result = strategy.get_signal(sample_ohlcv, "ETH/USDT")
        assert isinstance(result, TradeSignal)


class TestRegistry:
    def test_list_strategies_not_empty(self):
        strategies = list_strategies()
        assert len(strategies) > 0

    def test_get_known_strategy(self):
        s = get_strategy("ma_crossover")
        assert isinstance(s, MACrossoverStrategy)

    def test_get_unknown_strategy_raises(self):
        from core.exceptions import StrategyNotFoundError
        with pytest.raises(StrategyNotFoundError):
            get_strategy("nonexistent_strategy")


class TestTradeSignalConfidenceContract:
    """TradeSignal must enforce confidence ∈ [0, 1] (some strategy formulas
    produce out-of-range values, e.g. a negative bollinger/RSI confidence)."""

    def test_negative_confidence_clamped_to_zero(self):
        s = TradeSignal(signal=Signal.SELL, confidence=-4.0, pair="X",
                        price=1.0, reason="r")
        assert s.confidence == 0.0

    def test_over_one_clamped_to_one(self):
        s = TradeSignal(signal=Signal.BUY, confidence=1.7, pair="X",
                        price=1.0, reason="r")
        assert s.confidence == 1.0

    def test_valid_confidence_unchanged(self):
        s = TradeSignal(signal=Signal.BUY, confidence=0.7, pair="X",
                        price=1.0, reason="r")
        assert s.confidence == 0.7


class TestConfidenceBoundsAcrossStrategies:
    """Every strategy's get_signal must honour the [0, 1] contract, even on
    raw data missing the precomputed BB/RSI columns (the bollinger SELL bug)."""

    @pytest.mark.parametrize("strat", [
        BollingerBounceStrategy(),
        RSIReversalStrategy(),
        MACrossoverStrategy(),
    ])
    def test_get_signal_confidence_in_range(self, strat, sample_ohlcv):
        r = strat.get_signal(sample_ohlcv, "BTC/USDT")
        assert 0.0 <= r.confidence <= 1.0


class TestBollingerBounce:
    def test_signals_are_valid_values(self, sample_ohlcv):
        s = BollingerBounceStrategy().generate_signals(sample_ohlcv)
        assert set(s.unique()).issubset({-1, 0, 1})

    def test_get_signal_returns_trade_signal(self, sample_ohlcv):
        r = BollingerBounceStrategy().get_signal(sample_ohlcv, "BTC/USDT")
        assert isinstance(r, TradeSignal)
        assert r.signal in list(Signal)

    def test_squeeze_filter_reduces_or_equals_signal_count(self, sample_ohlcv):
        with_f = BollingerBounceStrategy(use_squeeze_filter=True).generate_signals(sample_ohlcv)
        without = BollingerBounceStrategy(use_squeeze_filter=False).generate_signals(sample_ohlcv)
        assert (with_f != 0).sum() <= (without != 0).sum()
