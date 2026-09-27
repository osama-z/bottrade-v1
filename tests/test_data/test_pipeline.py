"""Unit tests for the shared data pipeline (data/pipeline.build_indicator_frame)."""
import numpy as np
import pandas as pd

from data.pipeline import build_indicator_frame


class FakeFetcher:
    """Returns a canned frame; records the fetch args it was called with."""
    def __init__(self, df):
        self.df = df
        self.calls = []

    def fetch_ohlcv(self, pair, timeframe, limit):
        self.calls.append((pair, timeframe, limit))
        return self.df


def test_returns_indicator_frame(ohlcv_300):
    f = FakeFetcher(ohlcv_300)
    out = build_indicator_frame(f, "BTC/USDT", "1h", limit=300, drop_forming=False)
    assert out is not None
    # preprocessing + indicators ran → key indicator columns present
    for col in ("RSI", "ATR", "Supertrend_dir", "EMA_50", "ADX"):
        assert col in out.columns
    assert len(out) > 0
    assert f.calls == [("BTC/USDT", "1h", 300)]   # fetch args forwarded


def test_none_fetch_returns_none():
    assert build_indicator_frame(FakeFetcher(None), "BTC/USDT", "1h") is None


def test_too_few_rows_returns_none(ohlcv_300):
    # 50 rows with min_rows=100 → None, before indicators are even computed.
    f = FakeFetcher(ohlcv_300.head(50))
    assert build_indicator_frame(f, "BTC/USDT", "1h", drop_forming=False,
                                 min_rows=100) is None


def test_drop_forming_removes_still_forming_last_candle():
    # Last 4h candle is still open (opened 1h ago). Varying prices so the
    # indicators aren't all-NaN (which would empty the frame via dropna).
    now = pd.Timestamp.now(tz="UTC")
    idx = pd.date_range(end=now - pd.Timedelta(hours=1), periods=120, freq="4h")
    close = np.abs(np.cumsum(np.random.default_rng(0).normal(0, 1, 120))) + 50
    df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                       "close": close, "volume": 1000.0}, index=idx)
    f = FakeFetcher(df)
    kept = build_indicator_frame(f, "BTC/USDT", "4h", drop_forming=False, min_rows=1)
    dropped = build_indicator_frame(f, "BTC/USDT", "4h", drop_forming=True, min_rows=1)
    assert kept.index[-1] not in dropped.index          # forming candle excluded
    assert len(dropped) == len(kept) - 1
