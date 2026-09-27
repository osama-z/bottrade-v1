"""Unit tests for DataFetcher parsing/mapping, with a fake ccxt exchange."""
import ccxt
import pandas as pd
import pytest

from data.fetcher import DataFetcher
from core.exceptions import DataFetchError, InsufficientDataError


def _fetcher(exchange):
    """A DataFetcher wired to a fake exchange (no network, no __init__ ccxt)."""
    f = DataFetcher.__new__(DataFetcher)
    f._exchange = exchange
    f._public_only = True
    return f


class FakeEx:
    def __init__(self, ohlcv=None, ticker=None, raise_net=False):
        self._ohlcv = ohlcv
        self._ticker = ticker
        self._raise_net = raise_net

    def fetch_ohlcv(self, symbol, timeframe, limit, since=None):
        if self._raise_net:
            raise ccxt.NetworkError("down")
        return self._ohlcv

    def fetch_ticker(self, symbol):
        return self._ticker

    def fetch_funding_rate(self, symbol):
        return {"fundingRate": 0.0001}


class TestFetchOHLCV:
    def test_parses_raw_to_dataframe(self):
        raw = [[1704067200000, 100.0, 110.0, 95.0, 105.0, 1000.0],
               [1704070800000, 105.0, 112.0, 104.0, 108.0, 1200.0]]
        df = _fetcher(FakeEx(ohlcv=raw)).fetch_ohlcv("BTC/USDT", "1h", 2)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert str(df.index.tz) == "UTC"
        assert df["close"].iloc[-1] == 108.0
        assert df["open"].dtype == float

    def test_empty_raises_insufficient(self):
        with pytest.raises(InsufficientDataError):
            _fetcher(FakeEx(ohlcv=[])).fetch_ohlcv("BTC/USDT", "1h", 2)

    def test_network_error_wrapped_as_datafetcherror(self):
        with pytest.raises(DataFetchError):
            _fetcher(FakeEx(raise_net=True)).fetch_ohlcv("BTC/USDT", "1h", 2)


class TestFetchTicker:
    def test_maps_ccxt_ticker(self):
        t = {"symbol": "BTC/USDT", "last": 64000.0, "bid": 63999.0, "ask": 64001.0,
             "high": 65000.0, "low": 63000.0, "quoteVolume": 1e9, "change": 500.0,
             "percentage": 0.8}
        out = _fetcher(FakeEx(ticker=t)).fetch_ticker("BTC/USDT")
        assert out["price"] == 64000.0
        assert out["bid"] == 63999.0 and out["ask"] == 64001.0
        assert out["volume_24h"] == 1e9


class TestFundingRate:
    def test_returns_float(self):
        assert _fetcher(FakeEx()).fetch_funding_rate("BTC/USDT") == pytest.approx(0.0001)

    def test_defaults_to_zero_on_error(self):
        class Boom:
            def fetch_funding_rate(self, s): raise RuntimeError("nope")
        assert _fetcher(Boom()).fetch_funding_rate("BTC/USDT") == 0.0


class CountingEx:
    """Counts fetch_ohlcv calls, to prove cache hits skip the exchange."""
    def __init__(self, raw):
        self.raw = raw
        self.calls = 0

    def fetch_ohlcv(self, symbol, timeframe, limit, since=None):
        self.calls += 1
        return self.raw


_RAW = [[1704067200000, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1704070800000, 1.0, 1.0, 1.0, 1.0, 1.0]]


class TestOHLCVCache:
    @pytest.fixture(autouse=True)
    def _isolate_cache(self):
        from data.cache import cache
        cache.clear()
        yield
        cache.clear()

    def test_use_cache_serves_second_fetch_from_cache(self):
        ex = CountingEx(_RAW)
        f = _fetcher(ex)
        a = f.fetch_ohlcv("BTC/USDT", "4h", 100, use_cache=True)
        b = f.fetch_ohlcv("BTC/USDT", "4h", 100, use_cache=True)
        assert ex.calls == 1                 # second call served from cache
        assert a.equals(b)

    def test_default_never_caches(self):
        ex = CountingEx(_RAW)
        f = _fetcher(ex)
        f.fetch_ohlcv("BTC/USDT", "4h", 100)   # default use_cache=False
        f.fetch_ohlcv("BTC/USDT", "4h", 100)
        assert ex.calls == 2                 # decision path always fresh

    def test_different_limit_is_a_different_key(self):
        ex = CountingEx(_RAW)
        f = _fetcher(ex)
        f.fetch_ohlcv("BTC/USDT", "4h", 100, use_cache=True)
        f.fetch_ohlcv("BTC/USDT", "4h", 200, use_cache=True)
        assert ex.calls == 2

    def test_empty_result_is_not_cached(self):
        from data.cache import cache
        f = _fetcher(CountingEx([]))          # empty → InsufficientDataError
        with pytest.raises(InsufficientDataError):
            f.fetch_ohlcv("BTC/USDT", "4h", 100, use_cache=True)
        assert len(cache) == 0


class TestFundingRateHistory:
    class _HistEx:
        def fetch_funding_rate_history(self, symbol, since=None, limit=500):
            assert symbol == "BTC/USDT:USDT"      # spot symbol → swap symbol
            return [
                {"timestamp": 1704067200000, "fundingRate": 0.0004},
                {"timestamp": 1704096000000, "fundingRate": -0.0001},
            ]

    def test_parses_history_to_dataframe(self):
        df = _fetcher(self._HistEx()).fetch_funding_rate_history("BTC/USDT")
        assert list(df.columns) == ["funding_rate"]
        assert str(df.index.tz) == "UTC"
        assert df["funding_rate"].tolist() == [0.0004, -0.0001]

    def test_empty_history_returns_empty_frame(self):
        class Empty:
            def fetch_funding_rate_history(self, s, since=None, limit=500): return []
        df = _fetcher(Empty()).fetch_funding_rate_history("BTC/USDT")
        assert df.empty and list(df.columns) == ["funding_rate"]

    def test_error_returns_empty_frame(self):
        class Boom:
            def fetch_funding_rate_history(self, s, since=None, limit=500):
                raise RuntimeError("down")
        assert _fetcher(Boom()).fetch_funding_rate_history("BTC/USDT").empty
