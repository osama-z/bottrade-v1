"""Tier-13 tests: market-data venue split and training-history pagination.

Two bugs motivated this tier:

1. Every public endpoint followed ``BINANCE_TESTNET``, so the whole
   research stack ran on testnet candles — a few weeks of synthetic
   prices. A model trained on them scored 49.7% accuracy / 0.519 AUC.
   Market data is free and unauthenticated, so it now comes from
   production while account endpoints stay on testnet.
2. ``scripts/train_model.py`` called the single-request ``fetch_ohlcv``
   with ``limit=days * 24``. Binance caps a request at 1000 candles, so
   ``--days 365`` silently trained on ~41 days.
"""

from pathlib import Path

import pytest

from config.settings import settings as real_settings
from data.fetcher import BINANCE_TESTNET_REST, DataFetcher

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROD_HOST = "api.binance.com"


def _settings(*, market_data_testnet: bool, binance_testnet: bool):
    """Settings is frozen, so build a copy rather than mutating."""
    return real_settings.model_copy(
        update={
            "market_data_testnet": market_data_testnet,
            "binance_testnet": binance_testnet,
        }
    )


@pytest.fixture
def venue(monkeypatch):
    """Return a factory that builds an exchange under the given flags."""

    def _build(*, market_data_testnet: bool, binance_testnet: bool):
        fake = _settings(
            market_data_testnet=market_data_testnet,
            binance_testnet=binance_testnet,
        )
        monkeypatch.setattr("data.fetcher.settings", fake)
        return DataFetcher()._exchange.urls["api"]

    return _build


class TestVenueSplit:
    def test_defaults_are_production_data_and_testnet_account(self):
        """The shipped defaults must be safe: real candles, fake money."""
        assert real_settings.market_data_testnet is False
        assert real_settings.binance_testnet is True

    def test_market_data_from_production_while_account_on_testnet(self, venue):
        api = venue(market_data_testnet=False, binance_testnet=True)
        assert PROD_HOST in api["public"]
        assert api["private"] == BINANCE_TESTNET_REST

    def test_market_data_flag_switches_only_public(self, venue):
        api = venue(market_data_testnet=True, binance_testnet=True)
        assert api["public"] == BINANCE_TESTNET_REST
        assert api["private"] == BINANCE_TESTNET_REST

    def test_binance_testnet_false_puts_account_on_production(self, venue):
        api = venue(market_data_testnet=False, binance_testnet=False)
        assert PROD_HOST in api["public"]
        assert PROD_HOST in api["private"]

    def test_account_venue_is_independent_of_data_venue(self, venue):
        """Regression: wanting production candles must never be able to
        drag order/balance calls onto production as a side effect."""
        api = venue(market_data_testnet=True, binance_testnet=True)
        assert api["private"] == BINANCE_TESTNET_REST

    def test_url_override_preserves_other_venue_keys(self, venue):
        """ccxt deep-merges urls; a shallow replace would drop fapi/sapi
        and break funding-rate lookups."""
        api = venue(market_data_testnet=False, binance_testnet=True)
        for key in ("sapi", "fapiPublic", "dapiPublic"):
            assert key in api, f"{key} dropped by the urls override"


class TestNoInertConfig:
    def test_ccxt_binance_ignores_options_testnet(self):
        """The old code set options["testnet"]=True and believed it did
        something. ccxt's binance never reads that key — only the URL
        override switches venue. Pin the assumption so a future revert to
        the "simpler" option-based form fails loudly."""
        import ccxt

        assert "testnet" not in ccxt.binance().describe()["options"]

    def test_fetcher_does_not_set_options_testnet(self, monkeypatch):
        monkeypatch.setattr(
            "data.fetcher.settings",
            _settings(market_data_testnet=True, binance_testnet=True),
        )
        assert "testnet" not in DataFetcher()._exchange.options


class TestWebSocketFollowsMarketData:
    def test_ws_base_follows_market_data_flag(self, monkeypatch):
        """The WS stream and the REST snapshot must come from the same
        venue — their update-ID spaces are disjoint, so a mismatch means
        the book can never sync."""
        import sys

        from data.ws_depth_feed import (
            BINANCE_TESTNET_WS_BASE,
            BINANCE_WS_BASE,
            _ws_base,
        )

        # _ws_base imports settings inside the function body, so patch the
        # module attribute it reads. Reach for it via sys.modules:
        # config/__init__ does `from config.settings import settings`,
        # which rebinds the package's `settings` attribute to the instance
        # and shadows the submodule, so `import config.settings as m`
        # hands back the Settings object rather than the module.
        settings_module = sys.modules["config.settings"]
        monkeypatch.setattr(
            settings_module,
            "settings",
            _settings(market_data_testnet=True, binance_testnet=True),
        )
        assert _ws_base() == BINANCE_TESTNET_WS_BASE

        monkeypatch.setattr(
            settings_module,
            "settings",
            _settings(market_data_testnet=False, binance_testnet=True),
        )
        assert _ws_base() == BINANCE_WS_BASE, (
            "WS must follow market_data_testnet, not binance_testnet"
        )


class TestTrainingHistoryPagination:
    def test_train_model_uses_the_paginating_fetch(self):
        """fetch_ohlcv is one request, capped at 1000 candles by the
        exchange — --days above ~41 was silently truncated."""
        src = (PROJECT_ROOT / "scripts" / "train_model.py").read_text()
        assert "fetcher.get_historical_data(" in src
        assert "fetcher.fetch_ohlcv(" not in src
        assert "days=args.days" in src

    def test_get_historical_data_pages_past_the_batch_cap(self):
        """A 1000-candle cap with 365 days requested must issue multiple
        requests and return more than one batch worth of candles."""
        calls: list[int] = []
        base_ms = 1_700_000_000_000
        hour_ms = 3_600_000

        def fake_fetch_ohlcv(symbol, timeframe, since, limit):
            calls.append(since)
            if len(calls) > 3:
                return []
            start = since if since > base_ms else base_ms
            return [
                [start + i * hour_ms, 1.0, 2.0, 0.5, 1.5, 10.0]
                for i in range(limit)
            ]

        fetcher = DataFetcher.__new__(DataFetcher)
        fetcher._exchange = type(
            "FakeExchange", (), {"fetch_ohlcv": staticmethod(fake_fetch_ohlcv), "rateLimit": 0}
        )()

        df = fetcher.get_historical_data(pair="ETH/USDT", timeframe="1h", days=365)

        assert len(calls) > 1, "single request — pagination did not happen"
        assert len(df) > 1000, f"only {len(df)} candles; batch cap not exceeded"
