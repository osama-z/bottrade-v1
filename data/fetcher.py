"""
Data Fetcher — fetches OHLCV (candlestick) data and ticker info from Binance.
Supports both historical data fetching and live data polling.
Uses ccxt for exchange connectivity.
"""

import time
from datetime import datetime, timezone
from typing import Optional

import ccxt
import pandas as pd
from loguru import logger

from config.settings import settings
from config.constants import TIMEFRAME_SECONDS
from core.exceptions import DataFetchError, InsufficientDataError
from data.cache import cache

# Binance's spot testnet. ccxt's own describe() lists the same host under
# urls["test"], but we set it explicitly so the public/private split below
# is readable at the call site.
BINANCE_TESTNET_REST = "https://testnet.binance.vision/api"


class DataFetcher:
    """
    Fetches market data from Binance using ccxt.

    Supports:
    - Historical OHLCV data (candles)
    - Real-time ticker data
    - Order book snapshots
    - Account balance (for live trading)
    """

    def __init__(self, public_only: bool = False) -> None:
        # public_only: build the market-data exchange WITHOUT credentials.
        # Market data (candles/tickers/funding/OI) is public and needs no key;
        # this is required when the configured key belongs to a DIFFERENT venue
        # than the market-data venue (e.g. a TESTNET order key while candles
        # come from PRODUCTION — production rejects a testnet key even on
        # public calls). Order placement lives in LiveExecutor, not here.
        self._public_only = public_only
        self._exchange = self._create_exchange()
        logger.info(
            "DataFetcher initialized — market data: {} | account endpoints: {}{}",
            "testnet" if settings.market_data_testnet else "PRODUCTION",
            "testnet" if settings.binance_testnet else "PRODUCTION",
            " | PUBLIC-ONLY (no key)" if public_only else "",
        )

    def _create_exchange(self) -> ccxt.binance:
        """Create and configure ccxt Binance exchange instance.

        Public and private endpoints are routed independently:

        * **Market data (public)** follows ``MARKET_DATA_TESTNET``, default
          false — i.e. PRODUCTION even when ``BINANCE_TESTNET=true``.
          Testnet's public history is only a few weeks deep and its prices
          are synthetic, so training or paper trading on it fits noise
          (measured: an ETH model trained on testnet scored 49.7%
          accuracy / 0.519 AUC — a coin flip). Market data is free and
          unauthenticated, so there is no cost or risk to taking it from
          production. Flip the flag only to exercise the testnet feed
          itself.
        * **Account endpoints (private)** follow ``BINANCE_TESTNET``,
          default true. These must stay independent: pointing order and
          balance calls at production is the one mistake that costs real
          money, and it must never be a side effect of wanting good
          candles.

        ccxt deep-merges ``urls``, so overriding ``public``/``private``
        leaves the other venue keys (sapi, fapi, …) intact.
        """
        config = {
            "enableRateLimit": True,     # Respect rate limits automatically
            "options": {
                "defaultType": "spot",   # Spot trading (no leverage)
            },
        }
        # Only attach credentials when not in public-only mode. Public market
        # data needs none; omitting the key avoids sending a wrong-venue key.
        if not self._public_only:
            config["apiKey"] = settings.binance_api_key
            config["secret"] = settings.binance_api_secret

        # NB: ccxt's binance never reads options["testnet"] — the URL
        # override below is the only thing that actually switches venue.
        api_urls: dict[str, str] = {}
        if settings.market_data_testnet:
            api_urls["public"] = BINANCE_TESTNET_REST
        if settings.binance_testnet:
            api_urls["private"] = BINANCE_TESTNET_REST
        if api_urls:
            config["urls"] = {"api": api_urls}

        exchange = ccxt.binance(config)
        return exchange

    def fetch_ohlcv(
        self,
        pair: str,
        timeframe: str = "1h",
        limit: int = 500,
        since: Optional[datetime] = None,
        use_cache: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candlestick data.

        Args:
            pair: Trading pair (e.g., "BTC/USDT")
            timeframe: Candle size (e.g., "1h", "15m", "1d")
            limit: Number of candles to fetch (max 1000 for Binance)
            since: Fetch candles from this datetime onwards

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume

        Raises:
            DataFetchError: If exchange request fails
            InsufficientDataError: If fewer candles than expected are returned
        """
        try:
            since_ms = int(since.timestamp() * 1000) if since else None

            # Opt-in TTL cache for REPEATED fetches — e.g. slow-changing
            # higher-timeframe context (1d/4h) reused across cycles by a faster
            # decision loop. Only for non-`since` requests. The live decision
            # path keeps use_cache=False so its candle is always fresh.
            cache_key = None
            if use_cache and since is None:
                cache_key = cache.make_key(pair, timeframe, f"ohlcv:{limit}")
                cached = cache.get(cache_key)
                if cached is not None:
                    return cached

            logger.debug(
                "Fetching {} candles for {} on {}",
                limit, pair, timeframe
            )

            raw = self._exchange.fetch_ohlcv(
                symbol=pair,
                timeframe=timeframe,
                limit=limit,
                since=since_ms,
            )

            if not raw:
                raise InsufficientDataError(
                    f"No data returned for {pair} on {timeframe}"
                )

            df = pd.DataFrame(
                raw,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

            # Convert timestamp from milliseconds to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)

            # Ensure correct dtypes
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            logger.debug(
                "Fetched {} candles for {} | range: {} → {}",
                len(df), pair,
                df.index[0].strftime("%Y-%m-%d %H:%M"),
                df.index[-1].strftime("%Y-%m-%d %H:%M"),
            )

            # Store only successful, non-empty frames (never cache an error).
            if cache_key is not None:
                cache.set(cache_key, df, ttl=cache.ttl_for_timeframe(timeframe))

            return df

        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise DataFetchError(f"Failed to fetch OHLCV for {pair}: {e}") from e

    def fetch_ticker(self, pair: str) -> dict:
        """
        Fetch current ticker data (latest price, 24h stats).

        Returns:
            Dict with: symbol, last, bid, ask, high, low, volume, change, percentage
        """
        try:
            ticker = self._exchange.fetch_ticker(pair)
            logger.debug("Ticker {}: ${:.2f}", pair, ticker["last"])
            return {
                "symbol": ticker["symbol"],
                "price": ticker["last"],
                "bid": ticker["bid"],
                "ask": ticker["ask"],
                "high_24h": ticker["high"],
                "low_24h": ticker["low"],
                "volume_24h": ticker["quoteVolume"],
                "change_24h": ticker["change"],
                "change_pct_24h": ticker["percentage"],
                "timestamp": datetime.now(timezone.utc),
            }
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise DataFetchError(f"Failed to fetch ticker for {pair}: {e}") from e

    def fetch_multiple_tickers(self, pairs: list[str]) -> dict[str, dict]:
        """Fetch tickers for multiple pairs at once."""
        try:
            tickers = self._exchange.fetch_tickers(pairs)
            result = {}
            for pair, ticker in tickers.items():
                result[pair] = {
                    "symbol": ticker["symbol"],
                    "price": ticker["last"],
                    "volume_24h": ticker.get("quoteVolume", 0),
                    "change_pct_24h": ticker.get("percentage", 0),
                    "timestamp": datetime.now(timezone.utc),
                }
            return result
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise DataFetchError(f"Failed to fetch tickers: {e}") from e

    def fetch_balance(self) -> dict:
        """
        Fetch account balance.

        Returns:
            Dict with currency balances: {currency: {free, used, total}}
        """
        try:
            balance = self._exchange.fetch_balance()
            # Return only non-zero balances
            result = {
                currency: {
                    "free": info["free"],
                    "used": info["used"],
                    "total": info["total"],
                }
                for currency, info in balance["total"].items()
                if isinstance(info, (int, float)) and info > 0
            }
            return result
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise DataFetchError(f"Failed to fetch balance: {e}") from e

    def fetch_order_book(self, pair: str, depth: int = 10) -> dict:
        """
        Fetch order book snapshot.

        Returns:
            Dict with 'bids' and 'asks' lists of [price, amount]
        """
        try:
            book = self._exchange.fetch_order_book(pair, limit=depth)
            return {
                "bids": book["bids"][:depth],
                "asks": book["asks"][:depth],
                "timestamp": datetime.now(timezone.utc),
                "spread": book["asks"][0][0] - book["bids"][0][0] if book["asks"] and book["bids"] else None,
            }
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise DataFetchError(f"Failed to fetch order book for {pair}: {e}") from e

    def fetch_order_book_raw(self, pair: str, depth: int = 1000) -> dict:
        """Raw ccxt order book INCLUDING ``nonce`` (Binance lastUpdateId).

        The WS depth-sync algorithm needs the exchange's update id to
        bracket buffered diff events; the formatted fetch_order_book()
        wrapper strips it.
        """
        try:
            return self._exchange.fetch_order_book(pair, limit=depth)
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise DataFetchError(f"Failed to fetch order book for {pair}: {e}") from e

    def fetch_funding_rate(self, pair: str) -> float:
        """
        Fetch perpetual futures funding rate for a pair.
        Translates spot symbol (e.g. BTC/USDT) to swap symbol (e.g. BTC/USDT:USDT).
        Returns the funding rate as a float (e.g., 0.0001 for 0.01%).
        Returns 0.0 if not available or on error.
        """
        try:
            if ":" not in pair:
                swap_symbol = f"{pair}:USDT"
            else:
                swap_symbol = pair

            rate_data = self._exchange.fetch_funding_rate(swap_symbol)
            funding_rate = float(rate_data.get("fundingRate", 0.0))
            logger.debug("Fetched funding rate for {}: {:.6f}%", swap_symbol, funding_rate * 100)
            return funding_rate
        except Exception as e:
            logger.warning("Could not fetch funding rate for {}: {} — defaulting to 0.0", pair, e)
            return 0.0

    def fetch_funding_rate_history(
        self,
        pair: str,
        limit: int = 500,
        since: Optional[int] = None,
    ) -> "pd.DataFrame":
        """Historical perpetual funding rates (one row per 8h settlement epoch).

        Feeds the funding-carry strategy's backtest (Roadmap Task 1.4). Returns a
        DataFrame with a single ``funding_rate`` column on a UTC DatetimeIndex.
        Returns an EMPTY frame on error — callers treat absence as absence.
        """
        swap_symbol = pair if ":" in pair else f"{pair}:USDT"
        try:
            raw = self._exchange.fetch_funding_rate_history(
                swap_symbol, since=since, limit=limit
            )
            if not raw:
                return pd.DataFrame({"funding_rate": []},
                                    index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))
            df = pd.DataFrame(
                {"funding_rate": [float(r.get("fundingRate", 0.0)) for r in raw]},
                index=pd.to_datetime([r.get("timestamp") for r in raw], unit="ms", utc=True),
            )
            df.index.name = "timestamp"
            logger.debug("Fetched {} funding epochs for {}", len(df), swap_symbol)
            return df
        except Exception as e:
            logger.warning("Could not fetch funding history for {}: {} — returning empty", pair, e)
            return pd.DataFrame({"funding_rate": []},
                                index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))

    # ── Market structure (futures PUBLIC data — no key/permission needed) ─────
    # Binance only serves ~30 days of history for these series, so they
    # cannot be backtested from scratch: the bot RECORDS them each cycle
    # (storage.market_structure) and the accumulated series becomes
    # testable feature data after the paper run. Roadmap Step 3.

    def fetch_open_interest(self, pair: str) -> Optional[float]:
        """Current open interest (base units) for the pair's USDT-perp.
        None on failure — callers must treat absence as absence."""
        try:
            swap = pair if ":" in pair else f"{pair}:USDT"
            data = self._exchange.fetch_open_interest(swap)
            value = data.get("openInterestAmount") or data.get("openInterestValue")
            return float(value) if value is not None else None
        except Exception as e:
            logger.warning("Could not fetch open interest for {}: {}", pair, e)
            return None

    def fetch_taker_ratio(self, pair: str, period: str = "1h") -> Optional[float]:
        """Taker buy/sell volume ratio (>1 = aggressive buying dominates).
        Uses the public futures data endpoint; None on failure."""
        try:
            symbol = pair.replace("/", "").split(":")[0]
            # ccxt implicit method name is case-sensitive: the correct one is
            # ...Ratio (capital R). The lowercase form silently AttributeError'd,
            # so taker_ratio was recorded as NULL every cycle.
            rows = self._exchange.fapiDataGetTakerlongshortRatio({
                "symbol": symbol, "period": period, "limit": 1,
            })
            if rows:
                return float(rows[-1]["buySellRatio"])
            return None
        except Exception as e:
            logger.warning("Could not fetch taker ratio for {}: {}", pair, e)
            return None

    def fetch_order_book_imbalance(self, pair: str, percentage: float = 0.01) -> float:
        """
        Calculate the order book imbalance within a specified percentage range of the mid price.

        Formula:
            Ask Volume = Sum(ask_qty) for price <= mid * (1 + percentage)
            Bid Volume = Sum(bid_qty) for price >= mid * (1 - percentage)
            Ratio = Ask Volume / Bid Volume (if Bid Volume > 0, else float('inf'))

        Returns:
            imbalance_ratio (float) — Ask Volume / Bid Volume
        """
        try:
            # Fetch a deeper book (e.g., limit=100) to ensure we cover the 1% range
            book = self._exchange.fetch_order_book(pair, limit=100)
            bids = book.get("bids", [])
            asks = book.get("asks", [])

            if not bids or not asks:
                logger.warning("Empty order book for {} — imbalance defaulted to 1.0", pair)
                return 1.0

            best_bid = bids[0][0]
            best_ask = asks[0][0]
            mid_price = (best_bid + best_ask) / 2.0

            lower_limit = mid_price * (1.0 - percentage)
            upper_limit = mid_price * (1.0 + percentage)

            bid_volume = sum(qty for price, qty in bids if price >= lower_limit)
            ask_volume = sum(qty for price, qty in asks if price <= upper_limit)

            if bid_volume == 0:
                logger.warning("Bid volume is 0 within 1% price band for {} — imbalance ratio set to infinity", pair)
                return float("inf")

            imbalance_ratio = ask_volume / bid_volume
            logger.debug(
                "Order book imbalance for {} (1% band): Ask Vol={:.2f}, Bid Vol={:.2f}, Ratio={:.2f}",
                pair, ask_volume, bid_volume, imbalance_ratio
            )
            return imbalance_ratio
        except Exception as e:
            logger.warning("Could not calculate order book imbalance for {}: {} — defaulting to 1.0", pair, e)
            return 1.0

    def get_historical_data(
        self,
        pair: str,
        timeframe: str = "1h",
        days: int = 365,
    ) -> pd.DataFrame:
        """
        Fetch a large amount of historical data by paginating through the API.

        Args:
            pair: Trading pair
            timeframe: Candle size
            days: Number of days of history to fetch

        Returns:
            DataFrame with full historical OHLCV data
        """
        timeframe_secs = TIMEFRAME_SECONDS.get(timeframe, 3600)
        total_candles_needed = (days * 24 * 3600) // timeframe_secs
        batch_size = 1000  # Binance max per request

        logger.info(
            "Fetching {} days of {} history for {} (~{} candles)",
            days, timeframe, pair, total_candles_needed
        )

        all_data: list[pd.DataFrame] = []
        since = datetime.now(timezone.utc).timestamp() * 1000 - (days * 86400 * 1000)

        # Bounded pagination: termination previously relied entirely on the
        # exchange returning a short batch — a server echoing full batches
        # (or a `since` arithmetic bug) looped forever.
        max_batches = (days * 1440) // max(batch_size, 1) + 10
        batches_done = 0

        while batches_done < max_batches:
            batches_done += 1
            batch = self._exchange.fetch_ohlcv(
                symbol=pair,
                timeframe=timeframe,
                since=int(since),
                limit=batch_size,
            )

            if not batch:
                break

            df = pd.DataFrame(
                batch,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)

            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            all_data.append(df)

            # Move 'since' to the last candle's timestamp
            since = batch[-1][0] + 1

            # Stop if we got fewer candles than requested (end of data)
            if len(batch) < batch_size:
                break

            # Respect rate limits
            time.sleep(self._exchange.rateLimit / 1000)

        if not all_data:
            raise InsufficientDataError(f"No historical data for {pair}")

        result = pd.concat(all_data)
        result = result[~result.index.duplicated(keep="last")]  # Remove duplicates
        result.sort_index(inplace=True)

        logger.info(
            "Fetched {} candles for {} | {} → {}",
            len(result), pair,
            result.index[0].strftime("%Y-%m-%d"),
            result.index[-1].strftime("%Y-%m-%d"),
        )

        return result
