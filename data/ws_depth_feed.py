"""ws_depth_feed.py — Binance WebSocket depth transport (Stage 2).

Thin transport around DepthStreamSynchronizer: connects to the diff
stream, feeds raw messages to the synchronizer, reconnects with backoff.
Runs its own asyncio loop in a daemon thread so the synchronous trading
loop can consume the book without an event loop of its own.

The strategy consumes it through ``BookImbalanceSource`` (duck-typed to
the same contract as the REST fallback): ``get_imbalance(pair)`` returns
None whenever the book is not synced/fresh — the decide() filter treats
None as "unavailable → pass through", never a fabricated neutral value.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Optional

import aiohttp
from loguru import logger

from data.depth_sync import DepthStreamSynchronizer
from data.order_book import OrderBookManager

BINANCE_WS_BASE = "wss://stream.binance.com:9443/ws"
BINANCE_TESTNET_WS_BASE = "wss://stream.testnet.binance.vision/ws"


def _ws_base() -> str:
    """WS venue MUST match the REST venue that supplies the snapshot: the
    two update-ID spaces are disjoint, so a mismatched stream can never
    bracket the snapshot and the book never syncs. DataFetcher keys market
    data off `market_data_testnet` (not `binance_testnet`), so this must
    follow the same flag."""
    # Deliberately imported at call time (late binding): tier-13 tests
    # swap config.settings.settings to exercise both venues.
    from config.settings import settings

    return BINANCE_TESTNET_WS_BASE if settings.market_data_testnet else BINANCE_WS_BASE


def _stream_symbol(pair: str) -> str:
    """'BTC/USDT' → 'btcusdt' (Binance stream naming)."""
    return pair.replace("/", "").lower()


class WSDepthFeed:
    """One WS connection + synchronizer + book per trading pair."""

    def __init__(
        self,
        pair: str,
        fetcher,                      # DataFetcher — REST snapshot source
        depth_ms: int = 100,
        stale_after_seconds: float = 30.0,
        imbalance_band_pct: float = 0.01,
    ) -> None:
        self.pair = pair
        self.book = OrderBookManager(
            symbol=pair,
            imbalance_band_pct=imbalance_band_pct,
            stale_after_seconds=stale_after_seconds,
        )
        self._sync = DepthStreamSynchronizer(
            book=self.book,
            snapshot_provider=lambda: self._rest_snapshot(fetcher),
        )
        self._url = f"{_ws_base()}/{_stream_symbol(pair)}@depth@{depth_ms}ms"
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _rest_snapshot(self, fetcher) -> dict:
        # Raw ccxt book: binance exposes lastUpdateId as `nonce`, which the
        # sync algorithm requires to bracket buffered diff events.
        raw = fetcher.fetch_order_book_raw(self.pair, depth=1000)
        return {
            "bids": raw["bids"],
            "asks": raw["asks"],
            "lastUpdateId": int(raw.get("nonce") or 0),
            "timestamp_ms": int(raw.get("timestamp") or 0),
        }

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._thread = threading.Thread(
            target=lambda: asyncio.run(self._run()),
            daemon=True,
            name=f"ws-depth-{_stream_symbol(self.pair)}",
        )
        self._thread.start()
        logger.info("WS depth feed started | {} | {}", self.pair, self._url)

    def stop(self) -> None:
        self._stop.set()

    async def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(self._url, heartbeat=30) as ws:
                        logger.info("WS depth connected | {}", self.pair)
                        backoff = 1.0
                        async for frame in ws:
                            if self._stop.is_set():
                                return
                            if frame.type == aiohttp.WSMsgType.TEXT:
                                self._sync.on_depth_message(json.loads(frame.data))
                            elif frame.type in (
                                aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR
                            ):
                                break
            except Exception as e:
                logger.warning("WS depth feed error ({}): {} — reconnecting", self.pair, e)
            # Reconnect: the book is stale until resynced — reset so
            # is_ready is honest during the outage.
            self.book.reset()
            self._sync._synced = False
            await asyncio.sleep(min(backoff, 30.0))
            backoff *= 2


class BookImbalanceSource:
    """Adapter: live WS books → the strategy's imbalance contract.

    Returns None (filter pass-through, reported as unavailable) whenever
    the pair has no feed, the book is not synced, or the feed went stale —
    never a fabricated neutral value (claude.md retry/fabrication rule).
    """

    def __init__(self, feeds: dict[str, WSDepthFeed]) -> None:
        self._feeds = feeds

    def get_imbalance(self, pair: str) -> Optional[float]:
        feed = self._feeds.get(pair)
        if feed is None or not feed._sync.is_synced:
            return None
        snap = feed.book.snapshot()
        return snap.imbalance_ratio
