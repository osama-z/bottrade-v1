"""ws_book_smoke.py — live verification of the Stage 2 WS order book.

Connects the real Binance depth stream for one pair, runs the sync
algorithm, and prints a book snapshot every 2 seconds. Run this once
before enabling USE_WS_ORDER_BOOK=true.

Usage:
    python scripts/ws_book_smoke.py [PAIR] [SECONDS]
    python scripts/ws_book_smoke.py BTC/USDT 30
"""

import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from config.logging_config import setup_logging
from data.fetcher import DataFetcher
from data.ws_depth_feed import WSDepthFeed


def main() -> None:
    setup_logging()
    pair = sys.argv[1] if len(sys.argv) > 1 else "BTC/USDT"
    seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    feed = WSDepthFeed(pair, fetcher=DataFetcher())
    feed.start()

    deadline = time.time() + seconds
    try:
        while time.time() < deadline:
            time.sleep(2)
            if not feed._sync.is_synced:
                logger.info("syncing... (resyncs so far: {})", feed._sync.resync_count)
                continue
            snap = feed.book.snapshot()
            logger.info(
                "{} | bid={} ask={} spread={:.4f} | imbalance={:.3f} | "
                "levels={} | update_id={} | fresh={:.1f}s ago",
                pair, snap.best_bid, snap.best_ask, snap.spread or 0.0,
                snap.imbalance_ratio or 0.0, len(feed.book),
                snap.last_update_id, feed.book.seconds_since_update or -1,
            )
    finally:
        feed.stop()
        logger.info(
            "Smoke run done | synced={} | resyncs={}",
            feed._sync.is_synced, feed._sync.resync_count,
        )


if __name__ == "__main__":
    main()
