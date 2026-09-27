"""
In-memory data cache.
Stores recent OHLCV DataFrames to avoid hammering the exchange API.
"""

import time
from typing import Optional
import pandas as pd
from loguru import logger


class DataCache:
    """
    Simple TTL-based in-memory cache for market data.

    Prevents repeated API calls for the same data within a short window.
    TTL (time-to-live) is per key — shorter for faster timeframes.
    """

    def __init__(self, default_ttl: int = 60) -> None:
        """
        Args:
            default_ttl: Default cache expiry in seconds
        """
        self._default_ttl = default_ttl
        self._store: dict[str, dict] = {}  # key -> {data, expires_at}

    def get(self, key: str) -> Optional[pd.DataFrame]:
        """
        Retrieve cached data.

        Returns:
            DataFrame if found and not expired, else None
        """
        entry = self._store.get(key)
        if entry is None:
            return None

        if time.time() > entry["expires_at"]:
            del self._store[key]
            logger.debug("Cache MISS (expired): {}", key)
            return None

        logger.debug("Cache HIT: {}", key)
        return entry["data"]

    def set(self, key: str, data: pd.DataFrame, ttl: Optional[int] = None) -> None:
        """
        Store data in cache.

        Args:
            key: Cache key
            data: DataFrame to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        expires_at = time.time() + (ttl or self._default_ttl)
        self._store[key] = {"data": data, "expires_at": expires_at}
        logger.debug("Cache SET: {} (TTL={}s)", key, ttl or self._default_ttl)

    def invalidate(self, key: str) -> None:
        """Remove a specific key from cache."""
        self._store.pop(key, None)

    def clear(self) -> None:
        """Clear all cached data."""
        self._store.clear()
        logger.debug("Cache cleared")

    def make_key(self, pair: str, timeframe: str, suffix: str = "") -> str:
        """Generate a consistent cache key."""
        key = f"{pair}:{timeframe}"
        if suffix:
            key += f":{suffix}"
        return key

    @staticmethod
    def ttl_for_timeframe(timeframe: str) -> int:
        """
        Return an appropriate TTL based on the candle timeframe.
        No point caching 1-hour data for only 10 seconds.
        """
        ttl_map = {
            "1m": 30,
            "5m": 60,
            "15m": 120,
            "30m": 300,
            "1h": 300,
            "4h": 600,
            "1d": 1800,
            "1w": 3600,
        }
        return ttl_map.get(timeframe, 60)

    def __len__(self) -> int:
        return len(self._store)


# Global cache instance
cache = DataCache()
