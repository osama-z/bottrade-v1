"""
Sentiment fetcher — retrieves market sentiment data from external sources.
Includes: Fear & Greed Index, social sentiment scores.
"""

import httpx
from datetime import datetime, timezone
from loguru import logger

from config.constants import MarketSentiment
from core.exceptions import DataFetchError


class SentimentFetcher:
    """
    Fetches crypto market sentiment from multiple sources.

    Sources:
    - Alternative.me Fear & Greed Index (free, no API key needed)
    - (More sources can be added: LunarCrush, Santiment, etc.)
    """

    FEAR_GREED_URL = "https://api.alternative.me/fng/"

    def fetch_fear_greed_index(self, limit: int = 1) -> list[dict]:
        """
        Fetch Fear & Greed Index from alternative.me.
        Free API, no key required.

        Args:
            limit: Number of days of data to fetch (1 = today only)

        Returns:
            List of dicts: {value, classification, timestamp, sentiment_enum}
        """
        try:
            params = {"limit": limit, "format": "json"}

            with httpx.Client(timeout=10.0) as client:
                response = client.get(self.FEAR_GREED_URL, params=params)
                response.raise_for_status()
                data = response.json()

            results = []
            for entry in data.get("data", []):
                value = int(entry["value"])
                classification = entry["value_classification"]

                # Map to our MarketSentiment enum
                sentiment = self._value_to_enum(value)

                results.append({
                    "value": value,                    # 0-100 (0=extreme fear, 100=extreme greed)
                    "classification": classification,
                    "sentiment": sentiment,
                    "timestamp": datetime.fromtimestamp(
                        int(entry["timestamp"]), tz=timezone.utc
                    ),
                })

            if results:
                current = results[0]
                logger.info(
                    "Fear & Greed Index: {}/100 — {}",
                    current["value"], current["classification"]
                )

            return results

        except httpx.HTTPError as e:
            raise DataFetchError(f"Failed to fetch Fear & Greed Index: {e}") from e

    def get_current_sentiment(self) -> dict:
        """
        Get the current market sentiment (latest Fear & Greed value).

        Returns:
            Dict with value (0-100), classification, sentiment enum,
            and a normalized score (0.0-1.0)
        """
        results = self.fetch_fear_greed_index(limit=1)
        if not results:
            logger.warning("No sentiment data — returning neutral")
            return {
                "value": 50,
                "classification": "Neutral",
                "sentiment": MarketSentiment.NEUTRAL,
                "normalized": 0.5,
            }

        current = results[0]
        current["normalized"] = current["value"] / 100.0
        return current

    def get_sentiment_history(self, days: int = 30) -> list[dict]:
        """Fetch sentiment history for the past N days."""
        return self.fetch_fear_greed_index(limit=days)

    @staticmethod
    def _value_to_enum(value: int) -> MarketSentiment:
        """Convert Fear & Greed numeric value to MarketSentiment enum."""
        if value <= 20:
            return MarketSentiment.EXTREME_FEAR
        elif value <= 40:
            return MarketSentiment.FEAR
        elif value <= 60:
            return MarketSentiment.NEUTRAL
        elif value <= 80:
            return MarketSentiment.GREED
        else:
            return MarketSentiment.EXTREME_GREED

    @staticmethod
    def interpret_for_trading(sentiment: dict) -> str:
        """
        Give a trading interpretation of the sentiment.
        Classic strategy: buy extreme fear, sell extreme greed.
        """
        value = sentiment["value"]
        if value <= 20:
            return "CONTRARIAN_BUY"    # Extreme fear = potential buying opportunity
        elif value <= 35:
            return "SLIGHT_BUY"
        elif value <= 65:
            return "NEUTRAL"
        elif value <= 80:
            return "SLIGHT_CAUTION"
        else:
            return "CONTRARIAN_SELL"   # Extreme greed = potential selling opportunity
