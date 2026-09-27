"""
News fetcher — retrieves financial news headlines for sentiment analysis.
Uses NewsAPI.org for news data.
Get a free API key at: https://newsapi.org
"""

from datetime import datetime, timezone, timedelta
from typing import Optional
import httpx
from loguru import logger

from config.settings import settings
from core.exceptions import DataFetchError


class NewsFetcher:
    """
    Fetches crypto/financial news from NewsAPI.

    Returns structured news articles with title, description,
    source, published date, and URL.
    """

    BASE_URL = "https://newsapi.org/v2"

    CRYPTO_KEYWORDS = [
        "bitcoin", "ethereum", "crypto", "cryptocurrency",
        "blockchain", "BTC", "ETH", "binance", "altcoin",
        "defi", "web3", "nft",
    ]

    def __init__(self) -> None:
        self._api_key = settings.news_api_key
        if not self._api_key:
            logger.warning(
                "NEWS_API_KEY not set — news fetching will be disabled. "
                "Get a free key at https://newsapi.org"
            )

    def fetch_crypto_news(
        self,
        query: Optional[str] = None,
        hours_back: int = 24,
        max_articles: int = 20,
    ) -> list[dict]:
        """
        Fetch recent crypto news articles.

        Args:
            query: Search query (defaults to general crypto keywords)
            hours_back: How many hours of news to fetch
            max_articles: Maximum number of articles to return

        Returns:
            List of article dicts: {title, description, source, url, published_at, sentiment_text}
        """
        if not self._api_key:
            logger.warning("Skipping news fetch — no API key")
            return []

        if query is None:
            query = "bitcoin OR ethereum OR cryptocurrency OR crypto"

        from_time = (
            datetime.now(timezone.utc) - timedelta(hours=hours_back)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        params = {
            "q": query,
            "from": from_time,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(max_articles, 100),
            "apiKey": self._api_key,
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{self.BASE_URL}/everything", params=params)
                response.raise_for_status()
                data = response.json()

            if data.get("status") != "ok":
                raise DataFetchError(f"NewsAPI error: {data.get('message')}")

            articles = []
            for article in data.get("articles", []):
                # Skip articles with removed content
                if article.get("title") == "[Removed]":
                    continue

                articles.append({
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "source": article.get("source", {}).get("name", "Unknown"),
                    "url": article.get("url", ""),
                    "published_at": article.get("publishedAt", ""),
                    # Combined text for sentiment analysis
                    "sentiment_text": " ".join(filter(None, [
                        article.get("title", ""),
                        article.get("description", ""),
                    ])),
                })

            logger.info(
                "Fetched {} news articles (query: '{}')",
                len(articles), query
            )
            return articles

        except httpx.HTTPError as e:
            raise DataFetchError(f"Failed to fetch news: {e}") from e

    def fetch_pair_news(self, pair: str, hours_back: int = 12) -> list[dict]:
        """
        Fetch news relevant to a specific trading pair.

        Args:
            pair: Trading pair like "BTC/USDT"

        Returns:
            List of relevant articles
        """
        # Extract base currency (BTC from BTC/USDT)
        base = pair.split("/")[0]

        queries = {
            "BTC": "bitcoin OR BTC",
            "ETH": "ethereum OR ETH",
            "BNB": "BNB OR Binance coin",
        }

        query = queries.get(base, base)
        return self.fetch_crypto_news(query=query, hours_back=hours_back)
