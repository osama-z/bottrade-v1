"""Storage-backend factory (Roadmap Task 4.1).

Selects the audit logger (SQLite default, Postgres opt-in) and the market-data
time-series store (none / TimescaleDB / ClickHouse) from settings, so callers
depend on configuration rather than a concrete class. SQLite remains the default
so single-node/dev runs are unchanged.
"""

from __future__ import annotations

from loguru import logger


def get_trade_logger(db_path: str | None = None, *, settings=None):
    """Build the configured trade/signal/risk audit logger.

    DATABASE_BACKEND=postgres → pooled ACID PostgresTradeLogger (POSTGRES_DSN);
    otherwise the SQLite TradeLogger (default). ``settings`` overrides the global
    (for tests).
    """
    if settings is None:
        from config.settings import settings as settings

    if settings.database_backend == "postgres":
        from storage.postgres_logger import PostgresTradeLogger
        logger.info("Storage backend: PostgreSQL (pooled)")
        return PostgresTradeLogger(
            dsn=settings.postgres_dsn,
            minconn=settings.db_pool_min,
            maxconn=settings.db_pool_max,
        )

    from storage.trade_logger import TradeLogger
    return TradeLogger(db_path=db_path)


def get_market_data_store(*, settings=None):
    """Build the configured market-data store, or None when disabled.

    MARKET_DATA_BACKEND=timescale → TimescaleMarketDataStore (TIMESCALE_DSN, or
    POSTGRES_DSN); =clickhouse → ClickHouse stub; =none (default) → None.
    ``settings`` overrides the global (for tests).
    """
    if settings is None:
        from config.settings import settings as settings

    backend = settings.market_data_backend
    if backend == "timescale":
        from storage.market_data import TimescaleMarketDataStore
        return TimescaleMarketDataStore(
            dsn=settings.timescale_dsn or settings.postgres_dsn,
            minconn=settings.db_pool_min,
            maxconn=settings.db_pool_max,
        )
    if backend == "clickhouse":
        from storage.market_data import ClickHouseMarketDataStore
        return ClickHouseMarketDataStore()
    return None
