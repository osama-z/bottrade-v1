"""Time-series market-data stores (Roadmap Task 4.1).

Market data (OHLCV, order-book snapshots, ticks) is high-volume and append-only —
a poor fit for the relational trade audit DB. This provides a TimescaleDB store
(a Postgres extension: hypertables auto-partition by time) and a ClickHouse
interface stub, both behind one small write API so the caller doesn't care which
backend is configured.

TimescaleMarketDataStore is pooled + ACID like PostgresTradeLogger and takes an
injectable pool, so the schema/write logic is testable without a server.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime

from loguru import logger

try:
    import psycopg2
    import psycopg2.pool
    PSYCOPG2_AVAILABLE = True
except ImportError:                     # pragma: no cover
    PSYCOPG2_AVAILABLE = False


# TimescaleDB: create_hypertable partitions each table by time. Guarded so the
# schema still applies (as plain tables) on a Postgres without the extension.
_TS_SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlcv (
    time      TIMESTAMPTZ NOT NULL,
    symbol    TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    open      DOUBLE PRECISION,
    high      DOUBLE PRECISION,
    low       DOUBLE PRECISION,
    close     DOUBLE PRECISION,
    volume    DOUBLE PRECISION,
    PRIMARY KEY (symbol, timeframe, time)
);
CREATE TABLE IF NOT EXISTS order_book_snapshots (
    time     TIMESTAMPTZ NOT NULL,
    symbol   TEXT NOT NULL,
    best_bid DOUBLE PRECISION,
    best_ask DOUBLE PRECISION,
    bid_depth_usdt DOUBLE PRECISION,
    ask_depth_usdt DOUBLE PRECISION,
    imbalance DOUBLE PRECISION
);
CREATE TABLE IF NOT EXISTS ticks (
    time   TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    price  DOUBLE PRECISION,
    qty    DOUBLE PRECISION,
    side   TEXT
);
"""

_TS_HYPERTABLES = ("ohlcv", "order_book_snapshots", "ticks")


class TimescaleMarketDataStore:
    """Pooled TimescaleDB store for OHLCV / order-book snapshots / ticks."""

    def __init__(self, dsn: str = "", *, pool=None, minconn: int = 1,
                 maxconn: int = 10, ensure_schema: bool = True) -> None:
        if pool is not None:
            self._pool = pool
        elif PSYCOPG2_AVAILABLE and dsn:
            self._pool = psycopg2.pool.ThreadedConnectionPool(minconn, maxconn, dsn)
        else:
            raise RuntimeError("TimescaleMarketDataStore needs psycopg2 + a DSN, or an injected pool")
        if ensure_schema:
            self._ensure_schema()

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)

    @contextmanager
    def _cursor(self, *, commit: bool = False):
        conn = self._pool.getconn()
        try:
            cur = conn.cursor()
            try:
                yield cur
                if commit:
                    conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                cur.close()
        finally:
            self._pool.putconn(conn)

    def _ensure_schema(self) -> None:
        with self._cursor(commit=True) as cur:
            cur.execute(_TS_SCHEMA)
            for table in _TS_HYPERTABLES:
                # create_hypertable is a no-op if TimescaleDB isn't installed;
                # tolerate that so a plain Postgres still works.
                try:
                    cur.execute(
                        "SELECT create_hypertable(%s, 'time', if_not_exists => TRUE)",
                        (table,),
                    )
                except Exception as e:      # pragma: no cover - depends on extension
                    logger.debug("create_hypertable({}) skipped: {}", table, e)

    def insert_ohlcv(self, *, symbol, timeframe, time, open, high, low, close, volume) -> None:
        with self._cursor(commit=True) as cur:
            cur.execute(
                """INSERT INTO ohlcv (time, symbol, timeframe, open, high, low, close, volume)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (symbol, timeframe, time) DO UPDATE
                   SET open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low,
                       close=EXCLUDED.close, volume=EXCLUDED.volume""",
                (time, symbol, timeframe, open, high, low, close, volume),
            )

    def insert_order_book_snapshot(self, *, symbol, best_bid, best_ask,
                                   bid_depth_usdt, ask_depth_usdt, imbalance,
                                   time=None) -> None:
        with self._cursor(commit=True) as cur:
            cur.execute(
                """INSERT INTO order_book_snapshots
                   (time, symbol, best_bid, best_ask, bid_depth_usdt, ask_depth_usdt, imbalance)
                   VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                (time or self._now(), symbol, best_bid, best_ask,
                 bid_depth_usdt, ask_depth_usdt, imbalance),
            )

    def insert_tick(self, *, symbol, price, qty, side, time=None) -> None:
        with self._cursor(commit=True) as cur:
            cur.execute(
                "INSERT INTO ticks (time, symbol, price, qty, side) VALUES (%s,%s,%s,%s,%s)",
                (time or self._now(), symbol, price, qty, side),
            )


class ClickHouseMarketDataStore:
    """ClickHouse alternative for market data — interface stub.

    ClickHouse (columnar OLAP) is an alternative to TimescaleDB for very high
    ingest rates. The store is not wired to a client yet; the interface mirrors
    TimescaleMarketDataStore so it can be dropped in later. Methods raise until a
    driver (clickhouse-connect) is added.
    """

    _MSG = ("ClickHouse backend is a stub — install clickhouse-connect and "
            "implement, or use MARKET_DATA_BACKEND=timescale")

    def __init__(self, *args, **kwargs) -> None:
        logger.warning("ClickHouseMarketDataStore instantiated as a stub — writes will raise")

    def insert_ohlcv(self, **kwargs) -> None:
        raise NotImplementedError(self._MSG)

    def insert_order_book_snapshot(self, **kwargs) -> None:
        raise NotImplementedError(self._MSG)

    def insert_tick(self, **kwargs) -> None:
        raise NotImplementedError(self._MSG)
