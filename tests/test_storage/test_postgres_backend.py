"""Task 4.1 — Postgres/Timescale storage: pooling, ACID, write logic (mocked)."""
import types
from datetime import UTC, datetime

import pytest

from storage.postgres_logger import PostgresTradeLogger


# ─── Fake psycopg2 pool / connection / cursor ─────────────────────────────────
class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.closed = False

    def execute(self, sql, params=None):
        self.conn.log.append(("execute", sql, params))
        if self.conn.raise_on_execute:
            raise RuntimeError("boom")

    def fetchone(self):
        return self.conn.rows[0] if self.conn.rows else None

    def fetchall(self):
        return list(self.conn.rows)

    def close(self):
        self.closed = True


class FakeConn:
    def __init__(self):
        self.log = []
        self.rows = []
        self.raise_on_execute = False

    def cursor(self, **kwargs):
        return FakeCursor(self)

    def commit(self):
        self.log.append(("commit",))

    def rollback(self):
        self.log.append(("rollback",))


class FakePool:
    def __init__(self):
        self.conn = FakeConn()
        self.getconn_calls = 0
        self.putconn_calls = 0

    def getconn(self):
        self.getconn_calls += 1
        return self.conn

    def putconn(self, conn):
        self.putconn_calls += 1

    def closeall(self):
        pass


def _executes(pool):
    return [e for e in pool.conn.log if e[0] == "execute"]


class TestPoolingAndAcid:
    def test_connection_is_borrowed_and_returned(self):
        pool = FakePool()
        lg = PostgresTradeLogger(pool=pool, ensure_schema=False)
        lg.log_risk_event("pause", reason="cool-off", correlation_id="c1")
        assert pool.getconn_calls == 1 and pool.putconn_calls == 1   # balanced
        assert ("commit",) in pool.conn.log                          # ACID commit

    def test_correlation_id_is_written(self):
        pool = FakePool()
        PostgresTradeLogger(pool=pool, ensure_schema=False).log_signal(
            "BTC/USDT", 0.1, 0.2, 0.3, 0.4, 0.6, "BUY", correlation_id="corr-9")
        sql, params = _executes(pool)[0][1], _executes(pool)[0][2]
        assert "INSERT INTO signals" in sql and "corr-9" in params

    def test_rollback_and_return_on_error(self):
        pool = FakePool()
        pool.conn.raise_on_execute = True
        lg = PostgresTradeLogger(pool=pool, ensure_schema=False)
        with pytest.raises(RuntimeError):
            lg.log_order(symbol="BTC/USDT", side="buy", requested_qty=1, filled_qty=0,
                         avg_price=0, cost=0, status="failed", ok=False)
        assert ("rollback",) in pool.conn.log
        assert ("commit",) not in pool.conn.log
        assert pool.putconn_calls == 1                               # returned even on error

    def test_ensure_schema_creates_audit_tables(self):
        pool = FakePool()
        PostgresTradeLogger(pool=pool, ensure_schema=True)
        ddl = _executes(pool)[0][1]
        for t in ("trades", "order_log", "risk_events", "signals", "pending_orders"):
            assert t in ddl
        assert "correlation_id" in ddl                               # audit structure preserved


class TestWriteAndRead:
    def test_log_trade_open_returns_generated_id(self):
        pool = FakePool()
        pool.conn.rows = [[42]]                                       # RETURNING id
        tid = PostgresTradeLogger(pool=pool, ensure_schema=False).log_trade_open(
            "BTC/USDT", "buy", 0.5, 100.0, 97.0, 110.0, correlation_id="c")
        assert tid == 42

    def test_get_open_trades_returns_dicts(self):
        pool = FakePool()
        pool.conn.rows = [{"id": 1, "symbol": "BTC/USDT", "status": "open"}]
        rows = PostgresTradeLogger(pool=pool, ensure_schema=False).get_open_trades()
        assert rows == [{"id": 1, "symbol": "BTC/USDT", "status": "open"}]

    def test_pending_order_lifecycle_sql(self):
        pool = FakePool()
        lg = PostgresTradeLogger(pool=pool, ensure_schema=False)
        lg.record_pending_order(client_order_id="k1", symbol="BTC/USDT", side="buy",
                                action="entry", requested_qty=0.5)
        lg.resolve_pending_order("k1", "filled", "EX1")
        sqls = [e[1] for e in _executes(pool)]
        assert any("INSERT INTO pending_orders" in s for s in sqls)
        assert any("UPDATE pending_orders" in s for s in sqls)


# ─── TimescaleDB market-data store ─────────────────────────────────────────────
class TestTimescale:
    def test_schema_creates_hypertables(self):
        from storage.market_data import TimescaleMarketDataStore
        pool = FakePool()
        TimescaleMarketDataStore(pool=pool, ensure_schema=True)
        sqls = [e[1] for e in _executes(pool)]
        assert any("CREATE TABLE" in s and "ohlcv" in s for s in sqls)
        assert any("create_hypertable" in s for s in sqls)           # time-partitioned

    def test_insert_ohlcv_upserts(self):
        from storage.market_data import TimescaleMarketDataStore
        pool = FakePool()
        store = TimescaleMarketDataStore(pool=pool, ensure_schema=False)
        store.insert_ohlcv(symbol="BTC/USDT", timeframe="1h",
                           time=datetime(2024, 1, 1, tzinfo=UTC),
                           open=1, high=2, low=0.5, close=1.5, volume=100)
        assert any("INSERT INTO ohlcv" in e[1] for e in _executes(pool))
        assert pool.putconn_calls == 1

    def test_clickhouse_stub_raises(self):
        from storage.market_data import ClickHouseMarketDataStore
        s = ClickHouseMarketDataStore()
        with pytest.raises(NotImplementedError):
            s.insert_ohlcv(symbol="BTC/USDT")


# ─── Factory routing ───────────────────────────────────────────────────────────
class TestFactory:
    def test_default_backend_is_sqlite(self, tmp_path):
        from storage.factory import get_trade_logger
        from storage.trade_logger import TradeLogger
        lg = get_trade_logger(db_path=str(tmp_path / "t.db"))
        assert isinstance(lg, TradeLogger)                           # SQLite default unchanged

    def test_postgres_backend_routes_with_pool_config(self, monkeypatch):
        import storage.postgres_logger as pg
        captured = {}

        class FakePG:
            def __init__(self, dsn="", minconn=1, maxconn=10):
                captured.update(dsn=dsn, minconn=minconn, maxconn=maxconn)

        monkeypatch.setattr(pg, "PostgresTradeLogger", FakePG)
        from storage.factory import get_trade_logger
        cfg = types.SimpleNamespace(database_backend="postgres", postgres_dsn="postgresql://x",
                                    db_pool_min=2, db_pool_max=7)
        assert isinstance(get_trade_logger(settings=cfg), FakePG)
        assert captured == {"dsn": "postgresql://x", "minconn": 2, "maxconn": 7}

    def test_market_data_default_is_none(self):
        from storage.factory import get_market_data_store
        cfg = types.SimpleNamespace(market_data_backend="none")
        assert get_market_data_store(settings=cfg) is None

    def test_market_data_routes_to_timescale(self, monkeypatch):
        import storage.market_data as md
        captured = {}

        class FakeTS:
            def __init__(self, dsn="", minconn=1, maxconn=10):
                captured.update(dsn=dsn)

        monkeypatch.setattr(md, "TimescaleMarketDataStore", FakeTS)
        from storage.factory import get_market_data_store
        cfg = types.SimpleNamespace(market_data_backend="timescale", timescale_dsn="ts://y",
                                    postgres_dsn="pg://z", db_pool_min=1, db_pool_max=5)
        assert isinstance(get_market_data_store(settings=cfg), FakeTS)
        assert captured["dsn"] == "ts://y"
