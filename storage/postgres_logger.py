"""PostgreSQL trade/signal/risk audit store (Roadmap Task 4.1).

A scalable, ACID, connection-pooled drop-in for the SQLite ``TradeLogger`` audit
methods — same signatures, same correlation-id structure, so run_live /
run_testnet / the risk manager keep the identical audit trail on Postgres.

- Pooling: a ``ThreadedConnectionPool`` (psycopg2); every operation borrows a
  connection via ``_cursor`` and always returns it (``putconn`` in ``finally``).
- ACID: ``_cursor(commit=True)`` commits on success and ROLLS BACK on any
  exception before re-raising — no half-written audit rows.
- Testability: the pool is injectable, so tests exercise the pooling + write
  logic against a fake pool with no psycopg2 or Postgres server.

psycopg2 is imported lazily; importing this module never fails if it's absent.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime

from loguru import logger

try:
    import psycopg2
    import psycopg2.pool
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:                     # pragma: no cover - env without psycopg2
    PSYCOPG2_AVAILABLE = False
    RealDictCursor = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id             SERIAL PRIMARY KEY,
    symbol         TEXT NOT NULL,
    side           TEXT NOT NULL,
    quantity       DOUBLE PRECISION,
    entry_price    DOUBLE PRECISION,
    highest_price  DOUBLE PRECISION,
    stop_loss      DOUBLE PRECISION,
    take_profit    DOUBLE PRECISION,
    status         TEXT NOT NULL DEFAULT 'open',
    entry_time     TIMESTAMPTZ,
    exit_time      TIMESTAMPTZ,
    exit_price     DOUBLE PRECISION,
    pnl            DOUBLE PRECISION,
    exit_reason    TEXT,
    ai_score       DOUBLE PRECISION,
    ai_confidence  DOUBLE PRECISION,
    ai_reasoning   TEXT,
    scaled_out     INTEGER DEFAULT 0,
    correlation_id TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades (status);

CREATE TABLE IF NOT EXISTS order_log (
    id                SERIAL PRIMARY KEY,
    timestamp         TIMESTAMPTZ NOT NULL,
    symbol            TEXT NOT NULL,
    side              TEXT NOT NULL,
    requested_qty     DOUBLE PRECISION,
    filled_qty        DOUBLE PRECISION,
    avg_price         DOUBLE PRECISION,
    cost              DOUBLE PRECISION,
    status            TEXT,
    ok                INTEGER,
    reason            TEXT,
    exchange_order_id TEXT,
    correlation_id    TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS risk_events (
    id             SERIAL PRIMARY KEY,
    event_type     TEXT NOT NULL,
    symbol         TEXT DEFAULT '',
    reason         TEXT DEFAULT '',
    created_at     TIMESTAMPTZ NOT NULL,
    correlation_id TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS signals (
    id              SERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          TEXT NOT NULL,
    ml_score        DOUBLE PRECISION,
    llm_score       DOUBLE PRECISION,
    sentiment_score DOUBLE PRECISION,
    combined_score  DOUBLE PRECISION,
    confidence      DOUBLE PRECISION,
    decision        TEXT,
    correlation_id  TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS pending_orders (
    client_order_id   TEXT PRIMARY KEY,
    timestamp         TIMESTAMPTZ NOT NULL,
    symbol            TEXT NOT NULL,
    side              TEXT NOT NULL,
    action            TEXT NOT NULL,
    requested_qty     DOUBLE PRECISION,
    status            TEXT NOT NULL DEFAULT 'pending',
    exchange_order_id TEXT DEFAULT '',
    correlation_id    TEXT DEFAULT ''
);
"""


class PostgresTradeLogger:
    """Pooled, ACID Postgres backend mirroring the SQLite TradeLogger audit API."""

    def __init__(self, dsn: str = "", *, pool=None, minconn: int = 1,
                 maxconn: int = 10, ensure_schema: bool = True) -> None:
        if pool is not None:
            self._pool = pool                        # injected (tests / custom)
        elif PSYCOPG2_AVAILABLE and dsn:
            self._pool = psycopg2.pool.ThreadedConnectionPool(minconn, maxconn, dsn)
            logger.info("PostgresTradeLogger — pool [{}, {}] connected", minconn, maxconn)
        else:
            raise RuntimeError(
                "PostgresTradeLogger requires psycopg2 + a DSN (or an injected pool). "
                "Install psycopg2-binary and set POSTGRES_DSN."
            )
        if ensure_schema:
            self._ensure_schema()

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)

    @contextmanager
    def _cursor(self, *, commit: bool = False, dict_rows: bool = False):
        """Borrow a pooled connection; commit or ROLLBACK, then always return it."""
        conn = self._pool.getconn()
        try:
            if dict_rows and PSYCOPG2_AVAILABLE and RealDictCursor is not None:
                cur = conn.cursor(cursor_factory=RealDictCursor)
            else:
                cur = conn.cursor()
            try:
                yield cur
                if commit:
                    conn.commit()
            except Exception:
                conn.rollback()          # ACID: never leave a half-written audit row
                raise
            finally:
                cur.close()
        finally:
            self._pool.putconn(conn)     # pooling: connection always returned

    def _ensure_schema(self) -> None:
        with self._cursor(commit=True) as cur:
            cur.execute(_SCHEMA)

    def close(self) -> None:
        closeall = getattr(self._pool, "closeall", None)
        if callable(closeall):
            closeall()

    # ─── Trades ───────────────────────────────────────────────────────────────
    def log_trade_open(self, symbol, side, quantity, entry_price, stop_loss,
                       take_profit, ai_score=None, ai_confidence=None,
                       ai_reasoning=None, correlation_id: str = "") -> int:
        with self._cursor(commit=True) as cur:
            cur.execute(
                """INSERT INTO trades
                   (symbol, side, quantity, entry_price, highest_price, stop_loss,
                    take_profit, status, entry_time, ai_score, ai_confidence,
                    ai_reasoning, scaled_out, correlation_id)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,'open',%s,%s,%s,%s,0,%s)
                   RETURNING id""",
                (symbol, side, quantity, entry_price, entry_price, stop_loss,
                 take_profit, self._now(), ai_score, ai_confidence, ai_reasoning,
                 correlation_id),
            )
            trade_id = cur.fetchone()[0]
        logger.info("Trade opened (pg) id={} {} {} {} @ {}", trade_id, side, quantity, symbol, entry_price)
        return int(trade_id)

    def log_trade_close(self, trade_id: int, exit_price: float, pnl: float,
                        exit_reason: str, correlation_id: str = "") -> None:
        with self._cursor(commit=True) as cur:
            cur.execute(
                """UPDATE trades SET status='closed', exit_time=%s, exit_price=%s,
                          pnl=%s, exit_reason=%s, correlation_id=COALESCE(NULLIF(%s,''), correlation_id)
                   WHERE id=%s""",
                (self._now(), exit_price, pnl, exit_reason, correlation_id, trade_id),
            )

    def get_open_trades(self) -> list[dict]:
        with self._cursor(dict_rows=True) as cur:
            cur.execute("SELECT * FROM trades WHERE status='open' ORDER BY entry_time DESC")
            return [dict(r) for r in cur.fetchall()]

    # ─── Orders / risk / signals (audit trail + correlation ids) ──────────────
    def log_order(self, *, symbol, side, requested_qty, filled_qty, avg_price,
                  cost, status, ok, reason: str = "", exchange_order_id: str = "",
                  correlation_id: str = "") -> None:
        with self._cursor(commit=True) as cur:
            cur.execute(
                """INSERT INTO order_log
                   (timestamp, symbol, side, requested_qty, filled_qty, avg_price,
                    cost, status, ok, reason, exchange_order_id, correlation_id)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (self._now(), symbol, side, requested_qty, filled_qty, avg_price,
                 cost, status, 1 if ok else 0, reason, exchange_order_id, correlation_id),
            )

    def log_risk_event(self, event_type: str, symbol: str = "", reason: str = "",
                       correlation_id: str = "") -> None:
        with self._cursor(commit=True) as cur:
            cur.execute(
                """INSERT INTO risk_events (event_type, symbol, reason, created_at, correlation_id)
                   VALUES (%s,%s,%s,%s,%s)""",
                (event_type, symbol, reason, self._now(), correlation_id),
            )

    def log_signal(self, symbol, ml_score, llm_score, sentiment_score,
                   combined_score, confidence, decision, correlation_id: str = "") -> None:
        with self._cursor(commit=True) as cur:
            cur.execute(
                """INSERT INTO signals
                   (timestamp, symbol, ml_score, llm_score, sentiment_score,
                    combined_score, confidence, decision, correlation_id)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (self._now(), symbol, ml_score, llm_score, sentiment_score,
                 combined_score, confidence, decision, correlation_id),
            )

    # ─── Pending orders (Task 3.3 idempotency) ────────────────────────────────
    def record_pending_order(self, *, client_order_id, symbol, side, action,
                             requested_qty, correlation_id: str = "") -> None:
        with self._cursor(commit=True) as cur:
            cur.execute(
                """INSERT INTO pending_orders
                   (client_order_id, timestamp, symbol, side, action, requested_qty,
                    status, exchange_order_id, correlation_id)
                   VALUES (%s,%s,%s,%s,%s,%s,'pending','',%s)
                   ON CONFLICT (client_order_id) DO UPDATE
                   SET status='pending', requested_qty=EXCLUDED.requested_qty""",
                (client_order_id, self._now(), symbol, side, action, requested_qty, correlation_id),
            )

    def get_pending_orders(self) -> list[dict]:
        with self._cursor(dict_rows=True) as cur:
            cur.execute("SELECT * FROM pending_orders WHERE status='pending' ORDER BY timestamp")
            return [dict(r) for r in cur.fetchall()]

    def resolve_pending_order(self, client_order_id: str, status: str,
                              exchange_order_id: str = "") -> None:
        with self._cursor(commit=True) as cur:
            cur.execute(
                "UPDATE pending_orders SET status=%s, exchange_order_id=%s WHERE client_order_id=%s",
                (status, exchange_order_id, client_order_id),
            )
