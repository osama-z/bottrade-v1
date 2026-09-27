"""
Trade Logger — SQLite persistence layer for NeuronTrade.

Uses raw sqlite3 (no ORM) for maximum speed and simplicity.
Handles all database operations: trades, signals, and daily summaries.

Tables:
    trades       — Every position opened and closed by the bot
    signals      — Every AI decision made (full audit trail)
    daily_summary — End-of-day performance snapshot
"""

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional

from loguru import logger


class TradeLogger:
    """
    Thread-safe SQLite database manager for NeuronTrade.

    Usage:
        db = TradeLogger()
        trade_id = db.log_trade_open("BTC/USDT", "buy", 0.01, 64000.0, 62720.0, 67200.0)
        db.log_trade_close(trade_id, 67200.0, 32.0, "take_profit")
    """

    # SQL — Table Definitions
    _CREATE_TRADES = """
    CREATE TABLE IF NOT EXISTS trades (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol         TEXT    NOT NULL,
        side           TEXT    NOT NULL,
        quantity       REAL    NOT NULL,
        entry_price    REAL    NOT NULL,
        exit_price     REAL,
        stop_loss      REAL,
        take_profit    REAL,
        pnl            REAL    DEFAULT 0.0,
        status         TEXT    DEFAULT 'open',
        exit_reason    TEXT,
        entry_time     TEXT    NOT NULL,
        exit_time      TEXT,
        ai_score       REAL,
        ai_confidence  REAL,
        ai_reasoning   TEXT,
        highest_price  REAL    DEFAULT 0.0,
        scaled_out     INTEGER DEFAULT 0
    );
    """

    _CREATE_SIGNALS = """
    CREATE TABLE IF NOT EXISTS signals (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp       TEXT    NOT NULL,
        symbol          TEXT    NOT NULL,
        ml_score        REAL,
        llm_score       REAL,
        sentiment_score REAL,
        combined_score  REAL,
        confidence      REAL,
        decision        TEXT
    );
    """

    _CREATE_DAILY_SUMMARY = """
    CREATE TABLE IF NOT EXISTS daily_summary (
        date             TEXT PRIMARY KEY,
        starting_balance REAL,
        ending_balance   REAL,
        total_trades     INTEGER DEFAULT 0,
        wins             INTEGER DEFAULT 0,
        losses           INTEGER DEFAULT 0,
        net_pnl          REAL    DEFAULT 0.0
    );
    """

    _CREATE_MARKET_STRUCTURE = """
    CREATE TABLE IF NOT EXISTS market_structure (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp     TEXT NOT NULL,
        symbol        TEXT NOT NULL,
        open_interest REAL,
        taker_ratio   REAL,
        funding_rate  REAL
    );
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        """
        Initialize database, create tables if they don't exist.

        Args:
            db_path: Path to the SQLite database file. Defaults to the
                configured ``settings.database_path`` (absolute, anchored
                at the project root — never CWD-relative).
        """
        if db_path is None:
            from config.settings import settings
            db_path = settings.database_path
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Lock for thread safety (Telegram + trading loop run concurrently)
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None

        self._init_db()
        logger.info("TradeLogger initialized at {}", self.db_path.resolve())

    def _init_db(self) -> None:
        """Create all tables on first run."""
        with self._get_conn() as conn:
            conn.execute(self._CREATE_TRADES)
            conn.execute(self._CREATE_SIGNALS)
            conn.execute(self._CREATE_DAILY_SUMMARY)
            conn.execute(self._CREATE_MARKET_STRUCTURE)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS account_state (
                    id          INTEGER PRIMARY KEY CHECK (id = 1),
                    balance     REAL NOT NULL,
                    updated_at  TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_events (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type  TEXT NOT NULL,
                    symbol      TEXT,
                    reason      TEXT,
                    created_at  TEXT NOT NULL
                )
                """
            )
            # Append-only audit of EVERY order attempt (accepted or rejected) —
            # the source of truth for what the bot actually sent to the
            # exchange, distinct from `trades` (positions). Enables debugging,
            # reconciliation, and idempotency.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS order_log (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp         TEXT NOT NULL,
                    symbol            TEXT NOT NULL,
                    side              TEXT NOT NULL,
                    requested_qty     REAL,
                    filled_qty        REAL,
                    avg_price         REAL,
                    cost              REAL,
                    status            TEXT,
                    ok                INTEGER,
                    reason            TEXT,
                    exchange_order_id TEXT,
                    correlation_id    TEXT DEFAULT ''
                )
                """
            )

            # pending_orders (Task 3.3): an action's clientOrderId is written HERE
            # with status='pending' BEFORE the API call. On restart, run_testnet
            # queries the exchange for each pending coid and resolves it — so a
            # crash mid-order never duplicates or loses a fill.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_orders (
                    client_order_id   TEXT PRIMARY KEY,
                    timestamp         TEXT NOT NULL,
                    symbol            TEXT NOT NULL,
                    side              TEXT NOT NULL,
                    action            TEXT NOT NULL,
                    requested_qty     REAL,
                    status            TEXT NOT NULL DEFAULT 'pending',
                    exchange_order_id TEXT DEFAULT '',
                    correlation_id    TEXT DEFAULT ''
                )
                """
            )

            # shadow_decisions (Task 5.1): shadow-mode logs the signal + expected
            # fill for every decided candle, executing nothing — the basis for
            # the backtest-parity (< 5% divergence) gate.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shadow_decisions (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp         TEXT NOT NULL,
                    symbol            TEXT NOT NULL,
                    decision          TEXT NOT NULL,
                    price             REAL,
                    expected_slippage REAL,
                    expected_fill     REAL,
                    confidence        REAL,
                    reason            TEXT,
                    correlation_id    TEXT DEFAULT '',
                    strategy          TEXT DEFAULT '',
                    timeframe         TEXT DEFAULT ''
                )
                """
            )
            # Migrate older shadow_decisions tables (add strategy/timeframe).
            shadow_cols = [r[1] for r in conn.execute("PRAGMA table_info(shadow_decisions)")]
            if "strategy" not in shadow_cols:
                conn.execute("ALTER TABLE shadow_decisions ADD COLUMN strategy TEXT DEFAULT ''")
            if "timeframe" not in shadow_cols:
                conn.execute("ALTER TABLE shadow_decisions ADD COLUMN timeframe TEXT DEFAULT ''")

            # Check if migrations are needed for existing databases
            cursor = conn.execute("PRAGMA table_info(trades)")
            columns = [row[1] for row in cursor.fetchall()]
            if "highest_price" not in columns:
                conn.execute("ALTER TABLE trades ADD COLUMN highest_price REAL DEFAULT 0.0")
                logger.info("Database migration: Added highest_price column to trades table")
            if "scaled_out" not in columns:
                conn.execute("ALTER TABLE trades ADD COLUMN scaled_out INTEGER DEFAULT 0")
                logger.info("Database migration: Added scaled_out column to trades table")
            if "correlation_id" not in columns:
                conn.execute("ALTER TABLE trades ADD COLUMN correlation_id TEXT DEFAULT ''")
                logger.info("Database migration: Added correlation_id column to trades table")
            cursor = conn.execute("PRAGMA table_info(signals)")
            sig_columns = [row[1] for row in cursor.fetchall()]
            if "correlation_id" not in sig_columns:
                conn.execute("ALTER TABLE signals ADD COLUMN correlation_id TEXT DEFAULT ''")
                logger.info("Database migration: Added correlation_id column to signals table")
            cursor = conn.execute("PRAGMA table_info(risk_events)")
            re_columns = [row[1] for row in cursor.fetchall()]
            if "correlation_id" not in re_columns:
                conn.execute("ALTER TABLE risk_events ADD COLUMN correlation_id TEXT DEFAULT ''")
                logger.info("Database migration: Added correlation_id column to risk_events table")

            conn.commit()
        logger.debug("Database tables initialized")

    @contextmanager
    def _get_conn(self) -> Generator[sqlite3.Connection, None, None]:
        """Thread-safe access to the single persistent connection.

        One connection per TradeLogger, serialized by the RLock. The
        previous design opened (and closed) a fresh connection on every
        query — a file open, schema parse and cold page cache per call —
        and defeated WAL's reader/writer concurrency.
        """
        with self._lock:
            if self._conn is None:
                self._conn = sqlite3.connect(
                    str(self.db_path),
                    check_same_thread=False,  # cross-thread use is lock-guarded
                    timeout=10.0,
                )
                self._conn.row_factory = sqlite3.Row  # Access columns by name
                # WAL: readers (dashboard) never block the writer and vice
                # versa. busy_timeout: wait instead of raising "database is
                # locked". synchronous=FULL: a circuit-breaker trip must
                # survive power loss, not just a process crash.
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA busy_timeout=10000")
                self._conn.execute("PRAGMA synchronous=FULL")
            try:
                yield self._conn
            except sqlite3.Error as e:
                self._conn.rollback()
                logger.error("Database error: {}", e)
                raise

    def close(self) -> None:
        """Close the persistent connection (shutdown / tests)."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # ─── Trade Operations ──────────────────────────────────────────────────────

    def log_trade_open(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        ai_score: Optional[float] = None,
        ai_confidence: Optional[float] = None,
        ai_reasoning: Optional[str] = None,
        correlation_id: str = "",
    ) -> int:
        """
        Record a newly opened trade.

        Returns:
            trade_id (int) — use this to close the trade later
        """
        now = self._now()
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO trades
                    (symbol, side, quantity, entry_price, highest_price, stop_loss, take_profit,
                     status, entry_time, ai_score, ai_confidence, ai_reasoning, scaled_out,
                     correlation_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?, 0, ?)
                """,
                (symbol, side, quantity, entry_price, entry_price, stop_loss, take_profit,
                 now, ai_score, ai_confidence, ai_reasoning, correlation_id),
            )
            conn.commit()
            trade_id = cursor.lastrowid

        logger.info(
            "Trade opened | id={} | {} {} {} @ ${:.2f} | SL=${:.2f} TP=${:.2f}",
            trade_id, side.upper(), quantity, symbol, entry_price, stop_loss, take_profit
        )
        return trade_id

    def update_trade_trailing_state(
        self,
        trade_id: int,
        highest_price: float,
        scaled_out: int,
        stop_loss: float,
        quantity: Optional[float] = None,
    ) -> None:
        """
        Update the trailing stop price, highest price reached, scaled out status,
        and optionally reduce quantity if scaling out.
        """
        with self._get_conn() as conn:
            if quantity is not None:
                conn.execute(
                    """
                    UPDATE trades
                    SET highest_price = ?, scaled_out = ?, stop_loss = ?, quantity = ?
                    WHERE id = ?
                    """,
                    (highest_price, scaled_out, stop_loss, quantity, trade_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE trades
                    SET highest_price = ?, scaled_out = ?, stop_loss = ?
                    WHERE id = ?
                    """,
                    (highest_price, scaled_out, stop_loss, trade_id),
                )
            conn.commit()
        logger.debug(
            "Updated trade trailing state | id={} | highest_price=${:.2f} | scaled_out={} | SL=${:.2f}",
            trade_id, highest_price, scaled_out, stop_loss
        )

    def log_trade_close(
        self,
        trade_id: int,
        exit_price: float,
        pnl: float,
        exit_reason: str,
    ) -> bool:
        """
        Update an existing trade record with exit data.

        The UPDATE is guarded on ``status = 'open'`` so two threads racing to
        close the same trade cannot both succeed — exactly one caller gets
        True and may apply the balance credit.

        Args:
            trade_id: The ID returned by log_trade_open()
            exit_price: Price at which the position was closed
            pnl: Profit or loss in USDT (negative for losses)
            exit_reason: 'stop_loss', 'take_profit', 'signal', or 'force_sell'

        Returns:
            True if the trade was open and is now closed; False if it was
            already closed (duplicate close attempt).
        """
        now = self._now()
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                UPDATE trades
                SET exit_price = ?, pnl = ?, status = 'closed',
                    exit_reason = ?, exit_time = ?
                WHERE id = ? AND status = 'open'
                """,
                (exit_price, pnl, exit_reason, now, trade_id),
            )
            conn.commit()
            closed = cursor.rowcount > 0

        if not closed:
            logger.warning(
                "Duplicate close ignored | id={} | trade is not open", trade_id
            )
            return False

        emoji = "✅" if pnl >= 0 else "❌"
        logger.info(
            "{} Trade closed | id={} | PnL=${:.2f} | reason={}",
            emoji, trade_id, pnl, exit_reason
        )
        return True

    # ─── Account State ─────────────────────────────────────────────────────────

    def save_balance(self, balance: float) -> None:
        """Persist the current cash balance (single-row account_state table)."""
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO account_state (id, balance, updated_at)
                VALUES (1, ?, ?)
                """,
                (balance, self._now()),
            )
            conn.commit()

    def load_balance(self) -> Optional[float]:
        """Return the last persisted cash balance, or None if never saved."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT balance FROM account_state WHERE id = 1"
            ).fetchone()
        return float(row["balance"]) if row is not None else None

    # ─── Risk Event Audit Trail ────────────────────────────────────────────────

    def log_risk_event(
        self,
        event_type: str,
        symbol: str = "",
        reason: str = "",
        correlation_id: str = "",
    ) -> None:
        """Persist a risk decision to the audit trail.

        event_type: 'trade_blocked', 'sizing_failed', 'pause', 'resume',
        'kill_switch', ... — the mandated record of every risk check
        rejection and halt transition (claude.md Logging & Observability).
        """
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO risk_events (event_type, symbol, reason, created_at,
                                            correlation_id)
                   VALUES (?, ?, ?, ?, ?)""",
                (event_type, symbol, reason, self._now(), correlation_id),
            )
            conn.commit()

    def log_order(
        self,
        *,
        symbol: str,
        side: str,
        requested_qty: float,
        filled_qty: float,
        avg_price: float,
        cost: float,
        status: str,
        ok: bool,
        reason: str = "",
        exchange_order_id: str = "",
        correlation_id: str = "",
    ) -> None:
        """Append one order attempt (accepted OR rejected) to the audit log."""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO order_log
                   (timestamp, symbol, side, requested_qty, filled_qty, avg_price,
                    cost, status, ok, reason, exchange_order_id, correlation_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (self._now(), symbol, side, requested_qty, filled_qty, avg_price,
                 cost, status, 1 if ok else 0, reason, exchange_order_id,
                 correlation_id),
            )
            conn.commit()

    # ─── Pending orders (Task 3.3 idempotency / state recovery) ─────────────────
    def record_pending_order(
        self,
        *,
        client_order_id: str,
        symbol: str,
        side: str,
        action: str,
        requested_qty: float,
        correlation_id: str = "",
    ) -> None:
        """Persist an action's clientOrderId with status='pending' BEFORE the API
        call, so a crash between write and fill is recoverable on restart."""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO pending_orders
                   (client_order_id, timestamp, symbol, side, action,
                    requested_qty, status, exchange_order_id, correlation_id)
                   VALUES (?, ?, ?, ?, ?, ?, 'pending', '', ?)""",
                (client_order_id, self._now(), symbol, side, action,
                 requested_qty, correlation_id),
            )
            conn.commit()

    def get_pending_orders(self) -> list[dict]:
        """All orders still marked 'pending' (unresolved at last write)."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM pending_orders WHERE status = 'pending' ORDER BY timestamp"
            ).fetchall()
        return [dict(row) for row in rows]

    def resolve_pending_order(
        self, client_order_id: str, status: str, exchange_order_id: str = ""
    ) -> None:
        """Mark a pending order resolved (filled / cancelled / not_placed / failed)."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE pending_orders SET status = ?, exchange_order_id = ? "
                "WHERE client_order_id = ?",
                (status, exchange_order_id, client_order_id),
            )
            conn.commit()

    def log_shadow_decision(
        self,
        *,
        symbol: str,
        decision: str,
        price: float,
        expected_slippage: float,
        expected_fill: float,
        confidence: float = 0.0,
        reason: str = "",
        correlation_id: str = "",
        strategy: str = "",
        timeframe: str = "",
    ) -> None:
        """Record one shadow-mode decision (Task 5.1) — signal + expected fill,
        NO order placed. ``strategy``/``timeframe`` tag the row so parity can
        filter (e.g. a 5m rsi_reversal stress test vs a 4h trend_following run)."""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO shadow_decisions
                   (timestamp, symbol, decision, price, expected_slippage,
                    expected_fill, confidence, reason, correlation_id, strategy, timeframe)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (self._now(), symbol, decision, price, expected_slippage,
                 expected_fill, confidence, reason, correlation_id, strategy, timeframe),
            )
            conn.commit()

    def get_shadow_decisions(self, limit: int = 100, *, strategy: str | None = None,
                             timeframe: str | None = None) -> list[dict]:
        """Recent shadow-mode decisions (newest first), optionally filtered by
        strategy and/or timeframe."""
        where, params = [], []
        if strategy is not None:
            where.append("strategy = ?")
            params.append(strategy)
        if timeframe is not None:
            where.append("timeframe = ?")
            params.append(timeframe)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        params.append(limit)
        with self._get_conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM shadow_decisions {clause} ORDER BY timestamp DESC LIMIT ?",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def get_open_trades(self) -> list[dict]:
        """Return all currently open positions."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status = 'open' ORDER BY entry_time DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_trade(self, trade_id: int) -> Optional[dict]:
        """Fetch a specific trade by ID."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM trades WHERE id = ?", (trade_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_trade_history(self, limit: int = 50) -> list[dict]:
        """Return the most recent closed trades."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM trades
                WHERE status = 'closed'
                ORDER BY exit_time DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    # ─── Signal Logging ────────────────────────────────────────────────────────

    def log_signal(
        self,
        symbol: str,
        ml_score: Optional[float],
        llm_score: Optional[float],
        sentiment_score: Optional[float],
        combined_score: float,
        confidence: float,
        decision: str,
        correlation_id: str = "",
    ) -> None:
        """
        Record every AI decision for a full audit trail.
        Called on every candle, regardless of whether a trade is taken.
        The correlation_id joins this signal to the trade/risk events of
        the same candle cycle (claude.md Logging & Observability).
        """
        now = self._now()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO signals
                    (timestamp, symbol, ml_score, llm_score, sentiment_score,
                     combined_score, confidence, decision, correlation_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (now, symbol, ml_score, llm_score, sentiment_score,
                 combined_score, confidence, decision, correlation_id),
            )
            conn.commit()

    # ─── Performance Analytics ─────────────────────────────────────────────────

    def get_daily_pnl(self, date: Optional[str] = None) -> float:
        """
        Get the net PnL for a given date (default: today).

        Returns:
            Net PnL in USDT (negative = net loss)
        """
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        with self._get_conn() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(SUM(pnl), 0.0) as net_pnl
                FROM trades
                WHERE status = 'closed'
                  AND date(exit_time) = ?
                """,
                (date,),
            ).fetchone()
        return float(row["net_pnl"]) if row else 0.0

    def get_stats(self) -> dict:
        """
        Return overall performance statistics.

        Returns:
            Dict with total_trades, wins, losses, win_rate, total_pnl
        """
        with self._get_conn() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*)                          AS total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) AS losses,
                    COALESCE(SUM(pnl), 0.0)           AS total_pnl,
                    COALESCE(AVG(CASE WHEN pnl > 0 THEN pnl END), 0.0) AS avg_win,
                    COALESCE(AVG(CASE WHEN pnl <= 0 THEN pnl END), 0.0) AS avg_loss
                FROM trades
                WHERE status = 'closed'
                """
            ).fetchone()

        stats = dict(row) if row else {}
        total = stats.get("total_trades", 0)
        wins = stats.get("wins", 0)
        stats["win_rate"] = (wins / total * 100) if total > 0 else 0.0
        return stats

    def get_consecutive_losses(self) -> int:
        """
        Return the number of consecutive losing trades in the recent history.
        Stops counting as soon as a winning trade (pnl > 0) is encountered.
        """
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT pnl FROM trades
                WHERE status = 'closed'
                ORDER BY exit_time DESC
                LIMIT 10
                """
            ).fetchall()

        consecutive_losses = 0
        for row in rows:
            pnl_val = row["pnl"]
            if pnl_val < 0:
                consecutive_losses += 1
            else:
                break
        return consecutive_losses

    def get_open_position_count(self) -> int:
        """Count of currently open positions."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM trades WHERE status = 'open'"
            ).fetchone()
        return int(row["cnt"]) if row else 0

    def log_market_structure(
        self,
        *,
        symbol: str,
        open_interest: Optional[float],
        taker_ratio: Optional[float],
        funding_rate: Optional[float],
    ) -> None:
        """Append one market-structure observation (per pair per cycle).

        Binance serves only ~30 days of open-interest / taker-flow history,
        so backtesting these features requires recording them ourselves —
        this table accumulates the series during the paper run (roadmap
        Step 3). NULLs are stored as NULLs: absence is data.
        """
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO market_structure
                   (timestamp, symbol, open_interest, taker_ratio, funding_rate)
                   VALUES (?, ?, ?, ?, ?)""",
                (self._now(), symbol, open_interest, taker_ratio, funding_rate),
            )
            conn.commit()

    def save_daily_summary(
        self,
        starting_balance: float,
        ending_balance: float,
        date: Optional[str] = None,
    ) -> None:
        """Snapshot end-of-day stats into daily_summary table."""
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        stats = self.get_stats()
        net_pnl = self.get_daily_pnl(date)

        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO daily_summary
                    (date, starting_balance, ending_balance,
                     total_trades, wins, losses, net_pnl)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    date,
                    starting_balance,
                    ending_balance,
                    stats.get("total_trades", 0),
                    stats.get("wins", 0),
                    stats.get("losses", 0),
                    net_pnl,
                ),
            )
            conn.commit()

        logger.info(
            "Daily summary saved | date={} | PnL=${:.2f} | Balance=${:.2f}",
            date, net_pnl, ending_balance
        )

    # ─── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _now() -> str:
        """Current UTC time as ISO 8601 string."""
        return datetime.now(timezone.utc).isoformat()
