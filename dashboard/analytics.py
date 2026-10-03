"""
dashboard/analytics.py — SQLite analytics helpers used by the paper status command.

Reads directly from the SQLite database (no ORM, read-only queries).
Returns pandas DataFrames and statistics; no web dashboard UI is bundled.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


# Single source of truth for the DB location (audit V-71): the dashboard
# must read the same file the bot writes, regardless of launch directory.
from config.settings import settings
from risk import metrics

DB_PATH = Path(settings.database_path)


def _get_conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def db_exists(db_path: Path = DB_PATH) -> bool:
    return db_path.exists() and db_path.stat().st_size > 0


# ─── Trade Data ────────────────────────────────────────────────────────────────

def get_open_trades(db_path: Path = DB_PATH) -> pd.DataFrame:
    """All currently open positions."""
    if not db_exists(db_path):
        return pd.DataFrame()
    with _get_conn(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM trades WHERE status='open' ORDER BY entry_time DESC",
            conn,
        )
    return df


def get_trade_history(limit: int = 200, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Recent closed trades as a DataFrame."""
    if not db_exists(db_path):
        return pd.DataFrame()
    with _get_conn(db_path) as conn:
        df = pd.read_sql_query(
            """
            SELECT * FROM trades
            WHERE status='closed'
            ORDER BY exit_time DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )
    if not df.empty:
        df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
        df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    return df


# ─── Equity Curve ─────────────────────────────────────────────────────────────

def get_equity_curve(
    starting_balance: float = 10_000.0,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """Reconstruct equity curve from closed trades.

    Returns a DataFrame with columns: exit_time, pnl, equity.
    """
    trades = get_trade_history(limit=10_000, db_path=db_path)
    if trades.empty:
        return pd.DataFrame(columns=["exit_time", "pnl", "equity"])

    curve = trades[["exit_time", "pnl"]].copy()
    curve = curve.sort_values("exit_time")
    curve["equity"] = starting_balance + curve["pnl"].cumsum()
    return curve.reset_index(drop=True)


def get_drawdown_series(equity: pd.Series) -> pd.Series:
    """Compute running drawdown (%) from a cumulative equity series."""
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max * 100
    return drawdown


# ─── Performance Stats ─────────────────────────────────────────────────────────

def get_stats(db_path: Path = DB_PATH) -> dict:
    """Return overall performance metrics."""
    if not db_exists(db_path):
        return _empty_stats()

    with _get_conn(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*)                                        AS total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)       AS wins,
                SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END)      AS losses,
                COALESCE(SUM(pnl), 0.0)                        AS total_pnl,
                COALESCE(AVG(CASE WHEN pnl > 0 THEN pnl END), 0.0) AS avg_win,
                COALESCE(AVG(CASE WHEN pnl <= 0 THEN pnl END), 0.0) AS avg_loss,
                COALESCE(MAX(pnl), 0.0)                        AS best_trade,
                COALESCE(MIN(pnl), 0.0)                        AS worst_trade
            FROM trades WHERE status='closed'
            """
        ).fetchone()

    stats = dict(row) if row else _empty_stats()
    total = stats.get("total_trades", 0) or 0
    wins = stats.get("wins", 0) or 0
    avg_win = stats.get("avg_win", 0) or 0
    avg_loss = abs(stats.get("avg_loss", 0) or 0)

    stats["win_rate"] = (wins / total * 100) if total > 0 else 0.0
    stats["profit_factor"] = (
        (avg_win * wins) / (avg_loss * (total - wins))
        if (avg_loss > 0 and (total - wins) > 0)
        else 0.0
    )
    return stats


def get_sharpe_ratio(
    starting_balance: float = 10_000.0,
    periods_per_year: int = 8760,
    db_path: Path = DB_PATH,
) -> float:
    """Annualised Sharpe Ratio from hourly PnL series.

    Returns 0.0 if insufficient data.
    """
    curve = get_equity_curve(starting_balance, db_path)
    if len(curve) < 2:
        return 0.0
    returns = curve["equity"].pct_change().dropna()
    # Single Sharpe definition (risk/metrics.py) — annualised variant.
    return metrics.sharpe_annualized(returns.tolist(), periods_per_year)


def get_max_drawdown(
    starting_balance: float = 10_000.0,
    db_path: Path = DB_PATH,
) -> float:
    """Maximum drawdown as a negative percentage."""
    curve = get_equity_curve(starting_balance, db_path)
    if curve.empty:
        return 0.0
    dd = get_drawdown_series(curve["equity"])
    return float(dd.min())


# ─── Signal Audit Log ─────────────────────────────────────────────────────────

def get_signal_log(limit: int = 100, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Return AI signal audit log."""
    if not db_exists(db_path):
        return pd.DataFrame()
    with _get_conn(db_path) as conn:
        df = pd.read_sql_query(
            """
            SELECT * FROM signals
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


# ─── Daily Summary ─────────────────────────────────────────────────────────────

def get_daily_summary(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Daily summary table as a DataFrame."""
    if not db_exists(db_path):
        return pd.DataFrame()
    with _get_conn(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM daily_summary ORDER BY date DESC LIMIT 30",
            conn,
        )
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _empty_stats() -> dict:
    return {
        "total_trades": 0, "wins": 0, "losses": 0,
        "total_pnl": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
        "best_trade": 0.0, "worst_trade": 0.0,
        "win_rate": 0.0, "profit_factor": 0.0,
    }
