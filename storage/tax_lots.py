"""Tax-lot accounting ledger (Roadmap Task 6.1).

A tax-compliant, append-only ledger of acquisitions (lots) and disposals (sales
matched to lots), with FIFO and Specific-ID methods and realized/unrealized P&L.

Immutability: rows are NEVER deleted and quantities are NEVER rewritten — a
lot's *remaining* quantity is DERIVED from the disposals booked against it, so
history can't be altered. Every row carries ``created_at`` and an ``is_amended``
flag; a correction is a NEW row (``amends_id`` → the superseded row), with the
superseded row's flag set — the only sanctioned mutation, a marking not a
rewrite. Retention policy: 7 years (never hard-deleted).

The matching math (``match_fifo`` / ``match_specific`` / ``realized_pnl_of_matches``)
is pure so it is unit-tested without a database.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

_EPS = 1e-9


@dataclass(frozen=True)
class Lot:
    lot_id: int
    symbol: str
    remaining: float
    cost_basis: float          # per-unit acquisition price
    acquired_at: str


@dataclass(frozen=True)
class Match:
    lot_id: int
    quantity: float
    cost_basis: float          # per-unit
    acquired_at: str


@dataclass(frozen=True)
class DisposalResult:
    symbol: str
    quantity: float
    sell_price: float
    proceeds: float
    cost_basis_total: float
    realized_pnl: float
    matches: list[Match]


# ─── Pure matching / P&L math ──────────────────────────────────────────────────
def match_fifo(open_lots: list[Lot], sell_qty: float) -> list[Match]:
    """Match ``sell_qty`` against ``open_lots`` OLDEST-FIRST. Raises if the open
    lots can't cover the sale (no naked shorting in the ledger)."""
    remaining = float(sell_qty)
    matches: list[Match] = []
    for lot in open_lots:                       # caller supplies oldest-first
        if remaining <= _EPS:
            break
        take = min(lot.remaining, remaining)
        if take <= _EPS:
            continue
        matches.append(Match(lot.lot_id, take, lot.cost_basis, lot.acquired_at))
        remaining -= take
    if remaining > _EPS:
        raise ValueError(f"insufficient open lots to cover {sell_qty}: short by {remaining:g}")
    return matches


def match_specific(open_lots: list[Lot], sell_qty: float, lot_id: int) -> list[Match]:
    """Match ``sell_qty`` against one chosen lot (Specific-ID method)."""
    lot = next((lot_ for lot_ in open_lots if lot_.lot_id == lot_id), None)
    if lot is None:
        raise ValueError(f"lot {lot_id} is not open")
    if lot.remaining + _EPS < float(sell_qty):
        raise ValueError(f"lot {lot_id} has {lot.remaining:g}, cannot dispose {sell_qty:g}")
    return [Match(lot_id, float(sell_qty), lot.cost_basis, lot.acquired_at)]


def realized_pnl_of_matches(matches: list[Match], sell_price: float) -> float:
    """Realized P&L = Σ qty · (sell_price − cost_basis) over the matched lots."""
    return sum(m.quantity * (float(sell_price) - m.cost_basis) for m in matches)


# ─── Persistent ledger ─────────────────────────────────────────────────────────
class TaxLotLedger:
    """Append-only SQLite lot/disposal ledger with FIFO + Specific-ID methods."""

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()
        logger.debug("TaxLotLedger at {}", self.db_path)

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def _ensure_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS tax_lots (
                lot_id         INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol         TEXT NOT NULL,
                quantity       REAL NOT NULL,
                cost_basis     REAL NOT NULL,
                acquired_at    TEXT NOT NULL,
                created_at     TEXT NOT NULL,
                is_amended     INTEGER NOT NULL DEFAULT 0,
                amends_id      INTEGER,
                correlation_id TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS tax_disposals (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol         TEXT NOT NULL,
                lot_id         INTEGER NOT NULL,
                quantity       REAL NOT NULL,
                sell_price     REAL NOT NULL,
                cost_basis     REAL NOT NULL,
                proceeds       REAL NOT NULL,
                realized_pnl   REAL NOT NULL,
                acquired_at    TEXT,
                disposed_at    TEXT NOT NULL,
                holding_days   INTEGER,
                method         TEXT NOT NULL,
                created_at     TEXT NOT NULL,
                is_amended     INTEGER NOT NULL DEFAULT 0,
                amends_id      INTEGER,
                correlation_id TEXT DEFAULT ''
            );
            """
        )
        self._conn.commit()

    # ── Acquire (every trade gets a lot id) ───────────────────────────────────
    def acquire(self, symbol: str, quantity: float, price: float,
                acquired_at: str | None = None, correlation_id: str = "") -> int:
        cur = self._conn.execute(
            """INSERT INTO tax_lots (symbol, quantity, cost_basis, acquired_at,
                                     created_at, correlation_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (symbol, float(quantity), float(price), acquired_at or self._now(),
             self._now(), correlation_id),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def _open_lots(self, symbol: str) -> list[Lot]:
        rows = self._conn.execute(
            """SELECT l.lot_id, l.symbol, l.quantity, l.cost_basis, l.acquired_at,
                      COALESCE((SELECT SUM(d.quantity) FROM tax_disposals d
                                WHERE d.lot_id = l.lot_id AND d.is_amended = 0), 0) AS disposed
               FROM tax_lots l
               WHERE l.symbol = ? AND l.is_amended = 0
               ORDER BY l.acquired_at ASC, l.lot_id ASC""",
            (symbol,),
        ).fetchall()
        lots = []
        for r in rows:
            remaining = float(r["quantity"]) - float(r["disposed"])
            if remaining > _EPS:
                lots.append(Lot(int(r["lot_id"]), r["symbol"], remaining,
                                float(r["cost_basis"]), r["acquired_at"]))
        return lots

    def open_lots(self, symbol: str) -> list[Lot]:
        return self._open_lots(symbol)

    # ── Dispose (match against lots, book realized P&L) ───────────────────────
    def dispose(self, symbol: str, quantity: float, price: float,
                *, method: str = "fifo", lot_id: int | None = None,
                disposed_at: str | None = None, correlation_id: str = "") -> DisposalResult:
        disposed_at = disposed_at or self._now()
        open_lots = self._open_lots(symbol)
        if method == "specific":
            if lot_id is None:
                raise ValueError("method='specific' requires a lot_id")
            matches = match_specific(open_lots, quantity, lot_id)
        elif method == "fifo":
            matches = match_fifo(open_lots, quantity)
        else:
            raise ValueError(f"unknown accounting method {method!r} (use 'fifo' or 'specific')")

        total_pnl = total_cost = total_proceeds = 0.0
        for m in matches:
            proceeds = m.quantity * float(price)
            cost = m.quantity * m.cost_basis
            pnl = proceeds - cost
            total_pnl += pnl
            total_cost += cost
            total_proceeds += proceeds
            self._conn.execute(
                """INSERT INTO tax_disposals
                   (symbol, lot_id, quantity, sell_price, cost_basis, proceeds,
                    realized_pnl, acquired_at, disposed_at, holding_days, method,
                    created_at, correlation_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (symbol, m.lot_id, m.quantity, float(price), m.cost_basis, proceeds,
                 pnl, m.acquired_at, disposed_at,
                 _holding_days(m.acquired_at, disposed_at), method, self._now(),
                 correlation_id),
            )
        self._conn.commit()
        return DisposalResult(symbol, float(quantity), float(price), total_proceeds,
                              total_cost, total_pnl, matches)

    # ── P&L reporting ─────────────────────────────────────────────────────────
    def realized_pnl(self, symbol: str | None = None) -> float:
        if symbol is None:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(realized_pnl),0) p FROM tax_disposals WHERE is_amended=0"
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(realized_pnl),0) p FROM tax_disposals "
                "WHERE symbol=? AND is_amended=0", (symbol,)
            ).fetchone()
        return float(row["p"])

    def unrealized_pnl(self, symbol: str, current_price: float) -> float:
        return sum(lot.remaining * (float(current_price) - lot.cost_basis)
                   for lot in self._open_lots(symbol))

    def disposals(self, symbol: str | None = None) -> list[dict]:
        if symbol is None:
            rows = self._conn.execute(
                "SELECT * FROM tax_disposals ORDER BY id").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM tax_disposals WHERE symbol=? ORDER BY id", (symbol,)).fetchall()
        return [dict(r) for r in rows]

    # ── Amendment (immutable: supersede, never rewrite) ───────────────────────
    def amend_disposal(self, disposal_id: int, *, realized_pnl: float,
                       reason_correlation_id: str = "") -> int:
        """Correct a booked disposal WITHOUT rewriting it: flag the original as
        amended and insert a new superseding row referencing it. History stays."""
        orig = self._conn.execute(
            "SELECT * FROM tax_disposals WHERE id=?", (disposal_id,)).fetchone()
        if orig is None:
            raise ValueError(f"disposal {disposal_id} not found")
        self._conn.execute("UPDATE tax_disposals SET is_amended=1 WHERE id=?", (disposal_id,))
        cur = self._conn.execute(
            """INSERT INTO tax_disposals
               (symbol, lot_id, quantity, sell_price, cost_basis, proceeds,
                realized_pnl, acquired_at, disposed_at, holding_days, method,
                created_at, amends_id, correlation_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (orig["symbol"], orig["lot_id"], orig["quantity"], orig["sell_price"],
             orig["cost_basis"], orig["proceeds"], float(realized_pnl),
             orig["acquired_at"], orig["disposed_at"], orig["holding_days"],
             orig["method"], self._now(), disposal_id, reason_correlation_id),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def close(self) -> None:
        self._conn.close()


def _holding_days(acquired_at: str | None, disposed_at: str) -> int | None:
    if not acquired_at:
        return None
    try:
        a = datetime.fromisoformat(acquired_at)
        d = datetime.fromisoformat(disposed_at)
        return (d - a).days
    except ValueError:
        return None
