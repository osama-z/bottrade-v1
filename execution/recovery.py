"""Startup state recovery (Roadmap Task 3.3).

Every real action records its clientOrderId to SQLite (status='pending') BEFORE
the API call. If the process crashes between that write and the fill being
recorded, this module — run once on startup, before any trading — asks the
exchange the true fate of each pending order and reconciles it:

    filled          → mark filled (record the fill)
    unfilled / open → cancel the orphan, mark cancelled
    never landed    → mark not_placed (safe — it never reached the exchange)

It NEVER re-sends an order: recovery only queries, cancels, and records. That is
the whole point — a restart can neither duplicate a fill nor resurrect a dead
order.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger


@dataclass(frozen=True)
class RecoveryOutcome:
    client_order_id: str
    action: str        # filled | cancelled | not_placed | error
    detail: str


def recover_pending_orders(*, db, executor) -> list[RecoveryOutcome]:
    """Resolve every pending order against the exchange. Returns the outcomes."""
    pending = db.get_pending_orders()
    if not pending:
        return []

    outcomes: list[RecoveryOutcome] = []
    for p in pending:
        coid = str(p.get("client_order_id"))
        symbol = str(p.get("symbol"))
        try:
            order = executor.fetch_order_by_client_id(symbol, coid)
        except Exception as e:
            outcomes.append(RecoveryOutcome(coid, "error", f"status query failed: {e}"))
            continue

        if order is None:
            # The order never reached the exchange (crash before/at send). Safe to
            # close it out — and we must NOT re-send it.
            db.resolve_pending_order(coid, "not_placed")
            outcomes.append(RecoveryOutcome(coid, "not_placed",
                                            "order never reached the exchange"))
        elif order.filled > 0:
            # It filled (fully or partially) while we were down — record the fill.
            if order.status == "open":
                executor.cancel_order(order.id, symbol)   # cancel any unfilled remainder
            db.resolve_pending_order(coid, "filled", order.id)
            outcomes.append(RecoveryOutcome(coid, "filled",
                                            f"filled {order.filled:g} @ {order.avg_price:g}"))
        else:
            # Still resting / unfilled → cancel the orphan (never leave it, never re-send).
            executor.cancel_order(order.id, symbol)
            db.resolve_pending_order(coid, "cancelled", order.id)
            outcomes.append(RecoveryOutcome(coid, "cancelled",
                                            f"unfilled ({order.status}) → cancelled"))

    logger.warning(
        "State recovery resolved {} pending order(s): {}",
        len(outcomes), ", ".join(f"{o.client_order_id}={o.action}" for o in outcomes),
    )
    return outcomes
