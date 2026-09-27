"""Dead-man's switch + reconciliation loop (Roadmap Task 3.2).

Two safety nets against a crashed/desynced bot leaving orphaned state on the
exchange:

1. DeadMansSwitch — a heartbeat that re-arms the exchange auto-cancel-all
   countdown every ~30s. If the process dies, the countdown expires and the
   exchange cancels all open orders (no re-arm from a dead process).
2. Reconciler — every ~60s, compares the exchange's live open orders + held
   positions against the local SQLite state. Any divergence (orphan/missing
   order, orphan/missing position, quantity mismatch) HALTS trading via the
   circuit breaker and pushes a critical alert.

The comparison itself is pure (``reconcile``) so it is unit-testable without an
exchange or a database.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import UTC, datetime

from loguru import logger


# ─── Pure divergence detection ─────────────────────────────────────────────────
@dataclass(frozen=True)
class Divergence:
    kind: str        # orphan_order | missing_order | orphan_position | missing_position | qty_mismatch
    detail: str


def diff_open_orders(local_ids, exchange_ids) -> list[Divergence]:
    """Orders open on the exchange but not tracked locally (orphans), and orders
    tracked locally but no longer open on the exchange (filled/cancelled elsewhere)."""
    local, exch = set(map(str, local_ids)), set(map(str, exchange_ids))
    out = [Divergence("orphan_order", f"order {oid} is open on the exchange but not tracked locally")
           for oid in sorted(exch - local)]
    out += [Divergence("missing_order", f"order {oid} is tracked locally but not open on the exchange")
            for oid in sorted(local - exch)]
    return out


def diff_positions(local, exchange, *, qty_tol: float = 1e-6) -> list[Divergence]:
    """Compare per-asset held quantities ({asset: qty}). Detects assets held on
    the exchange with no local trade (orphan), local trades not held on the
    exchange (missing), and quantity mismatches beyond ``qty_tol``."""
    out: list[Divergence] = []
    for sym in sorted(exchange.keys() - local.keys()):
        out.append(Divergence("orphan_position",
                              f"{sym} held on exchange ({exchange[sym]:g}) with no open trade locally"))
    for sym in sorted(local.keys() - exchange.keys()):
        out.append(Divergence("missing_position",
                              f"{sym} open locally ({local[sym]:g}) but not held on the exchange"))
    for sym in sorted(local.keys() & exchange.keys()):
        if abs(float(local[sym]) - float(exchange[sym])) > qty_tol:
            out.append(Divergence("qty_mismatch",
                                  f"{sym} local {local[sym]:g} vs exchange {exchange[sym]:g}"))
    return out


def reconcile(local_orders, exchange_orders, local_positions, exchange_positions,
              *, qty_tol: float = 1e-6) -> list[Divergence]:
    """All divergences between local state and the exchange (empty = in sync)."""
    return (diff_open_orders(local_orders, exchange_orders)
            + diff_positions(local_positions, exchange_positions, qty_tol=qty_tol))


def _base_asset(symbol: str) -> str:
    return symbol.split("/")[0] if "/" in symbol else symbol


# ─── Dead-man's switch (heartbeat) ─────────────────────────────────────────────
class DeadMansSwitch:
    """Re-arms the exchange auto-cancel-all countdown on a heartbeat."""

    def __init__(self, executor, *, interval_s: float = 30.0, countdown_ms: int = 120_000) -> None:
        self._executor = executor
        self.interval_s = interval_s
        self.countdown_ms = countdown_ms
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def beat(self) -> bool:
        """One heartbeat: re-arm the countdown. Returns whether it was accepted."""
        return bool(self._executor.arm_dead_mans_switch(self.countdown_ms))

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="dead-mans-switch", daemon=True)
        self._thread.start()
        logger.info("Dead-man's switch armed — heartbeat every {}s, countdown {}ms",
                    self.interval_s, self.countdown_ms)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval_s + 1)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.beat()
            except Exception as e:  # pragma: no cover - defensive
                logger.error("dead-man's switch heartbeat error: {}", e)
            self._stop.wait(self.interval_s)


# ─── Reconciliation loop ───────────────────────────────────────────────────────
class Reconciler:
    """Periodically reconciles exchange state with the local DB; halts on drift.

    Args:
        executor:     provides fetch_open_orders() and fetch_positions().
        db:           provides get_open_trades().
        risk_manager: tripped (halt) on divergence; optional.
        telegram:     send_error() for the critical alert; optional.
        universe:     base assets to reconcile positions over (ignore faucet
                      dust); defaults to the base assets of local open trades.
    """

    def __init__(self, *, executor, db, risk_manager=None, telegram=None,
                 universe=None, interval_s: float = 60.0, qty_tol: float = 1e-6) -> None:
        self._executor = executor
        self._db = db
        self._risk = risk_manager
        self._telegram = telegram
        self._universe = {a.upper() for a in universe} if universe else None
        self.interval_s = interval_s
        self.qty_tol = qty_tol
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def run_once(self, now: datetime | None = None) -> list[Divergence]:
        """Fetch both sides, diff them, and halt + alert on any divergence."""
        now = now or datetime.now(UTC)

        exchange_orders = {str(o.get("id")) for o in self._executor.fetch_open_orders()}
        local_orders: set[str] = set()   # a market/spot bot holds no resting orders it tracks

        local_positions = {}
        for t in self._db.get_open_trades():
            sym = t.get("symbol")
            if sym:
                local_positions[_base_asset(str(sym)).upper()] = float(t.get("quantity") or 0.0)

        held = {a.upper(): q for a, q in self._executor.fetch_positions().items()}
        universe = self._universe or set(local_positions)
        exchange_positions = {a: q for a, q in held.items() if a in universe}

        divergences = reconcile(local_orders, exchange_orders,
                                local_positions, exchange_positions, qty_tol=self.qty_tol)
        if divergences:
            self._halt_and_alert(divergences, now)
        else:
            logger.debug("Reconciliation OK — exchange and local state agree")
        return divergences

    def _halt_and_alert(self, divergences: list[Divergence], now: datetime) -> None:
        summary = f"Reconciliation divergence: {len(divergences)} issue(s) — trading HALTED"
        logger.critical("{}\n{}", summary, "\n".join(d.detail for d in divergences))
        if self._risk is not None and not self._risk.circuit_breaker_tripped():
            self._risk.trip_circuit_breaker(summary, now)
        if self._telegram is not None:
            try:
                self._telegram.send_error("🚨 CRITICAL — " + summary + "\n"
                                          + "\n".join(f"• {d.detail}" for d in divergences))
            except Exception as e:  # pragma: no cover - alert must never crash the loop
                logger.error("reconciliation alert failed: {}", e)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="reconciler", daemon=True)
        self._thread.start()
        logger.info("Reconciliation loop started — every {}s", self.interval_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval_s + 1)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception as e:  # pragma: no cover - defensive
                logger.error("reconciliation error: {}", e)
            self._stop.wait(self.interval_s)
