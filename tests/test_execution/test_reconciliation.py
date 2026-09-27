"""Task 3.2 — dead-man's switch heartbeat + reconciliation divergence detection."""
from datetime import UTC, datetime


from execution.reconciliation import (
    DeadMansSwitch,
    Reconciler,
    diff_open_orders,
    diff_positions,
    reconcile,
)


# ─── Pure divergence detection ─────────────────────────────────────────────────
class TestDiffOrders:
    def test_in_sync_is_empty(self):
        assert diff_open_orders({"a", "b"}, {"a", "b"}) == []

    def test_orphan_on_exchange(self):
        d = diff_open_orders(set(), {"x99"})
        assert len(d) == 1 and d[0].kind == "orphan_order" and "x99" in d[0].detail

    def test_missing_from_exchange(self):
        d = diff_open_orders({"loc1"}, set())
        assert len(d) == 1 and d[0].kind == "missing_order"


class TestDiffPositions:
    def test_in_sync_is_empty(self):
        assert diff_positions({"BTC": 1.0}, {"BTC": 1.0}) == []

    def test_orphan_position_on_exchange(self):
        d = diff_positions({}, {"ETH": 2.0})
        assert [x.kind for x in d] == ["orphan_position"]

    def test_missing_position_locally_tracked(self):
        d = diff_positions({"BTC": 0.5}, {})
        assert [x.kind for x in d] == ["missing_position"]

    def test_quantity_mismatch_beyond_tolerance(self):
        assert diff_positions({"BTC": 1.0}, {"BTC": 1.0 + 1e-9}) == []       # within tol
        d = diff_positions({"BTC": 1.0}, {"BTC": 0.7})
        assert [x.kind for x in d] == ["qty_mismatch"]

    def test_reconcile_combines_both(self):
        divs = reconcile({"o1"}, set(), {"BTC": 1.0}, {"ETH": 3.0})
        kinds = sorted(d.kind for d in divs)
        assert kinds == ["missing_order", "missing_position", "orphan_position"]


# ─── Dead-man's switch heartbeat ───────────────────────────────────────────────
class _FakeExec:
    def __init__(self, arm_ok=True):
        self.arm_ok = arm_ok
        self.arm_calls = []
        self._open_orders = []
        self._positions = {}

    def arm_dead_mans_switch(self, countdown_ms):
        self.arm_calls.append(countdown_ms)
        return self.arm_ok

    def fetch_open_orders(self, symbol=None):
        return self._open_orders

    def fetch_positions(self):
        return dict(self._positions)


class TestDeadMansSwitch:
    def test_beat_arms_the_countdown(self):
        ex = _FakeExec()
        dms = DeadMansSwitch(ex, interval_s=30, countdown_ms=120_000)
        assert dms.beat() is True
        assert ex.arm_calls == [120_000]

    def test_beat_reports_failure(self):
        ex = _FakeExec(arm_ok=False)
        assert DeadMansSwitch(ex).beat() is False

    def test_thread_start_stop_arms_at_least_once(self):
        ex = _FakeExec()
        dms = DeadMansSwitch(ex, interval_s=0.01)
        dms.start()
        import time
        time.sleep(0.05)
        dms.stop()
        assert len(ex.arm_calls) >= 1


# ─── Reconciler: halt + alert on divergence ────────────────────────────────────
class _FakeDB:
    def __init__(self, open_trades):
        self._t = open_trades

    def get_open_trades(self):
        return self._t


class _FakeRisk:
    def __init__(self):
        self.tripped = False
        self.trip_reason = None

    def circuit_breaker_tripped(self):
        return self.tripped

    def trip_circuit_breaker(self, reason, ts):
        self.tripped = True
        self.trip_reason = reason


class _FakeTelegram:
    def __init__(self):
        self.errors = []

    def send_error(self, msg):
        self.errors.append(msg)


_NOW = datetime(2026, 8, 1, tzinfo=UTC)


def _reconciler(exchange_orders, positions, open_trades, universe=None):
    ex = _FakeExec()
    ex._open_orders = [{"id": oid} for oid in exchange_orders]
    ex._positions = positions
    risk, tg = _FakeRisk(), _FakeTelegram()
    rec = Reconciler(executor=ex, db=_FakeDB(open_trades), risk_manager=risk,
                     telegram=tg, universe=universe)
    return rec, risk, tg


class TestReconcilerRunOnce:
    def test_in_sync_does_not_halt(self):
        rec, risk, tg = _reconciler(
            exchange_orders=[], positions={"BTC": 0.5},
            open_trades=[{"symbol": "BTC/USDT", "quantity": 0.5}],
        )
        assert rec.run_once(_NOW) == []
        assert risk.tripped is False and tg.errors == []

    def test_orphan_exchange_order_halts_and_alerts(self):
        rec, risk, tg = _reconciler(
            exchange_orders=["ghost1"], positions={}, open_trades=[],
        )
        divs = rec.run_once(_NOW)
        assert any(d.kind == "orphan_order" for d in divs)
        assert risk.tripped is True and "HALTED" in risk.trip_reason
        assert len(tg.errors) == 1 and "CRITICAL" in tg.errors[0]

    def test_remote_fill_divergence_position_mismatch_halts(self):
        # Local believes 1.0 BTC open; exchange only holds 0.4 (filled/sold remotely).
        rec, risk, tg = _reconciler(
            exchange_orders=[], positions={"BTC": 0.4},
            open_trades=[{"symbol": "BTC/USDT", "quantity": 1.0}],
            universe={"BTC"},
        )
        divs = rec.run_once(_NOW)
        assert any(d.kind == "qty_mismatch" for d in divs)
        assert risk.tripped is True

    def test_universe_filter_ignores_faucet_dust(self):
        # Exchange holds faucet ETH/BNB the bot never traded; with a BTC-only
        # universe those are ignored → no false divergence.
        rec, risk, tg = _reconciler(
            exchange_orders=[], positions={"BTC": 0.5, "ETH": 9.0, "BNB": 3.0},
            open_trades=[{"symbol": "BTC/USDT", "quantity": 0.5}],
            universe={"BTC"},
        )
        assert rec.run_once(_NOW) == []
        assert risk.tripped is False

    def test_does_not_double_trip_when_already_tripped(self):
        rec, risk, tg = _reconciler(exchange_orders=["ghost"], positions={}, open_trades=[])
        risk.tripped = True                    # already halted
        rec.run_once(_NOW)
        assert risk.trip_reason is None        # trip_circuit_breaker not called again
        assert len(tg.errors) == 1             # but the alert still fires
