"""Task 3.3 — idempotency + startup state recovery."""
from execution.live_executor import OrderResult
from execution.recovery import recover_pending_orders
from storage.trade_logger import TradeLogger


# ─── Storage: pending-order round-trip ─────────────────────────────────────────
class TestPendingOrderStorage:
    def test_record_get_resolve(self, tmp_path):
        db = TradeLogger(db_path=str(tmp_path / "t.db"))
        db.record_pending_order(client_order_id="nt_abc", symbol="BTC/USDT",
                                side="buy", action="entry", requested_qty=0.5)
        pending = db.get_pending_orders()
        assert len(pending) == 1
        assert pending[0]["client_order_id"] == "nt_abc"
        assert pending[0]["status"] == "pending" and pending[0]["action"] == "entry"

        db.resolve_pending_order("nt_abc", "filled", "EX123")
        assert db.get_pending_orders() == []          # no longer pending


# ─── Recovery logic ────────────────────────────────────────────────────────────
def _order(status, filled, oid="EX1", avg=100.0):
    return OrderResult(id=oid, symbol="BTC/USDT", side="buy", amount=1.0,
                       filled=filled, avg_price=avg, cost=filled * avg,
                       status=status, ok=filled > 0)


class FakeExec:
    """Records calls; NEVER placing an order is the property we assert."""
    def __init__(self, by_coid):
        self._by_coid = by_coid          # coid -> OrderResult | None | Exception
        self.cancelled = []
        self.placed = []                 # must stay EMPTY — recovery never re-sends

    def fetch_order_by_client_id(self, symbol, coid):
        v = self._by_coid[coid]
        if isinstance(v, Exception):
            raise v
        return v

    def cancel_order(self, order_id, symbol):
        self.cancelled.append(order_id)
        return True

    def place_market_order(self, *a, **k):
        self.placed.append((a, k))       # if this ever fires, recovery re-sent → bug
        return _order("closed", 1.0)


class FakeDB:
    def __init__(self, pending):
        self._pending = pending
        self.resolved = {}               # coid -> (status, exchange_id)

    def get_pending_orders(self):
        return [p for p in self._pending if p["client_order_id"] not in self.resolved]

    def resolve_pending_order(self, coid, status, exchange_order_id=""):
        self.resolved[coid] = (status, exchange_order_id)


def _pending(*coids):
    return [{"client_order_id": c, "symbol": "BTC/USDT"} for c in coids]


class TestRecovery:
    def test_filled_order_is_recorded_not_resent(self):
        db = FakeDB(_pending("c1"))
        ex = FakeExec({"c1": _order("closed", 1.0, oid="EX9")})
        out = recover_pending_orders(db=db, executor=ex)
        assert [o.action for o in out] == ["filled"]
        assert db.resolved["c1"] == ("filled", "EX9")
        assert ex.placed == []                        # NEVER re-sent
        assert ex.cancelled == []

    def test_unfilled_order_is_cancelled(self):
        db = FakeDB(_pending("c2"))
        ex = FakeExec({"c2": _order("open", 0.0, oid="EX2")})
        out = recover_pending_orders(db=db, executor=ex)
        assert [o.action for o in out] == ["cancelled"]
        assert ex.cancelled == ["EX2"]
        assert db.resolved["c2"][0] == "cancelled"
        assert ex.placed == []

    def test_order_that_never_landed_is_marked_not_placed(self):
        db = FakeDB(_pending("c3"))
        ex = FakeExec({"c3": None})                   # exchange has no record of it
        out = recover_pending_orders(db=db, executor=ex)
        assert [o.action for o in out] == ["not_placed"]
        assert db.resolved["c3"][0] == "not_placed"
        assert ex.cancelled == [] and ex.placed == []  # not cancelled, not re-sent

    def test_partial_fill_records_fill_and_cancels_remainder(self):
        db = FakeDB(_pending("c4"))
        ex = FakeExec({"c4": _order("open", 0.4, oid="EX4")})  # partially filled, still open
        out = recover_pending_orders(db=db, executor=ex)
        assert [o.action for o in out] == ["filled"]
        assert ex.cancelled == ["EX4"]                # remainder cancelled
        assert db.resolved["c4"] == ("filled", "EX4")

    def test_query_error_is_reported_and_left_pending(self):
        db = FakeDB(_pending("c5"))
        ex = FakeExec({"c5": RuntimeError("api down")})
        out = recover_pending_orders(db=db, executor=ex)
        assert [o.action for o in out] == ["error"]
        assert "c5" not in db.resolved                # untouched → retried next startup
        assert ex.placed == []

    def test_no_pending_is_noop(self):
        assert recover_pending_orders(db=FakeDB([]), executor=FakeExec({})) == []

    def test_multiple_pending_all_resolved_none_resent(self):
        db = FakeDB(_pending("a", "b", "c"))
        ex = FakeExec({"a": _order("closed", 1.0), "b": _order("open", 0.0),
                       "c": None})
        out = recover_pending_orders(db=db, executor=ex)
        assert {o.action for o in out} == {"filled", "cancelled", "not_placed"}
        assert ex.placed == []                        # the core guarantee
