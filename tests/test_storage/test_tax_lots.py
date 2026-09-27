"""Task 6.1 — tax-lot accounting: FIFO/Specific-ID matching, realized P&L, immutability."""
import pytest

from storage.tax_lots import (
    Lot,
    TaxLotLedger,
    match_fifo,
    match_specific,
    realized_pnl_of_matches,
)


def _lots(*specs):
    # specs: (lot_id, remaining, cost_basis)
    return [Lot(i, "BTC/USDT", rem, cb, f"2024-01-{i:02d}") for (i, rem, cb) in specs]


# ─── Pure FIFO / Specific matching ─────────────────────────────────────────────
class TestMatchFifo:
    def test_single_lot_partial(self):
        m = match_fifo(_lots((1, 10.0, 100.0)), 4.0)
        assert len(m) == 1 and m[0].lot_id == 1 and m[0].quantity == 4.0

    def test_consumes_oldest_first_across_lots(self):
        m = match_fifo(_lots((1, 3.0, 100.0), (2, 5.0, 120.0)), 6.0)
        assert [(x.lot_id, x.quantity) for x in m] == [(1, 3.0), (2, 3.0)]
        assert m[0].cost_basis == 100.0 and m[1].cost_basis == 120.0

    def test_exact_fill(self):
        m = match_fifo(_lots((1, 2.0, 100.0), (2, 2.0, 110.0)), 4.0)
        assert sum(x.quantity for x in m) == pytest.approx(4.0)

    def test_oversell_raises(self):
        with pytest.raises(ValueError, match="insufficient"):
            match_fifo(_lots((1, 1.0, 100.0)), 5.0)


class TestMatchSpecific:
    def test_picks_the_named_lot(self):
        m = match_specific(_lots((1, 5.0, 100.0), (2, 5.0, 120.0)), 3.0, lot_id=2)
        assert m[0].lot_id == 2 and m[0].cost_basis == 120.0

    def test_insufficient_in_named_lot_raises(self):
        with pytest.raises(ValueError):
            match_specific(_lots((1, 1.0, 100.0)), 5.0, lot_id=1)

    def test_unknown_lot_raises(self):
        with pytest.raises(ValueError, match="not open"):
            match_specific(_lots((1, 5.0, 100.0)), 1.0, lot_id=99)


class TestRealizedPnlMath:
    def test_pnl_is_qty_times_price_minus_basis(self):
        m = match_fifo(_lots((1, 100.0, 10.0), (2, 100.0, 12.0)), 150.0)
        # 100·(15−10) + 50·(15−12) = 500 + 150 = 650
        assert realized_pnl_of_matches(m, 15.0) == pytest.approx(650.0)

    def test_loss_when_price_below_basis(self):
        m = match_fifo(_lots((1, 10.0, 100.0)), 10.0)
        assert realized_pnl_of_matches(m, 90.0) == pytest.approx(-100.0)


# ─── Persistent ledger ─────────────────────────────────────────────────────────
def _ledger(tmp_path):
    return TaxLotLedger(db_path=str(tmp_path / "lots.db"))


class TestLedger:
    def test_every_acquisition_gets_a_lot_id(self, tmp_path):
        led = _ledger(tmp_path)
        a = led.acquire("BTC/USDT", 1.0, 100.0)
        b = led.acquire("BTC/USDT", 1.0, 110.0)
        assert isinstance(a, int) and b == a + 1

    def test_fifo_disposal_books_realized_pnl_and_leaves_remainder(self, tmp_path):
        led = _ledger(tmp_path)
        led.acquire("BTC/USDT", 100.0, 10.0, acquired_at="2024-01-01T00:00:00+00:00")
        led.acquire("BTC/USDT", 100.0, 12.0, acquired_at="2024-01-05T00:00:00+00:00")
        res = led.dispose("BTC/USDT", 150.0, 15.0, method="fifo",
                          disposed_at="2024-01-11T00:00:00+00:00")
        assert res.realized_pnl == pytest.approx(650.0)
        assert res.proceeds == pytest.approx(150.0 * 15.0)
        # 50 units of the 2nd lot remain open.
        open_lots = led.open_lots("BTC/USDT")
        assert len(open_lots) == 1 and open_lots[0].remaining == pytest.approx(50.0)
        assert led.realized_pnl("BTC/USDT") == pytest.approx(650.0)

    def test_unrealized_pnl_on_open_lots(self, tmp_path):
        led = _ledger(tmp_path)
        led.acquire("BTC/USDT", 100.0, 10.0)
        led.acquire("BTC/USDT", 100.0, 12.0)
        led.dispose("BTC/USDT", 150.0, 15.0)          # 50 @ basis 12 remain
        assert led.unrealized_pnl("BTC/USDT", 20.0) == pytest.approx(50.0 * (20.0 - 12.0))

    def test_specific_id_targets_a_lot(self, tmp_path):
        led = _ledger(tmp_path)
        l1 = led.acquire("BTC/USDT", 10.0, 10.0)
        l2 = led.acquire("BTC/USDT", 10.0, 30.0)
        res = led.dispose("BTC/USDT", 5.0, 25.0, method="specific", lot_id=l2)
        assert res.realized_pnl == pytest.approx(5.0 * (25.0 - 30.0))   # from the pricey lot
        # l1 untouched (10), l2 has 5 left.
        rem = {lot.lot_id: lot.remaining for lot in led.open_lots("BTC/USDT")}
        assert rem[l1] == pytest.approx(10.0) and rem[l2] == pytest.approx(5.0)

    def test_holding_period_days_recorded(self, tmp_path):
        led = _ledger(tmp_path)
        led.acquire("BTC/USDT", 1.0, 100.0, acquired_at="2024-01-01T00:00:00+00:00")
        led.dispose("BTC/USDT", 1.0, 120.0, disposed_at="2024-01-11T00:00:00+00:00")
        assert led.disposals("BTC/USDT")[0]["holding_days"] == 10

    def test_oversell_raises_and_books_nothing(self, tmp_path):
        led = _ledger(tmp_path)
        led.acquire("BTC/USDT", 1.0, 100.0)
        with pytest.raises(ValueError):
            led.dispose("BTC/USDT", 5.0, 120.0)
        assert led.disposals("BTC/USDT") == []        # nothing partially booked


class TestImmutability:
    def test_rows_carry_created_at_and_amended_flag(self, tmp_path):
        led = _ledger(tmp_path)
        led.acquire("BTC/USDT", 1.0, 100.0)
        led.dispose("BTC/USDT", 1.0, 120.0)
        d = led.disposals("BTC/USDT")[0]
        assert d["created_at"] and d["is_amended"] == 0

    def test_amendment_supersedes_without_deleting(self, tmp_path):
        led = _ledger(tmp_path)
        led.acquire("BTC/USDT", 1.0, 100.0)
        led.dispose("BTC/USDT", 1.0, 120.0)
        orig_id = led.disposals("BTC/USDT")[0]["id"]
        led.amend_disposal(orig_id, realized_pnl=999.0)
        rows = led.disposals("BTC/USDT")
        assert len(rows) == 2                                  # original NOT deleted
        original = next(r for r in rows if r["id"] == orig_id)
        correction = next(r for r in rows if r["amends_id"] == orig_id)
        assert original["is_amended"] == 1                     # superseded, flagged
        assert correction["realized_pnl"] == 999.0
        # Realized total counts only the live (non-amended) row.
        assert led.realized_pnl("BTC/USDT") == pytest.approx(999.0)

    def test_ledger_exposes_no_delete_method(self, tmp_path):
        led = _ledger(tmp_path)
        assert not hasattr(led, "delete_lot") and not hasattr(led, "delete_disposal")
