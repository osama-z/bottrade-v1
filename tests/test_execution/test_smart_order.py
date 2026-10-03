"""Task 3.1 — smart order execution: post-only, TWAP slicing, depth caps."""
import pytest

from execution.smart_order import (
    depth_capped_quantity,
    post_only_price,
    top_n_liquidity,
    twap_plan,
)


# ─── Pure helpers (no exchange) ────────────────────────────────────────────────
class TestDepthCap:
    _BOOK = {"bids": [[100, 3], [99, 3], [98, 3], [97, 3], [96, 3], [95, 99]],
             "asks": [[101, 2], [102, 2], [103, 2], [104, 2], [105, 2], [106, 99]]}

    def test_top_n_liquidity_sums_first_n(self):
        assert top_n_liquidity(self._BOOK["asks"], 5) == 10.0   # 2*5, ignores 6th
        assert top_n_liquidity(self._BOOK["bids"], 5) == 15.0   # 3*5

    def test_buy_caps_against_ask_depth(self):
        # 25% of top-5 asks (10) = 2.5; a 100-lot order is capped to 2.5.
        assert depth_capped_quantity(100, "buy", self._BOOK) == pytest.approx(2.5)

    def test_sell_caps_against_bid_depth(self):
        assert depth_capped_quantity(100, "sell", self._BOOK) == pytest.approx(3.75)  # 25% of 15

    def test_small_order_is_not_capped(self):
        assert depth_capped_quantity(1.0, "buy", self._BOOK) == 1.0

    def test_no_depth_returns_zero(self):
        assert depth_capped_quantity(5.0, "buy", {"bids": [], "asks": []}) == 0.0


class TestPostOnlyPrice:
    def test_buy_joins_the_bid_sell_joins_the_ask(self):
        assert post_only_price("buy", 100.0, 101.0) == (100.0, False)
        assert post_only_price("sell", 100.0, 101.0) == (101.0, False)

    def test_crossing_request_is_repriced_to_passive(self):
        # buy at 105 would cross the 101 ask → re-priced down to the 100 bid.
        assert post_only_price("buy", 100.0, 101.0, requested_price=105.0) == (100.0, True)
        # sell at 95 would cross the 100 bid → re-priced up to the 101 ask.
        assert post_only_price("sell", 100.0, 101.0, requested_price=95.0) == (101.0, True)

    def test_passive_request_is_respected(self):
        # A non-crossing buy request below the bid is kept (never above the bid).
        assert post_only_price("buy", 100.0, 101.0, requested_price=99.5) == (99.5, False)

    def test_bad_side_raises(self):
        with pytest.raises(ValueError):
            post_only_price("hold", 100.0, 101.0)


class TestTwapPlan:
    def test_small_order_single_chunk_no_delay(self):
        assert twap_plan(1.0, 300.0) == ([1.0], 0.0)

    def test_large_order_splits_into_threshold_sized_chunks(self):
        chunks, delay = twap_plan(20.0, 2000.0)   # ceil(2000/500) = 4 slices
        assert len(chunks) == 4
        assert sum(chunks) == pytest.approx(20.0)
        assert delay == 60.0

    def test_slice_count_is_capped_at_max(self):
        chunks, _ = twap_plan(100.0, 100_000.0, max_slices=10)   # would be 200 → capped 10
        assert len(chunks) == 10
        assert sum(chunks) == pytest.approx(100.0)

    def test_zero_quantity_is_empty(self):
        assert twap_plan(0.0, 9999.0) == ([], 0.0)
