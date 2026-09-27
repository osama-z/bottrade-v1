"""Stage 2 tests — OrderBookManager and TickAggregator.

All tests are fully offline: no exchange connections, no network.
The WS feed is replayed from synthetic in-memory message sequences.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime

import pandas as pd
import pytest

from data.order_book import OrderBookError, OrderBookManager
from data.tick_aggregator import CompletedCandle, TickAggregator, merge_candle_into


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _snapshot_msgs() -> dict:
    """Synthetic REST snapshot — 5 bid levels, 5 ask levels."""
    return {
        "lastUpdateId": 1000,
        "timestamp": 1_700_000_000_000,
        "bids": [
            [50_000.0, 1.0],
            [49_999.0, 2.5],
            [49_998.0, 0.5],
            [49_997.0, 3.0],
            [49_996.0, 1.2],
        ],
        "asks": [
            [50_001.0, 0.8],
            [50_002.0, 1.5],
            [50_003.0, 2.0],
            [50_004.0, 0.3],
            [50_005.0, 4.0],
        ],
    }


def _diff_msgs() -> list[dict]:
    """Recorded WS diff sequence (6 messages)."""
    return [
        # 1. New ask at 50_001.50 — tightens spread
        {"u": 1001, "T": 1_700_000_001_000,
         "b": [], "a": [[50_001.5, 0.6]]},
        # 2. Remove best ask at 50_001.0 (qty = 0)
        {"u": 1002, "T": 1_700_000_002_000,
         "b": [], "a": [[50_001.0, 0.0]]},
        # 3. Large bid arrives at 50_000.5 — best bid improves
        {"u": 1003, "T": 1_700_000_003_000,
         "b": [[50_000.5, 5.0]], "a": []},
        # 4. Both sides update simultaneously
        {"u": 1004, "T": 1_700_000_004_000,
         "b": [[49_996.0, 0.0]],           # remove deep bid
         "a": [[50_005.0, 0.0]]},          # remove deep ask
        # 5. Best bid level size shrinks
        {"u": 1005, "T": 1_700_000_005_000,
         "b": [[50_000.5, 2.0]], "a": []},
        # 6. Best bid removed entirely — old best bid (50_000) becomes best
        {"u": 1006, "T": 1_700_000_006_000,
         "b": [[50_000.5, 0.0]], "a": []},
    ]


# ─── OrderBookManager tests ───────────────────────────────────────────────────

class TestOrderBookSnapshot:
    def test_best_bid_and_ask_after_snapshot(self) -> None:
        book = OrderBookManager("BTC/USDT")
        snap = _snapshot_msgs()
        book.apply_snapshot(
            bids=snap["bids"], asks=snap["asks"],
            last_update_id=snap["lastUpdateId"],
            timestamp_ms=snap["timestamp"],
        )
        assert book.best_bid() == 50_000.0
        assert book.best_ask() == 50_001.0

    def test_spread_is_ask_minus_bid(self) -> None:
        book = OrderBookManager("BTC/USDT")
        s = _snapshot_msgs()
        book.apply_snapshot(bids=s["bids"], asks=s["asks"],
                            last_update_id=s["lastUpdateId"], timestamp_ms=s["timestamp"])
        snap = book.snapshot()
        assert snap.spread == pytest.approx(1.0)

    def test_mid_price(self) -> None:
        book = OrderBookManager("BTC/USDT")
        s = _snapshot_msgs()
        book.apply_snapshot(bids=s["bids"], asks=s["asks"],
                            last_update_id=s["lastUpdateId"], timestamp_ms=s["timestamp"])
        snap = book.snapshot()
        assert snap.mid_price == pytest.approx(50_000.5)

    def test_snapshot_zero_qty_levels_excluded(self) -> None:
        """Levels with qty=0 in the snapshot must be silently ignored."""
        book = OrderBookManager("BTC/USDT")
        book.apply_snapshot(
            bids=[[50_000.0, 1.0], [49_999.0, 0.0]],   # second is qty=0
            asks=[[50_001.0, 0.5]],
            last_update_id=1, timestamp_ms=0,
        )
        top_bids = book.top_bids(10)
        prices = [p for p, _ in top_bids]
        assert 49_999.0 not in prices
        assert 50_000.0 in prices

    def test_depth_usdt_is_price_times_qty(self) -> None:
        book = OrderBookManager("ETH/USDT")
        book.apply_snapshot(
            bids=[[3_000.0, 2.0], [2_999.0, 1.0]],
            asks=[[3_001.0, 0.5]],
            last_update_id=1, timestamp_ms=0,
        )
        snap = book.snapshot()
        expected_bid_depth = 3_000.0 * 2.0 + 2_999.0 * 1.0
        assert snap.bid_depth_usdt == pytest.approx(expected_bid_depth)
        assert snap.ask_depth_usdt == pytest.approx(3_001.0 * 0.5)


class TestOrderBookDiff:
    def _make_book(self) -> OrderBookManager:
        book = OrderBookManager("BTC/USDT")
        s = _snapshot_msgs()
        book.apply_snapshot(bids=s["bids"], asks=s["asks"],
                            last_update_id=s["lastUpdateId"], timestamp_ms=s["timestamp"])
        return book

    def test_diff_raises_before_snapshot(self) -> None:
        book = OrderBookManager("BTC/USDT")
        with pytest.raises(OrderBookError, match="apply_snapshot"):
            book.apply_diff(bids=[], asks=[], last_update_id=1, timestamp_ms=0)

    def test_replay_full_sequence_best_bid_correct(self) -> None:
        """Replay all 6 diffs and verify best bid at each step."""
        book = self._make_book()
        diffs = _diff_msgs()

        # After snapshot: best bid = 50_000.0
        assert book.best_bid() == 50_000.0

        # msg 1: no bid changes
        d = diffs[0]
        book.apply_diff(bids=d["b"], asks=d["a"],
                        last_update_id=d["u"], timestamp_ms=d["T"])
        assert book.best_bid() == 50_000.0

        # msg 2: no bid changes
        d = diffs[1]
        book.apply_diff(bids=d["b"], asks=d["a"],
                        last_update_id=d["u"], timestamp_ms=d["T"])
        assert book.best_bid() == 50_000.0

        # msg 3: new bid at 50_000.5 → becomes new best bid
        d = diffs[2]
        book.apply_diff(bids=d["b"], asks=d["a"],
                        last_update_id=d["u"], timestamp_ms=d["T"])
        assert book.best_bid() == 50_000.5

        # msg 4: removes ask at 50_005; bid at 49_996 removed — best bid unchanged
        d = diffs[3]
        book.apply_diff(bids=d["b"], asks=d["a"],
                        last_update_id=d["u"], timestamp_ms=d["T"])
        assert book.best_bid() == 50_000.5

        # msg 5: best bid size shrinks from 5.0 to 2.0, price unchanged
        d = diffs[4]
        book.apply_diff(bids=d["b"], asks=d["a"],
                        last_update_id=d["u"], timestamp_ms=d["T"])
        assert book.best_bid() == 50_000.5
        top = dict(book.top_bids(1))
        assert top[50_000.5] == pytest.approx(2.0)

        # msg 6: best bid removed entirely → falls back to 50_000.0
        d = diffs[5]
        book.apply_diff(bids=d["b"], asks=d["a"],
                        last_update_id=d["u"], timestamp_ms=d["T"])
        assert book.best_bid() == 50_000.0

    def test_replay_full_sequence_best_ask_correct(self) -> None:
        book = self._make_book()
        diffs = _diff_msgs()

        # After snapshot: best ask = 50_001.0
        assert book.best_ask() == 50_001.0

        # msg 1: new ask at 50_001.5 — best ask unchanged (50_001.0 still there)
        d = diffs[0]
        book.apply_diff(bids=d["b"], asks=d["a"],
                        last_update_id=d["u"], timestamp_ms=d["T"])
        assert book.best_ask() == 50_001.0

        # msg 2: best ask at 50_001.0 removed → best ask becomes 50_001.5
        d = diffs[1]
        book.apply_diff(bids=d["b"], asks=d["a"],
                        last_update_id=d["u"], timestamp_ms=d["T"])
        assert book.best_ask() == 50_001.5

    def test_update_id_advances(self) -> None:
        book = self._make_book()
        assert book.snapshot().last_update_id == 1000
        d = _diff_msgs()[0]
        book.apply_diff(bids=d["b"], asks=d["a"],
                        last_update_id=d["u"], timestamp_ms=d["T"])
        assert book.snapshot().last_update_id == 1001

    def test_reset_clears_all_state(self) -> None:
        book = self._make_book()
        book.reset()
        assert not book.is_ready
        assert book.best_bid() is None
        assert book.best_ask() is None
        snap = book.snapshot()
        assert snap.spread is None

    def test_depth_usdt_changes_after_diff(self) -> None:
        book = self._make_book()
        before = book.snapshot().bid_depth_usdt
        # Remove the 3.0-unit bid at 49_997 (value = 149_991)
        book.apply_diff(bids=[[49_997.0, 0.0]], asks=[],
                        last_update_id=2000, timestamp_ms=1_700_000_010_000)
        after = book.snapshot().bid_depth_usdt
        assert after == pytest.approx(before - 49_997.0 * 3.0)


class TestOrderBookImbalance:
    def test_balanced_book_has_ratio_near_one(self) -> None:
        book = OrderBookManager("BTC/USDT", imbalance_band_pct=0.01)
        # Symmetric book: equal USDT on both sides within 1%
        mid = 50_000.0
        # Bids within 1% band: 49_500 to 50_000
        # Asks within 1% band: 50_000 to 50_500
        book.apply_snapshot(
            bids=[[mid - 1, 1.0], [mid - 2, 1.0]],
            asks=[[mid + 1, 1.0], [mid + 2, 1.0]],
            last_update_id=1, timestamp_ms=0,
        )
        snap = book.snapshot()
        assert snap.imbalance_ratio is not None
        assert 0.5 < snap.imbalance_ratio < 2.0  # roughly balanced

    def test_ask_heavy_book_has_ratio_above_one(self) -> None:
        book = OrderBookManager("BTC/USDT", imbalance_band_pct=0.01)
        mid = 50_000.0
        book.apply_snapshot(
            bids=[[mid - 1, 1.0]],
            asks=[[mid + 1, 10.0]],    # 10x more ask volume
            last_update_id=1, timestamp_ms=0,
        )
        snap = book.snapshot()
        assert snap.imbalance_ratio > 5.0

    def test_zero_bid_volume_returns_infinity(self) -> None:
        book = OrderBookManager("BTC/USDT", imbalance_band_pct=0.5)
        # Put all bids well outside the 50% band
        book.apply_snapshot(
            bids=[[1.0, 100.0]],   # far below mid
            asks=[[50_001.0, 1.0]],
            last_update_id=1, timestamp_ms=0,
        )
        snap = book.snapshot()
        assert snap.imbalance_ratio == float("inf")


class TestOrderBookMaxDepth:
    def test_prune_keeps_book_within_max_depth(self) -> None:
        book = OrderBookManager("BTC/USDT", max_depth=3)
        bids = [[50_000.0 - i, 1.0] for i in range(10)]
        asks = [[50_001.0 + i, 1.0] for i in range(10)]
        book.apply_snapshot(bids=bids, asks=asks,
                            last_update_id=1, timestamp_ms=0)
        # len counts both sides; each side pruned to 3
        assert len(book) <= 6

    def test_best_bid_survives_pruning(self) -> None:
        book = OrderBookManager("BTC/USDT", max_depth=3)
        bids = [[50_000.0 - i, 1.0] for i in range(10)]
        book.apply_snapshot(bids=bids, asks=[[50_001.0, 1.0]],
                            last_update_id=1, timestamp_ms=0)
        # Best bid (highest price) must be retained
        assert book.best_bid() == 50_000.0


class TestOrderBookThreadSafety:
    def test_concurrent_reads_and_writes_do_not_corrupt(self) -> None:
        """Apply 1,000 diffs from one thread while reading snapshots from another."""
        book = OrderBookManager("BTC/USDT")
        s = _snapshot_msgs()
        book.apply_snapshot(bids=s["bids"], asks=s["asks"],
                            last_update_id=s["lastUpdateId"], timestamp_ms=s["timestamp"])

        errors: list[Exception] = []

        def writer() -> None:
            for i in range(1_000):
                try:
                    book.apply_diff(
                        bids=[[50_000.0, float(i % 10 + 1)]],
                        asks=[[50_001.0, float(i % 5 + 1)]],
                        last_update_id=2000 + i,
                        timestamp_ms=1_700_000_000_000 + i * 1000,
                    )
                except Exception as e:
                    errors.append(e)

        def reader() -> None:
            for _ in range(1_000):
                try:
                    snap = book.snapshot()
                    # Invariants that must always hold
                    if snap.best_bid and snap.best_ask:
                        assert snap.best_bid < snap.best_ask
                except Exception as e:
                    errors.append(e)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == [], f"Thread safety errors: {errors}"


# ─── TickAggregator tests ─────────────────────────────────────────────────────

class TestTickAggregator:
    def test_close_candle_synthesised_when_no_trades(self) -> None:
        agg = TickAggregator(timeframe_seconds=3600)
        agg.seed(last_close=50_000.0)
        candle = agg.close_candle(open_ts_ms=1_700_000_000_000)
        assert candle.synthesised is True
        assert candle.open == 50_000.0
        assert candle.high == 50_000.0
        assert candle.low == 50_000.0
        assert candle.close == 50_000.0
        assert candle.volume == 0.0
        assert candle.trade_count == 0

    def test_push_trade_updates_ohlcv_correctly(self) -> None:
        agg = TickAggregator(timeframe_seconds=3600)
        agg.seed(last_close=50_000.0)
        agg.push_trade(price=50_100.0, qty=0.5, ts_ms=1_700_000_100_000)
        agg.push_trade(price=50_500.0, qty=1.0, ts_ms=1_700_000_200_000)  # new high
        agg.push_trade(price=49_900.0, qty=0.2, ts_ms=1_700_000_300_000)  # new low
        agg.push_trade(price=50_300.0, qty=0.3, ts_ms=1_700_000_400_000)  # close

        candle = agg.close_candle(open_ts_ms=1_700_000_000_000)
        assert candle.synthesised is False
        assert candle.open == pytest.approx(50_100.0)   # first trade
        assert candle.high == pytest.approx(50_500.0)
        assert candle.low == pytest.approx(49_900.0)
        assert candle.close == pytest.approx(50_300.0)  # last trade
        assert candle.volume == pytest.approx(0.5 + 1.0 + 0.2 + 0.3)
        assert candle.trade_count == 4

    def test_after_close_accumulator_resets(self) -> None:
        agg = TickAggregator(timeframe_seconds=3600)
        agg.seed(last_close=50_000.0)
        agg.push_trade(50_100.0, 1.0, ts_ms=1_700_000_100_000)
        agg.close_candle()  # close first candle

        # Push a trade into the second candle
        agg.push_trade(51_000.0, 0.5, ts_ms=1_700_000_200_000)
        candle2 = agg.close_candle(open_ts_ms=1_700_003_600_000)
        assert candle2.open == pytest.approx(51_000.0)
        assert candle2.volume == pytest.approx(0.5)
        assert candle2.trade_count == 1

    def test_timestamp_is_utc_aware(self) -> None:
        agg = TickAggregator(timeframe_seconds=3600)
        agg.seed(last_close=50_000.0)
        ts_ms = 1_700_000_000_000
        candle = agg.close_candle(open_ts_ms=ts_ms)
        assert candle.timestamp_utc.tzinfo is not None
        assert candle.timestamp_utc.tzinfo == UTC


class TestMergeCandle:
    def _base_df(self, rows: int = 10) -> pd.DataFrame:
        """Create a minimal historical OHLCV DataFrame."""
        rng = range(rows)
        idx = pd.date_range("2024-01-01", periods=rows, freq="1h", tz="UTC")
        return pd.DataFrame({
            "open":   [50_000.0 + i for i in rng],
            "high":   [50_100.0 + i for i in rng],
            "low":    [49_900.0 + i for i in rng],
            "close":  [50_050.0 + i for i in rng],
            "volume": [1.0 + i * 0.1 for i in rng],
        }, index=idx)

    def _make_candle(self, ts: datetime) -> CompletedCandle:
        return CompletedCandle(
            timestamp_utc=ts,
            open=51_000.0, high=51_200.0, low=50_800.0,
            close=51_100.0, volume=2.5,
            trade_count=42, synthesised=False,
        )

    def test_append_adds_one_row(self) -> None:
        df = self._base_df(10)
        new_ts = df.index[-1] + pd.Timedelta(hours=1)
        candle = self._make_candle(new_ts)
        result = merge_candle_into(candle, df, window=500)
        assert len(result) == 11
        assert result.index[-1] == new_ts

    def test_values_match_candle(self) -> None:
        df = self._base_df(5)
        new_ts = df.index[-1] + pd.Timedelta(hours=1)
        candle = self._make_candle(new_ts)
        result = merge_candle_into(candle, df)
        last = result.iloc[-1]
        assert last["open"] == pytest.approx(51_000.0)
        assert last["close"] == pytest.approx(51_100.0)
        assert last["volume"] == pytest.approx(2.5)

    def test_duplicate_timestamp_deduplicates(self) -> None:
        """Appending a candle with the same timestamp replaces the old row."""
        df = self._base_df(5)
        existing_ts = df.index[-1]
        candle = self._make_candle(existing_ts)
        result = merge_candle_into(candle, df)
        assert len(result) == 5   # no new row, last replaced
        assert result.iloc[-1]["close"] == pytest.approx(51_100.0)

    def test_window_trims_old_rows(self) -> None:
        df = self._base_df(100)
        new_ts = df.index[-1] + pd.Timedelta(hours=1)
        candle = self._make_candle(new_ts)
        result = merge_candle_into(candle, df, window=50)
        assert len(result) == 50

    def test_missing_columns_raises_value_error(self) -> None:
        df = pd.DataFrame({"open": [1.0]},
                          index=pd.date_range("2024-01-01", periods=1, freq="h", tz="UTC"))
        ts = datetime(2024, 1, 2, tzinfo=UTC)
        candle = self._make_candle(ts)
        with pytest.raises(ValueError, match="missing columns"):
            merge_candle_into(candle, df)
