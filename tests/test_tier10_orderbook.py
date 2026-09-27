"""Tier-10 tests: Stage 2 order-book wiring (issue #8, audit V-14/V-33/V-35).

The replay test claude.md names as the Stage 2 deliverable: scripted
diff sequences (with gaps, stale events, malformed frames) driven through
the full sync state machine, confirming the book state stays consistent
and desyncs trigger a snapshot resync.
"""

import time
from pathlib import Path

import pytest

from data.depth_sync import DepthStreamSynchronizer
from data.order_book import OrderBookDesyncError, OrderBookManager

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def snap_provider(last_update_id: int, bid=(100.0, 5.0), ask=(101.0, 5.0)):
    """Snapshot provider factory; counts calls for resync assertions."""
    calls = {"n": 0}

    def provider():
        calls["n"] += 1
        return {
            "bids": [bid], "asks": [ask],
            "lastUpdateId": last_update_id, "timestamp_ms": 1_000,
        }

    provider.calls = calls
    return provider


def msg(U, u, bids=(), asks=(), E=2_000):
    return {"U": U, "u": u, "b": list(bids), "a": list(asks), "E": E}


# ─── V-14: sequence continuity in the manager ─────────────────────────────────

class TestSequenceContinuity:
    def _book(self):
        book = OrderBookManager("BTC/USDT")
        book.apply_snapshot(bids=[(100.0, 1.0)], asks=[(101.0, 1.0)],
                            last_update_id=10, timestamp_ms=0)
        return book

    def test_contiguous_diff_applies(self):
        book = self._book()
        book.apply_diff(bids=[(99.5, 2.0)], asks=[], last_update_id=12,
                        first_update_id=11, timestamp_ms=1)
        assert book.snapshot().last_update_id == 12

    def test_stale_event_is_ignored(self):
        book = self._book()
        book.apply_diff(bids=[(50.0, 9.0)], asks=[], last_update_id=10,
                        first_update_id=9, timestamp_ms=1)
        assert 50.0 not in dict(book.top_bids(10))
        assert book.snapshot().last_update_id == 10

    def test_gap_raises_desync_and_unreadies_book(self):
        book = self._book()
        with pytest.raises(OrderBookDesyncError):
            book.apply_diff(bids=[], asks=[], last_update_id=20,
                            first_update_id=15, timestamp_ms=1)  # gap: 11..14 lost
        assert not book.is_ready

    def test_legacy_calls_without_U_still_work(self):
        book = self._book()
        book.apply_diff(bids=[(99.0, 1.0)], asks=[], last_update_id=99,
                        timestamp_ms=1)  # no first_update_id → no enforcement
        assert book.snapshot().last_update_id == 99


# ─── V-35: ingest validation ──────────────────────────────────────────────────

class TestIngestValidation:
    def test_nan_and_negative_levels_rejected(self):
        book = OrderBookManager("BTC/USDT")
        book.apply_snapshot(
            bids=[(100.0, 1.0), (float("nan"), 5.0), (-10.0, 5.0), (99.0, float("nan"))],
            asks=[(101.0, 1.0), (0.0, 5.0)],
            last_update_id=1, timestamp_ms=0,
        )
        assert book.best_bid() == 100.0
        assert book.best_ask() == 101.0
        assert len(book) == 2

    def test_diff_rejects_negative_quantity(self):
        book = OrderBookManager("BTC/USDT")
        book.apply_snapshot(bids=[(100.0, 1.0)], asks=[(101.0, 1.0)],
                            last_update_id=1, timestamp_ms=0)
        book.apply_diff(bids=[(100.5, -3.0)], asks=[], last_update_id=2,
                        first_update_id=2, timestamp_ms=1)
        assert book.best_bid() == 100.0  # invalid level never stored


# ─── Stale-book timer ─────────────────────────────────────────────────────────

class TestStaleBook:
    def test_frozen_feed_flips_is_ready(self):
        book = OrderBookManager("BTC/USDT", stale_after_seconds=0.05)
        book.apply_snapshot(bids=[(100.0, 1.0)], asks=[(101.0, 1.0)],
                            last_update_id=1, timestamp_ms=0)
        assert book.is_ready
        time.sleep(0.08)
        assert not book.is_ready  # no updates → stale, not "ready" forever


# ─── V-33: unit parity with the live filter ───────────────────────────────────

class TestImbalanceUnits:
    def test_imbalance_is_base_quantity_weighted(self):
        # 2.0 base on bid, 4.0 base on ask → ratio exactly 2.0 in base
        # units. Notional weighting would give (4*101)/(2*100) = 2.02.
        book = OrderBookManager("BTC/USDT", imbalance_band_pct=0.05)
        book.apply_snapshot(bids=[(100.0, 2.0)], asks=[(101.0, 4.0)],
                            last_update_id=1, timestamp_ms=0)
        assert book.snapshot().imbalance_ratio == pytest.approx(2.0)


# ─── The Stage 2 replay test: full sync state machine ─────────────────────────

class TestDepthStreamSynchronizer:
    def test_clean_sync_and_apply(self):
        book = OrderBookManager("BTC/USDT")
        provider = snap_provider(last_update_id=10)
        sync = DepthStreamSynchronizer(book, provider)

        sync.on_depth_message(msg(U=8, u=10))            # pre-snapshot: dropped
        sync.on_depth_message(msg(U=11, u=12, bids=[(99.0, 3.0)]))
        assert sync.is_synced
        assert dict(book.top_bids(10))[99.0] == 3.0
        assert book.snapshot().last_update_id == 12

    def test_first_event_must_bracket_snapshot(self):
        """Buffer starts beyond the snapshot → a fresh snapshot is fetched."""
        book = OrderBookManager("BTC/USDT")
        provider = snap_provider(last_update_id=10)
        sync = DepthStreamSynchronizer(book, provider)

        # U=15 > lastUpdateId+1=11 → retries snapshots (same id each time
        # here, so sync stays pending) — but never applies a gapped event.
        sync.on_depth_message(msg(U=15, u=16))
        assert not sync.is_synced
        assert provider.calls["n"] >= 1
        assert book.snapshot().last_update_id in (0, 10)

    def test_gap_mid_stream_triggers_resync(self):
        book = OrderBookManager("BTC/USDT")
        provider = snap_provider(last_update_id=10)
        sync = DepthStreamSynchronizer(book, provider)

        sync.on_depth_message(msg(U=11, u=12))
        assert sync.is_synced
        calls_before = provider.calls["n"]

        # Gap: 13..19 lost. Must resync via a new snapshot, not carry on.
        sync.on_depth_message(msg(U=20, u=21))
        assert sync.resync_count == 1
        assert provider.calls["n"] > calls_before

    def test_malformed_messages_discarded_individually(self):
        book = OrderBookManager("BTC/USDT")
        sync = DepthStreamSynchronizer(book, snap_provider(last_update_id=10))

        sync.on_depth_message({"garbage": True})          # no U/u — dropped
        sync.on_depth_message(msg(U=11, u=12, bids=[(99.0, 1.0)]))
        assert sync.is_synced

    def test_snapshot_failure_keeps_buffering(self):
        book = OrderBookManager("BTC/USDT")

        state = {"fail": True}
        def flaky_provider():
            if state["fail"]:
                raise ConnectionError("REST down")
            return {"bids": [(100.0, 5.0)], "asks": [(101.0, 5.0)],
                    "lastUpdateId": 10, "timestamp_ms": 0}

        sync = DepthStreamSynchronizer(book, flaky_provider)
        sync.on_depth_message(msg(U=11, u=12))
        assert not sync.is_synced   # snapshot failed; event buffered

        state["fail"] = False
        sync.on_depth_message(msg(U=13, u=14))
        assert sync.is_synced       # recovered on next message
        assert book.snapshot().last_update_id == 14


# ─── Wiring ───────────────────────────────────────────────────────────────────

class TestWiring:
    def test_strategy_prefers_injected_source(self):
        src = (PROJECT_ROOT / "strategies" / "ai_combined.py").read_text()
        assert "self._imbalance_source.get_imbalance(pair)" in src

    def test_run_live_wires_feeds_behind_flag(self):
        src = (PROJECT_ROOT / "scripts" / "run_live.py").read_text()
        assert "settings.use_ws_order_book" in src
        assert "BookImbalanceSource" in src
        assert "feed.stop()" in src

    def test_unsynced_book_yields_none_not_neutral(self):
        from data.ws_depth_feed import BookImbalanceSource

        assert BookImbalanceSource({}).get_imbalance("BTC/USDT") is None
