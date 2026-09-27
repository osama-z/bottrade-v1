"""Task 6.2 — anti-spoofing guardrail: fast place/cancel trips + halts placement."""
from execution.spoofing_guard import SpoofingGuard


# ─── Pure guard logic (time injected) ──────────────────────────────────────────
class TestSpoofingGuard:
    def test_slow_cancels_never_trip(self):
        g = SpoofingGuard(min_resting_seconds=2.0, max_fast_cancels=3, window_seconds=60)
        for i in range(10):
            g.record_placement(f"o{i}", now=i * 100.0)
            g.record_cancellation(f"o{i}", now=i * 100.0 + 5.0)   # 5s rest → not fast
        assert g.tripped is False and g.fast_cancel_count == 0

    def test_repeated_fast_cancels_trip(self):
        g = SpoofingGuard(min_resting_seconds=2.0, max_fast_cancels=3, window_seconds=60)
        for i in range(3):
            g.record_placement(f"o{i}", now=10.0 + i)
            fast = g.record_cancellation(f"o{i}", now=10.0 + i + 0.5)   # 0.5s rest → fast
            assert fast is True
        assert g.tripped is True

    def test_below_threshold_does_not_trip(self):
        g = SpoofingGuard(max_fast_cancels=3)
        g.record_placement("o1", now=1.0)
        g.record_cancellation("o1", now=1.2)
        assert g.tripped is False                 # 1 fast cancel < 3

    def test_fast_cancels_outside_window_are_pruned(self):
        g = SpoofingGuard(min_resting_seconds=2.0, max_fast_cancels=3, window_seconds=10)
        # Two fast cancels far in the past, then one now — old ones pruned, no trip.
        for oid, place, cancel in [("a", 0.0, 0.5), ("b", 1.0, 1.5), ("c", 100.0, 100.5)]:
            g.record_placement(oid, place)
            g.record_cancellation(oid, cancel)
        assert g.fast_cancel_count == 1 and g.tripped is False

    def test_cancel_without_placement_is_not_fast(self):
        g = SpoofingGuard()
        assert g.record_cancellation("unknown", now=5.0) is False

    def test_reset_clears_trip(self):
        g = SpoofingGuard(max_fast_cancels=1, min_resting_seconds=2.0)
        g.record_placement("o", 1.0)
        g.record_cancellation("o", 1.1)
        assert g.tripped is True
        g.reset()
        assert g.tripped is False and g.fast_cancel_count == 0
