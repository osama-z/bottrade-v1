"""Anti-spoofing guardrail (Roadmap Task 6.2).

Spoofing/layering = placing orders with the intent to CANCEL them (to project
false liquidity), not to trade. The bot's execution path does not do this by
design — post-only slices are placed to rest and fill; the only cancellations
are legitimate crash-recovery / reconciliation cleanups (see execution/recovery.py).

This guard is defense-in-depth: it watches the place→cancel timing and, if orders
are repeatedly placed and cancelled within a few seconds (the signature of
spoofing), it TRIPS — halting further order placement so the bot can never even
*appear* to spoof, whatever future cancel-replace logic is added.

Pure and time-injectable, so the trip condition is unit-tested without waiting.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class SpoofingGuard:
    """Trips when too many orders are cancelled within seconds of placement.

    min_resting_seconds: a cancel sooner than this after placement is "fast"
                         (the spoofing signature — an order never meant to rest).
    max_fast_cancels:    this many fast cancels...
    window_seconds:      ...within this rolling window trips the guard.
    """

    min_resting_seconds: float = 2.0
    max_fast_cancels: int = 3
    window_seconds: float = 60.0
    tripped: bool = False
    _placed_at: dict = field(default_factory=dict, repr=False)
    _fast_cancels: deque = field(default_factory=deque, repr=False)

    def record_placement(self, order_id, now: float) -> None:
        """Remember when an order was placed (to measure its resting time)."""
        if order_id:
            self._placed_at[str(order_id)] = float(now)

    def record_cancellation(self, order_id, now: float) -> bool:
        """Record a cancel; return True if it was a FAST cancel. Trips the guard
        once ``max_fast_cancels`` fast cancels fall inside ``window_seconds``."""
        placed = self._placed_at.pop(str(order_id), None)
        fast = placed is not None and (float(now) - placed) < self.min_resting_seconds
        if fast:
            self._fast_cancels.append(float(now))
            self._prune(now)
            if len(self._fast_cancels) >= self.max_fast_cancels:
                self.tripped = True
        return fast

    def _prune(self, now: float) -> None:
        cutoff = float(now) - self.window_seconds
        while self._fast_cancels and self._fast_cancels[0] < cutoff:
            self._fast_cancels.popleft()

    @property
    def fast_cancel_count(self) -> int:
        return len(self._fast_cancels)

    def reset(self) -> None:
        """Operator reset after review (mirrors the manual-reset breaker policy)."""
        self.tripped = False
        self._fast_cancels.clear()
        self._placed_at.clear()
