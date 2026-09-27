"""depth_sync.py — Binance depth-stream synchronization (Stage 2).

Implements the exchange's documented local-order-book algorithm
(claude.md Data Validation), decoupled from the transport so the whole
state machine is unit-testable with scripted messages:

1. Buffer incoming diff events.
2. Fetch a REST snapshot (``lastUpdateId``).
3. Drop buffered events with ``u <= lastUpdateId``.
4. The first applied event must bracket the snapshot:
   ``U <= lastUpdateId + 1 <= u``. If the buffer starts beyond that
   (gap), fetch a fresh snapshot and retry (bounded).
5. Thereafter every event must satisfy ``U == previous u + 1``;
   a gap discards the book and resyncs from step 1.

The transport (``data/ws_depth_feed.py``) only feeds raw messages to
``on_depth_message()`` and provides a ``snapshot_provider`` callable.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable

from loguru import logger

from data.order_book import OrderBookDesyncError, OrderBookManager

# snapshot_provider() -> {"bids": [...], "asks": [...],
#                         "lastUpdateId": int, "timestamp_ms": int}
SnapshotProvider = Callable[[], dict]


class DepthStreamSynchronizer:
    """Drives an OrderBookManager from a Binance-style diff stream."""

    def __init__(
        self,
        book: OrderBookManager,
        snapshot_provider: SnapshotProvider,
        max_buffer: int = 1000,
        max_snapshot_retries: int = 3,
    ) -> None:
        self._book = book
        self._get_snapshot = snapshot_provider
        self._buffer: deque[dict] = deque(maxlen=max_buffer)  # bounded memory
        self._max_retries = max_snapshot_retries
        self._synced = False
        self.resync_count = 0  # monitoring: how often the stream desynced

    @property
    def is_synced(self) -> bool:
        return self._synced and self._book.is_ready

    # ── Message entry point ────────────────────────────────────────────────────

    def on_depth_message(self, msg: dict[str, Any]) -> None:
        """Feed one raw diff message: {"U": int, "u": int, "b": [...],
        "a": [...], "E": event-time-ms}. Malformed messages are discarded
        individually (untrusted input; one bad frame must not kill the feed).
        """
        try:
            event = {
                "U": int(msg["U"]),
                "u": int(msg["u"]),
                "b": msg.get("b", []),
                "a": msg.get("a", []),
                "E": int(msg.get("E", 0)),
            }
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("Discarding malformed depth message: {}", e)
            return

        if not self._synced:
            self._buffer.append(event)
            self._try_sync()
            return

        try:
            self._apply(event)
        except OrderBookDesyncError as e:
            logger.warning("Order book desync ({}) — resyncing", e)
            self._start_resync(event)

    # ── Sync state machine ─────────────────────────────────────────────────────

    def _start_resync(self, pending_event: dict | None = None) -> None:
        self.resync_count += 1
        self._synced = False
        self._book.reset()
        self._buffer.clear()
        if pending_event is not None:
            self._buffer.append(pending_event)
        self._try_sync()

    def _try_sync(self) -> None:
        """Attempt snapshot + buffered-event replay (steps 2-5)."""
        for attempt in range(1, self._max_retries + 1):
            try:
                snap = self._get_snapshot()
            except Exception as e:
                logger.error("Snapshot fetch failed (attempt {}): {}", attempt, e)
                return  # keep buffering; retry on the next message

            last_id = int(snap["lastUpdateId"])
            # Step 3: drop events already contained in the snapshot
            while self._buffer and self._buffer[0]["u"] <= last_id:
                self._buffer.popleft()

            if not self._buffer:
                # Nothing to bridge yet — snapshot alone is a valid start.
                self._apply_snapshot(snap)
                self._synced = True
                logger.info(
                    "Order book synced from snapshot (lastUpdateId={})", last_id
                )
                return

            first = self._buffer[0]
            # Step 4: first event must bracket lastUpdateId + 1
            if first["U"] <= last_id + 1 <= first["u"]:
                self._apply_snapshot(snap)
                try:
                    while self._buffer:
                        self._apply(self._buffer.popleft())
                except OrderBookDesyncError as e:
                    logger.warning("Gap inside buffered events ({}) — retrying", e)
                    self._book.reset()
                    continue
                self._synced = True
                logger.info(
                    "Order book synced (lastUpdateId={}, replayed buffer)", last_id
                )
                return

            if first["U"] > last_id + 1:
                # Snapshot is older than our earliest buffered event —
                # fetch a newer snapshot and try again.
                logger.warning(
                    "Snapshot too old (lastUpdateId={} < first U={}) — retry {}/{}",
                    last_id, first["U"], attempt, self._max_retries,
                )
                continue

        logger.error(
            "Order book sync failed after {} snapshot attempts — will retry "
            "on next message", self._max_retries,
        )

    # ── Book application helpers ───────────────────────────────────────────────

    def _apply_snapshot(self, snap: dict) -> None:
        self._book.apply_snapshot(
            bids=snap["bids"],
            asks=snap["asks"],
            last_update_id=int(snap["lastUpdateId"]),
            timestamp_ms=int(snap.get("timestamp_ms", 0)),
        )

    def _apply(self, event: dict) -> None:
        self._book.apply_diff(
            bids=event["b"],
            asks=event["a"],
            last_update_id=event["u"],
            first_update_id=event["U"],
            timestamp_ms=event["E"],
        )
