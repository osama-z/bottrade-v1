"""
Event Bus — lightweight pub/sub system for decoupled communication.
Modules publish events and subscribe to events without knowing each other.

Usage:
    # Subscribe
    event_bus.subscribe("signal.generated", my_handler)

    # Publish
    event_bus.publish("signal.generated", payload={"signal": "BUY", "pair": "BTC/USDT"})
"""

from typing import Callable, Any
from collections import defaultdict
from loguru import logger


class EventBus:
    """Simple synchronous pub/sub event bus."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)

    def subscribe(self, event: str, handler: Callable) -> None:
        """Subscribe a handler to an event."""
        self._subscribers[event].append(handler)
        logger.debug("Subscribed '{}' to event '{}'", handler.__name__, event)

    def unsubscribe(self, event: str, handler: Callable) -> None:
        """Unsubscribe a handler from an event."""
        if handler in self._subscribers[event]:
            self._subscribers[event].remove(handler)

    def publish(self, event: str, **payload: Any) -> None:
        """Publish an event to all subscribers."""
        handlers = self._subscribers.get(event, [])
        if not handlers:
            logger.debug("Event '{}' published with no subscribers", event)
            return

        logger.debug("Publishing event '{}' to {} subscriber(s)", event, len(handlers))
        for handler in handlers:
            name = getattr(handler, "__name__", repr(handler))
            try:
                import asyncio
                if asyncio.iscoroutinefunction(handler):
                    # Calling a coroutine function here would create an
                    # un-awaited coroutine — the handler would silently
                    # never run. Reject loudly instead.
                    logger.error(
                        "Async handler '{}' for event '{}' NOT executed — "
                        "the sync EventBus cannot await it", name, event,
                    )
                    continue
                handler(**payload)
            except Exception as e:
                logger.error(
                    "Error in event handler '{}' for event '{}': {}",
                    name, event, e
                )

    def list_events(self) -> list[str]:
        """Return all events that have subscribers."""
        return list(self._subscribers.keys())


# ─── Global event bus instance ────────────────────────────────────────────────
# Import this in any module to publish or subscribe
event_bus = EventBus()

# ─── Standard event names ─────────────────────────────────────────────────────
# Use these constants to avoid typos in event names
class Events:
    # Data events
    CANDLE_UPDATED = "candle.updated"
    TICKER_UPDATED = "ticker.updated"

    # Signal events
    SIGNAL_GENERATED = "signal.generated"

    # Trade events
    ORDER_PLACED = "order.placed"
    ORDER_FILLED = "order.filled"
    ORDER_CANCELED = "order.canceled"
    POSITION_OPENED = "position.opened"
    POSITION_CLOSED = "position.closed"

    # Risk events
    DRAWDOWN_WARNING = "risk.drawdown_warning"
    MAX_DRAWDOWN_REACHED = "risk.max_drawdown_reached"

    # System events
    BOT_STARTED = "system.started"
    BOT_STOPPED = "system.stopped"
    BOT_ERROR = "system.error"
