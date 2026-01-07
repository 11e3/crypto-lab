"""
Event bus for event-driven architecture.

Provides publish-subscribe pattern for loose coupling between components.
"""

from collections import defaultdict
from collections.abc import Callable
from typing import TypeVar, overload

from src.execution.events import Event, EventType
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Type variable for event handlers
T = TypeVar("T", bound=Event)


class EventBus:
    """
    Event bus for publishing and subscribing to events.

    Implements publish-subscribe pattern for decoupled communication.
    """

    def __init__(self) -> None:
        """Initialize event bus."""
        self._subscribers: dict[EventType, list[Callable[[Event], None]]] = defaultdict(list)
        self._global_subscribers: list[Callable[[Event], None]] = []

    @overload
    def subscribe(
        self,
        event_type: EventType | None = None,
        handler: None = None,
    ) -> Callable[[Callable[[Event], None]], Callable[[Event], None]]: ...

    @overload
    def subscribe(
        self,
        event_type: EventType | None = None,
        handler: Callable[[Event], None] = ...,
    ) -> Callable[[Event], None]: ...

    def subscribe(
        self,
        event_type: EventType | None = None,
        handler: Callable[[Event], None] | None = None,
    ) -> Callable[[Event], None] | Callable[[Callable[[Event], None]], Callable[[Event], None]]:
        """
        Subscribe to events.

        Can be used as a decorator or called directly.

        Args:
            event_type: Specific event type to subscribe to (None for all events)
            handler: Handler function (required if not used as decorator)

        Returns:
            Decorator function if used as decorator, handler otherwise

        Example:
            As decorator::

                @event_bus.subscribe(EventType.ORDER_PLACED)
                def handle_order(event: OrderEvent):
                    ...

            Direct call::

                event_bus.subscribe(EventType.ORDER_PLACED, handle_order)
        """
        if handler is None:
            # Used as decorator
            def decorator(func: Callable[[Event], None]) -> Callable[[Event], None]:
                if event_type is None:
                    self._global_subscribers.append(func)
                else:
                    self._subscribers[event_type].append(func)
                return func

            return decorator
        else:
            # Direct call
            if event_type is None:
                self._global_subscribers.append(handler)
            else:
                self._subscribers[event_type].append(handler)
            return handler

    def unsubscribe(
        self,
        event_type: EventType,
        handler: Callable[[Event], None],
    ) -> bool:
        """
        Unsubscribe from events.

        Args:
            event_type: Event type to unsubscribe from
            handler: Handler function to remove

        Returns:
            True if handler was removed, False if not found
        """
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)
            return True
        return False

    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event: Event to publish
        """
        # Notify specific subscribers
        handlers = self._subscribers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(
                    f"Error in event handler for {event.event_type}: {e}",
                    exc_info=True,
                )

        # Notify global subscribers
        for handler in self._global_subscribers:
            try:
                handler(event)
            except Exception as e:
                logger.error(
                    f"Error in global event handler: {e}",
                    exc_info=True,
                )

    def clear(self) -> None:
        """Clear all subscribers."""
        self._subscribers.clear()
        self._global_subscribers.clear()

    def get_subscriber_count(self, event_type: EventType | None = None) -> int:
        """
        Get number of subscribers for an event type.

        Args:
            event_type: Event type (None for global subscribers)

        Returns:
            Number of subscribers
        """
        if event_type is None:
            return len(self._global_subscribers)
        return len(self._subscribers.get(event_type, []))


# Global event bus instance
_default_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """
    Get the default global event bus instance.

    Returns:
        Global EventBus instance
    """
    global _default_event_bus
    if _default_event_bus is None:
        _default_event_bus = EventBus()
    return _default_event_bus


def set_event_bus(event_bus: EventBus | None) -> None:
    """
    Set the default global event bus instance.

    Args:
        event_bus: EventBus instance to use as default, or None to clear
    """
    global _default_event_bus
    _default_event_bus = event_bus
