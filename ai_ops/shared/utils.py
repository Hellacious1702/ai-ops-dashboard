"""
Shared utilities and base classes for AI Ops Dashboard.
Provides common patterns used across core, web, and TUI components.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class EventType(str, Enum):
    """Types of events that can be emitted by the system."""

    LLM_CALL_START = "llm_call_start"
    LLM_CALL_END = "llm_call_end"
    LLM_CALL_ERROR = "llm_call_error"
    METRIC_FLUSH = "metric_flush"
    RECOMMENDATION_GENERATED = "recommendation_generated"
    ALERT_TRIGGERED = "alert_triggered"


@dataclass
class Event:
    """Base event for the event bus system."""

    type: EventType
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class EventBus:
    """
    Simple async event bus for decoupled communication between components.
    Used by instrumentation, metrics, recommendations, and alerting.
    """

    def __init__(self) -> None:
        self._subscribers: dict[EventType, list[Callable[[Event], Any]]] = {
            event_type: [] for event_type in EventType
        }

    def subscribe(self, event_type: EventType, handler: Callable[[Event], Any]) -> None:
        """Subscribe a handler to an event type."""
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: Callable[[Event], Any]) -> None:
        """Unsubscribe a handler from an event type."""
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)

    async def emit(self, event: Event) -> None:
        """Emit an event to all subscribers."""
        handlers = self._subscribers.get(event.type, [])
        if handlers:
            await asyncio.gather(
                *[handler(event) for handler in handlers],
                return_exceptions=True,
            )

    def emit_sync(self, event: Event) -> None:
        """Emit an event synchronously (for non-async contexts)."""
        for handler in self._subscribers.get(event.type, []):
            try:
                handler(event)
            except Exception:
                pass  # Silently ignore handler errors in sync mode


# Global event bus instance
event_bus = EventBus()


class Configurable(ABC):
    """Base class for components that can be configured from a dictionary."""

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict[str, Any]) -> Configurable:
        """Create instance from configuration dictionary."""
        ...


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True


async def with_retry(
    func: Callable[..., T],
    *args: Any,
    config: RetryConfig | None = None,
    **kwargs: Any,
) -> T:
    """Execute a function with retry logic."""
    config = config or RetryConfig()
    last_exception: Exception | None = None

    for attempt in range(config.max_attempts):
        try:
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                return await result
            return result
        except Exception as e:
            last_exception = e
            if attempt < config.max_attempts - 1:
                delay = min(
                    config.base_delay * (config.exponential_base ** attempt),
                    config.max_delay,
                )
                if config.jitter:
                    import random
                    delay *= (0.5 + random.random())
                await asyncio.sleep(delay)

    assert last_exception is not None
    raise last_exception


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix."""
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def now_utc() -> datetime:
    """Get current UTC datetime."""
    return datetime.utcnow()


def now_timestamp() -> float:
    """Get current timestamp."""
    return time.time()


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for common types."""

    def default(self, o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, BaseModel):
            return o.model_dump()
        if hasattr(o, "__dict__"):
            return o.__dict__
        return super().default(o)


def to_json(obj: Any) -> str:
    """Serialize object to JSON string."""
    return json.dumps(obj, cls=JSONEncoder)


def from_json(json_str: str, model: type[BaseModel]) -> BaseModel:
    """Deserialize JSON string to Pydantic model."""
    return model.model_validate_json(json_str)
