"""
Data models for AI Ops Dashboard.
Core domain models: LLMCall, MetricSnapshot, Recommendation, Alert.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field
from sqlalchemy import (
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from ai_ops.shared.utils import now_utc


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass


class ModelProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    VERTEX = "vertex"
    BEDROCK = "bedrock"
    OTHER = "other"


class CallStatus(str, Enum):
    """Status of an LLM call."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    CANCELLED = "cancelled"


class RecommendationType(str, Enum):
    """Types of cost optimization recommendations."""

    MODEL_ROUTING = "model_routing"
    CACHING = "caching"
    PROMPT_COMPRESSION = "prompt_compression"
    BATCHING = "batching"


class RecommendationPriority(str, Enum):
    """Priority levels for recommendations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of alerts."""

    COST_DAILY = "cost_daily"
    COST_HOURLY = "cost_hourly"
    LATENCY_P99 = "latency_p99"
    ERROR_RATE = "error_rate"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# SQLAlchemy Models (for persistence)

class LLMCallModel(Base):
    """SQLAlchemy model for LLM calls."""

    __tablename__ = "llm_calls"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=now_utc, index=True)
    provider: Mapped[str] = mapped_column(String(50), index=True)
    model: Mapped[str] = mapped_column(String(100), index=True)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0, index=True)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0, index=True)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    status: Mapped[str] = mapped_column(String(20), default=CallStatus.SUCCESS, index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    prompt_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    call_metadata: Mapped[dict] = mapped_column(SQLiteJSON, default=dict)

    __table_args__ = (
        Index("ix_llm_calls_timestamp_provider_model", "timestamp", "provider", "model"),
        Index("ix_llm_calls_timestamp_status", "timestamp", "status"),
    )


class MetricSnapshotModel(Base):
    """SQLAlchemy model for aggregated metric snapshots."""

    __tablename__ = "metric_snapshots"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=now_utc, index=True)
    window_seconds: Mapped[int] = mapped_column(Integer, default=60)
    provider: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    model: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    call_count: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    avg_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    p50_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    p95_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    p99_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    error_rate: Mapped[float] = mapped_column(Float, default=0.0)

    __table_args__ = (
        Index("ix_metric_snapshots_timestamp_provider_model", "timestamp", "provider", "model"),
    )


class RecommendationModel(Base):
    """SQLAlchemy model for recommendations."""

    __tablename__ = "recommendations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=now_utc, index=True)
    type: Mapped[str] = mapped_column(String(30), index=True)
    priority: Mapped[str] = mapped_column(String(10), default=RecommendationPriority.MEDIUM)
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text)
    estimated_savings_usd: Mapped[float] = mapped_column(Float, default=0.0)
    estimated_savings_percent: Mapped[float] = mapped_column(Float, default=0.0)
    affected_models: Mapped[list[str]] = mapped_column(SQLiteJSON, default=list)
    affected_providers: Mapped[list[str]] = mapped_column(SQLiteJSON, default=list)
    rec_metadata: Mapped[dict] = mapped_column(SQLiteJSON, default=dict)
    acknowledged: Mapped[bool] = mapped_column(default=False, index=True)
    dismissed: Mapped[bool] = mapped_column(default=False, index=True)


class AlertModel(Base):
    """SQLAlchemy model for alerts."""

    __tablename__ = "alerts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=now_utc, index=True)
    type: Mapped[str] = mapped_column(String(30), index=True)
    severity: Mapped[str] = mapped_column(String(10), default=AlertSeverity.WARNING)
    title: Mapped[str] = mapped_column(String(200))
    message: Mapped[str] = mapped_column(Text)
    value: Mapped[float] = mapped_column(Float, default=0.0)
    threshold: Mapped[float] = mapped_column(Float, default=0.0)
    provider: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    model: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    acknowledged: Mapped[bool] = mapped_column(default=False, index=True)
    resolved: Mapped[bool] = mapped_column(default=False, index=True)

    __table_args__ = (
        Index("ix_alerts_timestamp_type", "timestamp", "type"),
    )


# Pydantic Models (for API/serialization)

class LLMCall(BaseModel):
    """Represents a single LLM API call with all observability data."""

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    timestamp: datetime = Field(default_factory=now_utc)
    provider: ModelProvider
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = Field(default=0)
    latency_ms: int = 0
    cost_usd: float = 0.0
    status: CallStatus = CallStatus.SUCCESS
    error_message: str | None = None
    prompt_hash: str | None = None
    call_metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Compute total_tokens and cost_usd after initialization."""
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        # Auto-calculate cost if not provided
        if self.cost_usd == 0.0 and self.total_tokens > 0:
            self.cost_usd = estimate_cost(self.provider, self.model, self.prompt_tokens, self.completion_tokens)

    @computed_field
    @property
    def prompt_cost_usd(self) -> float:
        """Estimated prompt cost (rough approximation)."""
        return self.cost_usd * (self.prompt_tokens / max(self.total_tokens, 1))

    @computed_field
    @property
    def completion_cost_usd(self) -> float:
        """Estimated completion cost (rough approximation)."""
        return self.cost_usd * (self.completion_tokens / max(self.total_tokens, 1))

    def to_model(self) -> LLMCallModel:
        """Convert to SQLAlchemy model."""
        return LLMCallModel(
            id=self.id,
            timestamp=self.timestamp,
            provider=self.provider.value,
            model=self.model,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            latency_ms=self.latency_ms,
            cost_usd=self.cost_usd,
            status=self.status.value,
            error_message=self.error_message,
            prompt_hash=self.prompt_hash,
            call_metadata=self.call_metadata,
        )

    @classmethod
    def from_model(cls, model: LLMCallModel) -> LLMCall:
        """Create from SQLAlchemy model."""
        return cls(
            id=model.id,
            timestamp=model.timestamp,
            provider=ModelProvider(model.provider),
            model=model.model,
            prompt_tokens=model.prompt_tokens,
            completion_tokens=model.completion_tokens,
            total_tokens=model.total_tokens,
            latency_ms=model.latency_ms,
            cost_usd=model.cost_usd,
            status=CallStatus(model.status),
            error_message=model.error_message,
            prompt_hash=model.prompt_hash,
            call_metadata=model.call_metadata,
        )


class MetricSnapshot(BaseModel):
    """Aggregated metrics for a time window."""

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    timestamp: datetime = Field(default_factory=now_utc)
    window_seconds: int = 60
    provider: str | None = None
    model: str | None = None
    call_count: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    error_count: int = 0
    error_rate: float = Field(default=0.0)

    def model_post_init(self, __context: Any) -> None:
        """Compute error_rate after initialization."""
        if self.call_count > 0:
            self.error_rate = self.error_count / self.call_count

    def to_model(self) -> MetricSnapshotModel:
        """Convert to SQLAlchemy model."""
        return MetricSnapshotModel(
            id=self.id,
            timestamp=self.timestamp,
            window_seconds=self.window_seconds,
            provider=self.provider,
            model=self.model,
            call_count=self.call_count,
            total_tokens=self.total_tokens,
            total_cost_usd=self.total_cost_usd,
            avg_latency_ms=self.avg_latency_ms,
            p50_latency_ms=self.p50_latency_ms,
            p95_latency_ms=self.p95_latency_ms,
            p99_latency_ms=self.p99_latency_ms,
            error_count=self.error_count,
            error_rate=self.error_rate,
        )

    @classmethod
    def from_model(cls, model: MetricSnapshotModel) -> MetricSnapshot:
        """Create from SQLAlchemy model."""
        return cls(
            id=model.id,
            timestamp=model.timestamp,
            window_seconds=model.window_seconds,
            provider=model.provider,
            model=model.model,
            call_count=model.call_count,
            total_tokens=model.total_tokens,
            total_cost_usd=model.total_cost_usd,
            avg_latency_ms=model.avg_latency_ms,
            p50_latency_ms=model.p50_latency_ms,
            p95_latency_ms=model.p95_latency_ms,
            p99_latency_ms=model.p99_latency_ms,
            error_count=model.error_count,
            error_rate=model.error_rate,
        )


class Recommendation(BaseModel):
    """Cost optimization recommendation."""

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    timestamp: datetime = Field(default_factory=now_utc)
    type: RecommendationType
    priority: RecommendationPriority = RecommendationPriority.MEDIUM
    title: str
    description: str
    estimated_savings_usd: float = 0.0
    estimated_savings_percent: float = 0.0
    affected_models: list[str] = Field(default_factory=list)
    affected_providers: list[str] = Field(default_factory=list)
    rec_metadata: dict[str, Any] = Field(default_factory=dict)
    acknowledged: bool = False
    dismissed: bool = False

    def to_model(self) -> RecommendationModel:
        """Convert to SQLAlchemy model."""
        return RecommendationModel(
            id=self.id,
            timestamp=self.timestamp,
            type=self.type.value,
            priority=self.priority.value,
            title=self.title,
            description=self.description,
            estimated_savings_usd=self.estimated_savings_usd,
            estimated_savings_percent=self.estimated_savings_percent,
            affected_models=self.affected_models,
            affected_providers=self.affected_providers,
            rec_metadata=self.rec_metadata,
            acknowledged=self.acknowledged,
            dismissed=self.dismissed,
        )

    @classmethod
    def from_model(cls, model: RecommendationModel) -> Recommendation:
        """Create from SQLAlchemy model."""
        return cls(
            id=model.id,
            timestamp=model.timestamp,
            type=RecommendationType(model.type),
            priority=RecommendationPriority(model.priority),
            title=model.title,
            description=model.description,
            estimated_savings_usd=model.estimated_savings_usd,
            estimated_savings_percent=model.estimated_savings_percent,
            affected_models=model.affected_models,
            affected_providers=model.affected_providers,
            rec_metadata=model.rec_metadata,
            acknowledged=model.acknowledged,
            dismissed=model.dismissed,
        )


class Alert(BaseModel):
    """System alert."""

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    timestamp: datetime = Field(default_factory=now_utc)
    type: AlertType
    severity: AlertSeverity = AlertSeverity.WARNING
    title: str
    message: str
    value: float = 0.0
    threshold: float = 0.0
    provider: str | None = None
    model: str | None = None
    acknowledged: bool = False
    resolved: bool = False

    def to_model(self) -> AlertModel:
        """Convert to SQLAlchemy model."""
        return AlertModel(
            id=self.id,
            timestamp=self.timestamp,
            type=self.type.value,
            severity=self.severity.value,
            title=self.title,
            message=self.message,
            value=self.value,
            threshold=self.threshold,
            provider=self.provider,
            model=self.model,
            acknowledged=self.acknowledged,
            resolved=self.resolved,
        )

    @classmethod
    def from_model(cls, model: AlertModel) -> Alert:
        """Create from SQLAlchemy model."""
        return cls(
            id=model.id,
            timestamp=model.timestamp,
            type=AlertType(model.type),
            severity=AlertSeverity(model.severity),
            title=model.title,
            message=model.message,
            value=model.value,
            threshold=model.threshold,
            provider=model.provider,
            model=model.model,
            acknowledged=model.acknowledged,
            resolved=model.resolved,
        )


# Helper functions for cost calculation

MODEL_COSTS: dict[str, dict[str, float]] = {
    # Format: {model: {"prompt": cost_per_1k_tokens, "completion": cost_per_1k_tokens}}
    # OpenAI
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
    "gpt-4o": {"prompt": 0.005, "completion": 0.015},
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
    # Anthropic
    "claude-3-opus": {"prompt": 0.015, "completion": 0.075},
    "claude-3-sonnet": {"prompt": 0.003, "completion": 0.015},
    "claude-3-haiku": {"prompt": 0.00025, "completion": 0.00125},
    "claude-3.5-sonnet": {"prompt": 0.003, "completion": 0.015},
    # Default fallback
    "default": {"prompt": 0.001, "completion": 0.002},
}


def estimate_cost(provider: ModelProvider, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Estimate the cost of an LLM call based on token counts.
    
    WHY: Cost estimation is critical for budgeting and optimization.
    We use a lookup table with fallback to default rates. LiteLLM also
    provides cost tracking but this gives us a local estimate without
    needing to call their API.
    """
    model_lower = model.lower()

    # Try exact match first
    if model_lower in MODEL_COSTS:
        costs = MODEL_COSTS[model_lower]
    else:
        # Try partial match for model families
        costs = None
        for key, value in MODEL_COSTS.items():
            if key in model_lower:
                costs = value
                break
        if costs is None:
            costs = MODEL_COSTS["default"]

    prompt_cost = (prompt_tokens / 1000) * costs["prompt"]
    completion_cost = (completion_tokens / 1000) * costs["completion"]
    return prompt_cost + completion_cost


def calculate_percentile(values: list[float], percentile: float) -> float:
    """Calculate percentile from a list of values."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    # Use linear interpolation for more accurate percentile
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]

    # Position in 0-indexed array
    pos = percentile * (n - 1)
    lower = int(pos)
    upper = min(lower + 1, n - 1)
    weight = pos - lower

    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
