"""
Configuration management for AI Ops Dashboard.
Centralizes all configuration with environment variable support.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration."""

    model_config = SettingsConfigDict(env_prefix="DB_")

    sqlite_path: Path = Field(
        default=Path("data/ai_ops.db"),
        description="Path to SQLite database file",
    )
    parquet_export_path: Path = Field(
        default=Path("data/parquet"),
        description="Directory for Parquet exports",
    )
    export_interval_seconds: int = Field(
        default=300,
        description="Interval for Parquet exports in seconds",
    )
    connection_pool_size: int = Field(
        default=5,
        description="SQLAlchemy connection pool size",
    )


class InstrumentationSettings(BaseSettings):
    """LLM Instrumentation configuration."""

    model_config = SettingsConfigDict(env_prefix="INSTR_")

    enabled: bool = Field(default=True, description="Enable instrumentation")
    sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Sampling rate for instrumentation (0.0 to 1.0)",
    )
    capture_prompt: bool = Field(
        default=False,
        description="Capture prompt text (may contain sensitive data)",
    )
    capture_response: bool = Field(
        default=False,
        description="Capture response text (may contain sensitive data)",
    )
    max_prompt_length: int = Field(
        default=10000,
        description="Maximum prompt length to store",
    )
    max_response_length: int = Field(
        default=10000,
        description="Maximum response length to store",
    )
    async_buffer_size: int = Field(
        default=100,
        description="Buffer size for async metric writes",
    )
    flush_interval_seconds: float = Field(
        default=5.0,
        description="Flush interval for buffered metrics",
    )


class MetricsSettings(BaseSettings):
    """Metrics collection configuration."""

    model_config = SettingsConfigDict(env_prefix="METRICS_")

    enabled: bool = Field(default=True, description="Enable metrics collection")
    aggregation_window_seconds: int = Field(
        default=60,
        description="Time window for metric aggregation",
    )
    retention_days: int = Field(
        default=30,
        description="Days to retain raw metrics",
    )
    percentile_buckets: list[float] = Field(
        default=[0.5, 0.95, 0.99],
        description="Percentiles to calculate for latency",
    )


class RecommendationSettings(BaseSettings):
    """Cost optimization recommendation configuration."""

    model_config = SettingsConfigDict(env_prefix="REC_")

    enabled: bool = Field(default=True, description="Enable recommendations")
    check_interval_seconds: int = Field(
        default=3600,
        description="Interval for running recommendation checks",
    )
    # Model routing rules
    routing_enabled: bool = Field(default=True, description="Enable model routing recommendations")
    routing_cost_threshold: float = Field(
        default=0.10,
        description="Cost per 1k tokens threshold for routing to cheaper model",
    )
    routing_latency_threshold_ms: int = Field(
        default=5000,
        description="Latency threshold in ms for routing to faster model",
    )
    # Caching rules
    caching_enabled: bool = Field(default=True, description="Enable caching recommendations")
    caching_min_calls: int = Field(
        default=10,
        description="Minimum calls to same prompt for caching recommendation",
    )
    caching_similarity_threshold: float = Field(
        default=0.95,
        description="Similarity threshold for semantic caching",
    )
    # Prompt compression rules
    compression_enabled: bool = Field(default=True, description="Enable prompt compression recommendations")
    compression_min_tokens: int = Field(
        default=1000,
        description="Minimum prompt tokens for compression recommendation",
    )
    compression_target_reduction: float = Field(
        default=0.3,
        description="Target token reduction ratio",
    )
    # Batching rules
    batching_enabled: bool = Field(default=True, description="Enable batching recommendations")
    batching_min_calls: int = Field(
        default=5,
        description="Minimum calls in time window for batching",
    )
    batching_window_seconds: int = Field(
        default=60,
        description="Time window for batching calls",
    )


class AlertingSettings(BaseSettings):
    """Alerting configuration."""

    model_config = SettingsConfigDict(env_prefix="ALERT_")

    enabled: bool = Field(default=True, description="Enable alerting")
    # Cost alerts
    cost_daily_threshold: float = Field(
        default=100.0,
        description="Daily cost threshold for alert",
    )
    cost_hourly_threshold: float = Field(
        default=20.0,
        description="Hourly cost threshold for alert",
    )
    # Latency alerts
    latency_p99_threshold_ms: int = Field(
        default=10000,
        description="P99 latency threshold in ms",
    )
    # Error rate alerts
    error_rate_threshold: float = Field(
        default=0.05,
        description="Error rate threshold (0.0 to 1.0)",
    )
    # Cooldown
    alert_cooldown_seconds: int = Field(
        default=300,
        description="Cooldown between alerts of same type",
    )


class WebSettings(BaseSettings):
    """Web UI configuration."""

    model_config = SettingsConfigDict(env_prefix="WEB_")

    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=8000, description="Port to bind to")
    reload: bool = Field(default=False, description="Enable auto-reload")
    cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins",
    )
    ws_ping_interval: int = Field(
        default=30,
        description="WebSocket ping interval in seconds",
    )


class TUISettings(BaseSettings):
    """TUI configuration."""

    model_config = SettingsConfigDict(env_prefix="TUI_")

    refresh_rate: float = Field(
        default=2.0,
        description="Dashboard refresh rate in Hz",
    )
    theme: str = Field(
        default="dark",
        description="Color theme (dark, light, dracula, nord)",
    )
    show_charts: bool = Field(
        default=True,
        description="Show sparkline charts in TUI",
    )


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        nested_model_default_partial_update=True,
    )

    app_name: str = Field(default="AI Ops Dashboard", description="Application name")
    environment: str = Field(default="development", description="Environment name")
    log_level: str = Field(default="INFO", description="Logging level")
    data_dir: Path = Field(default=Path("data"), description="Data directory")

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    instrumentation: InstrumentationSettings = Field(default_factory=InstrumentationSettings)
    metrics: MetricsSettings = Field(default_factory=MetricsSettings)
    recommendations: RecommendationSettings = Field(default_factory=RecommendationSettings)
    alerting: AlertingSettings = Field(default_factory=AlertingSettings)
    web: WebSettings = Field(default_factory=WebSettings)
    tui: TUISettings = Field(default_factory=TUISettings)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def reload_settings() -> Settings:
    """Force reload of settings (clears cache)."""
    get_settings.cache_clear()
    return get_settings()
