"""
Metrics Collector with async buffering.
Aggregates LLM calls into time-windowed snapshots for dashboard visualization.
"""

from __future__ import annotations

import asyncio
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from ai_ops.core.storage.database import get_db_session
from ai_ops.shared.config import get_settings
from ai_ops.shared.models import MetricSnapshot, calculate_percentile
from ai_ops.shared.utils import Event, EventType, event_bus, now_utc


@dataclass
class AggregationWindow:
    """Holds calls for a specific time window and dimension combination."""

    window_start: datetime
    window_seconds: int
    provider: str | None
    model: str | None
    calls: list[dict] = field(default_factory=list)

    def add_call(self, call: dict) -> None:
        """Add a call to this window."""
        self.calls.append(call)

    def compute_snapshot(self) -> MetricSnapshot:
        """Compute aggregated metrics from buffered calls."""
        if not self.calls:
            return MetricSnapshot(
                timestamp=self.window_start,
                window_seconds=self.window_seconds,
                provider=self.provider,
                model=self.model,
            )

        latencies = [c["latency_ms"] for c in self.calls]
        costs = [c["cost_usd"] for c in self.calls]
        tokens = [c["total_tokens"] for c in self.calls]
        errors = [c for c in self.calls if c["status"] != "success"]

        return MetricSnapshot(
            timestamp=self.window_start,
            window_seconds=self.window_seconds,
            provider=self.provider,
            model=self.model,
            call_count=len(self.calls),
            total_tokens=sum(tokens),
            total_cost_usd=sum(costs),
            avg_latency_ms=statistics.mean(latencies) if latencies else 0.0,
            p50_latency_ms=calculate_percentile(latencies, 0.50),
            p95_latency_ms=calculate_percentile(latencies, 0.95),
            p99_latency_ms=calculate_percentile(latencies, 0.99),
            error_count=len(errors),
            error_rate=len(errors) / len(self.calls) if self.calls else 0.0,
        )


class MetricsCollector:
    """
    Collects and aggregates LLM call metrics into time-windowed snapshots.
    
    WHY: Raw call data is too granular for real-time dashboards. We aggregate
    into fixed time windows (default 60s) with dimensions (provider, model)
    to enable fast querying for charts and tables. Uses async buffering to
    avoid blocking the instrumentation path.
    """

    def __init__(self) -> None:
        self.settings = get_settings().metrics
        self._enabled = self.settings.enabled
        self._windows: dict[str, AggregationWindow] = {}
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None
        self._window_size = self.settings.aggregation_window_seconds

        # Subscribe to LLM call events
        event_bus.subscribe(EventType.LLM_CALL_END, self._on_call_end)
        event_bus.subscribe(EventType.LLM_CALL_ERROR, self._on_call_error)

    async def start(self) -> None:
        """Start the metrics collector."""
        if not self._enabled:
            return

        self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        """Stop the metrics collector."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self._flush_all_windows()

    def _get_window_key(self, timestamp: datetime, provider: str | None, model: str | None) -> str:
        """Generate a key for the aggregation window."""
        # Round timestamp to window boundary
        window_start = timestamp.replace(
            second=0, microsecond=0
        ) - timedelta(
            seconds=timestamp.second % self._window_size
        )
        return f"{window_start.isoformat()}|{provider or 'all'}|{model or 'all'}"

    def _get_or_create_window(
        self,
        timestamp: datetime,
        provider: str | None,
        model: str | None,
    ) -> AggregationWindow:
        """Get or create an aggregation window."""
        key = self._get_window_key(timestamp, provider, model)

        if key not in self._windows:
            window_start = timestamp.replace(
                second=0, microsecond=0
            ) - timedelta(
                seconds=timestamp.second % self._window_size
            )
            self._windows[key] = AggregationWindow(
                window_start=window_start,
                window_seconds=self._window_size,
                provider=provider,
                model=model,
            )

        return self._windows[key]

    async def _on_call_end(self, event: Event) -> None:
        """Handle successful LLM call event."""
        await self._process_call(event.payload)

    async def _on_call_error(self, event: Event) -> None:
        """Handle failed LLM call event."""
        await self._process_call(event.payload)

    async def _process_call(self, call_data: dict) -> None:
        """Process a call and add to appropriate aggregation window."""
        async with self._lock:
            timestamp = call_data.get("timestamp")
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            elif not isinstance(timestamp, datetime):
                timestamp = now_utc()

            provider = call_data.get("provider")
            model = call_data.get("model")

            window = self._get_or_create_window(timestamp, provider, model)
            window.add_call(call_data)

    async def _flush_loop(self) -> None:
        """Periodically flush completed windows to database."""
        while self._enabled:
            try:
                await asyncio.sleep(self._window_size)
                await self._flush_completed_windows()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Metrics flush error: {e}")

    async def _flush_completed_windows(self) -> None:
        """Flush windows that are older than the current window."""
        now = now_utc()
        current_window_start = now.replace(
            second=0, microsecond=0
        ) - timedelta(seconds=now.second % self._window_size)

        async with self._lock:
            keys_to_flush = []
            for key, window in self._windows.items():
                if window.window_start < current_window_start:
                    keys_to_flush.append(key)

            for key in keys_to_flush:
                window = self._windows.pop(key)
                await self._write_snapshot(window.compute_snapshot())

    async def _flush_all_windows(self) -> None:
        """Flush all remaining windows."""
        async with self._lock:
            for window in self._windows.values():
                await self._write_snapshot(window.compute_snapshot())
            self._windows.clear()

    async def _write_snapshot(self, snapshot: MetricSnapshot) -> None:
        """Write a metric snapshot to database."""
        async with get_db_session() as session:
            model = snapshot.to_model()
            session.add(model)
            await session.commit()

        # Emit flush event for real-time updates
        await event_bus.emit(Event(
            type=EventType.METRIC_FLUSH,
            payload=snapshot.model_dump(),
        ))

    async def get_latest_metrics(
        self,
        provider: str | None = None,
        model: str | None = None,
        limit: int = 100,
    ) -> list[MetricSnapshot]:
        """Get latest metric snapshots for dashboard."""
        async with get_db_session() as session:
            from sqlalchemy import desc, select

            from ai_ops.shared.models import MetricSnapshotModel

            query = select(MetricSnapshotModel).order_by(desc(MetricSnapshotModel.timestamp)).limit(limit)

            if provider:
                query = query.where(MetricSnapshotModel.provider == provider)
            if model:
                query = query.where(MetricSnapshotModel.model == model)

            result = await session.execute(query)
            models = result.scalars().all()

            return [MetricSnapshot.from_model(m) for m in models]

    async def get_aggregated_metrics(
        self,
        hours: int = 1,
        provider: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Get aggregated metrics for a time range."""
        async with get_db_session() as session:
            from sqlalchemy import func, select

            from ai_ops.shared.models import MetricSnapshotModel

            since = now_utc() - timedelta(hours=hours)

            query = select(
                func.sum(MetricSnapshotModel.call_count).label("total_calls"),
                func.sum(MetricSnapshotModel.total_tokens).label("total_tokens"),
                func.sum(MetricSnapshotModel.total_cost_usd).label("total_cost"),
                func.avg(MetricSnapshotModel.avg_latency_ms).label("avg_latency"),
                func.max(MetricSnapshotModel.p99_latency_ms).label("p99_latency"),
                func.sum(MetricSnapshotModel.error_count).label("total_errors"),
                func.avg(MetricSnapshotModel.error_rate).label("avg_error_rate"),
            ).where(MetricSnapshotModel.timestamp >= since)

            if provider:
                query = query.where(MetricSnapshotModel.provider == provider)
            if model:
                query = query.where(MetricSnapshotModel.model == model)

            result = await session.execute(query)
            row = result.one()

            total_calls = row.total_calls or 0
            total_errors = row.total_errors or 0

            return {
                "total_calls": total_calls,
                "total_tokens": row.total_tokens or 0,
                "total_cost_usd": row.total_cost or 0.0,
                "avg_latency_ms": row.avg_latency or 0.0,
                "p99_latency_ms": row.p99_latency or 0.0,
                "error_count": total_errors,
                "error_rate": total_errors / total_calls if total_calls > 0 else 0.0,
            }

    async def get_model_breakdown(self, hours: int = 1) -> list[dict[str, Any]]:
        """Get usage breakdown by model."""
        async with get_db_session() as session:
            from sqlalchemy import and_, desc, func, select

            from ai_ops.shared.models import MetricSnapshotModel

            since = now_utc() - timedelta(hours=hours)

            query = select(
                MetricSnapshotModel.provider,
                MetricSnapshotModel.model,
                func.sum(MetricSnapshotModel.call_count).label("calls"),
                func.sum(MetricSnapshotModel.total_tokens).label("tokens"),
                func.sum(MetricSnapshotModel.total_cost_usd).label("cost"),
                func.avg(MetricSnapshotModel.avg_latency_ms).label("avg_latency"),
                func.sum(MetricSnapshotModel.error_count).label("errors"),
            ).where(
                and_(
                    MetricSnapshotModel.timestamp >= since,
                    MetricSnapshotModel.model.isnot(None),
                )
            ).group_by(
                MetricSnapshotModel.provider,
                MetricSnapshotModel.model,
            ).order_by(desc("cost"))

            result = await session.execute(query)
            rows = result.all()

            return [
                {
                    "provider": row.provider,
                    "model": row.model,
                    "calls": row.calls,
                    "tokens": row.tokens,
                    "cost_usd": row.cost,
                    "avg_latency_ms": row.avg_latency,
                    "error_count": row.errors,
                }
                for row in rows
            ]

    async def get_time_series(
        self,
        hours: int = 1,
        metric: str = "cost_usd",
        provider: str | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get time series data for charts."""
        async with get_db_session() as session:
            from sqlalchemy import func, select

            from ai_ops.shared.models import MetricSnapshotModel

            since = now_utc() - timedelta(hours=hours)

            metric_column = getattr(MetricSnapshotModel, metric, MetricSnapshotModel.total_cost_usd)

            query = select(
                MetricSnapshotModel.timestamp,
                func.sum(metric_column).label("value"),
            ).where(MetricSnapshotModel.timestamp >= since)

            if provider:
                query = query.where(MetricSnapshotModel.provider == provider)
            if model:
                query = query.where(MetricSnapshotModel.model == model)

            query = query.group_by(MetricSnapshotModel.timestamp).order_by(MetricSnapshotModel.timestamp)

            result = await session.execute(query)
            rows = result.all()

            return [
                {
                    "timestamp": row.timestamp.isoformat(),
                    "value": row.value or 0,
                }
                for row in rows
            ]


# Global metrics collector instance
_metrics_collector: MetricsCollector | None = None


async def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
        await _metrics_collector.start()
    return _metrics_collector


async def shutdown_metrics_collector() -> None:
    """Shutdown the metrics collector."""
    global _metrics_collector
    if _metrics_collector:
        await _metrics_collector.stop()
        _metrics_collector = None
