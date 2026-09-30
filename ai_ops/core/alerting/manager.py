"""
Alerting System.
Monitors metrics and triggers alerts based on configurable thresholds.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from ai_ops.core.storage.database import get_db_session
from ai_ops.shared.config import get_settings
from ai_ops.shared.models import (
    Alert,
    AlertSeverity,
    AlertType,
)
from ai_ops.shared.utils import Event, EventType, event_bus, generate_id, now_utc


class AlertRule:
    """Base class for alert rules."""

    def __init__(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        threshold: float,
        cooldown_seconds: int = 300,
    ) -> None:
        self.alert_type = alert_type
        self.severity = severity
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self._last_triggered: dict[str, datetime] = {}

    def should_trigger(self, key: str, value: float) -> bool:
        """Check if alert should trigger based on threshold and cooldown."""
        if value < self.threshold:
            return False

        last_time = self._last_triggered.get(key)
        if last_time and (now_utc() - last_time).total_seconds() < self.cooldown_seconds:
            return False

        self._last_triggered[key] = now_utc()
        return True

    async def check(self, collector) -> list[Alert]:
        """Check rule and return alerts if triggered."""
        raise NotImplementedError


class CostDailyAlertRule(AlertRule):
    """Alert when daily cost exceeds threshold."""

    def __init__(self, threshold: float = 100.0, cooldown_seconds: int = 3600) -> None:
        super().__init__(
            alert_type=AlertType.COST_DAILY,
            severity=AlertSeverity.WARNING,
            threshold=threshold,
            cooldown_seconds=cooldown_seconds,
        )

    async def check(self, collector) -> list[Alert]:
        metrics = await collector.get_aggregated_metrics(hours=24)
        daily_cost = metrics.get("total_cost_usd", 0)

        if self.should_trigger("daily", daily_cost):
            return [Alert(
                id=generate_id("alert_"),
                timestamp=now_utc(),
                type=AlertType.COST_DAILY,
                severity=self.severity,
                title="Daily Cost Threshold Exceeded",
                message=f"Daily LLM cost (${daily_cost:.2f}) exceeds threshold (${self.threshold:.2f})",
                value=daily_cost,
                threshold=self.threshold,
            )]
        return []


class CostHourlyAlertRule(AlertRule):
    """Alert when hourly cost exceeds threshold."""

    def __init__(self, threshold: float = 20.0, cooldown_seconds: int = 300) -> None:
        super().__init__(
            alert_type=AlertType.COST_HOURLY,
            severity=AlertSeverity.WARNING,
            threshold=threshold,
            cooldown_seconds=cooldown_seconds,
        )

    async def check(self, collector) -> list[Alert]:
        metrics = await collector.get_aggregated_metrics(hours=1)
        hourly_cost = metrics.get("total_cost_usd", 0)

        if self.should_trigger("hourly", hourly_cost):
            return [Alert(
                id=generate_id("alert_"),
                timestamp=now_utc(),
                type=AlertType.COST_HOURLY,
                severity=self.severity,
                title="Hourly Cost Threshold Exceeded",
                message=f"Hourly LLM cost (${hourly_cost:.2f}) exceeds threshold (${self.threshold:.2f})",
                value=hourly_cost,
                threshold=self.threshold,
            )]
        return []


class LatencyP99AlertRule(AlertRule):
    """Alert when P99 latency exceeds threshold."""

    def __init__(self, threshold_ms: int = 10000, cooldown_seconds: int = 300) -> None:
        super().__init__(
            alert_type=AlertType.LATENCY_P99,
            severity=AlertSeverity.CRITICAL,
            threshold=threshold_ms,
            cooldown_seconds=cooldown_seconds,
        )

    async def check(self, collector) -> list[Alert]:
        alerts = []

        # Check overall P99
        metrics = await collector.get_aggregated_metrics(hours=1)
        p99 = metrics.get("p99_latency_ms", 0)

        if self.should_trigger("global_p99", p99):
            alerts.append(Alert(
                id=generate_id("alert_"),
                timestamp=now_utc(),
                type=AlertType.LATENCY_P99,
                severity=self.severity,
                title="Global P99 Latency Threshold Exceeded",
                message=f"P99 latency ({p99:.0f}ms) exceeds threshold ({self.threshold:.0f}ms)",
                value=p99,
                threshold=self.threshold,
            ))

        # Check per-model P99
        breakdown = await collector.get_model_breakdown(hours=1)
        for item in breakdown:
            # Note: We don't have per-model P99 in breakdown, would need separate query
            pass

        return alerts


class ErrorRateAlertRule(AlertRule):
    """Alert when error rate exceeds threshold."""

    def __init__(self, threshold: float = 0.05, cooldown_seconds: int = 300) -> None:
        super().__init__(
            alert_type=AlertType.ERROR_RATE,
            severity=AlertSeverity.CRITICAL,
            threshold=threshold,
            cooldown_seconds=cooldown_seconds,
        )

    async def check(self, collector) -> list[Alert]:
        metrics = await collector.get_aggregated_metrics(hours=1)
        error_rate = metrics.get("error_rate", 0)
        total_calls = metrics.get("total_calls", 0)

        # Only alert if we have enough data
        if total_calls < 10:
            return []

        if self.should_trigger("error_rate", error_rate):
            return [Alert(
                id=generate_id("alert_"),
                timestamp=now_utc(),
                type=AlertType.ERROR_RATE,
                severity=self.severity,
                title="Error Rate Threshold Exceeded",
                message=f"Error rate ({error_rate:.1%}) exceeds threshold ({self.threshold:.1%})",
                value=error_rate,
                threshold=self.threshold,
            )]
        return []


class AlertManager:
    """
    Manages alert rules and triggers alerts based on metrics.
    
    WHY: Centralizes alerting logic with configurable thresholds and cooldowns.
    Integrates with metrics collector for real-time monitoring.
    Supports acknowledgment and resolution workflow.
    """

    def __init__(self) -> None:
        self.settings = get_settings().alerting
        self._enabled = self.settings.enabled
        self._rules: list[AlertRule] = []
        self._check_task: asyncio.Task | None = None

        self._init_rules()

    def _init_rules(self) -> None:
        """Initialize alert rules from settings."""
        if self.settings.enabled:
            self._rules.append(CostDailyAlertRule(
                threshold=self.settings.cost_daily_threshold,
                cooldown_seconds=self.settings.alert_cooldown_seconds,
            ))
            self._rules.append(CostHourlyAlertRule(
                threshold=self.settings.cost_hourly_threshold,
                cooldown_seconds=self.settings.alert_cooldown_seconds,
            ))
            self._rules.append(LatencyP99AlertRule(
                threshold_ms=self.settings.latency_p99_threshold_ms,
                cooldown_seconds=self.settings.alert_cooldown_seconds,
            ))
            self._rules.append(ErrorRateAlertRule(
                threshold=self.settings.error_rate_threshold,
                cooldown_seconds=self.settings.alert_cooldown_seconds,
            ))

    async def start(self) -> None:
        """Start the alert manager."""
        if not self._enabled:
            return

        self._check_task = asyncio.create_task(self._check_loop())

    async def stop(self) -> None:
        """Stop the alert manager."""
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass

    async def _check_loop(self) -> None:
        """Periodically run alert checks."""
        while self._enabled:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self.run_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Alert check error: {e}")

    async def run_checks(self) -> list[Alert]:
        """Run all alert rules."""
        from ai_ops.core.metrics.collector import get_metrics_collector

        collector = await get_metrics_collector()
        all_alerts = []

        for rule in self._rules:
            try:
                alerts = await rule.check(collector)
                all_alerts.extend(alerts)
            except Exception as e:
                print(f"Alert rule {rule.alert_type} failed: {e}")

        # Persist alerts
        if all_alerts:
            await self._persist_alerts(all_alerts)

            # Emit events
            for alert in all_alerts:
                await event_bus.emit(Event(
                    type=EventType.ALERT_TRIGGERED,
                    payload=alert.model_dump(),
                ))

        return all_alerts

    async def _persist_alerts(self, alerts: list[Alert]) -> None:
        """Save alerts to database."""
        async with get_db_session() as session:
            for alert in alerts:
                model = alert.to_model()
                session.add(model)
            await session.commit()

    async def get_alerts(
        self,
        acknowledged: bool | None = None,
        resolved: bool | None = None,
        severity: AlertSeverity | None = None,
        limit: int = 100,
    ) -> list[Alert]:
        """Get stored alerts."""
        async with get_db_session() as session:
            from sqlalchemy import and_, desc, select

            from ai_ops.shared.models import AlertModel

            query = select(AlertModel).order_by(desc(AlertModel.timestamp)).limit(limit)

            conditions = []
            if acknowledged is not None:
                conditions.append(AlertModel.acknowledged == acknowledged)
            if resolved is not None:
                conditions.append(AlertModel.resolved == resolved)
            if severity:
                conditions.append(AlertModel.severity == severity.value)

            if conditions:
                query = query.where(and_(*conditions))

            result = await session.execute(query)
            models = result.scalars().all()

            return [Alert.from_model(m) for m in models]

    async def acknowledge(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged."""
        async with get_db_session() as session:
            from sqlalchemy import select

            from ai_ops.shared.models import AlertModel

            result = await session.execute(
                select(AlertModel).where(AlertModel.id == alert_id)
            )
            model = result.scalar_one_or_none()

            if model:
                model.acknowledged = True
                await session.commit()
                return True
            return False

    async def resolve(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        async with get_db_session() as session:
            from sqlalchemy import select

            from ai_ops.shared.models import AlertModel

            result = await session.execute(
                select(AlertModel).where(AlertModel.id == alert_id)
            )
            model = result.scalar_one_or_none()

            if model:
                model.resolved = True
                await session.commit()
                return True
            return False

    async def get_active_alerts(self) -> list[Alert]:
        """Get all active (unresolved) alerts."""
        return await self.get_alerts(resolved=False, limit=50)


# Global alert manager instance
_alert_manager: AlertManager | None = None


async def get_alert_manager() -> AlertManager:
    """Get or create the global alert manager."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
        await _alert_manager.start()
    return _alert_manager


async def shutdown_alert_manager() -> None:
    """Shutdown the alert manager."""
    global _alert_manager
    if _alert_manager:
        await _alert_manager.stop()
        _alert_manager = None
