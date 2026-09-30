"""
Tests for AI Ops Dashboard core functionality.
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from ai_ops.shared.config import DatabaseSettings, InstrumentationSettings, Settings
from ai_ops.shared.models import (
    Alert,
    AlertSeverity,
    AlertType,
    LLMCall,
    MetricSnapshot,
    ModelProvider,
    Recommendation,
    RecommendationPriority,
    RecommendationType,
    calculate_percentile,
    estimate_cost,
)
from ai_ops.shared.utils import Event, EventBus, EventType, generate_id, now_utc


class TestModels:
    """Test core data models."""

    def test_llm_call_creation(self):
        call = LLMCall(
            provider=ModelProvider.OPENAI,
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=1500,
        )
        assert call.total_tokens == 150
        assert call.provider == ModelProvider.OPENAI
        assert call.model == "gpt-4"

    def test_llm_call_cost_estimation(self):
        call = LLMCall(
            provider=ModelProvider.OPENAI,
            model="gpt-4",
            prompt_tokens=1000,
            completion_tokens=500,
        )
        # gpt-4: $0.03/1k prompt, $0.06/1k completion
        expected = (1000/1000)*0.03 + (500/1000)*0.06
        assert call.cost_usd == expected

    def test_metric_snapshot_creation(self):
        snapshot = MetricSnapshot(
            window_seconds=60,
            provider="openai",
            model="gpt-4",
            call_count=10,
            total_tokens=1500,
            total_cost_usd=0.05,
            avg_latency_ms=1200,
            p50_latency_ms=1000,
            p95_latency_ms=1800,
            p99_latency_ms=2500,
            error_count=1,
        )
        assert snapshot.error_rate == 0.1

    def test_recommendation_creation(self):
        rec = Recommendation(
            type=RecommendationType.MODEL_ROUTING,
            priority=RecommendationPriority.HIGH,
            title="Test recommendation",
            description="Test description",
            estimated_savings_usd=10.0,
            estimated_savings_percent=50.0,
            affected_models=["gpt-4"],
            affected_providers=["openai"],
        )
        assert rec.type == RecommendationType.MODEL_ROUTING
        assert rec.priority == RecommendationPriority.HIGH

    def test_alert_creation(self):
        alert = Alert(
            type=AlertType.COST_DAILY,
            severity=AlertSeverity.WARNING,
            title="Cost Alert",
            message="Daily cost exceeded",
            value=150.0,
            threshold=100.0,
        )
        assert alert.type == AlertType.COST_DAILY
        assert alert.severity == AlertSeverity.WARNING


class TestCostEstimation:
    """Test cost estimation functions."""

    def test_estimate_cost_openai_gpt4(self):
        cost = estimate_cost(ModelProvider.OPENAI, "gpt-4", 1000, 500)
        expected = (1000/1000)*0.03 + (500/1000)*0.06
        assert cost == expected

    def test_estimate_cost_anthropic_claude(self):
        cost = estimate_cost(ModelProvider.ANTHROPIC, "claude-3-sonnet", 1000, 500)
        expected = (1000/1000)*0.003 + (500/1000)*0.015
        assert cost == expected

    def test_estimate_cost_unknown_model(self):
        cost = estimate_cost(ModelProvider.OTHER, "unknown-model", 1000, 500)
        expected = (1000/1000)*0.001 + (500/1000)*0.002
        assert cost == expected


class TestPercentileCalculation:
    """Test percentile calculation."""

    def test_calculate_percentile_empty(self):
        assert calculate_percentile([], 0.5) == 0.0

    def test_calculate_percentile_single(self):
        assert calculate_percentile([100], 0.5) == 100

    def test_calculate_percentile_median(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert calculate_percentile(values, 0.5) == 30.0

    def test_calculate_percentile_p95(self):
        values = [float(i) for i in range(1, 101)]
        # Linear interpolation gives 95.05 for 1-100 range at 0.95
        assert abs(calculate_percentile(values, 0.95) - 95.05) < 0.1

    def test_calculate_percentile_p99(self):
        values = [float(i) for i in range(1, 101)]
        # Linear interpolation gives 99.01 for 1-100 range at 0.99
        assert abs(calculate_percentile(values, 0.99) - 99.01) < 0.1


class TestEventBus:
    """Test event bus functionality."""

    @pytest.mark.asyncio
    async def test_event_bus_subscribe_emit(self):
        bus = EventBus()
        events_received = []

        async def handler(event):
            events_received.append(event)

        bus.subscribe(EventType.LLM_CALL_END, handler)

        event = Event(
            type=EventType.LLM_CALL_END,
            payload={"test": "data"},
        )

        await bus.emit(event)

        assert len(events_received) == 1
        assert events_received[0].type == EventType.LLM_CALL_END
        assert events_received[0].payload["test"] == "data"

    def test_event_bus_sync_emit(self):
        bus = EventBus()
        events_received = []

        def handler(event):
            events_received.append(event)

        bus.subscribe(EventType.LLM_CALL_END, handler)

        event = Event(
            type=EventType.LLM_CALL_END,
            payload={"test": "data"},
        )

        bus.emit_sync(event)

        assert len(events_received) == 1


class TestConfig:
    """Test configuration loading."""

    def test_default_settings(self):
        settings = Settings()
        assert settings.app_name == "AI Ops Dashboard"
        assert settings.environment == "development"
        assert settings.log_level == "INFO"

    def test_database_settings(self):
        db_settings = DatabaseSettings()
        assert db_settings.sqlite_path.name == "ai_ops.db"
        assert db_settings.export_interval_seconds == 300

    def test_instrumentation_settings(self):
        instr_settings = InstrumentationSettings()
        assert instr_settings.enabled is True
        assert instr_settings.sample_rate == 1.0
        assert instr_settings.capture_prompt is False


class TestUtils:
    """Test utility functions."""

    def test_generate_id(self):
        id1 = generate_id()
        id2 = generate_id()
        assert len(id1) == 12
        assert len(id2) == 12
        assert id1 != id2

    def test_generate_id_with_prefix(self):
        id1 = generate_id("test_")
        assert id1.startswith("test_")
        assert len(id1) == 17  # "test_" + 12 chars

    def test_now_utc(self):
        now = now_utc()
        assert isinstance(now, datetime)
        # Should be close to now
        assert (datetime.utcnow() - now).total_seconds() < 1

    def test_to_json(self):
        from ai_ops.shared.utils import to_json
        data = {"key": "value", "number": 42}
        json_str = to_json(data)
        assert '"key": "value"' in json_str
        assert '"number": 42' in json_str


class TestRecommendationEngine:
    """Test recommendation engine (integration-style tests with mocks)."""

    @pytest.mark.asyncio
    async def test_model_routing_rule(self):
        from ai_ops.core.recommendations.engine import ModelRoutingRule

        rule = ModelRoutingRule(cost_threshold=0.05)
        assert rule.name == "model_routing"
        assert rule.type == RecommendationType.MODEL_ROUTING

    @pytest.mark.asyncio
    async def test_caching_rule(self):
        from ai_ops.core.recommendations.engine import CachingRule

        rule = CachingRule(min_calls=5)
        assert rule.name == "caching"
        assert rule.type == RecommendationType.CACHING


class TestAlertManager:
    """Test alert manager."""

    @pytest.mark.asyncio
    async def test_cost_daily_alert_rule(self):
        from ai_ops.core.alerting.manager import CostDailyAlertRule

        rule = CostDailyAlertRule(threshold=100.0)
        assert rule.alert_type == AlertType.COST_DAILY
        assert rule.threshold == 100.0

    @pytest.mark.asyncio
    async def test_error_rate_alert_rule(self):
        from ai_ops.core.alerting.manager import ErrorRateAlertRule

        rule = ErrorRateAlertRule(threshold=0.05)
        assert rule.alert_type == AlertType.ERROR_RATE
        assert rule.threshold == 0.05


# Integration tests (require database)
@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests requiring database."""

    @pytest.mark.asyncio
    async def test_database_manager_init(self):
        import tempfile
        from pathlib import Path

        from ai_ops.core.storage.database import DatabaseManager
        from ai_ops.shared.config import DatabaseSettings

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = MagicMock()
            settings.database = DatabaseSettings(
                sqlite_path=Path(tmpdir) / "test.db",
                parquet_export_path=Path(tmpdir) / "parquet",
                export_interval_seconds=0,  # Disable periodic export
            )

            db = DatabaseManager(settings)
            await db.initialize()

            stats = await db.get_stats()
            assert "llm_calls" in stats

            await db.close()

    @pytest.mark.asyncio
    async def test_parquet_export(self):
        import tempfile
        from pathlib import Path

        from ai_ops.core.storage.database import DatabaseManager
        from ai_ops.shared.config import DatabaseSettings
        from ai_ops.shared.models import LLMCall, ModelProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = MagicMock()
            settings.database = DatabaseSettings(
                sqlite_path=Path(tmpdir) / "test.db",
                parquet_export_path=Path(tmpdir) / "parquet",
                export_interval_seconds=0,
            )

            db = DatabaseManager(settings)
            await db.initialize()

            # Add a test call
            async with db.session() as session:
                call = LLMCall(
                    provider=ModelProvider.OPENAI,
                    model="gpt-4",
                    prompt_tokens=100,
                    completion_tokens=50,
                    latency_ms=1000,
                    cost_usd=0.01,
                )
                session.add(call.to_model())
                await session.commit()

            # Export
            result = await db.export_to_parquet()
            assert "llm_calls" in result
            assert result["llm_calls"] == 1

            await db.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
