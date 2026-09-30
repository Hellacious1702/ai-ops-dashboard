"""
Cost Optimization Recommendation Engine.
Analyzes usage patterns and generates actionable recommendations for cost reduction.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from ai_ops.core.storage.database import get_db_session
from ai_ops.shared.config import get_settings
from ai_ops.shared.models import (
    Recommendation,
    RecommendationPriority,
    RecommendationType,
)
from ai_ops.shared.utils import Event, EventType, event_bus, generate_id, now_utc


@dataclass
class RecommendationRule:
    """Base class for recommendation rules."""

    name: str
    type: RecommendationType
    enabled: bool = True
    priority: RecommendationPriority = RecommendationPriority.MEDIUM
    cooldown_hours: int = 24

    @property
    def cooldown_seconds(self) -> int:
        return self.cooldown_hours * 3600


class ModelRoutingRule(RecommendationRule):
    """
    Recommends routing to cheaper models when cost exceeds threshold.
    
    WHY: Many applications use expensive models (GPT-4) for tasks that cheaper
    models (GPT-3.5, GPT-4o-mini) can handle. This rule detects high-cost
    usage patterns and suggests alternatives.
    """

    def __init__(self, cost_threshold: float = 0.10, latency_threshold_ms: int = 5000) -> None:
        super().__init__(
            name="model_routing",
            type=RecommendationType.MODEL_ROUTING,
            priority=RecommendationPriority.HIGH,
        )
        self.cost_threshold = cost_threshold  # Cost per 1k tokens
        self.latency_threshold_ms = latency_threshold_ms

    async def check(self, collector) -> list[Recommendation]:
        recommendations = []

        # Get model breakdown for last 24 hours
        breakdown = await collector.get_model_breakdown(hours=24)

        # Define cheaper alternatives
        cheaper_alternatives = {
            "gpt-4": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            "gpt-4-turbo": ["gpt-4o", "gpt-3.5-turbo"],
            "claude-3-opus": ["claude-3.5-sonnet", "claude-3-haiku"],
            "claude-3-sonnet": ["claude-3.5-sonnet", "claude-3-haiku"],
        }

        for item in breakdown:
            model = item["model"].lower()
            cost_per_1k = (item["cost_usd"] / max(item["tokens"], 1)) * 1000

            if cost_per_1k > self.cost_threshold and model in cheaper_alternatives:
                alternatives = cheaper_alternatives[model]

                # Estimate savings (assume 70% cost reduction)
                estimated_savings = item["cost_usd"] * 0.7

                rec = Recommendation(
                    id=generate_id("rec_"),
                    timestamp=now_utc(),
                    type=RecommendationType.MODEL_ROUTING,
                    priority=RecommendationPriority.HIGH,
                    title=f"Route {item['model']} traffic to cheaper alternatives",
                    description=(
                        f"Model {item['model']} costs ${cost_per_1k:.4f}/1k tokens. "
                        f"Consider routing to: {', '.join(alternatives)}. "
                        f"Potential savings: ${estimated_savings:.2f}/day"
                    ),
                    estimated_savings_usd=estimated_savings,
                    estimated_savings_percent=70.0,
                    affected_models=[item["model"]],
                    affected_providers=[item["provider"]],
                    metadata={
                        "current_cost_per_1k": cost_per_1k,
                        "alternatives": alternatives,
                        "daily_cost": item["cost_usd"],
                        "daily_calls": item["calls"],
                    },
                )
                recommendations.append(rec)

        return recommendations


class CachingRule(RecommendationRule):
    """
    Recommends caching for repeated identical/similar prompts.
    
    WHY: Many applications send identical prompts repeatedly (e.g., system prompts,
    common queries). Caching can eliminate redundant LLM calls entirely.
    """

    def __init__(
        self,
        min_calls: int = 10,
        similarity_threshold: float = 0.95,
    ) -> None:
        super().__init__(
            name="caching",
            type=RecommendationType.CACHING,
            priority=RecommendationPriority.MEDIUM,
        )
        self.min_calls = min_calls
        self.similarity_threshold = similarity_threshold

    async def check(self, collector) -> list[Recommendation]:
        recommendations = []

        async with get_db_session() as session:
            from sqlalchemy import and_, func, select

            from ai_ops.shared.models import LLMCallModel

            since = now_utc() - timedelta(hours=24)

            # Find repeated prompt hashes
            query = select(
                LLMCallModel.prompt_hash,
                LLMCallModel.provider,
                LLMCallModel.model,
                func.count(LLMCallModel.id).label("call_count"),
                func.sum(LLMCallModel.total_tokens).label("total_tokens"),
                func.sum(LLMCallModel.cost_usd).label("total_cost"),
                func.avg(LLMCallModel.latency_ms).label("avg_latency"),
            ).where(
                and_(
                    LLMCallModel.timestamp >= since,
                    LLMCallModel.prompt_hash.isnot(None),
                    LLMCallModel.status == "success",
                )
            ).group_by(
                LLMCallModel.prompt_hash,
                LLMCallModel.provider,
                LLMCallModel.model,
            ).having(
                func.count(LLMCallModel.id) >= self.min_calls
            ).order_by(func.count(LLMCallModel.id).desc())

            result = await session.execute(query)
            rows = result.all()

            for row in rows:
                if row.call_count >= self.min_calls:
                    # Estimate savings from caching (eliminate all but first call)
                    cacheable_calls = row.call_count - 1
                    estimated_savings = (row.total_cost / row.call_count) * cacheable_calls

                    rec = Recommendation(
                        id=generate_id("rec_"),
                        timestamp=now_utc(),
                        type=RecommendationType.CACHING,
                        priority=RecommendationPriority.MEDIUM,
                        title=f"Cache repeated prompts for {row.model}",
                        description=(
                            f"Prompt hash {row.prompt_hash[:8]}... called {row.call_count} times "
                            f"in 24h. Implement semantic caching to save "
                            f"${estimated_savings:.2f}/day ({cacheable_calls} calls eliminated)"
                        ),
                        estimated_savings_usd=estimated_savings,
                        estimated_savings_percent=((cacheable_calls / row.call_count) * 100),
                        affected_models=[row.model],
                        affected_providers=[row.provider],
                        metadata={
                            "prompt_hash": row.prompt_hash,
                            "call_count": row.call_count,
                            "cacheable_calls": cacheable_calls,
                            "avg_latency_ms": row.avg_latency,
                            "similarity_threshold": self.similarity_threshold,
                        },
                    )
                    recommendations.append(rec)

        return recommendations


class PromptCompressionRule(RecommendationRule):
    """
    Recommends prompt compression for long prompts.
    
    WHY: Long prompts cost more and increase latency. Techniques like
    summarization, removing redundancy, or using RAG can reduce token usage.
    """

    def __init__(
        self,
        min_tokens: int = 1000,
        target_reduction: float = 0.3,
    ) -> None:
        super().__init__(
            name="prompt_compression",
            type=RecommendationType.PROMPT_COMPRESSION,
            priority=RecommendationPriority.MEDIUM,
        )
        self.min_tokens = min_tokens
        self.target_reduction = target_reduction

    async def check(self, collector) -> list[Recommendation]:
        recommendations = []

        async with get_db_session() as session:
            from sqlalchemy import and_, func, select

            from ai_ops.shared.models import LLMCallModel

            since = now_utc() - timedelta(hours=24)

            # Find models with high average prompt tokens
            query = select(
                LLMCallModel.provider,
                LLMCallModel.model,
                func.avg(LLMCallModel.prompt_tokens).label("avg_prompt_tokens"),
                func.avg(LLMCallModel.cost_usd).label("avg_cost"),
                func.count(LLMCallModel.id).label("call_count"),
                func.sum(LLMCallModel.cost_usd).label("total_cost"),
            ).where(
                and_(
                    LLMCallModel.timestamp >= since,
                    LLMCallModel.prompt_tokens >= self.min_tokens,
                    LLMCallModel.status == "success",
                )
            ).group_by(
                LLMCallModel.provider,
                LLMCallModel.model,
            ).having(
                func.count(LLMCallModel.id) >= 5  # At least 5 calls
            ).order_by(func.avg(LLMCallModel.prompt_tokens).desc())

            result = await session.execute(query)
            rows = result.all()

            for row in rows:
                if row.avg_prompt_tokens >= self.min_tokens:
                    # Estimate savings from compression
                    tokens_saved = row.avg_prompt_tokens * self.target_reduction
                    cost_per_token = row.avg_cost / max(row.avg_prompt_tokens + 1, 1)  # Rough estimate
                    estimated_savings = tokens_saved * cost_per_token * row.call_count

                    rec = Recommendation(
                        id=generate_id("rec_"),
                        timestamp=now_utc(),
                        type=RecommendationType.PROMPT_COMPRESSION,
                        priority=RecommendationPriority.MEDIUM,
                        title=f"Compress prompts for {row.model}",
                        description=(
                            f"Average prompt length: {row.avg_prompt_tokens:.0f} tokens. "
                            f"Target {self.target_reduction:.0%} reduction could save "
                            f"~{tokens_saved:.0f} tokens/call (${estimated_savings:.2f}/day)"
                        ),
                        estimated_savings_usd=estimated_savings,
                        estimated_savings_percent=self.target_reduction * 100,
                        affected_models=[row.model],
                        affected_providers=[row.provider],
                        metadata={
                            "avg_prompt_tokens": row.avg_prompt_tokens,
                            "target_reduction": self.target_reduction,
                            "tokens_saved_per_call": tokens_saved,
                            "daily_calls": row.call_count,
                        },
                    )
                    recommendations.append(rec)

        return recommendations


class BatchingRule(RecommendationRule):
    """
    Recommends request batching for high-frequency similar calls.
    
    WHY: Batching multiple requests into a single API call reduces overhead
    and can qualify for volume discounts on some providers.
    """

    def __init__(
        self,
        min_calls: int = 5,
        window_seconds: int = 60,
    ) -> None:
        super().__init__(
            name="batching",
            type=RecommendationType.BATCHING,
            priority=RecommendationPriority.LOW,
        )
        self.min_calls = min_calls
        self.window_seconds = window_seconds

    async def check(self, collector) -> list[Recommendation]:
        recommendations = []

        async with get_db_session() as session:
            from sqlalchemy import and_, func, select

            from ai_ops.shared.models import LLMCallModel

            since = now_utc() - timedelta(hours=1)

            # Find high-frequency model usage in short windows
            query = select(
                LLMCallModel.provider,
                LLMCallModel.model,
                func.count(LLMCallModel.id).label("call_count"),
                func.sum(LLMCallModel.total_tokens).label("total_tokens"),
                func.sum(LLMCallModel.cost_usd).label("total_cost"),
            ).where(
                and_(
                    LLMCallModel.timestamp >= since,
                    LLMCallModel.status == "success",
                )
            ).group_by(
                LLMCallModel.provider,
                LLMCallModel.model,
            ).having(
                func.count(LLMCallModel.id) >= self.min_calls * 60 / self.window_seconds
            ).order_by(func.count(LLMCallModel.id).desc())

            result = await session.execute(query)
            rows = result.all()

            for row in rows:
                calls_per_minute = row.call_count / 60  # Last hour

                if calls_per_minute >= self.min_calls / self.window_seconds * 60:
                    # Estimate savings from batching (10-20% overhead reduction)
                    estimated_savings = row.total_cost * 0.15

                    rec = Recommendation(
                        id=generate_id("rec_"),
                        timestamp=now_utc(),
                        type=RecommendationType.BATCHING,
                        priority=RecommendationPriority.LOW,
                        title=f"Batch requests for {row.model}",
                        description=(
                            f"~{calls_per_minute:.1f} calls/minute. "
                            f"Batching could reduce API overhead and save "
                            f"${estimated_savings:.2f}/hour"
                        ),
                        estimated_savings_usd=estimated_savings * 24,  # Daily estimate
                        estimated_savings_percent=15.0,
                        affected_models=[row.model],
                        affected_providers=[row.provider],
                        metadata={
                            "calls_per_minute": calls_per_minute,
                            "window_seconds": self.window_seconds,
                            "hourly_cost": row.total_cost,
                        },
                    )
                    recommendations.append(rec)

        return recommendations


class RecommendationEngine:
    """
    Orchestrates recommendation rules and generates cost optimization suggestions.
    
    WHY: Centralizes all recommendation logic with configurable rules.
    Runs periodically and emits events for real-time dashboard updates.
    Supports acknowledgment and dismissal to reduce noise.
    """

    def __init__(self) -> None:
        self.settings = get_settings().recommendations
        self._enabled = self.settings.enabled
        self._rules: list[RecommendationRule] = []
        self._check_task: asyncio.Task | None = None
        self._last_recommendations: dict[str, datetime] = {}  # For cooldown

        self._init_rules()

    def _init_rules(self) -> None:
        """Initialize recommendation rules from settings."""
        if self.settings.routing_enabled:
            self._rules.append(ModelRoutingRule(
                cost_threshold=self.settings.routing_cost_threshold,
                latency_threshold_ms=self.settings.routing_latency_threshold_ms,
            ))

        if self.settings.caching_enabled:
            self._rules.append(CachingRule(
                min_calls=self.settings.caching_min_calls,
                similarity_threshold=self.settings.caching_similarity_threshold,
            ))

        if self.settings.compression_enabled:
            self._rules.append(PromptCompressionRule(
                min_tokens=self.settings.compression_min_tokens,
                target_reduction=self.settings.compression_target_reduction,
            ))

        if self.settings.batching_enabled:
            self._rules.append(BatchingRule(
                min_calls=self.settings.batching_min_calls,
                window_seconds=self.settings.batching_window_seconds,
            ))

    async def start(self) -> None:
        """Start the recommendation engine."""
        if not self._enabled:
            return

        self._check_task = asyncio.create_task(self._check_loop())

    async def stop(self) -> None:
        """Stop the recommendation engine."""
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass

    async def _check_loop(self) -> None:
        """Periodically run recommendation checks."""
        while self._enabled:
            try:
                await asyncio.sleep(self.settings.check_interval_seconds)
                await self.run_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Recommendation check error: {e}")

    async def run_checks(self) -> list[Recommendation]:
        """Run all enabled recommendation rules."""
        from ai_ops.core.metrics.collector import get_metrics_collector

        collector = await get_metrics_collector()
        all_recommendations = []

        for rule in self._rules:
            if not rule.enabled:
                continue

            try:
                recommendations = await rule.check(collector)

                # Apply cooldown
                filtered = []
                for rec in recommendations:
                    key = f"{rule.name}:{','.join(rec.affected_models)}"
                    last_time = self._last_recommendations.get(key)

                    if last_time is None or (now_utc() - last_time).total_seconds() >= rule.cooldown_seconds:
                        self._last_recommendations[key] = now_utc()
                        filtered.append(rec)

                all_recommendations.extend(filtered)

            except Exception as e:
                print(f"Rule {rule.name} failed: {e}")

        # Persist recommendations
        if all_recommendations:
            await self._persist_recommendations(all_recommendations)

            # Emit events
            for rec in all_recommendations:
                await event_bus.emit(Event(
                    type=EventType.RECOMMENDATION_GENERATED,
                    payload=rec.model_dump(),
                ))

        return all_recommendations

    async def _persist_recommendations(self, recommendations: list[Recommendation]) -> None:
        """Save recommendations to database."""
        async with get_db_session() as session:
            for rec in recommendations:
                model = rec.to_model()
                session.add(model)
            await session.commit()

    async def get_recommendations(
        self,
        acknowledged: bool | None = None,
        dismissed: bool | None = None,
        limit: int = 50,
    ) -> list[Recommendation]:
        """Get stored recommendations."""
        async with get_db_session() as session:
            from sqlalchemy import and_, desc, select

            from ai_ops.shared.models import RecommendationModel

            query = select(RecommendationModel).order_by(desc(RecommendationModel.timestamp)).limit(limit)

            conditions = []
            if acknowledged is not None:
                conditions.append(RecommendationModel.acknowledged == acknowledged)
            if dismissed is not None:
                conditions.append(RecommendationModel.dismissed == dismissed)

            if conditions:
                query = query.where(and_(*conditions))

            result = await session.execute(query)
            models = result.scalars().all()

            return [Recommendation.from_model(m) for m in models]

    async def acknowledge(self, recommendation_id: str) -> bool:
        """Mark a recommendation as acknowledged."""
        async with get_db_session() as session:
            from sqlalchemy import select

            from ai_ops.shared.models import RecommendationModel

            result = await session.execute(
                select(RecommendationModel).where(RecommendationModel.id == recommendation_id)
            )
            model = result.scalar_one_or_none()

            if model:
                model.acknowledged = True
                await session.commit()
                return True
            return False

    async def dismiss(self, recommendation_id: str) -> bool:
        """Dismiss a recommendation."""
        async with get_db_session() as session:
            from sqlalchemy import select

            from ai_ops.shared.models import RecommendationModel

            result = await session.execute(
                select(RecommendationModel).where(RecommendationModel.id == recommendation_id)
            )
            model = result.scalar_one_or_none()

            if model:
                model.dismissed = True
                await session.commit()
                return True
            return False

    async def get_savings_summary(self) -> dict[str, Any]:
        """Get summary of potential savings from all recommendations."""
        recommendations = await self.get_recommendations(dismissed=False, limit=100)

        total_savings = sum(r.estimated_savings_usd for r in recommendations)
        by_type = {}
        by_priority = {}

        for rec in recommendations:
            by_type[rec.type.value] = by_type.get(rec.type.value, 0) + rec.estimated_savings_usd
            by_priority[rec.priority.value] = by_priority.get(rec.priority.value, 0) + rec.estimated_savings_usd

        return {
            "total_potential_savings_usd": total_savings,
            "recommendation_count": len(recommendations),
            "by_type": by_type,
            "by_priority": by_priority,
        }


# Global recommendation engine instance
_recommendation_engine: RecommendationEngine | None = None


async def get_recommendation_engine() -> RecommendationEngine:
    """Get or create the global recommendation engine."""
    global _recommendation_engine
    if _recommendation_engine is None:
        _recommendation_engine = RecommendationEngine()
        await _recommendation_engine.start()
    return _recommendation_engine


async def shutdown_recommendation_engine() -> None:
    """Shutdown the recommendation engine."""
    global _recommendation_engine
    if _recommendation_engine:
        await _recommendation_engine.stop()
        _recommendation_engine = None
