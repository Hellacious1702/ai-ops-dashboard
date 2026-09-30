"""
FastAPI Web Application for AI Ops Dashboard.
Provides REST API and HTMX-based real-time dashboard.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from ai_ops.core.alerting.manager import get_alert_manager, shutdown_alert_manager
from ai_ops.core.instrumentation.middleware import init_instrumentation, shutdown_instrumentation
from ai_ops.core.metrics.collector import get_metrics_collector, shutdown_metrics_collector
from ai_ops.core.recommendations.engine import (
    get_recommendation_engine,
    shutdown_recommendation_engine,
)
from ai_ops.core.storage.database import close_db_manager, get_db_manager
from ai_ops.shared.config import get_settings
from ai_ops.shared.models import (
    AlertSeverity,
)

# Global state
_app_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    settings = get_settings()

    # Initialize database
    db_manager = await get_db_manager()
    _app_state["db_manager"] = db_manager

    # Initialize instrumentation
    if settings.instrumentation.enabled:
        init_instrumentation()

    # Initialize metrics collector
    if settings.metrics.enabled:
        await get_metrics_collector()

    # Initialize recommendation engine
    if settings.recommendations.enabled:
        await get_recommendation_engine()

    # Initialize alert manager
    if settings.alerting.enabled:
        await get_alert_manager()

    yield

    # Shutdown
    await shutdown_alert_manager()
    await shutdown_recommendation_engine()
    await shutdown_metrics_collector()
    shutdown_instrumentation()
    await close_db_manager()


app = FastAPI(
    title="AI Ops Dashboard",
    description="LLM Observability, Cost Optimization, and Real-time Monitoring",
    version="0.1.0",
    lifespan=lifespan,
)

settings = get_settings()

# Mount static files
app.mount("/static", StaticFiles(directory="ai_ops/web/static"), name="static")

# Templates
templates = Jinja2Templates(directory="ai_ops/web/templates")


# Request/Response Models
class RecommendationAction(BaseModel):
    action: str  # "acknowledge" or "dismiss"


class AlertAction(BaseModel):
    action: str  # "acknowledge" or "resolve"


# Health Check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    db_manager = _app_state.get("db_manager")
    stats = await db_manager.get_stats() if db_manager else {}

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "database": stats,
    }


# Dashboard Page
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "settings": settings},
    )


# API Endpoints - Metrics
@app.get("/api/metrics/summary")
async def get_metrics_summary(hours: int = Query(1, ge=1, le=168)):
    """Get aggregated metrics summary."""
    collector = await get_metrics_collector()
    return await collector.get_aggregated_metrics(hours=hours)


@app.get("/api/metrics/breakdown")
async def get_model_breakdown(hours: int = Query(1, ge=1, le=168)):
    """Get model usage breakdown."""
    collector = await get_metrics_collector()
    return await collector.get_model_breakdown(hours=hours)


@app.get("/api/metrics/timeseries")
async def get_time_series(
    hours: int = Query(1, ge=1, le=168),
    metric: str = Query("cost_usd"),
    provider: str | None = None,
    model: str | None = None,
):
    """Get time series data for charts."""
    collector = await get_metrics_collector()
    return await collector.get_time_series(
        hours=hours,
        metric=metric,
        provider=provider,
        model=model,
    )


@app.get("/api/metrics/recent")
async def get_recent_metrics(limit: int = Query(100, ge=1, le=1000)):
    """Get recent metric snapshots."""
    collector = await get_metrics_collector()
    return await collector.get_latest_metrics(limit=limit)


# API Endpoints - Recommendations
@app.get("/api/recommendations")
async def get_recommendations(
    acknowledged: bool | None = None,
    dismissed: bool | None = None,
    limit: int = Query(50, ge=1, le=200),
):
    """Get recommendations."""
    engine = await get_recommendation_engine()
    return await engine.get_recommendations(
        acknowledged=acknowledged,
        dismissed=dismissed,
        limit=limit,
    )


@app.get("/api/recommendations/summary")
async def get_recommendations_summary():
    """Get savings summary from recommendations."""
    engine = await get_recommendation_engine()
    return await engine.get_savings_summary()


@app.post("/api/recommendations/{rec_id}/action")
async def recommendation_action(rec_id: str, action: RecommendationAction):
    """Acknowledge or dismiss a recommendation."""
    engine = await get_recommendation_engine()

    if action.action == "acknowledge":
        success = await engine.acknowledge(rec_id)
    elif action.action == "dismiss":
        success = await engine.dismiss(rec_id)
    else:
        raise HTTPException(status_code=400, detail="Invalid action")

    if not success:
        raise HTTPException(status_code=404, detail="Recommendation not found")

    return {"success": True}


# API Endpoints - Alerts
@app.get("/api/alerts")
async def get_alerts(
    acknowledged: bool | None = None,
    resolved: bool | None = None,
    severity: AlertSeverity | None = None,
    limit: int = Query(100, ge=1, le=500),
):
    """Get alerts."""
    manager = await get_alert_manager()
    return await manager.get_alerts(
        acknowledged=acknowledged,
        resolved=resolved,
        severity=severity,
        limit=limit,
    )


@app.get("/api/alerts/active")
async def get_active_alerts():
    """Get active (unresolved) alerts."""
    manager = await get_alert_manager()
    return await manager.get_active_alerts()


@app.post("/api/alerts/{alert_id}/action")
async def alert_action(alert_id: str, action: AlertAction):
    """Acknowledge or resolve an alert."""
    manager = await get_alert_manager()

    if action.action == "acknowledge":
        success = await manager.acknowledge(alert_id)
    elif action.action == "resolve":
        success = await manager.resolve(alert_id)
    else:
        raise HTTPException(status_code=400, detail="Invalid action")

    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")

    return {"success": True}


# API Endpoints - Database
@app.post("/api/database/export")
async def export_parquet():
    """Trigger Parquet export."""
    db_manager = _app_state.get("db_manager")
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not initialized")

    result = await db_manager.export_to_parquet()
    return {"success": True, "exported": result}


@app.get("/api/database/stats")
async def get_database_stats():
    """Get database statistics."""
    db_manager = _app_state.get("db_manager")
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not initialized")

    return await db_manager.get_stats()


# HTMX Partial Endpoints
@app.get("/partials/metrics/summary")
async def partial_metrics_summary(hours: int = Query(1)):
    """HTMX partial for metrics summary cards."""
    collector = await get_metrics_collector()
    metrics = await collector.get_aggregated_metrics(hours=hours)
    return templates.TemplateResponse(
        "partials/metrics_summary.html",
        {"request": {}, "metrics": metrics, "hours": hours},
    )


@app.get("/partials/metrics/breakdown")
async def partial_model_breakdown(hours: int = Query(1)):
    """HTMX partial for model breakdown table."""
    collector = await get_metrics_collector()
    breakdown = await collector.get_model_breakdown(hours=hours)
    return templates.TemplateResponse(
        "partials/model_breakdown.html",
        {"request": {}, "breakdown": breakdown, "hours": hours},
    )


@app.get("/partials/recommendations/list")
async def partial_recommendations_list(
    acknowledged: bool | None = None,
    dismissed: bool | None = None,
):
    """HTMX partial for recommendations list."""
    engine = await get_recommendation_engine()
    recommendations = await engine.get_recommendations(
        acknowledged=acknowledged,
        dismissed=dismissed,
        limit=20,
    )
    return templates.TemplateResponse(
        "partials/recommendations_list.html",
        {"request": {}, "recommendations": recommendations},
    )


@app.get("/partials/alerts/list")
async def partial_alerts_list(
    acknowledged: bool | None = None,
    resolved: bool | None = None,
):
    """HTMX partial for alerts list."""
    manager = await get_alert_manager()
    alerts = await manager.get_alerts(
        acknowledged=acknowledged,
        resolved=resolved,
        limit=20,
    )
    return templates.TemplateResponse(
        "partials/alerts_list.html",
        {"request": {}, "alerts": alerts},
    )


@app.get("/partials/charts/cost")
async def partial_cost_chart(hours: int = Query(1)):
    """HTMX partial for cost chart."""
    collector = await get_metrics_collector()
    data = await collector.get_time_series(hours=hours, metric="total_cost_usd")
    return templates.TemplateResponse(
        "partials/cost_chart.html",
        {"request": {}, "chart_data": data, "hours": hours},
    )


@app.get("/partials/charts/latency")
async def partial_latency_chart(hours: int = Query(1)):
    """HTMX partial for latency chart."""
    collector = await get_metrics_collector()
    data = await collector.get_time_series(hours=hours, metric="p99_latency_ms")
    return templates.TemplateResponse(
        "partials/latency_chart.html",
        {"request": {}, "chart_data": data, "hours": hours},
    )


@app.get("/partials/charts/tokens")
async def partial_tokens_chart(hours: int = Query(1)):
    """HTMX partial for tokens chart."""
    collector = await get_metrics_collector()
    data = await collector.get_time_series(hours=hours, metric="total_tokens")
    return templates.TemplateResponse(
        "partials/tokens_chart.html",
        {"request": {}, "chart_data": data, "hours": hours},
    )


@app.get("/partials/charts/errors")
async def partial_errors_chart(hours: int = Query(1)):
    """HTMX partial for error rate chart."""
    collector = await get_metrics_collector()
    data = await collector.get_time_series(hours=hours, metric="error_rate")
    return templates.TemplateResponse(
        "partials/errors_chart.html",
        {"request": {}, "chart_data": data, "hours": hours},
    )


# WebSocket for real-time updates (optional)
@app.websocket("/ws")
async def websocket_endpoint(websocket):
    """WebSocket for real-time dashboard updates."""
    await websocket.accept()
    try:
        while True:
            # Send periodic updates
            collector = await get_metrics_collector()
            metrics = await collector.get_aggregated_metrics(hours=1)

            await websocket.send_json({
                "type": "metrics_update",
                "data": metrics,
                "timestamp": datetime.utcnow().isoformat(),
            })

            await asyncio.sleep(5)
    except Exception:
        pass
    finally:
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "ai_ops.web.api:app",
        host=settings.web.host,
        port=settings.web.port,
        reload=settings.web.reload,
    )
