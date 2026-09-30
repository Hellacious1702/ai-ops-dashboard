"""
CLI Entry Point for AI Ops Dashboard.
Provides commands to run web server, TUI, and management utilities.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from ai_ops.core.alerting.manager import get_alert_manager
from ai_ops.core.instrumentation.middleware import init_instrumentation, shutdown_instrumentation
from ai_ops.core.recommendations.engine import (
    get_recommendation_engine,
)
from ai_ops.core.storage.database import get_db_manager
from ai_ops.shared.config import get_settings, reload_settings

app = typer.Typer(
    name="ai-ops",
    help="AI Operations Dashboard - LLM Observability & Cost Optimization",
    add_completion=False,
)

console = Console()


@app.command()
def web(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
) -> None:
    """Run the FastAPI web server."""
    import uvicorn

    settings = get_settings()
    settings.web.host = host
    settings.web.port = port
    settings.web.reload = reload

    console.print(f"[green]Starting web server on {host}:{port}[/green]")
    uvicorn.run(
        "ai_ops.web.api:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def tui() -> None:
    """Run the Textual TUI dashboard."""
    from ai_ops.tui.app import run_tui

    console.print("[green]Starting TUI dashboard...[/green]")
    run_tui()


@app.command()
def init_db() -> None:
    """Initialize the database."""
    async def _init() -> None:
        db_manager = await get_db_manager()
        stats = await db_manager.get_stats()
        console.print("[green]Database initialized[/green]")
        console.print(f"Tables: {stats}")

    asyncio.run(_init())


@app.command()
def export_parquet() -> None:
    """Export database to Parquet format."""
    async def _export() -> None:
        db_manager = await get_db_manager()
        result = await db_manager.export_to_parquet()
        console.print(f"[green]Parquet export complete:[/green] {result}")

    asyncio.run(_export())


@app.command()
def db_stats() -> None:
    """Show database statistics."""
    async def _stats() -> None:
        db_manager = await get_db_manager()
        stats = await db_manager.get_stats()

        table = Table(title="Database Statistics")
        table.add_column("Table", style="cyan")
        table.add_column("Rows", justify="right", style="green")

        for table_name, count in stats.items():
            table.add_row(table_name, str(count))

        console.print(table)

    asyncio.run(_stats())


@app.command()
def recommendations(
    acknowledged: bool | None = typer.Option(None, "--acknowledged/--unacknowledged"),
    dismissed: bool | None = typer.Option(None, "--dismissed/--active"),
    limit: int = typer.Option(20, "--limit", "-l"),
) -> None:
    """List cost optimization recommendations."""
    async def _list() -> None:
        engine = await get_recommendation_engine()
        recs = await engine.get_recommendations(
            acknowledged=acknowledged,
            dismissed=dismissed,
            limit=limit,
        )

        if not recs:
            console.print("[yellow]No recommendations found[/yellow]")
            return

        table = Table(title="Recommendations")
        table.add_column("ID", style="dim")
        table.add_column("Type", style="cyan")
        table.add_column("Priority", style="yellow")
        table.add_column("Title")
        table.add_column("Savings/day", justify="right", style="green")
        table.add_column("Models")

        for rec in recs:
            table.add_row(
                rec.id[:8],
                rec.type.value,
                rec.priority.value.upper(),
                rec.title[:50],
                f"${rec.estimated_savings_usd:.2f}",
                ", ".join(rec.affected_models[:2]),
            )

        console.print(table)

    asyncio.run(_list())


@app.command()
def alerts(
    acknowledged: bool | None = typer.Option(None, "--acknowledged/--unacknowledged"),
    resolved: bool | None = typer.Option(None, "--resolved/--active"),
    limit: int = typer.Option(20, "--limit", "-l"),
) -> None:
    """List alerts."""
    async def _list() -> None:
        manager = await get_alert_manager()
        alert_list = await manager.get_alerts(
            acknowledged=acknowledged,
            resolved=resolved,
            limit=limit,
        )

        if not alert_list:
            console.print("[green]No alerts found[/green]")
            return

        table = Table(title="Alerts")
        table.add_column("ID", style="dim")
        table.add_column("Type", style="cyan")
        table.add_column("Severity", style="red")
        table.add_column("Title")
        table.add_column("Value", justify="right")
        table.add_column("Threshold", justify="right")
        table.add_column("Status")

        for alert in alert_list:
            status = []
            if alert.acknowledged:
                status.append("ACK")
            if alert.resolved:
                status.append("RESOLVED")

            table.add_row(
                alert.id[:8],
                alert.type.value,
                alert.severity.value.upper(),
                alert.title[:40],
                f"{alert.value:.2f}",
                f"{alert.threshold:.2f}",
                " ".join(status) or "ACTIVE",
            )

        console.print(table)

    asyncio.run(_list())


@app.command()
def test_instrumentation(
    model: str = typer.Option("gpt-3.5-turbo", "--model", "-m"),
    provider: str = typer.Option("openai", "--provider"),
    calls: int = typer.Option(5, "--calls", "-n"),
) -> None:
    """Test instrumentation by making sample LLM calls."""
    import litellm

    init_instrumentation()

    console.print(f"[green]Making {calls} test calls to {model}...[/green]")

    for i in range(calls):
        try:
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": f"Test message {i+1}"}],
                max_tokens=10,
            )
            console.print(f"  Call {i+1}: {response.usage.total_tokens} tokens, ${response._hidden_params.get('response_cost', 0):.6f}")
        except Exception as e:
            console.print(f"  Call {i+1}: [red]Error: {e}[/red]")

    shutdown_instrumentation()
    console.print("[green]Test complete[/green]")


@app.command()
def config() -> None:
    """Show current configuration."""
    settings = get_settings()

    table = Table(title="Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    def add_settings(prefix: str, obj: Any) -> None:
        for key, value in obj.__dict__.items():
            if not key.startswith("_"):
                table.add_row(f"{prefix}.{key}", str(value))

    table.add_row("app_name", settings.app_name)
    table.add_row("environment", settings.environment)
    table.add_row("log_level", settings.log_level)
    table.add_row("data_dir", str(settings.data_dir))

    add_settings("database", settings.database)
    add_settings("instrumentation", settings.instrumentation)
    add_settings("metrics", settings.metrics)
    add_settings("recommendations", settings.recommendations)
    add_settings("alerting", settings.alerting)
    add_settings("web", settings.web)
    add_settings("tui", settings.tui)

    console.print(table)


@app.callback()
def main(
    env_file: Path | None = typer.Option(None, "--env-file", "-e", help="Path to .env file"),
    log_level: str = typer.Option("INFO", "--log-level", help="Log level"),
) -> None:
    """AI Ops Dashboard - LLM Observability & Cost Optimization."""
    if env_file:
        from dotenv import load_dotenv
        load_dotenv(env_file, override=True)

    # Reload settings with new env
    reload_settings()

    import logging
    logging.basicConfig(level=log_level)


if __name__ == "__main__":
    app()
