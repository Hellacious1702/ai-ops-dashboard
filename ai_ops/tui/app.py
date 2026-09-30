"""
Textual TUI Dashboard for AI Ops Dashboard.
Provides terminal-based real-time monitoring with live updates.
"""

from __future__ import annotations

from rich.align import Align
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import (
    Footer,
    Header,
    Sparkline,
    Static,
    TabbedContent,
    TabPane,
)

from ai_ops.core.alerting.manager import get_alert_manager
from ai_ops.core.metrics.collector import get_metrics_collector
from ai_ops.core.recommendations.engine import get_recommendation_engine
from ai_ops.core.storage.database import get_db_manager


class MetricCard(Static):
    """A metric display card for the TUI."""

    value = reactive("0")
    label = reactive("Metric")
    trend = reactive("")
    style_class = reactive("")

    def render(self) -> Panel:
        content = Text()
        content.append(f"{self.value}\n", style="bold white")
        content.append(self.label, style="dim white")
        if self.trend:
            trend_style = "green" if self.trend.startswith("+") else "red"
            content.append(f" {self.trend}", style=trend_style)

        return Panel(
            Align.center(content, vertical="middle"),
            border_style=self.style_class or "blue",
            padding=(1, 2),
        )


class CostChart(Static):
    """Sparkline chart for cost trends."""

    data = reactive([])

    def render(self) -> Panel:
        if not self.data:
            return Panel("No data", border_style="blue", title="Cost Trend")

        values = [d.get("value", 0) for d in self.data[-60:]]
        labels = [d.get("timestamp", "")[-8:-3] for d in self.data[-60:]]

        sparkline = Sparkline(values, width=80)

        content = Text()
        content.append("Cost (USD)\n", style="dim")
        content.append(str(sparkline))

        return Panel(content, border_style="green", title="Cost Trend")


class LatencyChart(Static):
    """Sparkline chart for latency percentiles."""

    data = reactive([])

    def render(self) -> Panel:
        if not self.data:
            return Panel("No data", border_style="blue", title="Latency")

        p50 = [d.get("p50_latency_ms", 0) for d in self.data[-60:]]
        p95 = [d.get("p95_latency_ms", 0) for d in self.data[-60:]]
        p99 = [d.get("p99_latency_ms", 0) for d in self.data[-60:]]

        content = Text()
        content.append("P50: ", style="green")
        content.append(str(Sparkline(p50, width=40)))
        content.append("\nP95: ", style="yellow")
        content.append(str(Sparkline(p95, width=40)))
        content.append("\nP99: ", style="red")
        content.append(str(Sparkline(p99, width=40)))

        return Panel(content, border_style="blue", title="Latency (ms)")


class ModelTable(Static):
    """Table showing model usage breakdown."""

    data = reactive([])

    def render(self) -> Panel:
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Provider", style="white")
        table.add_column("Model", style="dim white")
        table.add_column("Calls", justify="right", style="white")
        table.add_column("Tokens", justify="right", style="dim white")
        table.add_column("Cost", justify="right", style="green")
        table.add_column("Latency", justify="right", style="yellow")
        table.add_column("Errors", justify="right", style="red")

        for item in self.data[:20]:
            table.add_row(
                item.get("provider", "N/A"),
                item.get("model", "N/A")[:30],
                str(item.get("calls", 0)),
                f"{item.get('tokens', 0):,}",
                f"${item.get('cost_usd', 0):.4f}",
                f"{item.get('avg_latency_ms', 0):.0f}ms",
                str(item.get("errors", 0)),
            )

        return Panel(table, border_style="cyan", title="Model Breakdown")


class RecommendationsList(Static):
    """List of cost optimization recommendations."""

    data = reactive([])

    def render(self) -> Panel:
        if not self.data:
            return Panel("No recommendations", border_style="yellow", title="Recommendations")

        table = Table(show_header=True, header_style="bold yellow", box=None)
        table.add_column("Type", style="cyan", width=18)
        table.add_column("Priority", style="white", width=10)
        table.add_column("Title", style="white", min_width=30)
        table.add_column("Savings", justify="right", style="green", width=12)
        table.add_column("Models", style="dim white", width=20)

        for rec in self.data[:15]:
            priority_style = {
                "critical": "red bold",
                "high": "red",
                "medium": "yellow",
                "low": "green",
            }.get(rec.get("priority", "medium"), "white")

            table.add_row(
                rec.get("type", "N/A"),
                Text(rec.get("priority", "medium").upper(), style=priority_style),
                rec.get("title", "N/A")[:40],
                f"${rec.get('estimated_savings_usd', 0):.2f}/day",
                ", ".join(rec.get("affected_models", []))[:18],
            )

        return Panel(table, border_style="yellow", title="Recommendations")


class AlertsList(Static):
    """List of active alerts."""

    data = reactive([])

    def render(self) -> Panel:
        if not self.data:
            return Panel("No active alerts", border_style="green", title="Alerts ✓")

        table = Table(show_header=True, header_style="bold red", box=None)
        table.add_column("Severity", style="white", width=10)
        table.add_column("Type", style="cyan", width=18)
        table.add_column("Message", style="white", min_width=40)
        table.add_column("Value", justify="right", style="yellow", width=12)
        table.add_column("Threshold", justify="right", style="dim white", width=12)

        for alert in self.data[:15]:
            severity = alert.get("severity", "info")
            severity_style = {
                "critical": "red bold",
                "warning": "yellow bold",
                "info": "blue",
            }.get(severity, "white")

            table.add_row(
                Text(severity.upper(), style=severity_style),
                alert.get("type", "N/A"),
                alert.get("message", "N/A")[:50],
                f"{alert.get('value', 0):.2f}",
                f"{alert.get('threshold', 0):.2f}",
            )

        return Panel(table, border_style="red", title="Active Alerts")


class AIOpsDashboard(App):
    """Main TUI Application for AI Ops Dashboard."""

    CSS = """
    Screen {
        background: $surface;
    }
    
    #header {
        height: 3;
        background: $primary;
        color: $text;
        dock: top;
    }
    
    #main-container {
        layout: grid;
        grid-size: 2 3;
        grid-gutter: 1;
        padding: 1;
        height: 1fr;
    }
    
    .metric-card {
        border: solid $primary;
        height: 100%;
    }
    
    .chart-panel {
        border: solid $secondary;
        height: 100%;
    }
    
    .table-panel {
        border: solid $accent;
        height: 100%;
    }
    
    #metrics-row {
        height: 3;
    }
    
    #charts-row {
        height: 1fr;
    }
    
    #tables-row {
        height: 1fr;
    }
    
    DataTable {
        height: 1fr;
    }
    
    Sparkline {
        height: 1fr;
    }
    
    TabbedContent {
        height: 1fr;
    }
    
    TabPane {
        height: 1fr;
        padding: 1;
    }
    
    .status-healthy {
        color: $success;
    }
    
    .status-warning {
        color: $warning;
    }
    
    .status-critical {
        color: $error;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("t", "toggle_theme", "Toggle Theme"),
        Binding("1", "focus_metrics", "Metrics"),
        Binding("2", "focus_recommendations", "Recommendations"),
        Binding("3", "focus_alerts", "Alerts"),
    ]

    refresh_interval = reactive(5.0)
    time_range_hours = reactive(1)
    auto_refresh = reactive(True)

    def __init__(self) -> None:
        super().__init__()
        self._metrics_collector = None
        self._recommendation_engine = None
        self._alert_manager = None
        self._db_manager = None
        self._refresh_timer: Timer | None = None

    async def on_mount(self) -> None:
        """Initialize components on mount."""
        self._metrics_collector = await get_metrics_collector()
        self._recommendation_engine = await get_recommendation_engine()
        self._alert_manager = await get_alert_manager()
        self._db_manager = await get_db_manager()

        # Start auto-refresh timer
        self._refresh_timer = self.set_interval(self.refresh_interval, self.refresh_data)

        # Initial data load
        await self.refresh_data()

    async def on_unmount(self) -> None:
        """Cleanup on unmount."""
        if self._refresh_timer:
            self._refresh_timer.stop()

    def compose(self) -> ComposeResult:
        """Compose the UI layout."""
        yield Header(show_clock=True)

        with Container(id="main-container"):
            # Metrics Summary Row
            with Horizontal(id="metrics-row"):
                yield MetricCard(id="total-calls", label="Total Calls", style_class="blue")
                yield MetricCard(id="total-cost", label="Total Cost (USD)", style_class="green")
                yield MetricCard(id="avg-latency", label="Avg Latency (ms)", style_class="purple")
                yield MetricCard(id="error-rate", label="Error Rate (%)", style_class="red")

            # Charts Row
            with Horizontal(id="charts-row"):
                yield CostChart(id="cost-chart", classes="chart-panel")
                yield LatencyChart(id="latency-chart", classes="chart-panel")

            # Tables Row
            with Horizontal(id="tables-row"):
                with TabbedContent():
                    with TabPane("Models", id="models-tab"):
                        yield ModelTable(id="model-table", classes="table-panel")
                    with TabPane("Recommendations", id="recs-tab"):
                        yield RecommendationsList(id="recommendations-list", classes="table-panel")
                    with TabPane("Alerts", id="alerts-tab"):
                        yield AlertsList(id="alerts-list", classes="table-panel")

        yield Footer()

    async def refresh_data(self) -> None:
        """Refresh all dashboard data."""
        try:
            # Fetch metrics
            metrics = await self._metrics_collector.get_aggregated_metrics(hours=self.time_range_hours)

            # Update metric cards
            self.query_one("#total-calls", MetricCard).value = str(metrics.get("total_calls", 0))
            self.query_one("#total-cost", MetricCard).value = f"${metrics.get('total_cost_usd', 0):.2f}"
            self.query_one("#avg-latency", MetricCard).value = f"{metrics.get('avg_latency_ms', 0):.0f}ms"
            self.query_one("#error-rate", MetricCard).value = f"{metrics.get('error_rate', 0) * 100:.2f}%"

            # Fetch time series for charts
            cost_data = await self._metrics_collector.get_time_series(
                hours=self.time_range_hours, metric="total_cost_usd"
            )
            latency_data = await self._metrics_collector.get_time_series(
                hours=self.time_range_hours, metric="p99_latency_ms"
            )

            # Format latency data with percentiles
            latency_with_percentiles = []
            for d in latency_data:
                latency_with_percentiles.append({
                    "timestamp": d["timestamp"],
                    "p50_latency_ms": d["value"] * 0.5,  # Approximation
                    "p95_latency_ms": d["value"] * 0.8,
                    "p99_latency_ms": d["value"],
                })

            self.query_one("#cost-chart", CostChart).data = cost_data
            self.query_one("#latency-chart", LatencyChart).data = latency_with_percentiles

            # Fetch model breakdown
            breakdown = await self._metrics_collector.get_model_breakdown(hours=self.time_range_hours)
            self.query_one("#model-table", ModelTable).data = breakdown

            # Fetch recommendations
            recommendations = await self._recommendation_engine.get_recommendations(
                dismissed=False, limit=20
            )
            self.query_one("#recommendations-list", RecommendationsList).data = [
                r.model_dump() for r in recommendations
            ]

            # Fetch alerts
            alerts = await self._alert_manager.get_active_alerts()
            self.query_one("#alerts-list", AlertsList).data = [
                a.model_dump() for a in alerts
            ]

        except Exception as e:
            self.notify(f"Refresh error: {e}", severity="error")

    async def action_refresh(self) -> None:
        """Manual refresh action."""
        await self.refresh_data()
        self.notify("Data refreshed", severity="information")

    async def action_toggle_theme(self) -> None:
        """Toggle dark/light theme."""
        self.dark = not self.dark
        self.notify(f"Theme: {'Dark' if self.dark else 'Light'}", severity="information")

    async def action_focus_metrics(self) -> None:
        """Focus metrics tab."""
        self.query_one("#models-tab", TabPane).focus()

    async def action_focus_recommendations(self) -> None:
        """Focus recommendations tab."""
        self.query_one("#recs-tab", TabPane).focus()

    async def action_focus_alerts(self) -> None:
        """Focus alerts tab."""
        self.query_one("#alerts-tab", TabPane).focus()


def run_tui() -> None:
    """Run the TUI application."""
    app = AIOpsDashboard()
    app.run()


if __name__ == "__main__":
    run_tui()
