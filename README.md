# AI Ops Dashboard

LLM Observability, Cost Optimization, and Real-time Monitoring Dashboard.

## Features

- **LLM Instrumentation**: Auto-instrument all LiteLLM calls for latency, tokens, cost, errors, model info
- **Metrics Storage**: SQLite with periodic Parquet export for analytics
- **Real-time Dashboard**: 
  - Web UI (FastAPI + HTMX + Chart.js) with live charts
  - Terminal UI (Textual) with live updates
- **Cost Optimization**: Rule engine for model routing, caching, prompt compression, batching
- **Alerting**: Configurable thresholds for cost, latency, error rates
- **Developer Experience**: CLI, Docker, Fly.io, GitHub Actions

## Quick Start

### Installation

```bash
# Clone and install
git clone <repo>
cd ai-ops-dashboard
pip install -e .[dev]

# Or with Docker
docker-compose up -d
```

### Configuration

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
# Edit .env with your settings
```

### Running

```bash
# Web dashboard (FastAPI + HTMX)
ai-ops web --host 0.0.0.0 --port 8000

# Terminal dashboard (Textual)
ai-ops tui

# Test instrumentation
ai-ops test-instrumentation --model gpt-3.5-turbo --calls 5
```

### Using the Instrumentation

```python
import litellm
from ai_ops.core.instrumentation.middleware import init_instrumentation

# Initialize once at startup
init_instrumentation()

# All LiteLLM calls are now automatically instrumented
response = litellm.completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)

# View metrics in the dashboard at http://localhost:8000
```

## Architecture

```
ai_ops/
├── shared/           # Shared utilities, config, models
├── core/
│   ├── instrumentation/  # LiteLLM middleware wrapper
│   ├── metrics/         # Async metrics collector with aggregation
│   ├── recommendations/ # Rule-based cost optimization engine
│   ├── alerting/        # Threshold-based alerting
│   └── storage/         # Database + Parquet export
├── web/              # FastAPI + HTMX web dashboard
├── tui/              # Textual terminal dashboard
├── cli.py            # CLI entry point
└── tests/            # Unit and integration tests
```

## Dashboard Screenshots

### Web Dashboard (HTMX + Chart.js)
- Real-time cost, latency, token, error rate charts
- Model comparison table with cost breakdown
- Cost optimization recommendations with savings estimates
- Alert management with acknowledge/resolve actions

### Terminal Dashboard (Textual)
- Live metric cards with sparklines
- Model breakdown table
- Recommendations and alerts tabs
- Keyboard shortcuts for navigation

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Main dashboard |
| `GET /health` | Health check |
| `GET /api/metrics/summary` | Aggregated metrics |
| `GET /api/metrics/breakdown` | Model usage breakdown |
| `GET /api/metrics/timeseries` | Time series for charts |
| `GET /api/recommendations` | Cost optimization recommendations |
| `GET /api/recommendations/summary` | Savings summary |
| `GET /api/alerts` | Alerts list |
| `GET /api/alerts/active` | Active alerts only |
| `WS /ws` | WebSocket for real-time updates |

## Deployment

### Docker
```bash
docker-compose up -d
```

### Fly.io
```bash
fly launch --config fly.toml
fly deploy
```

### Manual
```bash
# Set environment variables
export DB_SQLITE_PATH=/data/ai_ops.db
export WEB_HOST=0.0.0.0
export WEB_PORT=8080

# Run
python -m ai_ops.web.api
```

## Configuration

All settings via environment variables (see `.env.example`):

| Category | Prefix | Key Settings |
|----------|--------|--------------|
| Database | `DB_` | sqlite_path, export_interval |
| Instrumentation | `INSTR_` | enabled, sample_rate, buffer_size |
| Metrics | `METRICS_` | aggregation_window, retention_days |
| Recommendations | `REC_` | routing, caching, compression, batching rules |
| Alerting | `ALERT_` | cost, latency, error thresholds |
| Web | `WEB_` | host, port, cors_origins |
| TUI | `TUI_` | refresh_rate, theme |

## Development

```bash
# Install dev dependencies
pip install -e .[dev]

# Run tests
pytest ai_ops/tests -v

# Lint
ruff check .
ruff format .

# Type check
mypy ai_ops
```

## How It Works

### Instrumentation Middleware
The `LLMInstrumentor` wraps LiteLLM's `completion`/`acompletion` functions to capture:
- Wall-clock latency (perf_counter)
- Token counts from response.usage
- Cost estimation via model pricing table
- Error classification (timeout, rate_limit, etc.)
- Model/provider identification

Buffered writes prevent blocking the LLM call path.

### Metrics Collection
The `MetricsCollector` subscribes to call events and aggregates into fixed time windows (default 60s) with dimensions (provider, model). Computes:
- Call count, total tokens, total cost
- Avg, P50, P95, P99 latency
- Error count and rate

### Recommendations Engine
Runs configurable rules periodically:
- **Model Routing**: Detects expensive models, suggests cheaper alternatives
- **Caching**: Finds repeated prompt hashes, suggests semantic caching
- **Prompt Compression**: Identifies long prompts, suggests compression
- **Batching**: Detects high-frequency calls, suggests request batching

### Alerting
Monitors metrics against thresholds with cooldown:
- Daily/hourly cost
- P99 latency
- Error rate

## License

MIT