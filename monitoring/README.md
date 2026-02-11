# Crypto Lab - Monitoring Stack

Production-grade monitoring infrastructure for the crypto trading system.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Monitoring Stack                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐            │
│  │ Trading Bot  │   │  ML Server   │   │   Airflow    │            │
│  │   :8000      │   │    :8001     │   │    :8080     │            │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘            │
│         │                  │                   │                    │
│         └──────────────────┼───────────────────┘                    │
│                            │                                        │
│                            ▼                                        │
│  ┌─────────────────────────────────────────┐                       │
│  │              Prometheus :9090            │◄── Alertmanager :9093│
│  │         (Metrics Collection)             │                       │
│  └─────────────────┬───────────────────────┘                       │
│                    │                                                │
│                    ▼                                                │
│  ┌─────────────────────────────────────────┐                       │
│  │              Grafana :3000               │                       │
│  │           (Visualization)                │                       │
│  │                                          │                       │
│  │   ┌──────────┐ ┌──────────┐ ┌────────┐ │                       │
│  │   │ Trading  │ │    ML    │ │ System │ │                       │
│  │   │Dashboard │ │Dashboard │ │Overview│ │                       │
│  │   └──────────┘ └──────────┘ └────────┘ │                       │
│  └─────────────────────────────────────────┘                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Start Monitoring Stack

```bash
cd monitoring
docker-compose up -d
```

### 2. Access Dashboards

| Service | URL | Credentials |
|---------|-----|-------------|
| Grafana | http://localhost:3000 | admin / crypto_admin_2024 |
| Prometheus | http://localhost:9090 | - |
| Alertmanager | http://localhost:9093 | - |
| Pushgateway | http://localhost:9091 | - |

### 3. Integrate with Applications

```python
from src.monitoring import TradingMetrics, get_logger

# Structured logging
logger = get_logger("trading-bot")
logger.info("Order executed", symbol="BTC", amount=1000000)

# Prometheus metrics
metrics = TradingMetrics(port=8000)
metrics.start_server()
metrics.record_order("BTC", "BUY", "filled", 1000000, latency=0.5)
```

## Components

### Prometheus
- Metrics collection and storage
- 30-day retention
- Alert rule evaluation

### Grafana
Pre-configured dashboards:
- **Trading Performance**: P&L, positions, orders, win rate
- **ML Pipeline**: Model accuracy, predictions, drift
- **System Overview**: Service health, CPU, memory, disk

### Alertmanager
Alert routing and notification:
- Critical trading alerts (immediate)
- System alerts (grouped)
- Configurable receivers (Slack, webhook, email)

### Pushgateway
For batch job metrics from Airflow DAGs and other scheduled tasks.

## Metrics Reference

### Trading Metrics (`trading_*`)
| Metric | Type | Description |
|--------|------|-------------|
| `trading_orders_total` | Counter | Total orders by symbol/action/status |
| `trading_order_volume_krw_total` | Counter | Total order volume in KRW |
| `trading_open_positions_count` | Gauge | Number of open positions |
| `trading_total_pnl_krw` | Gauge | Total realized P&L |
| `trading_win_rate` | Gauge | Win rate (0-1) |
| `trading_order_latency_seconds` | Histogram | Order execution latency |

### ML Metrics (`ml_*`)
| Metric | Type | Description |
|--------|------|-------------|
| `ml_predictions_total` | Counter | Total predictions |
| `ml_prediction_duration_seconds` | Histogram | Prediction latency |
| `ml_model_accuracy` | Gauge | Model accuracy |
| `ml_data_drift_score` | Gauge | Data drift score |
| `ml_feature_freshness_seconds` | Gauge | Feature freshness timestamp |

### Pipeline Metrics (`pipeline_*`)
| Metric | Type | Description |
|--------|------|-------------|
| `pipeline_dag_runs_total` | Counter | Total DAG runs |
| `pipeline_dag_duration_seconds` | Histogram | DAG execution time |
| `pipeline_records_processed_total` | Counter | Records processed |
| `pipeline_errors_total` | Counter | Pipeline errors |

## Alert Rules

### Trading Alerts
- `LargePositionOpened`: Position > 10M KRW
- `HighUnrealizedLoss`: Unrealized P&L < -500K KRW
- `TradingBotDown`: Bot unreachable for 2min
- `NoTradingActivity`: No orders in 2h while active

### ML Alerts
- `HighPredictionLatency`: p95 > 1s
- `ModelAccuracyDrop`: Accuracy < 60%
- `DataDriftDetected`: Drift score > 0.3
- `StaleFeatures`: Features > 1h old

### System Alerts
- `HighCPUUsage`: CPU > 85% for 10min
- `HighMemoryUsage`: Memory > 90% for 5min
- `LowDiskSpace`: Disk > 85% used
- `ServiceDown`: Any service unreachable

## Structured Logging

JSON-formatted logs with context support:

```python
from src.monitoring import get_logger

logger = get_logger("my-service", log_file="logs/service.log")

# Simple logging
logger.info("Processing started", batch_size=100)

# With context
with logger.context(request_id="abc123", user="trader1"):
    logger.info("Order received")
    logger.info("Order validated")

# Permanent binding
logger.bind(environment="production")
logger.info("All logs now include environment")
```

Output:
```json
{
  "timestamp": "2024-01-15T10:30:00.000000+00:00",
  "level": "INFO",
  "logger": "my-service",
  "message": "Processing started",
  "service": "my-service",
  "batch_size": 100
}
```

## Development

### Adding New Metrics

```python
from prometheus_client import Counter, Gauge

class CustomMetrics(MetricsExporter):
    def __init__(self):
        super().__init__(prefix="custom")

        self.my_counter = Counter(
            self._make_name("my_counter_total"),
            "Description",
            ["label1", "label2"],
            registry=self.registry,
        )
```

### Testing Alerts

```bash
# Send test alert to Alertmanager
curl -H "Content-Type: application/json" -d '[
  {
    "labels": {"alertname": "TestAlert", "severity": "warning"},
    "annotations": {"summary": "Test alert"}
  }
]' http://localhost:9093/api/v1/alerts
```

## Maintenance

### Backup Grafana
```bash
docker exec crypto-grafana grafana-cli admin data-migration encrypt
docker cp crypto-grafana:/var/lib/grafana ./grafana-backup
```

### Clean Old Metrics
```bash
curl -X POST http://localhost:9090/api/v1/admin/tsdb/clean_tombstones
```
