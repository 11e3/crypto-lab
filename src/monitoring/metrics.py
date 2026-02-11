"""
Prometheus metrics exporter for crypto-lab.

Provides metrics for trading, ML, and pipeline operations with support for:
- Counter, Gauge, Histogram, Summary metrics
- Labels for dimensionality
- Pushgateway support for batch jobs
- HTTP server for scraping
"""

from __future__ import annotations

import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from functools import wraps
from typing import Any

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    push_to_gateway,
    start_http_server,
)

# =============================================================================
# Metrics Exporter Base
# =============================================================================


class MetricsExporter:
    """
    Base class for Prometheus metrics exporters.

    Usage:
        exporter = MetricsExporter(port=8000, prefix="myapp")
        exporter.start_server()

        # Push to gateway for batch jobs
        exporter.push_to_gateway("localhost:9091", job="my-batch-job")
    """

    def __init__(
        self,
        port: int = 8000,
        prefix: str = "",
        registry: CollectorRegistry | None = None,
    ):
        self.port = port
        self.prefix = prefix
        self.registry = registry or REGISTRY
        self._server_started = False

    def _make_name(self, name: str) -> str:
        """Create metric name with prefix."""
        if self.prefix:
            return f"{self.prefix}_{name}"
        return name

    def start_server(self) -> None:
        """Start HTTP server for Prometheus scraping."""
        if not self._server_started:
            start_http_server(self.port, registry=self.registry)
            self._server_started = True

    def push_to_gateway(
        self,
        gateway: str,
        job: str,
        grouping_key: dict[str, str] | None = None,
    ) -> None:
        """Push metrics to Pushgateway."""
        push_to_gateway(
            gateway,
            job=job,
            registry=self.registry,
            grouping_key=grouping_key,
        )


# =============================================================================
# Trading Metrics
# =============================================================================


class TradingMetrics(MetricsExporter):
    """
    Metrics exporter for trading operations.

    Tracks:
    - Order counts and volumes
    - Position sizes and P&L
    - Win rates and returns
    - Execution latency
    """

    def __init__(self, port: int = 8000, prefix: str = "trading"):
        super().__init__(port=port, prefix=prefix)

        # Order metrics
        self.orders_total = Counter(
            self._make_name("orders_total"),
            "Total number of orders",
            ["symbol", "action", "status"],
            registry=self.registry,
        )

        self.order_volume_krw = Counter(
            self._make_name("order_volume_krw_total"),
            "Total order volume in KRW",
            ["symbol", "action"],
            registry=self.registry,
        )

        # Position metrics
        self.open_positions = Gauge(
            self._make_name("open_positions_count"),
            "Number of open positions",
            registry=self.registry,
        )

        self.position_size_krw = Gauge(
            self._make_name("position_size_krw"),
            "Current position size in KRW",
            ["symbol"],
            registry=self.registry,
        )

        # P&L metrics
        self.total_pnl_krw = Gauge(
            self._make_name("total_pnl_krw"),
            "Total realized P&L in KRW",
            registry=self.registry,
        )

        self.unrealized_pnl_krw = Gauge(
            self._make_name("unrealized_pnl_krw"),
            "Total unrealized P&L in KRW",
            registry=self.registry,
        )

        self.cumulative_pnl_krw = Gauge(
            self._make_name("cumulative_pnl_krw"),
            "Cumulative P&L in KRW",
            registry=self.registry,
        )

        # Performance metrics
        self.win_rate = Gauge(
            self._make_name("win_rate"),
            "Trading win rate (0-1)",
            registry=self.registry,
        )

        self.sharpe_ratio = Gauge(
            self._make_name("sharpe_ratio"),
            "Sharpe ratio",
            registry=self.registry,
        )

        # Execution latency
        self.order_latency = Histogram(
            self._make_name("order_latency_seconds"),
            "Order execution latency in seconds",
            ["action"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry,
        )

        # Bot status
        self.bot_active = Gauge(
            self._make_name("bot_active"),
            "Whether trading bot is active (1) or paused (0)",
            registry=self.registry,
        )

    def record_order(
        self,
        symbol: str,
        action: str,
        status: str,
        amount_krw: float,
        latency: float | None = None,
    ) -> None:
        """Record an order execution."""
        self.orders_total.labels(symbol=symbol, action=action, status=status).inc()

        if status == "filled":
            self.order_volume_krw.labels(symbol=symbol, action=action).inc(amount_krw)

        if latency is not None:
            self.order_latency.labels(action=action).observe(latency)

    def update_position(
        self,
        symbol: str,
        size_krw: float,
        unrealized_pnl: float,
    ) -> None:
        """Update position metrics."""
        self.position_size_krw.labels(symbol=symbol).set(size_krw)
        self.unrealized_pnl_krw.set(unrealized_pnl)

    def update_pnl(
        self,
        realized: float,
        cumulative: float,
        win_rate: float,
    ) -> None:
        """Update P&L metrics."""
        self.total_pnl_krw.set(realized)
        self.cumulative_pnl_krw.set(cumulative)
        self.win_rate.set(win_rate)

    @contextmanager
    def track_order(self, symbol: str, action: str) -> Generator[dict[str, Any], None, None]:
        """Context manager to track order execution."""
        start_time = time.time()
        result: dict[str, Any] = {"status": "pending"}

        try:
            yield result
        except Exception:
            result["status"] = "failed"
            raise
        finally:
            latency = time.time() - start_time
            self.record_order(
                symbol=symbol,
                action=action,
                status=result.get("status", "unknown"),
                amount_krw=result.get("amount_krw", 0),
                latency=latency,
            )


# =============================================================================
# ML Metrics
# =============================================================================


class MLMetrics(MetricsExporter):
    """
    Metrics exporter for ML operations.

    Tracks:
    - Model predictions and accuracy
    - Prediction latency
    - Data drift
    - Feature freshness
    """

    def __init__(self, port: int = 8001, prefix: str = "ml"):
        super().__init__(port=port, prefix=prefix)

        # Prediction metrics
        self.predictions_total = Counter(
            self._make_name("predictions_total"),
            "Total number of predictions",
            ["model"],
            registry=self.registry,
        )

        self.prediction_class_total = Counter(
            self._make_name("prediction_class_total"),
            "Predictions by class",
            ["model", "class"],
            registry=self.registry,
        )

        self.prediction_latency = Histogram(
            self._make_name("prediction_duration_seconds"),
            "Prediction latency in seconds",
            ["model"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=self.registry,
        )

        # Model performance
        self.model_accuracy = Gauge(
            self._make_name("model_accuracy"),
            "Model accuracy",
            ["model"],
            registry=self.registry,
        )

        self.model_f1_score = Gauge(
            self._make_name("model_f1_score"),
            "Model F1 score",
            ["model"],
            registry=self.registry,
        )

        self.model_precision = Gauge(
            self._make_name("model_precision"),
            "Model precision",
            ["model"],
            registry=self.registry,
        )

        self.model_recall = Gauge(
            self._make_name("model_recall"),
            "Model recall",
            ["model"],
            registry=self.registry,
        )

        # Drift metrics
        self.data_drift_score = Gauge(
            self._make_name("data_drift_score"),
            "Data drift score (0-1)",
            registry=self.registry,
        )

        self.feature_drift = Gauge(
            self._make_name("feature_drift_score"),
            "Feature drift score",
            ["feature"],
            registry=self.registry,
        )

        # Feature store metrics
        self.feature_freshness = Gauge(
            self._make_name("feature_freshness_seconds"),
            "Timestamp of last feature update",
            ["feature_view"],
            registry=self.registry,
        )

        self.feature_retrieval_total = Counter(
            self._make_name("feature_retrieval_total"),
            "Feature retrieval count",
            ["feature_view"],
            registry=self.registry,
        )

        self.feature_retrieval_latency = Histogram(
            self._make_name("feature_retrieval_duration_seconds"),
            "Feature retrieval latency",
            ["feature_view"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
            registry=self.registry,
        )

    def record_prediction(
        self,
        model: str,
        predicted_class: str,
        latency: float,
    ) -> None:
        """Record a model prediction."""
        self.predictions_total.labels(model=model).inc()
        self.prediction_class_total.labels(model=model, class_=predicted_class).inc()
        self.prediction_latency.labels(model=model).observe(latency)

    def update_model_metrics(
        self,
        model: str,
        accuracy: float,
        f1: float,
        precision: float,
        recall: float,
    ) -> None:
        """Update model performance metrics."""
        self.model_accuracy.labels(model=model).set(accuracy)
        self.model_f1_score.labels(model=model).set(f1)
        self.model_precision.labels(model=model).set(precision)
        self.model_recall.labels(model=model).set(recall)

    def update_drift_metrics(
        self,
        overall_drift: float,
        feature_drifts: dict[str, float],
    ) -> None:
        """Update drift metrics."""
        self.data_drift_score.set(overall_drift)
        for feature, drift in feature_drifts.items():
            self.feature_drift.labels(feature=feature).set(drift)

    @contextmanager
    def track_prediction(self, model: str) -> Generator[dict[str, Any], None, None]:
        """Context manager to track prediction latency."""
        start_time = time.time()
        result: dict[str, Any] = {}

        yield result

        latency = time.time() - start_time
        if "class" in result:
            self.record_prediction(model, result["class"], latency)

    def track_prediction_decorator(self, model: str) -> Callable[..., Any]:
        """Decorator to track prediction latency."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.track_prediction(model) as result:
                    prediction = func(*args, **kwargs)
                    result["class"] = str(prediction)
                    return prediction

            return wrapper

        return decorator


# =============================================================================
# Pipeline Metrics
# =============================================================================


class PipelineMetrics(MetricsExporter):
    """
    Metrics exporter for data pipeline operations.

    Tracks:
    - DAG runs and task statuses
    - Data freshness and volumes
    - Processing latency
    - Error rates
    """

    def __init__(self, port: int = 8002, prefix: str = "pipeline"):
        super().__init__(port=port, prefix=prefix)

        # DAG metrics
        self.dag_runs_total = Counter(
            self._make_name("dag_runs_total"),
            "Total DAG runs",
            ["dag_id", "state"],
            registry=self.registry,
        )

        self.task_runs_total = Counter(
            self._make_name("task_runs_total"),
            "Total task runs",
            ["dag_id", "task_id", "state"],
            registry=self.registry,
        )

        self.dag_duration = Histogram(
            self._make_name("dag_duration_seconds"),
            "DAG execution duration",
            ["dag_id"],
            buckets=[60, 120, 300, 600, 1200, 1800, 3600, 7200],
            registry=self.registry,
        )

        # Data metrics
        self.data_freshness = Gauge(
            self._make_name("data_freshness_seconds"),
            "Timestamp of last data update",
            ["data_source"],
            registry=self.registry,
        )

        self.records_processed = Counter(
            self._make_name("records_processed_total"),
            "Total records processed",
            ["pipeline", "stage"],
            registry=self.registry,
        )

        self.data_volume_bytes = Counter(
            self._make_name("data_volume_bytes_total"),
            "Total data volume processed",
            ["pipeline"],
            registry=self.registry,
        )

        # Error metrics
        self.errors_total = Counter(
            self._make_name("errors_total"),
            "Total errors",
            ["pipeline", "error_type"],
            registry=self.registry,
        )

        # DuckDB metrics
        self.duckdb_query_duration = Histogram(
            self._make_name("duckdb_query_duration_seconds"),
            "DuckDB query duration",
            ["query_type"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry,
        )

        self.duckdb_rows_scanned = Counter(
            self._make_name("duckdb_rows_scanned_total"),
            "Total rows scanned by DuckDB",
            registry=self.registry,
        )

    def record_dag_run(
        self,
        dag_id: str,
        state: str,
        duration: float | None = None,
    ) -> None:
        """Record a DAG run."""
        self.dag_runs_total.labels(dag_id=dag_id, state=state).inc()
        if duration is not None:
            self.dag_duration.labels(dag_id=dag_id).observe(duration)

    def record_task_run(
        self,
        dag_id: str,
        task_id: str,
        state: str,
    ) -> None:
        """Record a task run."""
        self.task_runs_total.labels(dag_id=dag_id, task_id=task_id, state=state).inc()

    def record_processing(
        self,
        pipeline: str,
        stage: str,
        records: int,
        bytes_processed: int | None = None,
    ) -> None:
        """Record data processing."""
        self.records_processed.labels(pipeline=pipeline, stage=stage).inc(records)
        if bytes_processed:
            self.data_volume_bytes.labels(pipeline=pipeline).inc(bytes_processed)

    def record_error(self, pipeline: str, error_type: str) -> None:
        """Record a pipeline error."""
        self.errors_total.labels(pipeline=pipeline, error_type=error_type).inc()

    @contextmanager
    def track_query(self, query_type: str) -> Generator[None, None, None]:
        """Context manager to track query duration."""
        start_time = time.time()

        yield

        duration = time.time() - start_time
        self.duckdb_query_duration.labels(query_type=query_type).observe(duration)
