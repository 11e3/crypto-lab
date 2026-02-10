"""Tests for monitoring.metrics module."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from prometheus_client import CollectorRegistry

from src.monitoring.metrics import (
    MetricsExporter,
    MLMetrics,
    PipelineMetrics,
    TradingMetrics,
)


@pytest.fixture()
def registry() -> CollectorRegistry:
    """Create isolated registry for each test."""
    return CollectorRegistry()


@pytest.fixture()
def _patch_registry(registry: CollectorRegistry) -> None:  # noqa: PT004
    """Patch default REGISTRY so subclass constructors use isolated registry."""
    with patch("src.monitoring.metrics.REGISTRY", registry):
        yield


# =============================================================================
# MetricsExporter base
# =============================================================================


class TestMetricsExporter:
    """Tests for MetricsExporter base class."""

    def test_init_defaults(self, registry: CollectorRegistry) -> None:
        exporter = MetricsExporter(registry=registry)
        assert exporter.port == 8000
        assert exporter.prefix == ""
        assert exporter._server_started is False

    def test_make_name_with_prefix(self, registry: CollectorRegistry) -> None:
        exporter = MetricsExporter(prefix="myapp", registry=registry)
        assert exporter._make_name("orders") == "myapp_orders"

    def test_make_name_without_prefix(self, registry: CollectorRegistry) -> None:
        exporter = MetricsExporter(prefix="", registry=registry)
        assert exporter._make_name("orders") == "orders"

    def test_start_server(self, registry: CollectorRegistry) -> None:
        exporter = MetricsExporter(port=9999, registry=registry)
        with patch("src.monitoring.metrics.start_http_server") as mock_start:
            exporter.start_server()
            mock_start.assert_called_once_with(9999, registry=registry)
            assert exporter._server_started is True

    def test_start_server_idempotent(self, registry: CollectorRegistry) -> None:
        exporter = MetricsExporter(port=9999, registry=registry)
        with patch("src.monitoring.metrics.start_http_server") as mock_start:
            exporter.start_server()
            exporter.start_server()
            assert mock_start.call_count == 1

    def test_push_to_gateway(self, registry: CollectorRegistry) -> None:
        exporter = MetricsExporter(registry=registry)
        with patch("src.monitoring.metrics.push_to_gateway") as mock_push:
            exporter.push_to_gateway("localhost:9091", job="test-job")
            mock_push.assert_called_once_with(
                "localhost:9091",
                job="test-job",
                registry=registry,
                grouping_key=None,
            )

    def test_push_to_gateway_with_grouping_key(self, registry: CollectorRegistry) -> None:
        exporter = MetricsExporter(registry=registry)
        with patch("src.monitoring.metrics.push_to_gateway") as mock_push:
            exporter.push_to_gateway(
                "localhost:9091", job="test-job",
                grouping_key={"env": "prod"},
            )
            mock_push.assert_called_once_with(
                "localhost:9091",
                job="test-job",
                registry=registry,
                grouping_key={"env": "prod"},
            )


# =============================================================================
# TradingMetrics
# =============================================================================


@pytest.mark.usefixtures("_patch_registry")
class TestTradingMetrics:
    """Tests for TradingMetrics."""

    def test_init(self) -> None:
        metrics = TradingMetrics()
        assert metrics.prefix == "trading"

    def test_record_order_filled(self) -> None:
        metrics = TradingMetrics()
        metrics.record_order(
            symbol="KRW-BTC", action="buy", status="filled",
            amount_krw=1000000, latency=0.5,
        )
        assert metrics.orders_total.labels(
            symbol="KRW-BTC", action="buy", status="filled",
        )._value.get() == 1.0

    def test_record_order_non_filled_skips_volume(self) -> None:
        metrics = TradingMetrics()
        metrics.record_order(
            symbol="KRW-BTC", action="buy", status="rejected",
            amount_krw=1000000,
        )
        assert metrics.order_volume_krw.labels(
            symbol="KRW-BTC", action="buy",
        )._value.get() == 0.0

    def test_record_order_no_latency(self) -> None:
        metrics = TradingMetrics()
        metrics.record_order(
            symbol="KRW-BTC", action="buy", status="filled",
            amount_krw=500000, latency=None,
        )
        assert metrics.orders_total.labels(
            symbol="KRW-BTC", action="buy", status="filled",
        )._value.get() == 1.0

    def test_update_position(self) -> None:
        metrics = TradingMetrics()
        metrics.update_position(
            symbol="KRW-BTC", size_krw=5000000, unrealized_pnl=100000,
        )
        assert metrics.position_size_krw.labels(symbol="KRW-BTC")._value.get() == 5000000
        assert metrics.unrealized_pnl_krw._value.get() == 100000

    def test_update_pnl(self) -> None:
        metrics = TradingMetrics()
        metrics.update_pnl(realized=50000, cumulative=150000, win_rate=0.65)
        assert metrics.total_pnl_krw._value.get() == 50000
        assert metrics.cumulative_pnl_krw._value.get() == 150000
        assert metrics.win_rate._value.get() == 0.65

    def test_track_order_success(self) -> None:
        metrics = TradingMetrics()
        with metrics.track_order("KRW-BTC", "buy") as result:
            result["status"] = "filled"
            result["amount_krw"] = 1000000

        assert metrics.orders_total.labels(
            symbol="KRW-BTC", action="buy", status="filled",
        )._value.get() == 1.0

    def test_track_order_failure(self) -> None:
        metrics = TradingMetrics()
        with pytest.raises(RuntimeError), metrics.track_order("KRW-BTC", "buy"):
            raise RuntimeError("connection error")

        assert metrics.orders_total.labels(
            symbol="KRW-BTC", action="buy", status="failed",
        )._value.get() == 1.0


# =============================================================================
# MLMetrics
# =============================================================================


@pytest.mark.usefixtures("_patch_registry")
class TestMLMetrics:
    """Tests for MLMetrics."""

    def test_init(self) -> None:
        metrics = MLMetrics()
        assert metrics.prefix == "ml"

    def test_record_prediction_class_label_bug(self) -> None:
        """record_prediction uses class_ kwarg but label is 'class' (Python keyword conflict)."""
        metrics = MLMetrics()
        with pytest.raises(ValueError, match="Incorrect label names"):
            metrics.record_prediction(model="regime_v1", predicted_class="bullish", latency=0.05)

    def test_update_model_metrics(self) -> None:
        metrics = MLMetrics()
        metrics.update_model_metrics(
            model="regime_v1", accuracy=0.85, f1=0.82, precision=0.88, recall=0.80,
        )
        assert metrics.model_accuracy.labels(model="regime_v1")._value.get() == 0.85
        assert metrics.model_f1_score.labels(model="regime_v1")._value.get() == 0.82

    def test_update_drift_metrics(self) -> None:
        metrics = MLMetrics()
        metrics.update_drift_metrics(
            overall_drift=0.15,
            feature_drifts={"volume": 0.2, "rsi": 0.1},
        )
        assert metrics.data_drift_score._value.get() == 0.15
        assert metrics.feature_drift.labels(feature="volume")._value.get() == 0.2

    def test_track_prediction_context_class_label_bug(self) -> None:
        """track_prediction calls record_prediction which has class_ label bug."""
        metrics = MLMetrics()
        with pytest.raises(ValueError, match="Incorrect label names"), metrics.track_prediction("regime_v1") as result:
            result["class"] = "bearish"

    def test_track_prediction_no_class(self) -> None:
        metrics = MLMetrics()
        with metrics.track_prediction("regime_v1"):
            pass  # No class set

        assert metrics.predictions_total.labels(model="regime_v1")._value.get() == 0.0

    def test_track_prediction_decorator_class_label_bug(self) -> None:
        """track_prediction_decorator calls record_prediction which has class_ label bug."""
        metrics = MLMetrics()

        @metrics.track_prediction_decorator("regime_v1")
        def predict() -> str:
            return "bullish"

        with pytest.raises(ValueError, match="Incorrect label names"):
            predict()


# =============================================================================
# PipelineMetrics
# =============================================================================


@pytest.mark.usefixtures("_patch_registry")
class TestPipelineMetrics:
    """Tests for PipelineMetrics."""

    def test_init(self) -> None:
        metrics = PipelineMetrics()
        assert metrics.prefix == "pipeline"

    def test_record_dag_run(self) -> None:
        metrics = PipelineMetrics()
        metrics.record_dag_run(dag_id="ohlcv_collect", state="success", duration=120.5)
        assert metrics.dag_runs_total.labels(
            dag_id="ohlcv_collect", state="success",
        )._value.get() == 1.0

    def test_record_dag_run_no_duration(self) -> None:
        metrics = PipelineMetrics()
        metrics.record_dag_run(dag_id="cleanup", state="success")
        assert metrics.dag_runs_total.labels(
            dag_id="cleanup", state="success",
        )._value.get() == 1.0

    def test_record_task_run(self) -> None:
        metrics = PipelineMetrics()
        metrics.record_task_run(dag_id="ohlcv", task_id="fetch", state="success")
        assert metrics.task_runs_total.labels(
            dag_id="ohlcv", task_id="fetch", state="success",
        )._value.get() == 1.0

    def test_record_processing(self) -> None:
        metrics = PipelineMetrics()
        metrics.record_processing(
            pipeline="ohlcv", stage="transform", records=1000, bytes_processed=50000,
        )
        assert metrics.records_processed.labels(
            pipeline="ohlcv", stage="transform",
        )._value.get() == 1000

    def test_record_processing_no_bytes(self) -> None:
        metrics = PipelineMetrics()
        metrics.record_processing(pipeline="ohlcv", stage="load", records=500)
        assert metrics.records_processed.labels(
            pipeline="ohlcv", stage="load",
        )._value.get() == 500

    def test_record_error(self) -> None:
        metrics = PipelineMetrics()
        metrics.record_error(pipeline="ohlcv", error_type="ConnectionError")
        assert metrics.errors_total.labels(
            pipeline="ohlcv", error_type="ConnectionError",
        )._value.get() == 1.0

    def test_track_query(self) -> None:
        metrics = PipelineMetrics()
        with metrics.track_query("select"):
            pass  # simulate query
        # Verify no exception raised - histogram observation is internal
