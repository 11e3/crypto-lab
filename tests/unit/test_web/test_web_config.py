"""Tests for web config modules (constants.py and app_settings.py)."""

from __future__ import annotations

from src.web.config.constants import (
    BINANCE_DATA_COLLECT_INTERVALS,
    BINANCE_DATA_COLLECT_TICKERS,
    DATA_COLLECT_INTERVALS,
    DATA_COLLECT_TICKERS,
    DEFAULT_TICKERS,
    INTERVAL_DISPLAY_MAP,
    OPTIMIZATION_METRICS,
)

# =========================================================================
# constants.py
# =========================================================================


class TestWebConstants:
    """Tests for web config constants."""

    def test_default_tickers_not_empty(self) -> None:
        assert len(DEFAULT_TICKERS) >= 3
        assert all(t.startswith("KRW-") for t in DEFAULT_TICKERS)

    def test_data_collect_tickers_superset_of_default(self) -> None:
        """DATA_COLLECT_TICKERS should include all DEFAULT_TICKERS."""
        for ticker in DEFAULT_TICKERS:
            assert ticker in DATA_COLLECT_TICKERS

    def test_data_collect_intervals_has_day(self) -> None:
        interval_keys = [i[0] for i in DATA_COLLECT_INTERVALS]
        assert "day" in interval_keys

    def test_binance_tickers_format(self) -> None:
        """Binance tickers should end with USDT or BTC."""
        for ticker in BINANCE_DATA_COLLECT_TICKERS:
            assert ticker.endswith("USDT") or ticker.endswith("BTC")

    def test_binance_intervals_has_1d(self) -> None:
        interval_keys = [i[0] for i in BINANCE_DATA_COLLECT_INTERVALS]
        assert "1d" in interval_keys

    def test_interval_display_map_keys(self) -> None:
        assert "day" in INTERVAL_DISPLAY_MAP
        assert "minute240" in INTERVAL_DISPLAY_MAP

    def test_optimization_metrics_not_empty(self) -> None:
        assert len(OPTIMIZATION_METRICS) >= 4
        metric_keys = [m[0] for m in OPTIMIZATION_METRICS]
        assert "sharpe_ratio" in metric_keys
        assert "cagr" in metric_keys

    def test_all_intervals_are_tuples(self) -> None:
        """All interval entries should be (key, display) tuples."""
        for item in DATA_COLLECT_INTERVALS:
            assert isinstance(item, tuple)
            assert len(item) == 2
        for item in BINANCE_DATA_COLLECT_INTERVALS:
            assert isinstance(item, tuple)
            assert len(item) == 2
