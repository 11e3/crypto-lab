"""Tests for optimization service — pure computation logic.

Tests:
- VboOptimizationResult dataclass
- get_default_param_range (int, float, bool types)
- parse_dynamic_param_grid (parsing, validation, error handling)
- extract_vbo_metric (all metric types)
- execute_vbo_optimization (mock-based, grid/random, progress callback)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.web.services.optimization_service import (
    VboOptimizationResult,
    extract_vbo_metric,
    get_default_param_range,
    parse_dynamic_param_grid,
)
from src.web.services.parameter_models import ParameterSpec

# ── VboOptimizationResult ──


class TestVboOptimizationResult:
    """Test VboOptimizationResult dataclass."""

    def test_creation(self) -> None:
        result = VboOptimizationResult(
            best_params={"lookback": 5},
            best_score=1.5,
            all_params=[{"lookback": 5}, {"lookback": 10}],
            all_scores=[1.5, 1.2],
        )
        assert result.best_params == {"lookback": 5}
        assert result.best_score == 1.5
        assert len(result.all_params) == 2

    def test_default_factory(self) -> None:
        result = VboOptimizationResult(best_params={}, best_score=0.0)
        assert result.all_params == []
        assert result.all_scores == []


# ── get_default_param_range ──


class TestGetDefaultParamRange:
    """Test default parameter range generation."""

    def test_int_param_generates_range(self) -> None:
        spec = ParameterSpec(
            name="lookback", type="int", default=5, min_value=2, max_value=20, step=1
        )
        result = get_default_param_range(spec)
        values = [int(v) for v in result.split(",")]
        assert 5 in values
        assert all(isinstance(v, int) for v in values)
        assert all(2 <= v <= 20 for v in values)

    def test_int_param_centers_around_default(self) -> None:
        spec = ParameterSpec(
            name="period", type="int", default=10, min_value=1, max_value=100, step=2
        )
        result = get_default_param_range(spec)
        values = [int(v) for v in result.split(",")]
        # All values should be within step*3 of default
        for v in values:
            assert abs(v - 10) <= 2 * 3

    def test_int_param_fallback_to_default(self) -> None:
        """When no values in range, falls back to default."""
        spec = ParameterSpec(
            name="period", type="int", default=50, min_value=1, max_value=5, step=1
        )
        result = get_default_param_range(spec)
        values = [int(v) for v in result.split(",")]
        assert 50 in values

    def test_float_param_generates_range(self) -> None:
        spec = ParameterSpec(
            name="ratio", type="float", default=0.5, min_value=0.1, max_value=1.0, step=0.1
        )
        result = get_default_param_range(spec)
        values = [float(v) for v in result.split(",")]
        assert len(values) <= 5
        assert all(0.1 <= v <= 1.0 for v in values)

    def test_float_param_max_5_values(self) -> None:
        spec = ParameterSpec(
            name="rate", type="float", default=0.0, min_value=0.0, max_value=10.0, step=0.01
        )
        result = get_default_param_range(spec)
        values = result.split(",")
        assert len(values) <= 5

    def test_bool_param(self) -> None:
        spec = ParameterSpec(name="flag", type="bool", default=True)
        assert get_default_param_range(spec) == "True,False"

    def test_unknown_type_returns_default(self) -> None:
        spec = ParameterSpec(name="x", type="choice", default="abc")
        assert get_default_param_range(spec) == "abc"

    def test_none_min_max(self) -> None:
        """Handles None min/max gracefully."""
        spec = ParameterSpec(name="x", type="int", default=5, min_value=None, max_value=None)
        result = get_default_param_range(spec)
        values = [int(v) for v in result.split(",")]
        assert len(values) >= 1


# ── parse_dynamic_param_grid ──


class TestParseDynamicParamGrid:
    """Test parameter grid parsing."""

    def test_int_parsing(self) -> None:
        spec = ParameterSpec(name="lookback", type="int", default=5)
        result = parse_dynamic_param_grid({"lookback": "3,5,7,10"}, {"lookback": spec})
        assert result == {"lookback": [3, 5, 7, 10]}

    def test_float_parsing(self) -> None:
        spec = ParameterSpec(name="ratio", type="float", default=0.5)
        result = parse_dynamic_param_grid({"ratio": "0.1,0.3,0.5"}, {"ratio": spec})
        assert result == {"ratio": [0.1, 0.3, 0.5]}

    def test_bool_parsing(self) -> None:
        spec = ParameterSpec(name="flag", type="bool", default=True)
        result = parse_dynamic_param_grid({"flag": "True,False"}, {"flag": spec})
        assert result == {"flag": [True, False]}

    def test_bool_parsing_various_formats(self) -> None:
        spec = ParameterSpec(name="flag", type="bool", default=True)
        result = parse_dynamic_param_grid({"flag": "true,1,yes,false,0,no"}, {"flag": spec})
        assert result == {"flag": [True, True, True, False, False, False]}

    def test_multiple_params(self) -> None:
        specs = {
            "lookback": ParameterSpec(name="lookback", type="int", default=5),
            "ratio": ParameterSpec(name="ratio", type="float", default=0.5),
        }
        result = parse_dynamic_param_grid({"lookback": "3,5,7", "ratio": "0.1,0.5"}, specs)
        assert result["lookback"] == [3, 5, 7]
        assert result["ratio"] == [0.1, 0.5]

    def test_empty_string_raises(self) -> None:
        spec = ParameterSpec(name="x", type="int", default=5)
        with pytest.raises(ValueError, match="Please enter values"):
            parse_dynamic_param_grid({"x": "  "}, {"x": spec})

    def test_invalid_int_raises(self) -> None:
        spec = ParameterSpec(name="x", type="int", default=5)
        with pytest.raises(ValueError, match="Invalid value"):
            parse_dynamic_param_grid({"x": "abc"}, {"x": spec})

    def test_invalid_float_raises(self) -> None:
        spec = ParameterSpec(name="x", type="float", default=0.5)
        with pytest.raises(ValueError, match="Invalid value"):
            parse_dynamic_param_grid({"x": "not_a_float"}, {"x": spec})

    def test_skips_empty_values_in_list(self) -> None:
        spec = ParameterSpec(name="x", type="int", default=5)
        result = parse_dynamic_param_grid({"x": "3,,5,,7"}, {"x": spec})
        assert result == {"x": [3, 5, 7]}

    def test_all_empty_after_split_raises(self) -> None:
        spec = ParameterSpec(name="x", type="int", default=5)
        # Only commas → no valid values after stripping
        with pytest.raises(ValueError, match="No valid values"):
            parse_dynamic_param_grid({"x": ",,,"}, {"x": spec})

    def test_unknown_param_type_uses_string(self) -> None:
        """When spec not in param_specs, defaults to int parsing."""
        result = parse_dynamic_param_grid({"unknown": "1,2,3"}, {})
        assert result == {"unknown": [1, 2, 3]}

    def test_whitespace_trimming(self) -> None:
        spec = ParameterSpec(name="x", type="int", default=5)
        result = parse_dynamic_param_grid({"x": " 3 , 5 , 7 "}, {"x": spec})
        assert result == {"x": [3, 5, 7]}


# ── extract_vbo_metric ──


class TestExtractVboMetric:
    """Test metric extraction from VboBacktestResult."""

    @pytest.fixture
    def mock_result(self) -> MagicMock:
        result = MagicMock()
        result.sharpe_ratio = 1.5
        result.cagr = 25.0
        result.mdd = -10.0
        result.total_return = 50.0
        result.win_rate = 0.6
        result.profit_factor = 2.0
        result.sortino_ratio = 2.5
        return result

    def test_sharpe_ratio(self, mock_result: MagicMock) -> None:
        assert extract_vbo_metric(mock_result, "sharpe_ratio") == 1.5

    def test_cagr(self, mock_result: MagicMock) -> None:
        assert extract_vbo_metric(mock_result, "cagr") == 25.0

    def test_total_return(self, mock_result: MagicMock) -> None:
        assert extract_vbo_metric(mock_result, "total_return") == 50.0

    def test_win_rate(self, mock_result: MagicMock) -> None:
        assert extract_vbo_metric(mock_result, "win_rate") == 0.6

    def test_profit_factor(self, mock_result: MagicMock) -> None:
        assert extract_vbo_metric(mock_result, "profit_factor") == 2.0

    def test_sortino_ratio(self, mock_result: MagicMock) -> None:
        assert extract_vbo_metric(mock_result, "sortino_ratio") == 2.5

    def test_calmar_ratio(self, mock_result: MagicMock) -> None:
        # calmar = cagr / |mdd| = 25.0 / 10.0 = 2.5
        assert extract_vbo_metric(mock_result, "calmar_ratio") == 2.5

    def test_calmar_ratio_zero_mdd(self, mock_result: MagicMock) -> None:
        mock_result.mdd = 0.0
        # When mdd is 0, uses 1.0 as denominator
        assert extract_vbo_metric(mock_result, "calmar_ratio") == 25.0

    def test_unknown_metric_defaults_to_sharpe(self, mock_result: MagicMock) -> None:
        assert extract_vbo_metric(mock_result, "nonexistent") == 1.5


# ── execute_vbo_optimization ──


class TestExecuteVboOptimization:
    """Test VBO optimization execution with mocked backtest runners."""

    @patch("src.web.services.optimization_service.run_vbo_backtest_service")
    def test_grid_search_all_combinations(self, mock_bt: MagicMock) -> None:
        from src.web.services.optimization_service import execute_vbo_optimization

        mock_result = MagicMock()
        mock_result.sharpe_ratio = 1.5
        mock_result.mdd = -10.0
        mock_result.cagr = 20.0
        mock_bt.return_value = mock_result

        result = execute_vbo_optimization(
            strategy_name="VBO",
            param_grid={"lookback": [3, 5], "multiplier": [2, 3]},
            symbols=["BTC"],
            metric="sharpe_ratio",
            method="grid",
            n_iter=100,
            initial_capital=10_000_000,
            fee_rate=0.0005,
        )

        assert mock_bt.call_count == 4  # 2x2 grid
        assert isinstance(result, VboOptimizationResult)
        assert result.best_score == 1.5
        assert len(result.all_params) == 4

    @patch("src.web.services.optimization_service.run_vbo_backtest_service")
    def test_random_search_limits_iterations(self, mock_bt: MagicMock) -> None:
        from src.web.services.optimization_service import execute_vbo_optimization

        mock_result = MagicMock()
        mock_result.sharpe_ratio = 1.0
        mock_result.mdd = -5.0
        mock_result.cagr = 10.0
        mock_bt.return_value = mock_result

        result = execute_vbo_optimization(
            strategy_name="VBO",
            param_grid={"lookback": [3, 5, 7, 10], "multiplier": [1, 2, 3]},
            symbols=["BTC"],
            metric="sharpe_ratio",
            method="random",
            n_iter=3,
            initial_capital=10_000_000,
            fee_rate=0.0005,
        )

        assert mock_bt.call_count == 3  # Limited to n_iter
        assert len(result.all_params) == 3

    @patch("src.web.services.optimization_service.run_vbo_backtest_service")
    def test_progress_callback(self, mock_bt: MagicMock) -> None:
        from src.web.services.optimization_service import execute_vbo_optimization

        mock_result = MagicMock()
        mock_result.sharpe_ratio = 1.0
        mock_result.mdd = -5.0
        mock_bt.return_value = mock_result

        progress_calls: list[tuple[int, int]] = []

        def on_progress(current: int, total: int) -> None:
            progress_calls.append((current, total))

        execute_vbo_optimization(
            strategy_name="VBO",
            param_grid={"lookback": [3, 5]},
            symbols=["BTC"],
            metric="sharpe_ratio",
            method="grid",
            n_iter=100,
            initial_capital=10_000_000,
            fee_rate=0.0005,
            on_progress=on_progress,
        )

        assert len(progress_calls) == 2
        assert progress_calls[0] == (1, 2)
        assert progress_calls[1] == (2, 2)

    @patch("src.web.services.optimization_service.run_vbo_backtest_service")
    def test_all_backtests_fail_raises(self, mock_bt: MagicMock) -> None:
        from src.web.services.optimization_service import execute_vbo_optimization

        mock_bt.return_value = None

        with pytest.raises(RuntimeError, match="All backtests failed"):
            execute_vbo_optimization(
                strategy_name="VBO",
                param_grid={"lookback": [3]},
                symbols=["BTC"],
                metric="sharpe_ratio",
                method="grid",
                n_iter=100,
                initial_capital=10_000_000,
                fee_rate=0.0005,
            )

    @patch("src.web.services.optimization_service.run_vbo_backtest_service")
    def test_partial_failures_skip_failed(self, mock_bt: MagicMock) -> None:
        """If some backtests fail, they are excluded from results."""
        from src.web.services.optimization_service import execute_vbo_optimization

        good_result = MagicMock()
        good_result.sharpe_ratio = 2.0
        good_result.mdd = -5.0
        good_result.cagr = 15.0

        # First call succeeds, second returns None
        mock_bt.side_effect = [good_result, None]

        result = execute_vbo_optimization(
            strategy_name="VBO",
            param_grid={"lookback": [3, 5]},
            symbols=["BTC"],
            metric="sharpe_ratio",
            method="grid",
            n_iter=100,
            initial_capital=10_000_000,
            fee_rate=0.0005,
        )

        assert result.best_score == 2.0
        assert len(result.all_params) == 1  # Only successful one

    @patch("src.web.services.optimization_service.run_vbo_backtest_service")
    def test_exception_during_backtest_logs_warning(self, mock_bt: MagicMock) -> None:
        """Exceptions are caught and logged, not propagated."""
        from src.web.services.optimization_service import execute_vbo_optimization

        good_result = MagicMock()
        good_result.sharpe_ratio = 1.0
        good_result.mdd = -5.0
        good_result.cagr = 10.0

        mock_bt.side_effect = [RuntimeError("boom"), good_result]

        result = execute_vbo_optimization(
            strategy_name="VBO",
            param_grid={"lookback": [3, 5]},
            symbols=["BTC"],
            metric="sharpe_ratio",
            method="grid",
            n_iter=100,
            initial_capital=10_000_000,
            fee_rate=0.0005,
        )

        # Only the successful one is in results
        assert result.best_score == 1.0
        assert len(result.all_params) == 1
