"""Tests for backtester.analysis.robustness_models."""

from src.backtester.analysis.robustness_models import RobustnessReport, RobustnessResult


class TestRobustnessResult:
    def test_to_dict_merges_params_and_metrics(self) -> None:
        r = RobustnessResult(
            params={"k": 0.5, "ma": 20},
            total_return=10.0,
            sharpe=1.5,
            max_drawdown=-5.0,
            win_rate=0.6,
            trade_count=10,
        )
        d = r.to_dict()
        assert d["k"] == 0.5
        assert d["ma"] == 20
        assert d["total_return"] == 10.0
        assert d["sharpe"] == 1.5
        assert d["trade_count"] == 10

    def test_to_dict_empty_params(self) -> None:
        r = RobustnessResult(
            params={},
            total_return=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            trade_count=0,
        )
        d = r.to_dict()
        assert d["total_return"] == 0.0
        assert d["trade_count"] == 0


class TestRobustnessReport:
    def test_sensitivity_scores_initialized_to_empty_dict(self) -> None:
        report = RobustnessReport(optimal_params={"k": 0.5}, results=[])
        assert report.sensitivity_scores == {}

    def test_aggregate_fields_default_zero(self) -> None:
        report = RobustnessReport(optimal_params={}, results=[])
        assert report.mean_return == 0.0
        assert report.std_return == 0.0
        assert report.neighbor_success_rate == 0.0

    def test_results_stored(self) -> None:
        r = RobustnessResult(
            params={"k": 0.6},
            total_return=5.0,
            sharpe=1.2,
            max_drawdown=-3.0,
            win_rate=0.55,
            trade_count=8,
        )
        report = RobustnessReport(optimal_params={"k": 0.6}, results=[r])
        assert len(report.results) == 1
        assert report.results[0].sharpe == 1.2
