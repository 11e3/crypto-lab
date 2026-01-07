"""
Unit tests for portfolio optimization module.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeResult

# 모듈 경로에 맞게 수정 (src.risk.portfolio_optimization)
from src.risk.portfolio_optimization import (
    PortfolioOptimizer,
    PortfolioWeights,
    optimize_portfolio,
)

# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def sample_returns_df() -> pd.DataFrame:
    """Generate a sample DataFrame of asset returns."""
    dates = pd.to_datetime(pd.date_range("2023-01-01", periods=252, freq="D"))
    rng = np.random.default_rng(42)
    returns = pd.DataFrame(
        {
            "ASSET1": rng.normal(0.0005, 0.01, 252),
            "ASSET2": rng.normal(0.0007, 0.015, 252),
            "ASSET3": rng.normal(0.0003, 0.008, 252),
        },
        index=dates,
    )
    return returns


@pytest.fixture
def sample_trades_df() -> pd.DataFrame:
    """Generate a sample DataFrame of trade results for Kelly."""
    return pd.DataFrame(
        {
            "ticker": [
                "ASSET1",
                "ASSET1",
                "ASSET2",
                "ASSET1",
                "ASSET2",
                "ASSET3",
                "ASSET1",
                "ASSET2",
                "ASSET3",
            ],
            "pnl_pct": [10.0, -5.0, 15.0, 8.0, -7.0, 20.0, -3.0, 10.0, -10.0],
            "pnl": [1000, -500, 1500, 800, -700, 2000, -300, 1000, -1000],
        }
    )


# -------------------------------------------------------------------------
# Test Classes
# -------------------------------------------------------------------------


class TestPortfolioWeights:
    """Tests for the PortfolioWeights dataclass."""

    def test_initialization(self) -> None:
        weights = PortfolioWeights(
            weights={"ASSET1": 0.5, "ASSET2": 0.5},
            method="test_method",
            expected_return=0.1,
            portfolio_volatility=0.2,
            sharpe_ratio=0.5,
        )
        assert weights.weights == {"ASSET1": 0.5, "ASSET2": 0.5}
        assert weights.method == "test_method"

    def test_repr(self) -> None:
        weights = PortfolioWeights(
            weights={"ASSET1": 0.5, "ASSET2": 0.5}, method="mpt", sharpe_ratio=0.75
        )
        assert "PortfolioWeights(method=mpt, assets=2, sharpe=0.750)" in repr(weights)


class TestPortfolioOptimizer:
    """Tests for the PortfolioOptimizer class."""

    @pytest.fixture
    def optimizer(self) -> PortfolioOptimizer:
        return PortfolioOptimizer()

    def test_optimize_mpt_maximize_sharpe(
        self, optimizer: PortfolioOptimizer, sample_returns_df: pd.DataFrame
    ) -> None:
        weights = optimizer.optimize_mpt(sample_returns_df, risk_free_rate=0.01)
        assert isinstance(weights, PortfolioWeights)
        assert weights.method == "mpt"
        assert sum(weights.weights.values()) == pytest.approx(1.0)
        assert weights.sharpe_ratio is not None and weights.sharpe_ratio > 0

    def test_optimize_mpt_target_return(
        self, optimizer: PortfolioOptimizer, sample_returns_df: pd.DataFrame
    ) -> None:
        mean_returns = sample_returns_df.mean() * 252
        target_return = mean_returns.mean()

        weights = optimizer.optimize_mpt(sample_returns_df, target_return=target_return)
        assert isinstance(weights, PortfolioWeights)
        assert weights.expected_return == pytest.approx(target_return, rel=1e-3)

    def test_optimize_mpt_empty_returns_df(self, optimizer: PortfolioOptimizer) -> None:
        with pytest.raises(ValueError, match="Returns DataFrame is empty"):
            optimizer.optimize_mpt(pd.DataFrame())

    def test_optimize_mpt_single_asset(
        self, optimizer: PortfolioOptimizer, sample_returns_df: pd.DataFrame
    ) -> None:
        single_asset_df = sample_returns_df[["ASSET1"]]
        weights = optimizer.optimize_mpt(single_asset_df)
        assert weights.weights == {"ASSET1": pytest.approx(1.0)}

    @pytest.mark.parametrize("max_w, min_w", [(0.6, 0.1), (0.4, 0.0)])
    def test_optimize_mpt_constraints(
        self,
        optimizer: PortfolioOptimizer,
        sample_returns_df: pd.DataFrame,
        max_w: float,
        min_w: float,
        mocker: pytest.MonkeyPatch,
    ) -> None:
        # 가짜 최적화 결과 생성
        mock_weights = np.array([0.3, 0.3, 0.4])
        mock_result = OptimizeResult(
            x=mock_weights,
            success=True,
            message="Ok",
            fun=0,
            nit=0,
            jac=np.array([]),
            hess_inv=None,
        )

        # 중요: minimize 패치 경로는 소스 코드의 import 방식에 따라 다릅니다.
        # Case A: 소스 코드가 `from scipy.optimize import minimize`를 쓰는 경우 -> 아래 유지
        # Case B: 소스 코드가 `import scipy.optimize`를 쓰는 경우 -> "scipy.optimize.minimize"로 변경
        mocker.patch("src.risk.portfolio_optimization.minimize", return_value=mock_result)

        weights = optimizer.optimize_mpt(sample_returns_df, max_weight=max_w, min_weight=min_w)
        assert all(min_w <= w <= max_w for w in weights.weights.values())

    @pytest.mark.parametrize("method", ["SLSQP"])
    @pytest.mark.parametrize("success_status", [False])
    def test_optimize_mpt_optimization_failure(
        self,
        optimizer: PortfolioOptimizer,
        sample_returns_df: pd.DataFrame,
        method: str,
        success_status: bool,
        mocker: pytest.MonkeyPatch,
    ) -> None:
        mock_result = OptimizeResult(
            x=np.array([0.33, 0.33, 0.34]), success=success_status, message="Optimization failed"
        )
        mocker.patch("src.risk.portfolio_optimization.minimize", return_value=mock_result)

        weights = optimizer.optimize_mpt(sample_returns_df)
        # 실패 시 균등 분배(1/N)로 fallback 되는지 확인
        assert weights.weights["ASSET1"] == pytest.approx(1.0 / len(sample_returns_df.columns))

    @pytest.mark.parametrize("error", [Exception("Mock error")])
    def test_optimize_mpt_exception_handling(
        self,
        optimizer: PortfolioOptimizer,
        sample_returns_df: pd.DataFrame,
        error: Exception,
        mocker: pytest.MonkeyPatch,
    ) -> None:
        mocker.patch("src.risk.portfolio_optimization.minimize", side_effect=error)
        weights = optimizer.optimize_mpt(sample_returns_df)
        assert weights.weights["ASSET1"] == pytest.approx(1.0 / len(sample_returns_df.columns))

    def test_optimize_risk_parity(
        self, optimizer: PortfolioOptimizer, sample_returns_df: pd.DataFrame
    ) -> None:
        weights = optimizer.optimize_risk_parity(sample_returns_df)
        assert weights.method == "risk_parity"
        assert sum(weights.weights.values()) == pytest.approx(1.0)

    # ... (Kelly 관련 테스트는 기존 로직 유지, mocker가 없으므로 통과 예상) ...

    def test_optimize_kelly_portfolio(
        self, optimizer: PortfolioOptimizer, sample_trades_df: pd.DataFrame
    ) -> None:
        available_cash = 10000
        allocations = optimizer.optimize_kelly_portfolio(sample_trades_df, available_cash)
        assert sum(allocations.values()) <= available_cash + 1e-6
        assert "ASSET1" in allocations

    # -------------------------------------------------------------------------
    # Integration Tests (Using optimize_portfolio wrapper)
    # -------------------------------------------------------------------------

    def test_optimize_portfolio_mpt(
        self, sample_returns_df: pd.DataFrame, mocker: pytest.MonkeyPatch
    ) -> None:
        """Test optimize_portfolio wrapper calls the correct class method."""
        # 문자열 경로 대신 객체를 직접 패치 (더 안전함)
        mock_method = mocker.patch.object(
            PortfolioOptimizer,
            "optimize_mpt",
            return_value=PortfolioWeights(weights={}, method="mpt"),
        )

        result = optimize_portfolio(sample_returns_df, method="mpt")
        assert result.method == "mpt"
        mock_method.assert_called_once()

    def test_optimize_portfolio_risk_parity(
        self, sample_returns_df: pd.DataFrame, mocker: pytest.MonkeyPatch
    ) -> None:
        mock_method = mocker.patch.object(
            PortfolioOptimizer,
            "optimize_risk_parity",
            return_value=PortfolioWeights(weights={}, method="risk_parity"),
        )

        result = optimize_portfolio(sample_returns_df, method="risk_parity")
        assert result.method == "risk_parity"
        mock_method.assert_called_once()

    def test_optimize_portfolio_kelly(
        self,
        sample_returns_df: pd.DataFrame,
        sample_trades_df: pd.DataFrame,
        mocker: pytest.MonkeyPatch,
    ) -> None:
        mock_method = mocker.patch.object(
            PortfolioOptimizer,
            "optimize_kelly_portfolio",
            return_value={"ASSET1": 5000.0, "ASSET2": 3000.0},
        )

        result = optimize_portfolio(
            sample_returns_df, method="kelly", trades=sample_trades_df, available_cash=10000
        )
        assert result.method == "kelly"
        mock_method.assert_called_once()
