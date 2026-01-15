"""
Tests for factor-based investment framework.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.factors.base import (
    Factor,
    FactorScore,
    FactorDirection,
    NormalizationMethod,
    calculate_factor_ic,
    calculate_factor_returns,
)
from src.factors.momentum import MomentumFactor, TimeSeriesMomentum
from src.factors.volatility import VolatilityFactor, BetaFactor
from src.factors.value import ValueFactor, RelativeValueFactor
from src.factors.quality import QualityFactor, SharpeQualityFactor
from src.factors.composite import (
    CompositeFactorModel,
    FactorWeight,
    create_standard_model,
)


@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=300, freq="D")

    # Generate prices with different characteristics
    data = {
        "BTC": 50000 * np.cumprod(1 + np.random.normal(0.001, 0.03, 300)),  # High momentum
        "ETH": 3000 * np.cumprod(1 + np.random.normal(0.0005, 0.04, 300)),  # High vol
        "SOL": 100 * np.cumprod(1 + np.random.normal(-0.001, 0.05, 300)),  # Negative momentum
        "ADA": 1.0 * np.cumprod(1 + np.random.normal(0.0002, 0.02, 300)),  # Low vol
        "DOT": 10 * np.cumprod(1 + np.random.normal(0.0003, 0.03, 300)),  # Medium
    }

    return pd.DataFrame(data, index=dates)


class TestMomentumFactor:
    """Tests for MomentumFactor."""

    def test_basic_momentum(self, sample_price_data):
        """Test basic momentum calculation."""
        factor = MomentumFactor(lookback_periods=(21, 63))
        scores = factor.calculate(sample_price_data)

        assert not scores.empty
        assert "raw" in scores.columns
        assert "normalized" in scores.columns
        assert "rank" in scores.columns
        assert len(scores) == len(sample_price_data.columns)

    def test_momentum_ranking(self, sample_price_data):
        """Test momentum produces correct ranking."""
        factor = MomentumFactor(lookback_periods=(63,))
        scores = factor.calculate(sample_price_data)

        # BTC should have high momentum (positive drift)
        # SOL should have low momentum (negative drift)
        assert scores.loc["BTC", "rank"] < scores.loc["SOL", "rank"]

    def test_risk_adjusted_momentum(self, sample_price_data):
        """Test risk-adjusted momentum."""
        factor = MomentumFactor(
            lookback_periods=(63,),
            momentum_type="risk_adjusted",
        )
        scores = factor.calculate(sample_price_data)

        assert not scores.empty
        # Risk-adjusted should favor lower volatility with same return

    def test_get_top_quintile(self, sample_price_data):
        """Test getting top quintile."""
        factor = MomentumFactor(lookback_periods=(63,))
        top = factor.get_top_quintile(sample_price_data)

        assert isinstance(top, list)
        assert len(top) <= len(sample_price_data.columns)

    def test_momentum_strength(self, sample_price_data):
        """Test momentum strength calculation."""
        factor = MomentumFactor(lookback_periods=(21, 63, 126))
        strength = factor.calculate_momentum_strength(sample_price_data)

        assert "momentum" in strength.columns
        assert "consistency" in strength.columns
        assert "acceleration" in strength.columns


class TestVolatilityFactor:
    """Tests for VolatilityFactor."""

    def test_historical_volatility(self, sample_price_data):
        """Test historical volatility calculation."""
        factor = VolatilityFactor(lookback=60, volatility_type="historical")
        scores = factor.calculate(sample_price_data)

        assert not scores.empty
        # ADA has lowest volatility, should rank highest (lower vol = better)
        # Check direction is SHORT (lower raw = higher score)
        assert factor.direction == FactorDirection.SHORT

    def test_low_vol_ranking(self, sample_price_data):
        """Test that low volatility assets rank higher."""
        factor = VolatilityFactor(lookback=60)
        scores = factor.calculate(sample_price_data)

        # ADA (2% vol) should rank better than SOL (5% vol)
        # Since direction is SHORT, lower raw value = higher rank
        assert scores.loc["ADA", "raw"] < scores.loc["SOL", "raw"]

    def test_beta_calculation(self, sample_price_data):
        """Test beta factor."""
        factor = BetaFactor(lookback=60)
        scores = factor.calculate(sample_price_data)

        assert not scores.empty
        # All betas should be positive relative to market

    def test_volatility_metrics(self, sample_price_data):
        """Test comprehensive volatility metrics."""
        factor = VolatilityFactor(lookback=60)
        metrics = factor.calculate_volatility_metrics(sample_price_data)

        assert "historical_vol" in metrics.columns
        assert "downside_vol" in metrics.columns
        assert "max_drawdown" in metrics.columns


class TestValueFactor:
    """Tests for ValueFactor."""

    def test_price_to_ma(self, sample_price_data):
        """Test price to MA calculation."""
        factor = ValueFactor(value_type="price_to_ma", ma_period=50)
        scores = factor.calculate(sample_price_data)

        assert not scores.empty
        # Lower ratio = cheaper

    def test_mean_reversion(self, sample_price_data):
        """Test mean reversion score."""
        factor = ValueFactor(value_type="mean_reversion", z_score_lookback=60)
        scores = factor.calculate(sample_price_data)

        assert not scores.empty

    def test_value_metrics(self, sample_price_data):
        """Test comprehensive value metrics."""
        factor = ValueFactor()
        metrics = factor.calculate_value_metrics(sample_price_data)

        assert "price_to_ma50" in metrics.columns
        assert "z_score_60d" in metrics.columns


class TestQualityFactor:
    """Tests for QualityFactor."""

    def test_sharpe_quality(self, sample_price_data):
        """Test Sharpe ratio calculation."""
        factor = QualityFactor(quality_type="sharpe", lookback=60)
        scores = factor.calculate(sample_price_data)

        assert not scores.empty

    def test_consistency_quality(self, sample_price_data):
        """Test consistency calculation."""
        factor = QualityFactor(quality_type="consistency", lookback=60)
        scores = factor.calculate(sample_price_data)

        assert not scores.empty
        # All consistency scores should be between 0 and 1
        assert (scores["raw"] >= 0).all()
        assert (scores["raw"] <= 1).all()

    def test_quality_metrics(self, sample_price_data):
        """Test comprehensive quality metrics."""
        factor = QualityFactor(lookback=60)
        metrics = factor.calculate_quality_metrics(sample_price_data)

        assert "sharpe" in metrics.columns
        assert "sortino" in metrics.columns
        assert "win_rate" in metrics.columns


class TestCompositeFactorModel:
    """Tests for CompositeFactorModel."""

    def test_basic_composite(self, sample_price_data):
        """Test basic composite model."""
        model = CompositeFactorModel(
            factors=[
                MomentumFactor(lookback_periods=(21, 63)),
                VolatilityFactor(lookback=60),
            ],
        )
        scores = model.calculate(sample_price_data)

        assert not scores.empty
        assert "composite" in scores.columns
        assert "rank" in scores.columns
        assert "momentum_contribution" in scores.columns

    def test_weighted_composite(self, sample_price_data):
        """Test weighted composite model."""
        model = CompositeFactorModel(
            factors=[
                MomentumFactor(lookback_periods=(21,)),
                VolatilityFactor(lookback=60),
            ],
            weights=[
                FactorWeight("momentum", weight=0.7),
                FactorWeight("volatility", weight=0.3),
            ],
        )
        scores = model.calculate(sample_price_data)

        # Momentum should have higher contribution
        assert "momentum_contribution" in scores.columns
        assert "volatility_contribution" in scores.columns

    def test_get_top_n(self, sample_price_data):
        """Test getting top N from composite model."""
        model = CompositeFactorModel(
            factors=[
                MomentumFactor(lookback_periods=(63,)),
                QualityFactor(quality_type="sharpe"),
            ],
        )
        top = model.get_top_n(sample_price_data, n=3)

        assert len(top) == 3
        assert all(t in sample_price_data.columns for t in top)

    def test_factor_exposures(self, sample_price_data):
        """Test portfolio factor exposure calculation."""
        model = CompositeFactorModel(
            factors=[
                MomentumFactor(lookback_periods=(63,)),
                VolatilityFactor(lookback=60),
            ],
        )
        portfolio_weights = {"BTC": 0.5, "ETH": 0.3, "ADA": 0.2}
        exposures = model.get_factor_exposures(sample_price_data, portfolio_weights)

        assert "momentum" in exposures
        assert "volatility" in exposures

    def test_factor_correlation(self, sample_price_data):
        """Test factor correlation calculation."""
        model = CompositeFactorModel(
            factors=[
                MomentumFactor(lookback_periods=(63,)),
                VolatilityFactor(lookback=60),
                QualityFactor(quality_type="sharpe"),
            ],
        )
        corr = model.get_factor_correlation(sample_price_data)

        assert corr.shape == (3, 3)
        # Diagonal should be 1
        assert np.allclose(np.diag(corr), 1.0)

    def test_toggle_factor(self, sample_price_data):
        """Test toggling factors on/off."""
        model = CompositeFactorModel(
            factors=[
                MomentumFactor(lookback_periods=(63,)),
                VolatilityFactor(lookback=60),
            ],
        )

        # Disable momentum
        model.toggle_factor("momentum", False)
        scores = model.calculate(sample_price_data)

        # Only volatility should contribute
        assert "volatility_contribution" in scores.columns


class TestStandardModel:
    """Tests for standard model factory."""

    def test_create_standard_model(self, sample_price_data):
        """Test creating standard multi-factor model."""
        model = create_standard_model(
            momentum_weight=0.30,
            volatility_weight=0.25,
            value_weight=0.20,
            quality_weight=0.25,
        )

        assert len(model.factors) == 4
        scores = model.calculate(sample_price_data)
        assert not scores.empty


class TestFactorMetrics:
    """Tests for factor performance metrics."""

    def test_calculate_factor_ic(self, sample_price_data):
        """Test Information Coefficient calculation."""
        factor = MomentumFactor(lookback_periods=(21,))
        scores = factor.calculate(sample_price_data)

        # Create forward returns
        returns = sample_price_data.pct_change(21).iloc[-1]

        ic = calculate_factor_ic(scores["normalized"], returns)

        assert isinstance(ic, float)
        assert -1 <= ic <= 1

    def test_calculate_factor_returns(self, sample_price_data):
        """Test factor return calculation."""
        factor = MomentumFactor(lookback_periods=(21,))
        scores = factor.calculate(sample_price_data)

        returns = sample_price_data.pct_change(21).iloc[-1]

        spread = calculate_factor_returns(
            scores["normalized"],
            returns,
            n_quantiles=3,
        )

        assert isinstance(spread, float)


class TestNormalization:
    """Tests for normalization methods."""

    def test_zscore_normalization(self, sample_price_data):
        """Test z-score normalization."""
        factor = MomentumFactor(
            lookback_periods=(63,),
            normalization=NormalizationMethod.ZSCORE,
        )
        scores = factor.calculate(sample_price_data)

        # Z-scores should have mean ~0, std ~1
        assert abs(scores["normalized"].mean()) < 0.5
        assert 0.5 < scores["normalized"].std() < 2.0

    def test_rank_normalization(self, sample_price_data):
        """Test rank normalization."""
        factor = MomentumFactor(
            lookback_periods=(63,),
            normalization=NormalizationMethod.RANK,
        )
        scores = factor.calculate(sample_price_data)

        # Rank should be between 0 and 1
        assert (scores["normalized"] >= 0).all()
        assert (scores["normalized"] <= 1).all()

    def test_minmax_normalization(self, sample_price_data):
        """Test min-max normalization."""
        factor = MomentumFactor(
            lookback_periods=(63,),
            normalization=NormalizationMethod.MINMAX,
        )
        scores = factor.calculate(sample_price_data)

        # Should be between 0 and 1
        assert (scores["normalized"] >= 0).all()
        assert (scores["normalized"] <= 1).all()
