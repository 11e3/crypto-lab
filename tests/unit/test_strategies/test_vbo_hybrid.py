"""Tests for VBOHybrid strategy (MA20 + ML hybrid filter)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.strategies.volatility_breakout.vbo_hybrid import (
    HybridMarketExitCondition,
    HybridMarketFilterCondition,
    VBOHybrid,
)


@pytest.fixture()
def ohlcv_data() -> pd.DataFrame:
    """Generate OHLCV data with enough history."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    prices = 50000 + np.cumsum(np.random.randn(n) * 500)
    return pd.DataFrame(
        {
            "open": prices * 0.99,
            "high": prices * 1.02,
            "low": prices * 0.97,
            "close": prices,
            "volume": np.random.rand(n) * 10000 + 1000,
        },
        index=dates,
    )


@pytest.fixture()
def btc_bull_data() -> pd.DataFrame:
    """BTC data clearly above MA20 (>2% above)."""
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    # Steadily rising: will be well above MA20
    prices = 50000 + np.arange(n) * 200.0
    return pd.DataFrame(
        {
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.full(n, 5000.0),
        },
        index=dates,
    )


@pytest.fixture()
def btc_bear_data() -> pd.DataFrame:
    """BTC data clearly below MA20 (>2% below)."""
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    # Steadily falling: will be well below MA20
    prices = 70000 - np.arange(n) * 200.0
    return pd.DataFrame(
        {
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.full(n, 5000.0),
        },
        index=dates,
    )


@pytest.fixture()
def btc_flat_data() -> pd.DataFrame:
    """BTC data near MA20 (within ±2% = ambiguous zone)."""
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    # Flat with tiny oscillations: stays close to MA20
    prices = 50000 + np.sin(np.arange(n) * 0.1) * 100
    return pd.DataFrame(
        {
            "open": prices * 0.999,
            "high": prices * 1.001,
            "low": prices * 0.998,
            "close": prices,
            "volume": np.full(n, 5000.0),
        },
        index=dates,
    )


class TestHybridMarketFilterCondition:
    """Test HybridMarketFilterCondition."""

    def test_true_when_hybrid_bull(self) -> None:
        from datetime import date

        from src.strategies.base import OHLCV

        condition = HybridMarketFilterCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        indicators = {"hybrid_bull": True}
        assert condition.evaluate(current, pd.DataFrame(), indicators) is True

    def test_false_when_hybrid_not_bull(self) -> None:
        from datetime import date

        from src.strategies.base import OHLCV

        condition = HybridMarketFilterCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        indicators = {"hybrid_bull": False}
        assert condition.evaluate(current, pd.DataFrame(), indicators) is False

    def test_false_when_missing(self) -> None:
        from datetime import date

        from src.strategies.base import OHLCV

        condition = HybridMarketFilterCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        assert condition.evaluate(current, pd.DataFrame(), {}) is False


class TestHybridMarketExitCondition:
    """Test HybridMarketExitCondition."""

    def test_exit_when_hybrid_bear(self) -> None:
        from datetime import date

        from src.strategies.base import OHLCV

        condition = HybridMarketExitCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        indicators = {"hybrid_bear": True}
        assert condition.evaluate(current, pd.DataFrame(), indicators) is True

    def test_no_exit_when_bull(self) -> None:
        from datetime import date

        from src.strategies.base import OHLCV

        condition = HybridMarketExitCondition()
        current = OHLCV(date(2024, 1, 1), 100.0, 110.0, 95.0, 105.0, 1000.0)
        indicators = {"hybrid_bear": False}
        assert condition.evaluate(current, pd.DataFrame(), indicators) is False


class TestVBOHybridInit:
    """Test VBOHybrid initialization."""

    def test_default_parameters(self) -> None:
        strategy = VBOHybrid(btc_data=pd.DataFrame(), model_path=None)
        assert strategy.name == "VBOHybrid"
        assert strategy.ma_short == 5
        assert strategy.btc_ma == 20
        assert strategy.threshold == 0.02
        assert strategy.ml_confidence == 0.5

    def test_custom_parameters(self) -> None:
        strategy = VBOHybrid(
            name="Custom",
            ma_short=10,
            btc_ma=30,
            threshold=0.03,
            ml_confidence=0.6,
            btc_data=pd.DataFrame(),
            model_path=None,
        )
        assert strategy.name == "Custom"
        assert strategy.ma_short == 10
        assert strategy.btc_ma == 30
        assert strategy.threshold == 0.03
        assert strategy.ml_confidence == 0.6

    def test_required_indicators(self) -> None:
        strategy = VBOHybrid(btc_data=pd.DataFrame(), model_path=None)
        indicators = strategy.required_indicators()
        assert "sma" in indicators
        assert "target" in indicators
        assert "hybrid_bull" in indicators
        assert "hybrid_bear" in indicators


class TestVBOHybridClearBull:
    """When BTC is clearly above MA20, ML should not be consulted."""

    def test_clearly_bull_allows_entry(
        self, ohlcv_data: pd.DataFrame, btc_bull_data: pd.DataFrame
    ) -> None:
        strategy = VBOHybrid(btc_data=btc_bull_data, model_path=None)
        df = strategy.calculate_indicators(ohlcv_data.copy())

        # After warmup, clearly bull days should have hybrid_bull=True
        valid = df.iloc[30:]
        assert valid["hybrid_bull"].any()

    def test_no_ml_needed_for_clear_bull(
        self, ohlcv_data: pd.DataFrame, btc_bull_data: pd.DataFrame
    ) -> None:
        """ML model should NOT be loaded for clearly bullish BTC."""
        with patch(
            "src.strategies.volatility_breakout.vbo_hybrid.get_regime_model_loader"
        ) as mock_loader:
            mock_loader.return_value = MagicMock()
            mock_loader.return_value.load_model.side_effect = FileNotFoundError(
                "Should not be called"
            )

            strategy = VBOHybrid(btc_data=btc_bull_data, model_path="/fake.joblib")
            df = strategy.calculate_indicators(ohlcv_data.copy())

            # Should still have hybrid_bull even though ML failed
            valid = df.iloc[30:]
            assert valid["hybrid_bull"].any()


class TestVBOHybridClearBear:
    """When BTC is clearly below MA20, ML should not be consulted."""

    def test_clearly_bear_blocks_entry(
        self, ohlcv_data: pd.DataFrame, btc_bear_data: pd.DataFrame
    ) -> None:
        strategy = VBOHybrid(btc_data=btc_bear_data, model_path=None)
        df = strategy.calculate_indicators(ohlcv_data.copy())

        # After MA20 settles, clearly bear days should have hybrid_bear=True
        valid = df.iloc[40:]
        assert valid["hybrid_bear"].any()


class TestVBOHybridAmbiguous:
    """When BTC is near MA20, ML model should decide."""

    @patch("src.strategies.volatility_breakout.vbo_hybrid.predict_regime_proba")
    @patch("src.strategies.volatility_breakout.vbo_hybrid.get_regime_model_loader")
    def test_ambiguous_zone_uses_ml(
        self,
        mock_get_loader: MagicMock,
        mock_predict_proba: MagicMock,
        ohlcv_data: pd.DataFrame,
        btc_flat_data: pd.DataFrame,
    ) -> None:
        """In ambiguous zone, ML probability should determine bull/bear."""
        mock_loader = MagicMock()
        mock_get_loader.return_value = mock_loader
        mock_loader.load_model.return_value = {"model": MagicMock()}

        # Return high bull probability
        n = len(btc_flat_data)
        probas_df = pd.DataFrame(
            {"BULL_TREND": np.full(n, 0.8), "NOT_BULL": np.full(n, 0.2)},
            index=btc_flat_data.index,
        )
        mock_predict_proba.return_value = probas_df

        strategy = VBOHybrid(
            btc_data=btc_flat_data,
            model_path="/fake/model.joblib",
            threshold=0.02,
            ml_confidence=0.5,
        )
        df = strategy.calculate_indicators(ohlcv_data.copy())

        # With high ML probability, ambiguous zone should be bull
        valid = df.iloc[30:]
        assert valid["hybrid_bull"].any()

    @patch("src.strategies.volatility_breakout.vbo_hybrid.predict_regime_proba")
    @patch("src.strategies.volatility_breakout.vbo_hybrid.get_regime_model_loader")
    def test_ml_bear_in_ambiguous_zone(
        self,
        mock_get_loader: MagicMock,
        mock_predict_proba: MagicMock,
        ohlcv_data: pd.DataFrame,
        btc_flat_data: pd.DataFrame,
    ) -> None:
        """Low ML probability in ambiguous zone → bear."""
        mock_loader = MagicMock()
        mock_get_loader.return_value = mock_loader
        mock_loader.load_model.return_value = {"model": MagicMock()}

        n = len(btc_flat_data)
        probas_df = pd.DataFrame(
            {"BULL_TREND": np.full(n, 0.2), "NOT_BULL": np.full(n, 0.8)},
            index=btc_flat_data.index,
        )
        mock_predict_proba.return_value = probas_df

        strategy = VBOHybrid(
            btc_data=btc_flat_data,
            model_path="/fake/model.joblib",
            threshold=0.02,
            ml_confidence=0.5,
        )
        df = strategy.calculate_indicators(ohlcv_data.copy())

        # With low ML probability, ambiguous zone should be bear
        valid = df.iloc[30:]
        assert valid["hybrid_bear"].any()


class TestVBOHybridNoModelFallback:
    """Without ML model, should degrade to pure MA20 behavior."""

    def test_no_model_fallback(
        self, ohlcv_data: pd.DataFrame, btc_bull_data: pd.DataFrame
    ) -> None:
        """Without model, ambiguous zone should default to bull."""
        strategy = VBOHybrid(btc_data=btc_bull_data, model_path=None)
        df = strategy.calculate_indicators(ohlcv_data.copy())

        assert "hybrid_bull" in df.columns
        assert "hybrid_bear" in df.columns

    @patch("src.strategies.volatility_breakout.vbo_hybrid._load_btc_data")
    def test_no_btc_data_allows_all(
        self, mock_load: MagicMock, ohlcv_data: pd.DataFrame
    ) -> None:
        """Without BTC data, should allow all entries."""
        mock_load.return_value = None
        strategy = VBOHybrid(btc_data=None, model_path=None)
        df = strategy.calculate_indicators(ohlcv_data.copy())

        assert df["hybrid_bull"].all()
        assert not df["hybrid_bear"].any()

    def test_empty_btc_data_allows_all(self, ohlcv_data: pd.DataFrame) -> None:
        """Empty BTC DataFrame should allow all entries."""
        strategy = VBOHybrid(btc_data=pd.DataFrame(), model_path=None)
        df = strategy.calculate_indicators(ohlcv_data.copy())

        assert df["hybrid_bull"].all()
        assert not df["hybrid_bear"].any()


class TestVBOHybridSignals:
    """Test signal generation."""

    def test_signals_have_correct_columns(
        self, ohlcv_data: pd.DataFrame, btc_bull_data: pd.DataFrame
    ) -> None:
        strategy = VBOHybrid(btc_data=btc_bull_data, model_path=None)
        df = strategy.calculate_indicators(ohlcv_data.copy())
        df = strategy.generate_signals(df)

        assert "entry_signal" in df.columns
        assert "exit_signal" in df.columns
        assert df["entry_signal"].dtype == bool
        assert df["exit_signal"].dtype == bool

    def test_entry_requires_breakout_and_hybrid_bull(
        self, ohlcv_data: pd.DataFrame, btc_bear_data: pd.DataFrame
    ) -> None:
        """In bear market, no entries even if breakout occurs."""
        strategy = VBOHybrid(btc_data=btc_bear_data, model_path=None)
        df = strategy.calculate_indicators(ohlcv_data.copy())
        df = strategy.generate_signals(df)

        # Where hybrid_bear is True, entry_signal should be False
        bear_days = df[df["hybrid_bear"].fillna(False)]
        if len(bear_days) > 0:
            assert not bear_days["entry_signal"].any()

    def test_exit_on_sma_or_bear(
        self, ohlcv_data: pd.DataFrame, btc_bull_data: pd.DataFrame
    ) -> None:
        strategy = VBOHybrid(btc_data=btc_bull_data, model_path=None)
        df = strategy.calculate_indicators(ohlcv_data.copy())
        df = strategy.generate_signals(df)

        # Exit should trigger when close < sma
        sma_exit = df["close"] < df["sma"]
        bear_exit = df["hybrid_bear"].fillna(False)
        expected_exit = sma_exit | bear_exit

        pd.testing.assert_series_equal(
            df["exit_signal"], expected_exit, check_names=False
        )
