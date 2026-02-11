"""Tests for common indicator-based conditions."""

from datetime import date

import pandas as pd
import pytest

from src.strategies.base_models import OHLCV
from src.strategies.common_conditions import (
    PriceAboveSMACondition,
    PriceBelowSMACondition,
    RSIOverboughtCondition,
    RSIOversoldCondition,
)


@pytest.fixture()
def sample_ohlcv() -> OHLCV:
    """Create sample OHLCV bar."""
    return OHLCV(
        date=date(2024, 1, 1),
        open=100.0,
        high=110.0,
        low=90.0,
        close=105.0,
        volume=1000.0,
    )


@pytest.fixture()
def empty_history() -> pd.DataFrame:
    """Empty DataFrame for history."""
    return pd.DataFrame()


# =========================================================================
# PriceAboveSMACondition
# =========================================================================


class TestPriceAboveSMACondition:
    """Tests for PriceAboveSMACondition."""

    def test_returns_true_when_close_above_sma(
        self, sample_ohlcv: OHLCV, empty_history: pd.DataFrame
    ) -> None:
        cond = PriceAboveSMACondition()
        assert cond.evaluate(sample_ohlcv, empty_history, {"sma": 100.0}) is True

    def test_returns_false_when_close_below_sma(
        self, sample_ohlcv: OHLCV, empty_history: pd.DataFrame
    ) -> None:
        cond = PriceAboveSMACondition()
        assert cond.evaluate(sample_ohlcv, empty_history, {"sma": 110.0}) is False

    def test_returns_false_when_sma_missing(
        self, sample_ohlcv: OHLCV, empty_history: pd.DataFrame
    ) -> None:
        cond = PriceAboveSMACondition()
        assert cond.evaluate(sample_ohlcv, empty_history, {}) is False

    def test_custom_sma_key(self, sample_ohlcv: OHLCV, empty_history: pd.DataFrame) -> None:
        cond = PriceAboveSMACondition(sma_key="sma_20")
        assert cond.evaluate(sample_ohlcv, empty_history, {"sma_20": 100.0}) is True
        assert cond.evaluate(sample_ohlcv, empty_history, {"sma": 100.0}) is False

    def test_name_default(self) -> None:
        cond = PriceAboveSMACondition()
        assert cond.name == "PriceAboveSMA"

    def test_name_custom(self) -> None:
        cond = PriceAboveSMACondition(name="CustomName")
        assert cond.name == "CustomName"


# =========================================================================
# PriceBelowSMACondition
# =========================================================================


class TestPriceBelowSMACondition:
    """Tests for PriceBelowSMACondition."""

    def test_returns_true_when_close_below_sma(
        self, sample_ohlcv: OHLCV, empty_history: pd.DataFrame
    ) -> None:
        cond = PriceBelowSMACondition()
        assert cond.evaluate(sample_ohlcv, empty_history, {"sma": 110.0}) is True

    def test_returns_false_when_close_above_sma(
        self, sample_ohlcv: OHLCV, empty_history: pd.DataFrame
    ) -> None:
        cond = PriceBelowSMACondition()
        assert cond.evaluate(sample_ohlcv, empty_history, {"sma": 100.0}) is False

    def test_returns_false_when_sma_missing(
        self, sample_ohlcv: OHLCV, empty_history: pd.DataFrame
    ) -> None:
        cond = PriceBelowSMACondition()
        assert cond.evaluate(sample_ohlcv, empty_history, {}) is False

    def test_custom_sma_key(self, sample_ohlcv: OHLCV, empty_history: pd.DataFrame) -> None:
        cond = PriceBelowSMACondition(sma_key="sma_50")
        assert cond.evaluate(sample_ohlcv, empty_history, {"sma_50": 110.0}) is True

    def test_name_default(self) -> None:
        cond = PriceBelowSMACondition()
        assert cond.name == "PriceBelowSMA"


# =========================================================================
# RSIOversoldCondition
# =========================================================================


class TestRSIOversoldCondition:
    """Tests for RSIOversoldCondition."""

    def test_returns_true_when_rsi_below_threshold(
        self, sample_ohlcv: OHLCV, empty_history: pd.DataFrame
    ) -> None:
        cond = RSIOversoldCondition()
        assert cond.evaluate(sample_ohlcv, empty_history, {"rsi": 25.0}) is True

    def test_returns_false_when_rsi_above_threshold(
        self, sample_ohlcv: OHLCV, empty_history: pd.DataFrame
    ) -> None:
        cond = RSIOversoldCondition()
        assert cond.evaluate(sample_ohlcv, empty_history, {"rsi": 50.0}) is False

    def test_returns_false_when_rsi_missing(
        self, sample_ohlcv: OHLCV, empty_history: pd.DataFrame
    ) -> None:
        cond = RSIOversoldCondition()
        assert cond.evaluate(sample_ohlcv, empty_history, {}) is False

    def test_custom_threshold(self, sample_ohlcv: OHLCV, empty_history: pd.DataFrame) -> None:
        cond = RSIOversoldCondition(oversold_threshold=40.0)
        assert cond.evaluate(sample_ohlcv, empty_history, {"rsi": 35.0}) is True
        assert cond.evaluate(sample_ohlcv, empty_history, {"rsi": 45.0}) is False

    def test_custom_rsi_key(self, sample_ohlcv: OHLCV, empty_history: pd.DataFrame) -> None:
        cond = RSIOversoldCondition(rsi_key="rsi_14")
        assert cond.evaluate(sample_ohlcv, empty_history, {"rsi_14": 25.0}) is True
        assert cond.evaluate(sample_ohlcv, empty_history, {"rsi": 25.0}) is False

    def test_name_default(self) -> None:
        cond = RSIOversoldCondition()
        assert cond.name == "RSIOversold"


# =========================================================================
# RSIOverboughtCondition
# =========================================================================


class TestRSIOverboughtCondition:
    """Tests for RSIOverboughtCondition."""

    def test_returns_true_when_rsi_above_threshold(
        self, sample_ohlcv: OHLCV, empty_history: pd.DataFrame
    ) -> None:
        cond = RSIOverboughtCondition()
        assert cond.evaluate(sample_ohlcv, empty_history, {"rsi": 80.0}) is True

    def test_returns_false_when_rsi_below_threshold(
        self, sample_ohlcv: OHLCV, empty_history: pd.DataFrame
    ) -> None:
        cond = RSIOverboughtCondition()
        assert cond.evaluate(sample_ohlcv, empty_history, {"rsi": 50.0}) is False

    def test_returns_false_when_rsi_missing(
        self, sample_ohlcv: OHLCV, empty_history: pd.DataFrame
    ) -> None:
        cond = RSIOverboughtCondition()
        assert cond.evaluate(sample_ohlcv, empty_history, {}) is False

    def test_custom_threshold(self, sample_ohlcv: OHLCV, empty_history: pd.DataFrame) -> None:
        cond = RSIOverboughtCondition(overbought_threshold=60.0)
        assert cond.evaluate(sample_ohlcv, empty_history, {"rsi": 65.0}) is True
        assert cond.evaluate(sample_ohlcv, empty_history, {"rsi": 55.0}) is False

    def test_custom_rsi_key(self, sample_ohlcv: OHLCV, empty_history: pd.DataFrame) -> None:
        cond = RSIOverboughtCondition(rsi_key="rsi_7")
        assert cond.evaluate(sample_ohlcv, empty_history, {"rsi_7": 80.0}) is True
        assert cond.evaluate(sample_ohlcv, empty_history, {"rsi": 80.0}) is False

    def test_name_default(self) -> None:
        cond = RSIOverboughtCondition()
        assert cond.name == "RSIOverbought"

    def test_boundary_value_not_triggered(
        self, sample_ohlcv: OHLCV, empty_history: pd.DataFrame
    ) -> None:
        """RSI exactly at threshold should NOT trigger."""
        cond = RSIOverboughtCondition(overbought_threshold=70.0)
        assert cond.evaluate(sample_ohlcv, empty_history, {"rsi": 70.0}) is False
