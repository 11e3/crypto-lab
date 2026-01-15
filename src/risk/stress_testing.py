"""
Stress testing module.

Provides:
- Historical stress tests (replay past crises)
- Hypothetical stress tests (custom scenarios)
- Monte Carlo stress simulation
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CrisisType(str, Enum):
    """Historical crisis types."""

    FINANCIAL_2008 = "financial_2008"
    COVID_2020 = "covid_2020"
    CRYPTO_2018 = "crypto_2018"
    CRYPTO_2022 = "crypto_2022"
    DOT_COM_2000 = "dot_com_2000"
    BLACK_MONDAY_1987 = "black_monday_1987"


@dataclass
class StressScenario:
    """Definition of a stress scenario."""

    name: str
    description: str

    # Shock parameters
    equity_shock: float = 0.0  # % change in equities
    crypto_shock: float = 0.0  # % change in crypto
    bond_shock: float = 0.0  # % change in bonds
    volatility_shock: float = 0.0  # Multiplier for volatility

    # Correlation shock
    correlation_spike: float = 0.0  # Increase in correlations (0-1)

    # Duration
    duration_days: int = 30  # Scenario duration

    # Market conditions
    liquidity_reduction: float = 0.0  # % reduction in liquidity


@dataclass
class StressTestResult:
    """Result of stress test."""

    scenario_name: str
    portfolio_loss: float  # Total portfolio loss
    var_breach: bool  # Whether VaR was breached
    margin_call: bool  # Whether margin call would occur

    # Asset-level impacts
    asset_losses: dict[str, float] = field(default_factory=dict)

    # Risk metrics under stress
    stressed_var: float = 0.0
    stressed_volatility: float = 0.0

    # Recovery estimate
    estimated_recovery_days: int | None = None

    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"Stress Test: {self.scenario_name}\n"
            f"{'=' * 40}\n"
            f"Portfolio Loss: {self.portfolio_loss:.2%}\n"
            f"VaR Breach: {'Yes' if self.var_breach else 'No'}\n"
            f"Margin Call: {'Yes' if self.margin_call else 'No'}\n"
            f"Stressed VaR: {self.stressed_var:.2%}\n"
            f"Stressed Volatility: {self.stressed_volatility:.2%}\n"
        )


# Predefined historical scenarios
HISTORICAL_SCENARIOS = {
    CrisisType.FINANCIAL_2008: StressScenario(
        name="2008 Financial Crisis",
        description="Global financial crisis triggered by subprime mortgage collapse",
        equity_shock=-0.50,
        crypto_shock=0.0,  # Crypto didn't exist
        bond_shock=0.10,  # Flight to quality
        volatility_shock=3.0,
        correlation_spike=0.30,
        duration_days=365,
        liquidity_reduction=0.50,
    ),
    CrisisType.COVID_2020: StressScenario(
        name="COVID-19 Crash (March 2020)",
        description="Market crash due to COVID-19 pandemic",
        equity_shock=-0.35,
        crypto_shock=-0.50,
        bond_shock=0.05,
        volatility_shock=4.0,
        correlation_spike=0.40,
        duration_days=30,
        liquidity_reduction=0.40,
    ),
    CrisisType.CRYPTO_2018: StressScenario(
        name="2018 Crypto Winter",
        description="Prolonged crypto bear market",
        equity_shock=-0.10,
        crypto_shock=-0.85,
        bond_shock=0.02,
        volatility_shock=2.0,
        correlation_spike=0.20,
        duration_days=365,
        liquidity_reduction=0.60,
    ),
    CrisisType.CRYPTO_2022: StressScenario(
        name="2022 Crypto Crash (Luna/FTX)",
        description="Crypto market crash due to Luna and FTX collapses",
        equity_shock=-0.20,
        crypto_shock=-0.70,
        bond_shock=-0.15,
        volatility_shock=2.5,
        correlation_spike=0.35,
        duration_days=180,
        liquidity_reduction=0.70,
    ),
}


class StressTester:
    """
    Portfolio stress testing engine.

    Evaluates portfolio resilience under various stress scenarios.

    Example:
        >>> tester = StressTester()
        >>> result = tester.historical_stress_test(
        ...     portfolio_weights={"BTC": 0.5, "ETH": 0.3, "SPY": 0.2},
        ...     crisis=CrisisType.COVID_2020,
        ... )
        >>> print(result.summary())
    """

    def __init__(
        self,
        var_confidence: float = 0.95,
        margin_threshold: float = 0.30,  # 30% loss triggers margin call
    ) -> None:
        """
        Initialize stress tester.

        Args:
            var_confidence: VaR confidence level
            margin_threshold: Loss threshold for margin call
        """
        self.var_confidence = var_confidence
        self.margin_threshold = margin_threshold

    def historical_stress_test(
        self,
        portfolio_weights: dict[str, float],
        crisis: CrisisType,
        asset_classes: dict[str, Literal["equity", "crypto", "bond"]] | None = None,
        current_var: float | None = None,
    ) -> StressTestResult:
        """
        Run historical stress test.

        Args:
            portfolio_weights: Current portfolio weights
            crisis: Historical crisis to simulate
            asset_classes: Asset class mapping for each holding
            current_var: Current portfolio VaR

        Returns:
            StressTestResult
        """
        scenario = HISTORICAL_SCENARIOS.get(crisis)
        if not scenario:
            raise ValueError(f"Unknown crisis type: {crisis}")

        return self.scenario_stress_test(
            portfolio_weights=portfolio_weights,
            scenario=scenario,
            asset_classes=asset_classes,
            current_var=current_var,
        )

    def scenario_stress_test(
        self,
        portfolio_weights: dict[str, float],
        scenario: StressScenario,
        asset_classes: dict[str, Literal["equity", "crypto", "bond"]] | None = None,
        current_var: float | None = None,
    ) -> StressTestResult:
        """
        Run stress test with custom scenario.

        Args:
            portfolio_weights: Current portfolio weights
            scenario: Stress scenario definition
            asset_classes: Asset class mapping
            current_var: Current portfolio VaR

        Returns:
            StressTestResult
        """
        # Default asset class assumptions
        if asset_classes is None:
            asset_classes = self._infer_asset_classes(portfolio_weights)

        # Calculate asset-level losses
        asset_losses = {}
        for asset, weight in portfolio_weights.items():
            asset_class = asset_classes.get(asset, "equity")

            if asset_class == "equity":
                loss = scenario.equity_shock
            elif asset_class == "crypto":
                loss = scenario.crypto_shock
            elif asset_class == "bond":
                loss = scenario.bond_shock
            else:
                loss = scenario.equity_shock

            asset_losses[asset] = loss

        # Portfolio loss
        portfolio_loss = sum(
            weight * asset_losses[asset]
            for asset, weight in portfolio_weights.items()
        )

        # VaR breach check
        var_breach = False
        if current_var:
            var_breach = abs(portfolio_loss) > abs(current_var)

        # Margin call check
        margin_call = abs(portfolio_loss) > self.margin_threshold

        # Stressed VaR (rough estimate)
        stressed_var = (current_var or 0) * (1 + scenario.volatility_shock)

        # Stressed volatility
        base_vol = 0.20  # Assume 20% base volatility
        stressed_vol = base_vol * scenario.volatility_shock

        # Recovery estimate
        recovery_days = scenario.duration_days * 2

        return StressTestResult(
            scenario_name=scenario.name,
            portfolio_loss=portfolio_loss,
            var_breach=var_breach,
            margin_call=margin_call,
            asset_losses=asset_losses,
            stressed_var=stressed_var,
            stressed_volatility=stressed_vol,
            estimated_recovery_days=recovery_days,
        )

    def monte_carlo_stress(
        self,
        returns: pd.DataFrame,
        portfolio_weights: dict[str, float],
        n_simulations: int = 10000,
        stress_factor: float = 2.0,
        horizon_days: int = 30,
    ) -> dict[str, float]:
        """
        Monte Carlo stress simulation.

        Simulates portfolio returns under stressed conditions.

        Args:
            returns: Historical returns DataFrame
            portfolio_weights: Portfolio weights
            n_simulations: Number of simulations
            stress_factor: Volatility multiplier
            horizon_days: Simulation horizon

        Returns:
            Dict with stress metrics
        """
        # Align weights with returns
        assets = [a for a in portfolio_weights if a in returns.columns]
        weights = np.array([portfolio_weights[a] for a in assets])
        returns_aligned = returns[assets]

        # Calculate stressed parameters
        mean_returns = returns_aligned.mean()
        cov_matrix = returns_aligned.cov() * stress_factor ** 2

        # Simulate
        simulated_returns = np.random.multivariate_normal(
            mean_returns.values,
            cov_matrix.values,
            size=(n_simulations, horizon_days),
        )

        # Portfolio returns
        portfolio_returns = simulated_returns @ weights

        # Cumulative returns over horizon
        cumulative = np.prod(1 + portfolio_returns, axis=1) - 1

        # Stress metrics
        return {
            "expected_loss": np.mean(cumulative),
            "worst_case_95": np.percentile(cumulative, 5),
            "worst_case_99": np.percentile(cumulative, 1),
            "probability_of_loss_10pct": np.mean(cumulative < -0.10),
            "probability_of_loss_20pct": np.mean(cumulative < -0.20),
            "max_simulated_loss": np.min(cumulative),
        }

    def run_all_historical_tests(
        self,
        portfolio_weights: dict[str, float],
        asset_classes: dict[str, Literal["equity", "crypto", "bond"]] | None = None,
    ) -> dict[CrisisType, StressTestResult]:
        """
        Run all historical stress tests.

        Args:
            portfolio_weights: Portfolio weights
            asset_classes: Asset class mapping

        Returns:
            Dict of crisis -> result
        """
        results = {}
        for crisis in CrisisType:
            try:
                results[crisis] = self.historical_stress_test(
                    portfolio_weights=portfolio_weights,
                    crisis=crisis,
                    asset_classes=asset_classes,
                )
            except Exception as e:
                logger.warning(f"Failed stress test for {crisis}: {e}")

        return results

    def _infer_asset_classes(
        self,
        portfolio_weights: dict[str, float],
    ) -> dict[str, Literal["equity", "crypto", "bond"]]:
        """Infer asset classes from symbol names."""
        result = {}

        crypto_keywords = ["BTC", "ETH", "SOL", "ADA", "DOT", "USDT", "USDC", "KRW-"]
        bond_keywords = ["TLT", "BND", "AGG", "IEF", "SHY"]

        for symbol in portfolio_weights:
            upper = symbol.upper()

            if any(kw in upper for kw in crypto_keywords):
                result[symbol] = "crypto"
            elif any(kw in upper for kw in bond_keywords):
                result[symbol] = "bond"
            else:
                result[symbol] = "equity"

        return result


__all__ = [
    "StressTester",
    "StressScenario",
    "StressTestResult",
    "CrisisType",
    "HISTORICAL_SCENARIOS",
]
