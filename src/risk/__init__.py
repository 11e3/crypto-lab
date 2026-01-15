"""Risk management module for portfolio-level risk analysis."""

from src.risk.metrics import (
    PortfolioRiskMetrics,
    calculate_cvar,
    calculate_portfolio_correlation,
    calculate_portfolio_volatility,
    calculate_var,
)
from src.risk.portfolio_optimization import (
    PortfolioOptimizer,
    PortfolioWeights,
    optimize_portfolio,
)
from src.risk.position_sizing import (
    PositionSizingMethod,
    calculate_multi_asset_position_sizes,
    calculate_position_size,
)
from src.risk.stress_testing import (
    StressTester,
    StressScenario,
    StressTestResult,
    CrisisType,
    HISTORICAL_SCENARIOS,
)
from src.risk.drawdown_control import (
    DrawdownController,
    DrawdownState,
    RiskState,
    RiskBudgetManager,
    RiskBudget,
)

__all__ = [
    # Metrics
    "PortfolioRiskMetrics",
    "calculate_var",
    "calculate_cvar",
    "calculate_portfolio_volatility",
    "calculate_portfolio_correlation",
    # Position sizing
    "PositionSizingMethod",
    "calculate_position_size",
    "calculate_multi_asset_position_sizes",
    # Portfolio optimization
    "PortfolioOptimizer",
    "PortfolioWeights",
    "optimize_portfolio",
    # Stress testing
    "StressTester",
    "StressScenario",
    "StressTestResult",
    "CrisisType",
    "HISTORICAL_SCENARIOS",
    # Drawdown control
    "DrawdownController",
    "DrawdownState",
    "RiskState",
    "RiskBudgetManager",
    "RiskBudget",
]
