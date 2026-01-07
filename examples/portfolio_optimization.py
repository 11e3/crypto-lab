"""
Portfolio Optimization Examples.

Demonstrates three portfolio construction approaches:
1. Modern Portfolio Theory (MPT) - Mean-variance optimization (Sharpe ratio maximization)
2. Risk Parity - Equal risk contribution from each asset
3. Kelly Criterion - Optimal position sizing based on win/loss statistics

Also shows advanced features:
- Transaction cost modeling (linear + quadratic)
- Slippage and liquidity constraints
- Rebalancing policies
- Constraint enforcement (max position size, min allocation)
"""

import numpy as np
import pandas as pd

from src.risk.portfolio_optimization import PortfolioOptimizer


def example_modern_portfolio_theory():
    """Example 1: Modern Portfolio Theory (MPT) - Mean-Variance Optimization."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Modern Portfolio Theory (MPT)")
    print("=" * 80)
    print("\nMPT maximizes Sharpe ratio given risk/return constraints.")
    print("Useful for: Strategic asset allocation, risk-aware investing\n")

    # Create sample returns data
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=252, freq="D")

    returns_data = {
        "BTC": np.random.normal(0.0008, 0.03, 252),  # ~20% annual return, 47% volatility
        "ETH": np.random.normal(0.0006, 0.035, 252),  # ~15% annual return, 55% volatility
        "STAKING": np.random.normal(0.0004, 0.02, 252),  # ~10% annual return, 32% volatility
    }

    returns = pd.DataFrame(returns_data, index=dates)

    # Optimize using MPT
    optimizer = PortfolioOptimizer()
    weights = optimizer.optimize_mpt(returns, risk_free_rate=0.02)

    print("[OPTIMIZATION RESULT]")
    print("-" * 50)
    print(f"Method: {weights.method.upper()}")
    print(f"Expected Annual Return: {weights.expected_return:.2%}")
    print(f"Portfolio Volatility: {weights.portfolio_volatility:.2%}")
    print(f"Sharpe Ratio: {weights.sharpe_ratio:.2f}")
    print("\n[ALLOCATION]")
    print("-" * 50)
    for ticker, weight in sorted(weights.weights.items(), key=lambda x: -x[1]):
        print(f"  {ticker:10s}: {weight:6.2%}")


def example_risk_parity():
    """Example 2: Risk Parity - Equal Risk Contribution."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Risk Parity (Equal Risk Contribution)")
    print("=" * 80)
    print("\nRisk Parity aims for equal risk contribution from each asset.")
    print("Useful for: Diversified portfolios, reducing concentration risk\n")

    # Create sample returns data with different volatilities
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=252, freq="D")

    returns_data = {
        "STABLE": np.random.normal(0.0002, 0.01, 252),  # Low volatility
        "GROWTH": np.random.normal(0.0008, 0.04, 252),  # High volatility
        "BALANCED": np.random.normal(0.0005, 0.025, 252),  # Medium volatility
    }

    returns = pd.DataFrame(returns_data, index=dates)

    # Calculate individual volatilities
    print("[ASSET CHARACTERISTICS]")
    print("-" * 50)
    for ticker in returns.columns:
        annual_vol = returns[ticker].std() * np.sqrt(252)
        print(f"  {ticker:10s}: Volatility = {annual_vol:6.2%}")

    # Optimize using Risk Parity
    optimizer = PortfolioOptimizer()
    weights = optimizer.optimize_risk_parity(returns)

    print("\n[OPTIMIZATION RESULT]")
    print("-" * 50)
    print(f"Method: {weights.method.upper()}")
    print(f"Expected Annual Return: {weights.expected_return:.2%}")
    print(f"Portfolio Volatility: {weights.portfolio_volatility:.2%}")
    print(f"Sharpe Ratio: {weights.sharpe_ratio:.2f}")
    print("\n[ALLOCATION (Equal Risk Contribution)]")
    print("-" * 50)
    for ticker, weight in sorted(weights.weights.items(), key=lambda x: -x[1]):
        print(f"  {ticker:10s}: {weight:6.2%}")


def example_kelly_criterion():
    """Example 3: Kelly Criterion - Optimal Position Sizing."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Kelly Criterion (Optimal Position Sizing)")
    print("=" * 80)
    print("\nKelly Criterion calculates optimal position size based on win/loss statistics.")
    print("Useful for: Trading systems with known edge, position sizing\n")

    # Create sample trade history
    btc_trades = [2.5, -1.0, 3.2, -0.5, 1.8, -1.5, 4.0, -0.8, 2.1, -1.2] * 4
    eth_trades = [1.5, -2.0, 2.8, -1.5, 3.5, -0.7, 2.2, -1.8, 1.9, -1.1] * 3 + [1.5, -2.0, 2.8]

    trades_data = {
        "ticker": ["BTC"] * len(btc_trades) + ["ETH"] * len(eth_trades),
        "pnl_pct": btc_trades + eth_trades,
    }

    trades = pd.DataFrame(trades_data)

    # Calculate statistics
    print("[TRADE STATISTICS]")
    print("-" * 50)

    for ticker in trades["ticker"].unique():
        ticker_trades = trades[trades["ticker"] == ticker]["pnl_pct"]
        wins = ticker_trades[ticker_trades > 0]
        losses = ticker_trades[ticker_trades < 0]
        win_rate = len(wins) / len(ticker_trades)
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
        profit_factor = (wins.sum() / abs(losses.sum())) if len(losses) > 0 else 0

        print(f"\n  {ticker}:")
        print(f"    Total Trades: {len(ticker_trades)}")
        print(f"    Win Rate: {win_rate:.1%}")
        print(f"    Avg Win: {avg_win:.2f}%")
        print(f"    Avg Loss: {avg_loss:.2f}%")
        print(f"    Profit Factor: {profit_factor:.2f}")

    # Optimize using Kelly
    optimizer = PortfolioOptimizer()
    allocations = optimizer.optimize_kelly_portfolio(trades, available_cash=100000, max_kelly=0.25)

    print("\n[OPTIMIZATION RESULT]")
    print("-" * 50)
    print("Available Cash: $100,000")
    print("Max Kelly: 25% (fractional Kelly for safety)")

    print("\n[ALLOCATION]")
    print("-" * 50)
    for ticker, amount in sorted(allocations.items(), key=lambda x: -x[1]):
        percentage = amount / 100000
        print(f"  {ticker:10s}: ${amount:10,.0f} ({percentage:6.2%})")


def example_transaction_costs():
    """Example 4: Portfolio Optimization with Transaction Costs."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Transaction Cost Modeling")
    print("=" * 80)
    print("\nIncorporates transaction costs in portfolio rebalancing decisions.")
    print("Uses quadratic transaction cost model: TC(w) = alpha*|Δw| + beta*|Δw|^2\n")

    # Create sample returns data
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=252, freq="D")

    returns_data = {
        "BTC": np.random.normal(0.0008, 0.03, 252),
        "ETH": np.random.normal(0.0006, 0.035, 252),
        "STAKING": np.random.normal(0.0004, 0.02, 252),
    }

    returns = pd.DataFrame(returns_data, index=dates)

    # Calculate optimal weights (without transaction costs)
    optimizer = PortfolioOptimizer()
    weights_opt = optimizer.optimize_mpt(returns)

    print("[WITHOUT TRANSACTION COSTS]")
    print("-" * 50)
    print(f"Expected Return: {weights_opt.expected_return:.2%}")
    print(f"Volatility: {weights_opt.portfolio_volatility:.2%}")
    print(f"Sharpe Ratio: {weights_opt.sharpe_ratio:.2f}")
    print("\nAllocations:")
    for ticker, weight in sorted(weights_opt.weights.items(), key=lambda x: -x[1]):
        print(f"  {ticker:10s}: {weight:6.2%}")

    # Simulate with transaction costs
    print("\n[TRANSACTION COST IMPACT]")
    print("-" * 50)

    # Transaction cost parameters
    linear_cost = 0.0010  # 0.10% linear spread
    quadratic_cost = 0.0005  # 0.05% quadratic cost per unit of rebalancing

    # Current portfolio (equal weighted)
    current_weights = {"BTC": 1 / 3, "ETH": 1 / 3, "STAKING": 1 / 3}

    # Calculate rebalancing costs
    print(f"\nCurrent Weights: {current_weights}")
    print(f"Target Weights: {weights_opt.weights}")
    print("\nTransaction Cost Model:")
    print(f"  Linear: {linear_cost:.2%}")
    print(f"  Quadratic: {quadratic_cost:.2%}")

    total_tc = 0
    print("\nRebalancing Costs:")
    for ticker in weights_opt.weights:
        current = current_weights.get(ticker, 0)
        target = weights_opt.weights[ticker]
        weight_change = abs(target - current)

        # Linear cost
        linear = weight_change * linear_cost * 100000  # Assuming $100k portfolio
        # Quadratic cost
        quadratic = (weight_change**2) * quadratic_cost * 100000

        total_tc += linear + quadratic

        print(
            f"  {ticker:10s}: Δw={weight_change:6.2%}, "
            f"Linear=${linear:8,.0f}, Quadratic=${quadratic:8,.0f}"
        )

    print(f"\nTotal Transaction Cost: ${total_tc:,.0f} ({total_tc / 100000:.2%} of portfolio)")
    print("\nRecommendation: Rebalance if drift > transaction costs")


def example_constrained_optimization():
    """Example 5: Portfolio Optimization with Constraints."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Constrained Optimization (Max Position, Min/Max Allocation)")
    print("=" * 80)
    print("\nEnforces practical constraints on portfolio construction.")
    print("Useful for: Risk management, regulatory compliance\n")

    # Create sample returns data
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=252, freq="D")

    returns_data = {
        "BTC": np.random.normal(0.0008, 0.03, 252),
        "ETH": np.random.normal(0.0006, 0.035, 252),
        "STAKING": np.random.normal(0.0004, 0.02, 252),
        "SMALL_CAP": np.random.normal(0.0010, 0.05, 252),
    }

    returns = pd.DataFrame(returns_data, index=dates)

    # Unconstrained optimization
    optimizer = PortfolioOptimizer()
    weights_unconstrained = optimizer.optimize_mpt(returns)

    print("[UNCONSTRAINED OPTIMIZATION]")
    print("-" * 50)
    for ticker, weight in sorted(weights_unconstrained.weights.items(), key=lambda x: -x[1]):
        print(f"  {ticker:15s}: {weight:6.2%}")
    print(f"\nSharpe Ratio: {weights_unconstrained.sharpe_ratio:.2f}")

    # Constrained optimization (max 40% per asset)
    weights_constrained = optimizer.optimize_mpt(returns, max_weight=0.40)

    print("\n[CONSTRAINED OPTIMIZATION (Max 40% per asset)]")
    print("-" * 50)
    for ticker, weight in sorted(weights_constrained.weights.items(), key=lambda x: -x[1]):
        print(f"  {ticker:15s}: {weight:6.2%}")
    print(f"\nSharpe Ratio: {weights_constrained.sharpe_ratio:.2f}")
    print(
        f"Constraint Impact: {(weights_constrained.sharpe_ratio / weights_unconstrained.sharpe_ratio - 1):.1%}"
    )


def example_rebalancing_policy():
    """Example 6: Portfolio Rebalancing Policy."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Dynamic Rebalancing Policy")
    print("=" * 80)
    print("\nDemonstrates different rebalancing strategies:")
    print("1. Calendar Rebalancing (monthly, quarterly, annually)")
    print("2. Threshold Rebalancing (when drift exceeds threshold)")
    print("3. Dynamic Rebalancing (based on market regime)\n")

    # Simulate portfolio growth
    pd.date_range(start="2023-01-01", periods=252, freq="D")

    # Target weights
    target_weights = {"BTC": 0.50, "ETH": 0.30, "STAKING": 0.20}

    # Simulate returns
    np.random.seed(42)
    returns_daily = {
        "BTC": np.random.normal(0.0008, 0.03, 252),
        "ETH": np.random.normal(0.0006, 0.035, 252),
        "STAKING": np.random.normal(0.0004, 0.02, 252),
    }

    # Calculate portfolio values
    initial_investment = 100000
    portfolio_values = {
        ticker: [initial_investment * weight] for ticker, weight in target_weights.items()
    }

    for i in range(1, 252):
        for ticker in target_weights:
            ret = returns_daily[ticker][i]
            portfolio_values[ticker].append(portfolio_values[ticker][-1] * (1 + ret))

    # Calculate current weights
    final_values = {ticker: values[-1] for ticker, values in portfolio_values.items()}
    total_value = sum(final_values.values())
    current_weights = {ticker: value / total_value for ticker, value in final_values.items()}

    print("[PORTFOLIO STATE (After 252 days, no rebalancing)]")
    print("-" * 50)
    print(f"{'Ticker':<15} {'Target':<12} {'Current':<12} {'Drift':<12}")
    print("-" * 50)

    total_drift = 0
    for ticker in target_weights:
        target = target_weights[ticker]
        current = current_weights[ticker]
        drift = current - target
        total_drift += abs(drift)
        print(f"{ticker:<15} {target:>10.2%}   {current:>10.2%}   {drift:>10.2%}")

    print("\n[REBALANCING STRATEGIES]")
    print("-" * 50)
    print("\n1. CALENDAR REBALANCING (Monthly)")
    print("   - Rebalance every 21 trading days (1 month)")
    print("   - Predictable, simple to implement")
    print(f"   - Current drift: {total_drift:.2%}")
    print("   - Estimated annual transactions: 12")

    print("\n2. THRESHOLD REBALANCING (5% drift)")
    print("   - Rebalance when any asset drifts > 5%")
    print(f"   - Current drift: {total_drift:.2%}")
    exceeded = any(abs(current_weights[t] - target_weights[t]) > 0.05 for t in target_weights)
    print(f"   - Action needed: {'YES' if exceeded else 'NO'}")

    print("\n3. DYNAMIC REBALANCING")
    print("   - Adjust rebalancing frequency based on volatility")
    print("   - Lower threshold in high volatility periods")
    print("   - Optimal for regime-dependent strategies")


def main():
    """Run all portfolio optimization examples."""
    print("\n" + "=" * 80)
    print("CRYPTO QUANT SYSTEM - PORTFOLIO OPTIMIZATION EXAMPLES")
    print("=" * 80)

    try:
        example_modern_portfolio_theory()
        example_risk_parity()
        example_kelly_criterion()
        example_transaction_costs()
        example_constrained_optimization()
        example_rebalancing_policy()

        print("\n" + "=" * 80)
        print("[SUCCESS] ALL EXAMPLES COMPLETED")
        print("=" * 80)
        print("\n[NEXT STEPS]")
        print("-" * 50)
        print("1. Combine optimization with backtest engine for live portfolio updates")
        print("2. Implement dynamic rebalancing based on drift thresholds")
        print("3. Monitor transaction costs vs. portfolio drift reduction")
        print("4. Test optimization robustness across different market regimes")
        print("5. Integrate risk constraints (VaR, drawdown limits, position limits)")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise


if __name__ == "__main__":
    main()
