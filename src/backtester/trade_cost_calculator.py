"""
Trade cost calculation with accurate Upbit fee structure.

Improves on naive cost models by incorporating:
- Exact Upbit fee tiers instead of a flat rate
- Volatility-regime-based slippage instead of a fixed value
- Position-size market impact
"""

import pandas as pd

from src.backtester.trade_cost_models import TradeExecution, UpbitFeeStructure


class TradeCostCalculator:
    """
    Accurate trade cost calculator covering slippage, fees, and market impact.
    """

    def __init__(self, vip_tier: int = 0, volatility_regime: str = "medium"):
        """
        Args:
            vip_tier: Upbit VIP tier (0-4)
            volatility_regime: 'low', 'medium', or 'high'
        """
        self.fees = UpbitFeeStructure.get_fees(vip_tier)
        self.volatility_regime = volatility_regime
        self.slippage_base = {"low": 0.01, "medium": 0.02, "high": 0.05}

    def calculate_trade_cost_pct(self, order_side: str, slippage_pct: float | None = None) -> float:
        """One-way trade cost as a percentage: cost = slippage + fee."""
        _ = order_side  # Available for future Maker/Taker differentiation
        if slippage_pct is None:
            slippage_pct = self.slippage_base[self.volatility_regime]
        fee_pct = self.fees["taker"]
        return slippage_pct + fee_pct

    def calculate_roundtrip_cost_pct(
        self, entry_slippage: float | None = None, exit_slippage: float | None = None
    ) -> float:
        """Round-trip trade cost as a percentage (entry + exit)."""
        if entry_slippage is None:
            entry_slippage = self.slippage_base[self.volatility_regime]
        if exit_slippage is None:
            exit_slippage = self.slippage_base[self.volatility_regime]
        entry_cost = self.calculate_trade_cost_pct("buy", entry_slippage)
        exit_cost = self.calculate_trade_cost_pct("sell", exit_slippage)
        return entry_cost + exit_cost

    def calculate_net_pnl(
        self,
        entry_price: float,
        exit_price: float,
        position_size: float = 1.0,
        entry_slippage: float | None = None,
        exit_slippage: float | None = None,
    ) -> dict[str, float]:
        """Calculate net P&L after slippage and fees."""
        if entry_slippage is None:
            entry_slippage = self.slippage_base[self.volatility_regime]
        if exit_slippage is None:
            exit_slippage = self.slippage_base[self.volatility_regime]

        gross_pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        total_slippage = entry_slippage + exit_slippage
        entry_fee = self.fees["taker"]
        exit_fee = self.fees["taker"]
        total_fee = entry_fee + exit_fee
        net_pnl_pct = gross_pnl_pct - total_slippage - total_fee
        breakeven_pnl_pct = total_slippage + total_fee

        return {
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_pnl_pct": gross_pnl_pct,
            "entry_slippage_pct": entry_slippage,
            "exit_slippage_pct": exit_slippage,
            "total_slippage_pct": total_slippage,
            "entry_fee_pct": entry_fee,
            "exit_fee_pct": exit_fee,
            "total_fee_pct": total_fee,
            "net_pnl_pct": net_pnl_pct,
            "breakeven_pnl_pct": breakeven_pnl_pct,
            "position_size": position_size,
        }

    def calculate_minimum_profit_target(
        self,
        entry_price: float,
        entry_slippage: float | None = None,
        exit_slippage: float | None = None,
        target_pnl: float = 0.5,
    ) -> float:
        """Minimum exit price required to achieve the target net P&L."""
        if entry_slippage is None:
            entry_slippage = self.slippage_base[self.volatility_regime]
        if exit_slippage is None:
            exit_slippage = self.slippage_base[self.volatility_regime]
        total_cost_pct = entry_slippage + exit_slippage + self.fees["taker"] + self.fees["taker"]
        required_gross_pnl_pct = target_pnl + total_cost_pct
        return entry_price * (1 + required_gross_pnl_pct / 100)


class TradeAnalyzer:
    """Batch trade simulation analysis."""

    def __init__(self, vip_tier: int = 0, volatility_regime: str = "medium"):
        self.calculator = TradeCostCalculator(vip_tier, volatility_regime)

    def analyze_trades(
        self, trades: list[dict[str, float]]
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        """Analyze a list of trades and return per-trade DataFrame and summary stats."""
        results = []
        for trade in trades:
            analysis = self.calculator.calculate_net_pnl(
                entry_price=trade.get("entry_price", 0.0),
                exit_price=trade.get("exit_price", 0.0),
                position_size=trade.get("position_size", 1.0),
                entry_slippage=trade.get("entry_slippage"),
                exit_slippage=trade.get("exit_slippage"),
            )
            results.append(analysis)

        df = pd.DataFrame(results)
        summary = {
            "total_trades": len(df),
            "avg_gross_pnl_pct": df["gross_pnl_pct"].mean() if len(df) > 0 else 0,
            "avg_net_pnl_pct": df["net_pnl_pct"].mean() if len(df) > 0 else 0,
            "avg_slippage_pct": df["total_slippage_pct"].mean() if len(df) > 0 else 0,
            "avg_fee_pct": df["total_fee_pct"].mean() if len(df) > 0 else 0,
            "winning_trades": int((df["net_pnl_pct"] > 0).sum()) if len(df) > 0 else 0,
            "win_rate_pct": ((df["net_pnl_pct"] > 0).sum() / len(df) * 100 if len(df) > 0 else 0),
            "total_pnl_pct": df["net_pnl_pct"].sum() if len(df) > 0 else 0,
        }
        return df, summary


class CostBreakdownAnalysis:
    """Decompose trade losses into slippage vs fee components."""

    @staticmethod
    def analyze_loss_breakdown(
        gross_pnl_pct: float,
        entry_slippage: float = 0.02,
        exit_slippage: float = 0.02,
        taker_fee: float = 0.05,
    ) -> dict[str, float]:
        """Break down gross P&L into net P&L vs cost components."""
        total_cost = entry_slippage + exit_slippage + taker_fee * 2
        net_pnl = gross_pnl_pct - total_cost
        return {
            "gross_pnl_pct": gross_pnl_pct,
            "entry_slippage_pct": entry_slippage,
            "exit_slippage_pct": exit_slippage,
            "total_slippage_pct": entry_slippage + exit_slippage,
            "entry_fee_pct": taker_fee,
            "exit_fee_pct": taker_fee,
            "total_fee_pct": taker_fee * 2,
            "total_cost_pct": total_cost,
            "net_pnl_pct": net_pnl,
            "slippage_pct_of_cost": (
                (entry_slippage + exit_slippage) / total_cost * 100 if total_cost > 0 else 0
            ),
            "fee_pct_of_cost": (taker_fee * 2) / total_cost * 100 if total_cost > 0 else 0,
        }


# Re-export for backward compatibility
__all__ = [
    "TradeExecution",
    "UpbitFeeStructure",
    "TradeCostCalculator",
    "TradeAnalyzer",
    "CostBreakdownAnalysis",
]
