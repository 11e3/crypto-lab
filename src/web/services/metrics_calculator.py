"""Extended metrics calculator service.

Sortino, Calmar, VaR, CVaR, 상방/하방 변동성, z-score, p-value 등
고급 백테스팅 메트릭 계산.

Uses focused metric calculators for each domain (SRP).
"""

from dataclasses import dataclass

import numpy as np

from src.web.services.metrics import (
    RatioMetrics,
    ReturnMetrics,
    RiskMetrics,
    StatisticalMetrics,
    TradeMetrics,
)

__all__ = [
    "ExtendedMetrics",
    "calculate_extended_metrics",
    # Re-export metric calculators
    "ReturnMetrics",
    "RiskMetrics",
    "RatioMetrics",
    "StatisticalMetrics",
    "TradeMetrics",
]


@dataclass(frozen=True)
class ExtendedMetrics:
    """확장 백테스팅 메트릭."""

    # 기본 수익률 메트릭
    total_return_pct: float
    cagr_pct: float

    # 리스크 메트릭
    max_drawdown_pct: float
    volatility_pct: float
    upside_volatility_pct: float
    downside_volatility_pct: float

    # 리스크 조정 수익률
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # VaR & CVaR
    var_95_pct: float
    var_99_pct: float
    cvar_95_pct: float
    cvar_99_pct: float

    # 통계적 검정
    z_score: float
    p_value: float
    skewness: float
    kurtosis: float

    # 거래 메트릭
    num_trades: int
    win_rate_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    expectancy: float

    # 기간 정보
    trading_days: int
    years: float


def calculate_extended_metrics(
    equity: np.ndarray,
    trade_returns: list[float] | None = None,
    risk_free_rate: float = 0.02,
) -> ExtendedMetrics:
    """확장 메트릭 계산.

    Args:
        equity: 포트폴리오 가치 배열
        trade_returns: 개별 거래 수익률 리스트 (선택)
        risk_free_rate: 무위험 수익률 (연간, 기본: 2%)

    Returns:
        ExtendedMetrics 데이터클래스
    """
    if len(equity) < 2:
        return _empty_metrics()

    # 기간 정보
    trading_days = len(equity)
    years = trading_days / 365

    # 일간 수익률
    returns = ReturnMetrics.calculate_returns(equity)

    # 수익률 메트릭
    initial_value = float(equity[0])
    final_value = float(equity[-1])
    total_return = ReturnMetrics.calculate_total_return(initial_value, final_value)
    cagr = ReturnMetrics.calculate_cagr(initial_value, final_value, years)
    max_dd = ReturnMetrics.calculate_max_drawdown(equity)

    # 리스크 메트릭
    volatility = RiskMetrics.calculate_volatility(returns)
    upside_vol = RiskMetrics.calculate_upside_volatility(returns)
    downside_vol = RiskMetrics.calculate_downside_volatility(returns)
    var_95 = RiskMetrics.calculate_var(returns, 0.95)
    var_99 = RiskMetrics.calculate_var(returns, 0.99)
    cvar_95 = RiskMetrics.calculate_cvar(returns, 0.95)
    cvar_99 = RiskMetrics.calculate_cvar(returns, 0.99)

    # 리스크 조정 수익률
    sharpe = RatioMetrics.calculate_sharpe_ratio(returns, risk_free_rate)
    sortino = RatioMetrics.calculate_sortino_ratio(returns, risk_free_rate)
    calmar = RatioMetrics.calculate_calmar_ratio(cagr, max_dd)

    # 통계적 메트릭
    z_score, p_value = StatisticalMetrics.calculate_z_score_and_pvalue(returns)
    skewness = StatisticalMetrics.calculate_skewness(returns)
    kurtosis = StatisticalMetrics.calculate_kurtosis(returns)

    # 거래 메트릭
    trade_returns = trade_returns or []
    num_trades = len(trade_returns)
    win_rate, avg_win, avg_loss, profit_factor, expectancy = TradeMetrics.calculate(
        trade_returns
    )

    return ExtendedMetrics(
        total_return_pct=total_return,
        cagr_pct=cagr,
        max_drawdown_pct=max_dd,
        volatility_pct=volatility,
        upside_volatility_pct=upside_vol,
        downside_volatility_pct=downside_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        var_95_pct=var_95,
        var_99_pct=var_99,
        cvar_95_pct=cvar_95,
        cvar_99_pct=cvar_99,
        z_score=z_score,
        p_value=p_value,
        skewness=skewness,
        kurtosis=kurtosis,
        num_trades=num_trades,
        win_rate_pct=win_rate,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        profit_factor=profit_factor,
        expectancy=expectancy,
        trading_days=trading_days,
        years=years,
    )


def _empty_metrics() -> ExtendedMetrics:
    """Return empty metrics for insufficient data."""
    return ExtendedMetrics(
        total_return_pct=0.0,
        cagr_pct=0.0,
        max_drawdown_pct=0.0,
        volatility_pct=0.0,
        upside_volatility_pct=0.0,
        downside_volatility_pct=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        calmar_ratio=0.0,
        var_95_pct=0.0,
        var_99_pct=0.0,
        cvar_95_pct=0.0,
        cvar_99_pct=0.0,
        z_score=0.0,
        p_value=1.0,
        skewness=0.0,
        kurtosis=0.0,
        num_trades=0,
        win_rate_pct=0.0,
        avg_win_pct=0.0,
        avg_loss_pct=0.0,
        profit_factor=0.0,
        expectancy=0.0,
        trading_days=0,
        years=0.0,
    )
