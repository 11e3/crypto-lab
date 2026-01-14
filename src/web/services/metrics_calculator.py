"""Extended metrics calculator service.

Sortino, Calmar, VaR, CVaR, 상방/하방 변동성, z-score, p-value 등
고급 백테스팅 메트릭 계산.

Uses core metrics module for unified calculations.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats

from src.metrics.core import (
    ANNUALIZATION_FACTOR,
    calculate_calmar_ratio,
    calculate_cvar,
    calculate_downside_volatility,
    calculate_max_drawdown,
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_total_return,
    calculate_upside_volatility,
    calculate_var,
    calculate_volatility,
)

__all__ = ["ExtendedMetrics", "calculate_extended_metrics"]


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


def _calculate_cagr(
    initial_value: float,
    final_value: float,
    years: float,
) -> float:
    """CAGR 계산."""
    if years <= 0 or initial_value <= 0:
        return 0.0
    return ((final_value / initial_value) ** (1 / years) - 1) * 100


def _calculate_statistical_tests(returns: np.ndarray) -> tuple[float, float]:
    """Z-score 및 P-value 계산.

    Returns:
        (z_score, p_value) 튜플
    """
    if len(returns) < 2:
        return 0.0, 1.0

    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)

    if std_return == 0:
        return 0.0, 1.0

    # H0: mean return = 0
    z_score = mean_return / (std_return / np.sqrt(len(returns)))

    # 양측 검정
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    return float(z_score), float(p_value)


def _calculate_trade_metrics(
    trade_returns: list[float],
) -> tuple[float, float, float, float, float]:
    """거래 메트릭 계산.

    Returns:
        (win_rate, avg_win, avg_loss, profit_factor, expectancy)
    """
    if not trade_returns:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    wins = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r < 0]

    win_rate = len(wins) / len(trade_returns) * 100
    avg_win = sum(wins) / len(wins) * 100 if wins else 0.0
    avg_loss = sum(losses) / len(losses) * 100 if losses else 0.0

    total_wins = sum(wins) if wins else 0
    total_losses = abs(sum(losses)) if losses else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

    expectancy = sum(trade_returns) / len(trade_returns) * 100

    return win_rate, avg_win, avg_loss, profit_factor, expectancy


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

    # 일간 수익률 (using core)
    returns = calculate_returns(equity)

    # 기간 정보
    trading_days = len(equity)
    years = trading_days / ANNUALIZATION_FACTOR

    # 총 수익률 (using core)
    initial_value = float(equity[0])
    final_value = float(equity[-1])
    total_return = calculate_total_return(initial_value, final_value)

    # CAGR
    cagr = _calculate_cagr(initial_value, final_value, years)

    # MDD (using core)
    max_dd = calculate_max_drawdown(equity)

    # 변동성 (using core)
    volatility = calculate_volatility(returns, annualize=True, annualization_factor=ANNUALIZATION_FACTOR)
    upside_vol = calculate_upside_volatility(returns, annualize=True, annualization_factor=ANNUALIZATION_FACTOR)
    downside_vol = calculate_downside_volatility(returns, annualize=True, annualization_factor=ANNUALIZATION_FACTOR)

    # 리스크 조정 수익률 (using core)
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate, ANNUALIZATION_FACTOR)
    sortino = calculate_sortino_ratio(returns, risk_free_rate, ANNUALIZATION_FACTOR)
    calmar = calculate_calmar_ratio(cagr, max_dd)

    # VaR & CVaR (using core)
    var_95 = calculate_var(returns, 0.95)
    var_99 = calculate_var(returns, 0.99)
    cvar_95 = calculate_cvar(returns, 0.95)
    cvar_99 = calculate_cvar(returns, 0.99)

    # 통계적 검정
    z_score, p_value = _calculate_statistical_tests(returns)

    # Skewness & Kurtosis
    skewness = float(stats.skew(returns)) if len(returns) > 2 else 0.0
    kurtosis = float(stats.kurtosis(returns)) if len(returns) > 2 else 0.0

    # 거래 메트릭
    trade_returns = trade_returns or []
    num_trades = len(trade_returns)
    win_rate, avg_win, avg_loss, profit_factor, expectancy = _calculate_trade_metrics(
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
