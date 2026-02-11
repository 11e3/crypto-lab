"""
WFA용 간단한 백테스트 실행기.

Walk-Forward Analysis에서 사용하는 가벼운 벡터화 백테스트.
"""

import numpy as np
import pandas as pd

from src.backtester.models import BacktestResult
from src.strategies.base import Strategy
from src.utils.logger import get_logger
from src.utils.metrics_core import calculate_daily_returns, calculate_mdd, calculate_sharpe_ratio

logger = get_logger(__name__)


def simple_backtest(
    data: pd.DataFrame,
    strategy: Strategy,
    initial_capital: float = 10000000,
) -> BacktestResult:
    """
    간단한 벡터화 백테스트 (BacktestEngine 대신 직접 구현).

    Args:
        data: OHLCV 데이터
        strategy: 트레이딩 전략
        initial_capital: 초기 자본금

    Returns:
        BacktestResult 객체
    """
    try:
        # 데이터 복사
        df = data.copy()

        # 지표 계산 및 신호 생성
        df = strategy.calculate_indicators(df)
        df = strategy.generate_signals(df)

        # 신호 확인
        if "signal" not in df.columns:
            return _create_empty_result()

        # 포지션 시뮬레이션
        trades, equity = _simulate_positions(df, initial_capital)

        # 메트릭 계산
        return _calculate_metrics(trades, equity, initial_capital)

    except Exception as e:
        logger.error(f"Simple backtest error: {e}")
        return _create_empty_result()


def _simulate_positions(
    df: pd.DataFrame,
    initial_capital: float,
) -> tuple[list[float], list[float]]:
    """포지션 시뮬레이션 및 거래 기록."""
    position = 0  # 0: 없음, 1: 롱, -1: 숏
    entry_price = 0.0
    trades: list[float] = []
    equity: list[float] = [initial_capital]

    for _idx, row in df.iterrows():
        signal = row.get("signal", 0)
        close = float(row.get("close", 0))

        if signal != 0 and position == 0:
            # 엔트리
            entry_price = close
            position = int(signal)
        elif signal * position < 0:
            # 엑싯 (반대 신호)
            if position != 0:
                pnl = (close - entry_price) * position / entry_price
                trades.append(pnl)
                equity.append(equity[-1] * (1 + pnl))
                position = int(signal)
                entry_price = close if signal != 0 else 0.0
            else:
                position = 0

        if position == 0 and len(equity) > 1:
            equity.append(equity[-1])

    # 보유 중인 포지션 정리
    if position != 0 and len(df) > 0:
        last_close = float(df.iloc[-1].get("close", entry_price))
        pnl = (last_close - entry_price) * position / entry_price
        trades.append(pnl)
        equity.append(equity[-1] * (1 + pnl))

    return trades, equity


def _calculate_metrics(
    trades: list[float],
    equity: list[float],
    initial_capital: float,
) -> BacktestResult:
    """거래 결과에서 메트릭 계산."""
    # 총 수익률
    total_return = (equity[-1] - initial_capital) / initial_capital if equity else 0.0

    # Sharpe 비율
    equity_arr = np.array(equity)
    returns = calculate_daily_returns(equity_arr)
    sharpe = calculate_sharpe_ratio(returns, 252)

    # Max Drawdown
    max_drawdown = calculate_mdd(equity_arr)

    # 승률
    winning_trades, win_rate = _calculate_win_rate(trades)

    result = BacktestResult()
    result.total_return = total_return
    result.sharpe_ratio = sharpe
    result.mdd = max_drawdown
    result.total_trades = len(trades)
    result.winning_trades = winning_trades
    result.win_rate = win_rate
    result.equity_curve = np.array(equity)

    return result


def _calculate_win_rate(trades: list[float]) -> tuple[int, float]:
    """승률 계산."""
    if not trades:
        return 0, 0.0

    winning = sum(1 for t in trades if t > 0)
    return winning, winning / len(trades)


def _create_empty_result() -> BacktestResult:
    """빈 결과 생성."""
    result = BacktestResult()
    result.total_return = 0.0
    result.sharpe_ratio = 0.0
    result.mdd = 0.0
    result.total_trades = 0
    result.winning_trades = 0
    result.win_rate = 0.0
    return result
