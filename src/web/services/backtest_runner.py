"""Backtest execution service.

백테스트 실행 및 결과 관리 서비스.
"""

from datetime import date
from pathlib import Path

import streamlit as st

from src.backtester.engine import BacktestEngine, EventDrivenBacktestEngine
from src.backtester.models import BacktestConfig, BacktestResult
from src.strategies.base import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["run_backtest_service", "BacktestService"]


class BacktestService:
    """백테스트 실행 서비스.

    BacktestEngine을 래핑하여 Streamlit 환경에서
    백테스트를 실행하고 결과를 관리합니다.
    """

    def __init__(
        self,
        config: BacktestConfig,
        engine: BacktestEngine | None = None,
    ) -> None:
        """백테스트 서비스 초기화.

        Args:
            config: 백테스트 설정
            engine: Optional BacktestEngine (uses EventDrivenBacktestEngine if not provided)
        """
        self.config = config
        self.engine = engine if engine else EventDrivenBacktestEngine(config)

    def run(
        self,
        strategy: Strategy,
        data_files: dict[str, Path],
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> BacktestResult | None:
        """백테스트 실행.

        Args:
            strategy: 전략 인스턴스
            data_files: {ticker: file_path} 딕셔너리
            start_date: 시작일 (선택)
            end_date: 종료일 (선택)

        Returns:
            BacktestResult 또는 None (실패 시)
        """
        try:
            logger.info(f"Starting backtest: {strategy.name} with {len(data_files)} assets")

            result = self.engine.run(
                strategy=strategy,
                data_files=data_files,
                start_date=start_date,
                end_date=end_date,
            )

            logger.info(
                f"Backtest completed: "
                f"Return={result.total_return:.2f}%, "
                f"Trades={result.total_trades}"
            )

            return result

        except Exception as e:
            logger.exception(f"Backtest failed: {e}")
            return None


@st.cache_data(show_spinner="백테스트 실행 중...")
def run_backtest_service(
    strategy_name: str,
    strategy_params: dict,
    data_files_dict: dict[str, str],  # {ticker: file_path_str}
    config_dict: dict,
    start_date_str: str | None,
    end_date_str: str | None,
) -> BacktestResult | None:
    """캐시 가능한 백테스트 실행 래퍼.

    Streamlit 캐싱을 위해 모든 파라미터를 직렬화 가능한 타입으로 변환.

    Args:
        strategy_name: 전략 이름
        strategy_params: 전략 파라미터
        data_files_dict: 데이터 파일 경로 딕셔너리
        config_dict: 백테스트 설정 딕셔너리
        start_date_str: 시작일 문자열
        end_date_str: 종료일 문자열

    Returns:
        BacktestResult 또는 None
    """
    try:
        # Strategy 인스턴스 생성
        from src.web.components.sidebar.strategy_selector import (
            create_strategy_instance,
        )

        strategy = create_strategy_instance(strategy_name, strategy_params)
        if not strategy:
            logger.error("Failed to create strategy instance")
            return None

        # Path 객체 복원
        data_files = {ticker: Path(path) for ticker, path in data_files_dict.items()}

        # 날짜 복원
        start_date = date.fromisoformat(start_date_str) if start_date_str else None
        end_date = date.fromisoformat(end_date_str) if end_date_str else None

        # BacktestConfig 생성
        config = BacktestConfig(**config_dict)

        # 백테스트 실행
        service = BacktestService(config)
        result = service.run(strategy, data_files, start_date, end_date)

        return result

    except Exception as e:
        logger.exception(f"Backtest service failed: {e}")
        return None
