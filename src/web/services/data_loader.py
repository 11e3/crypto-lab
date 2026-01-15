"""Data loading service.

OHLCV 데이터 로딩 및 캐싱 서비스.
"""

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import RAW_DATA_DIR
from src.data.collector_fetch import Interval
from src.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["load_ticker_data", "get_data_files"]


@st.cache_data(ttl=3600, show_spinner="데이터 로딩 중...")
def load_ticker_data(
    ticker: str,
    interval: Interval,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame | None:
    """OHLCV 데이터 로딩 (1시간 캐시).

    Args:
        ticker: 티커 (예: KRW-BTC)
        interval: 캔들 인터벌
        start_date: 시작일 (선택)
        end_date: 종료일 (선택)

    Returns:
        OHLCV DataFrame 또는 None (실패 시)
    """
    try:
        # 데이터 파일 경로
        file_path = RAW_DATA_DIR / ticker / f"{ticker}_{interval}.parquet"

        if not file_path.exists():
            logger.warning(f"Data file not found: {file_path}")
            return None

        # 데이터 로드
        df = pd.read_parquet(file_path)

        # 날짜 필터링
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]

        logger.info(f"Loaded {ticker} {interval}: {len(df)} rows ({df.index[0]} ~ {df.index[-1]})")

        return df

    except Exception as e:
        logger.exception(f"Failed to load data for {ticker}: {e}")
        return None


def get_data_files(
    tickers: list[str],
    interval: Interval,
) -> dict[str, Path]:
    """티커 목록에 대한 데이터 파일 경로 딕셔너리 생성.

    Args:
        tickers: 티커 리스트
        interval: 캔들 인터벌

    Returns:
        {ticker: file_path} 딕셔너리
    """
    data_files: dict[str, Path] = {}

    for ticker in tickers:
        file_path = RAW_DATA_DIR / ticker / f"{ticker}_{interval}.parquet"
        if file_path.exists():
            data_files[ticker] = file_path
        else:
            logger.warning(f"Data file not found for {ticker}: {file_path}")

    return data_files


def validate_data_availability(
    tickers: list[str],
    interval: Interval,
) -> tuple[list[str], list[str]]:
    """데이터 가용성 검증.

    Args:
        tickers: 티커 리스트
        interval: 캔들 인터벌

    Returns:
        (available_tickers, missing_tickers) 튜플
    """
    available: list[str] = []
    missing: list[str] = []

    for ticker in tickers:
        file_path = RAW_DATA_DIR / ticker / f"{ticker}_{interval}.parquet"
        if file_path.exists():
            available.append(ticker)
        else:
            missing.append(ticker)

    return available, missing
