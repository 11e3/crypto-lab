# Data Layer API

데이터 레이어는 시장 데이터 수집, 캐싱, 변환을 담당합니다.

## 모듈 개요

- `src.data.base`: 데이터 소스 인터페이스
- `src.data.upbit_source`: Upbit 데이터 소스 구현
- `src.data.collector`: 데이터 수집기
- `src.data.cache`: 지표 캐싱
- `src.data.converters`: 데이터 변환 유틸리티

## 핵심 인터페이스

### `DataSource` (추상 클래스)

모든 데이터 소스의 기본 인터페이스입니다.

#### 메서드

##### `get_ohlcv(ticker: str, interval: str, start_date: date | None = None, end_date: date | None = None) -> pd.DataFrame`

OHLCV 데이터를 조회합니다.

**매개변수:**
- `ticker`: 티커
- `interval`: 시간 간격 ("day", "minute240" 등)
- `start_date`: 시작 날짜 (선택)
- `end_date`: 종료 날짜 (선택)

**반환값:**
- `pd.DataFrame`: OHLCV 데이터

##### `get_multiple_ohlcv(tickers: list[str], interval: str, start_date: date | None = None, end_date: date | None = None) -> dict[str, pd.DataFrame]`

여러 티커의 OHLCV 데이터를 조회합니다.

**매개변수:**
- `tickers`: 티커 목록
- `interval`: 시간 간격
- `start_date`: 시작 날짜 (선택)
- `end_date`: 종료 날짜 (선택)

**반환값:**
- `dict[str, pd.DataFrame]`: 티커별 DataFrame 딕셔너리

## 구현 클래스

### `UpbitDataSource`

Upbit 거래소의 데이터 소스 구현입니다.

#### 생성자

```python
UpbitDataSource(
    access_key: str | None = None,
    secret_key: str | None = None
)
```

**매개변수:**
- `access_key`: Upbit API Access Key (선택)
- `secret_key`: Upbit API Secret Key (선택)

### `UpbitDataCollector`

Upbit 데이터를 수집하고 저장하는 클래스입니다.

#### 생성자

```python
UpbitDataCollector(
    data_dir: Path | None = None
)
```

**매개변수:**
- `data_dir`: 데이터 저장 디렉토리 (기본값: `data/raw`)

#### 메서드

##### `collect(ticker: str, interval: str, force_full_refresh: bool = False) -> dict`

단일 티커의 데이터를 수집합니다.

**매개변수:**
- `ticker`: 티커
- `interval`: 시간 간격
- `force_full_refresh`: 전체 새로고침 여부

**반환값:**
- `dict`: 수집 결과 정보

**예시:**
```python
from src.data.collector import UpbitDataCollector

collector = UpbitDataCollector()
result = collector.collect("KRW-BTC", "day", force_full_refresh=False)
print(f"수집된 데이터: {result['new_records']}개")
```

##### `collect_multiple(tickers: list[str], intervals: list[str], force_full_refresh: bool = False) -> dict`

여러 티커와 간격의 데이터를 수집합니다.

**매개변수:**
- `tickers`: 티커 목록
- `intervals`: 시간 간격 목록
- `force_full_refresh`: 전체 새로고침 여부

**반환값:**
- `dict`: 수집 결과 요약

**예시:**
```python
from src.data.collector import UpbitDataCollector, Interval

collector = UpbitDataCollector()
results = collector.collect_multiple(
    tickers=["KRW-BTC", "KRW-ETH"],
    intervals=[Interval.DAY, Interval.MINUTE240],
    force_full_refresh=False
)
```

### `IndicatorCache`

지표 계산 결과를 캐싱하는 클래스입니다.

#### 함수

##### `get_cache() -> IndicatorCache`

전역 캐시 인스턴스를 반환합니다.

#### 메서드

##### `get(ticker: str, interval: str, indicator_name: str, params: dict) -> pd.Series | None`

캐시에서 지표를 조회합니다.

**매개변수:**
- `ticker`: 티커
- `interval`: 시간 간격
- `indicator_name`: 지표 이름
- `params`: 지표 파라미터

**반환값:**
- `pd.Series | None`: 캐시된 지표 또는 None

##### `set(ticker: str, interval: str, indicator_name: str, params: dict, values: pd.Series) -> None`

지표를 캐시에 저장합니다.

**매개변수:**
- `ticker`: 티커
- `interval`: 시간 간격
- `indicator_name`: 지표 이름
- `params`: 지표 파라미터
- `values`: 지표 값

##### `clear() -> None`

캐시를 모두 지웁니다.

##### `clear_ticker(ticker: str) -> None`

특정 티커의 캐시를 지웁니다.

## 데이터 변환

### `Interval` (Enum)

지원하는 시간 간격을 정의합니다.

```python
class Interval(Enum):
    MINUTE1 = "minute1"
    MINUTE3 = "minute3"
    MINUTE5 = "minute5"
    MINUTE15 = "minute15"
    MINUTE30 = "minute30"
    MINUTE60 = "minute60"
    MINUTE240 = "minute240"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
```

### 변환 함수

#### `convert_to_parquet(df: pd.DataFrame, filepath: Path) -> None`

DataFrame을 Parquet 형식으로 저장합니다.

**매개변수:**
- `df`: 저장할 DataFrame
- `filepath`: 저장할 파일 경로

#### `load_from_parquet(filepath: Path) -> pd.DataFrame`

Parquet 파일에서 DataFrame을 로드합니다.

**매개변수:**
- `filepath`: 파일 경로

**반환값:**
- `pd.DataFrame`: 로드된 DataFrame

## 사용 예제

### 데이터 수집

```python
from src.data.collector import UpbitDataCollector, Interval

collector = UpbitDataCollector()

# 단일 티커 수집
result = collector.collect("KRW-BTC", Interval.DAY)
print(f"새 데이터: {result['new_records']}개")

# 여러 티커 수집
results = collector.collect_multiple(
    tickers=["KRW-BTC", "KRW-ETH", "KRW-XRP"],
    intervals=[Interval.DAY, Interval.MINUTE240]
)
```

### 데이터 소스 사용

```python
from src.data.upbit_source import UpbitDataSource
from datetime import date

source = UpbitDataSource()

# OHLCV 데이터 조회
df = source.get_ohlcv(
    ticker="KRW-BTC",
    interval="day",
    start_date=date(2023, 1, 1),
    end_date=date(2023, 12, 31)
)

# 여러 티커 조회
data = source.get_multiple_ohlcv(
    tickers=["KRW-BTC", "KRW-ETH"],
    interval="day"
)
```

### 캐싱 사용

```python
from src.data.cache import get_cache
import pandas as pd

cache = get_cache()

# 지표 계산
def calculate_sma(df: pd.DataFrame, period: int) -> pd.Series:
    cache_key = f"sma_{period}"
    
    # 캐시 확인
    cached = cache.get("KRW-BTC", "day", "sma", {"period": period})
    if cached is not None:
        return cached
    
    # 계산
    sma = df["close"].rolling(window=period).mean()
    
    # 캐시 저장
    cache.set("KRW-BTC", "day", "sma", {"period": period}, sma)
    
    return sma
```

### CLI를 통한 데이터 수집

```bash
# 기본 수집
upbit-quant collect

# 특정 티커 및 간격 지정
upbit-quant collect -t KRW-BTC KRW-ETH -i day minute240

# 전체 새로고침
upbit-quant collect --full-refresh
```

## 관련 문서

- [시작 가이드](../guides/getting_started.md)
- [아키텍처 문서](../architecture.md)
