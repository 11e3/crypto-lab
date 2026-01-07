# type: ignore
"""
================================================================================
[LEGACY / PROTOTYPE] 초기 단일 스크립트 버전 (Deprecated)
================================================================================

[개요]
이 스크립트는 프로젝트 초기에 '변동성 돌파 전략'의 논리를 빠르게 검증하기 위해
작성된 프로토타입(PoC) 코드입니다. 현재는 사용되지 않으며, 전략의 핵심 로직을
이해하기 위한 참고용으로만 보존합니다.

[시스템 재구축(Refactoring) 및 폐기 사유]

1. 자료구조의 한계 (Dictionary Hell vs Type Safety)
   - 문제점: 모든 시장 데이터를 `dict`로 관리하여 `row['target']`과 같은 문자열 키 접근 방식 사용.
     이는 오타로 인한 런타임 에러 위험이 높고, IDE의 자동완성 및 정적 분석(MyPy)의 도움을 받기 어려움.
   - 개선: 현재 시스템은 `dataclass`와 `Pydantic`을 도입하여 명시적인 타입 정의와
     데이터 무결성 검증을 수행함.

2. 아키텍처 및 확장성 부족 (Monolithic vs Modular)
   - 문제점: 데이터 로딩, 전략 계산, 백테스팅 루프, 성과 분석이 하나의 함수에 강하게 결합됨(Coupled).
     전략을 변경하거나 거래소를 추가하려면 코드 전체를 수정해야 하는 구조.
   - 개선: SOLID 원칙에 입각하여 Data Layer, Strategy Interface, Execution Engine으로
     관심사를 분리(Separation of Concerns)한 모듈형 아키텍처로 전환함.

3. 테스트 불가능성 (Untestable)
   - 문제점: 거대한 함수 단위로 작성되어 세부 로직(예: 노이즈 계산, 슬리피지 반영)에 대한
     단위 테스트(Unit Test) 작성이 불가능함.
   - 개선: 현재 900개 이상의 테스트 케이스를 통해 각 모듈의 동작을 독립적으로 검증함.

4. 실행 환경의 차이 (Simulation Only)
   - 문제점: 실제 매매(Live Trading) 환경의 비동기 이벤트 처리, 에러 핸들링,
     웹소켓 연결 관리 기능이 전무함.
   - 개선: `asyncio` 기반의 이벤트 루프와 상태 관리 머신을 도입하여 실전 운영 가능한 시스템 구축.

[현재 상태]
본 파일의 로직은 `src/strategies/` 및 `src/backtester/` 패키지로
완전히 이관 및 고도화되었습니다.
================================================================================
"""

import datetime
import os

# ... (이하 코드)
import pandas as pd  # 데이터 분석을 위한 pandas 라이브러리 임포트

# Configuration: 시스템 전역 설정
FEE = 0.0005  # 거래 수수료 설정 (0.05%)
SLIPPAGE = 0.0005  # 시장가 체결 시 발생하는 슬리피지 설정 (0.05%)
Initial_Capital = 1.0  # 백테스팅 시작 자산 (단위: 1.0 = 100%)
TARGET_SLOTS = 4  # 동시 투자 가능한 최대 종목 수 (포트폴리오 분산)

# --- [Dynamic Filter Configuration] ---
SMA_PERIOD = 5  # 기준이 되는 단기 이동평균선 및 노이즈 계산 기간 ($SMA$)
N = 2  # 단기 기간에 대한 장기 기간의 배수 ($N$). 자유도를 낮추기 위한 설계 요소
# --------------------------------------

TREND_SMA_PERIOD = SMA_PERIOD * N  # 추세 필터로 사용할 장기 이평선 기간 ($SMA \times N$)
SHORT_TERM_NOISE_PERIOD = SMA_PERIOD  # $K$값(변동성 계수)으로 사용할 단기 노이즈 이평 기간
LONG_TERM_NOISE_PERIOD = SMA_PERIOD * N  # 상대적 시장 상태를 비교할 장기 노이즈 이평 기간


def parse_float(value):
    """문자열을 실수로 변환하며, 실패 시 0.0을 반환하여 에러 방지"""
    try:
        return float(value)
    except ValueError:
        return 0.0


def parse_date(dt_str):
    """다양한 날짜 형식을 datetime 객체로 안전하게 변환"""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    return None


def calculate_cagr(start_val, end_val, days):
    """연평균 복리 수익률($CAGR$) 계산 공식 적용"""
    if days <= 0:
        return 0.0
    if start_val <= 0:
        return 0.0
    if end_val <= 0:
        return -100.0
    return (end_val / start_val) ** (365.25 / days) - 1.0


def calculate_mdd(cum_returns):
    """최대 낙폭($MDD$) 계산: 최고점 대비 현재 자산의 최대 하락 비율 측정"""
    max_val = -float("inf")  # 역대 최고점을 저장할 변수
    mdd = 0.0  # 최대 낙폭값을 저장할 변수
    for val in cum_returns:
        if val > max_val:
            max_val = val  # 새로운 고점 갱신
        if max_val > 0:
            dd = (max_val - val) / max_val  # 고점 대비 하락률 계산
            if dd > mdd:
                mdd = dd  # 최대 낙폭 갱신
    return mdd * 100.0


def load_and_prep_data(filepath, noise_period=SHORT_TERM_NOISE_PERIOD, sma_period=SMA_PERIOD):
    """Parquet 파일 로드 및 전략에 필요한 지표(노이즈, 이평선, 타겟가) 계산"""
    df = pd.read_parquet(filepath)  # 효율적인 데이터 로딩을 위해 parquet 포맷 읽기

    # 인덱스가 날짜 형식인 경우 컬럼으로 재설정하여 처리 용이성 확보
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        if "datetime" in df.columns:
            df = df.rename(columns={"datetime": "date"})
        elif "index" in df.columns:
            df = df.rename(columns={"index": "date"})

    if "date" not in df.columns:
        df["date"] = df.index  # 날짜 컬럼 부재 시 인덱스를 날짜로 사용

    raw_data = df.to_dict(orient="records")  # 빠른 순회를 위해 DataFrame을 사전 리스트로 변환

    if not raw_data:
        return []
    raw_data.sort(key=lambda x: x["date"])  # 날짜순 정렬 보장

    processed_data = {}  # 가공된 데이터를 저장할 사전
    noise_ratios = []  # 매일의 노이즈 비율을 저장할 리스트

    trend_sma_period = TREND_SMA_PERIOD  # 장기 이평 기간 설정
    long_noise_period = LONG_TERM_NOISE_PERIOD  # 장기 노이즈 기간 설정

    for i in range(len(raw_data)):
        d = raw_data[i]
        rng = d["high"] - d["low"]  # 당일 변동폭(Range) 계산
        # 노이즈 비율 계산: $1 - \frac{|종가-시가|}{고가-저가}$. 값이 클수록 추세가 없고 혼조세임.
        noise = (1 - abs(d["close"] - d["open"]) / rng) if rng > 0 else 0.0
        noise_ratios.append(noise)

        # 지표 계산에 필요한 최소 데이터 개수가 확보되지 않으면 스킵
        if i < max(noise_period, trend_sma_period, long_noise_period):
            continue

        # 단기 노이즈 이평: 전략의 변동성 계수 $K$값으로 활용
        short_noise = sum(noise_ratios[i - noise_period : i]) / noise_period

        # 장기 노이즈 이평: 시장의 평균적인 노이즈 수준(Baseline)으로 활용
        long_noise = sum(noise_ratios[i - long_noise_period : i]) / long_noise_period

        # 장기 추세 필터용 SMA 계산 ($SMA_{trend}$)
        closes_trend = [x["close"] for x in raw_data[i - trend_sma_period : i]]
        sma_trend = sum(closes_trend) / trend_sma_period

        # 단기 이평선 계산 ($SMA$): 매도 시그널 및 기본 필터로 활용
        closes = [x["close"] for x in raw_data[i - SMA_PERIOD : i]]
        sma = sum(closes) / SMA_PERIOD

        prev_day = raw_data[i - 1]  # 전일 데이터 참조

        # 계산된 지표들을 날짜별 키로 저장
        processed_data[d["date"]] = {
            "open": d["open"],
            "high": d["high"],
            "low": d["low"],
            "close": d["close"],
            # 타겟가(진입가): 당일 시가 + (전일 변동폭 * 단기 노이즈 이평)
            "target": d["open"] + (prev_day["high"] - prev_day["low"]) * short_noise,
            "sma": sma,
            "sma_trend": sma_trend,
            "short_noise": short_noise,  # 변동성 돌파 계수 $K$
            "long_noise": long_noise,  # 상대 노이즈 필터 기준점
        }
    return processed_data


def run_portfolio_simulation(file_paths):
    """다중 종목 포트폴리오 백테스팅 시뮬레이션 실행"""
    print(f"\n{'=' * 20} PORTFOLIO SIMULATION {'=' * 20}")
    print(f"SMA: {SMA_PERIOD}, N: {N}")

    market_data = {}  # 전체 시장 데이터를 저장할 사전
    all_dates = set()  # 모든 종목의 날짜를 통합한 집합
    coin_names = []  # 테스트 대상 코인 이름 리스트

    # 각 파일(코인)별로 데이터를 로드하여 통합 데이터 구조 생성
    for fp in file_paths:
        c_name = os.path.basename(fp).replace("KRW-", "").replace("_day.parquet", "")
        coin_names.append(c_name)
        p_data = load_and_prep_data(fp)
        for dt, row in p_data.items():
            if dt not in market_data:
                market_data[dt] = {}
            market_data[dt][c_name] = row
            all_dates.add(dt)

    sorted_dates = sorted(all_dates)  # 날짜 순으로 정렬하여 시뮬레이션 준비
    if not sorted_dates:
        return

    cash = Initial_Capital  # 현재 가용 현금
    positions = {}  # 현재 보유 중인 포지션 정보
    equity_curve = []  # 자산 변화 곡선을 저장할 리스트

    # 설정된 슬롯 수와 실제 코인 수 중 최소값을 최대 슬롯으로 설정
    MAX_SLOTS = min(len(file_paths), TARGET_SLOTS)

    for dt in sorted_dates:
        daily_market = market_data.get(dt, {})  # 당일 시장 데이터 추출

        # [Step 1] 매도 (Exit) 로직 검사
        sold_coins = []
        for coin, pos in positions.items():
            if coin not in daily_market:
                continue
            row = daily_market[coin]

            # 종가가 단기 이평선($SMA$)을 이탈하면 즉시 매도 (스윙 매도 조건)
            if row["close"] < row["sma"]:
                sell_price = row["close"] * (1 - SLIPPAGE)  # 슬리피지 반영 매도가
                revenue = pos["amount"] * sell_price * (1 - FEE)  # 수수료 제외 수익금
                cash += revenue
                sold_coins.append(coin)

        for c in sold_coins:
            del positions[c]  # 보유 목록에서 제거

        # [Step 2] 매수 (Entry) 로직 검사
        candidates = [c for c in coin_names if c in daily_market]
        # 노이즈가 낮은(추세가 명확할 가능성이 높은) 종목부터 검토하도록 정렬
        candidates.sort(key=lambda x: daily_market[x]["short_noise"])

        for coin in candidates:
            if coin in positions:
                continue
            row = daily_market[coin]

            available_slots = MAX_SLOTS - len(positions)  # 진입 가능한 남은 슬롯 계산
            if available_slots <= 0:
                break

            # --- [진입 조건] ---
            # 1. 가격 돌파: 당일 고가가 타겟가를 상향 돌파하고, 타겟가가 단기 이평 위에 있을 것
            basic_cond = (row["target"] > row["sma"]) and (row["high"] >= row["target"])

            # 2. 장기 추세 필터: 타겟가가 장기 이평선($SMA_{trend}$) 위에 있을 것
            trend_cond = row["target"] > row["sma_trend"]

            # 3. 상대적 노이즈 필터: 최근 노이즈($K$)가 평균 노이즈(Long Noise)보다 낮아 시장이 안정적일 것
            noise_cond = row["short_noise"] < row["long_noise"]

            if basic_cond and trend_cond and noise_cond:
                invest_money = cash / available_slots  # 슬롯당 동일 비중 투자

                # 휩소(Whipsaw) 체크: 돌파 후 당일 종가가 다시 이평선을 하향 돌파한 경우 당일 청산 처리
                if row["close"] < row["sma"]:
                    buy_price = row["target"] * (1 + SLIPPAGE)
                    sell_price = row["close"] * (1 - SLIPPAGE)
                    amount = invest_money / buy_price * (1 - FEE)
                    return_money = amount * sell_price * (1 - FEE)
                    cash = cash - invest_money + return_money
                else:
                    # 조건 만족 시 포지션 진입 및 현금 차감
                    buy_price = row["target"] * (1 + SLIPPAGE)
                    amount = invest_money / buy_price * (1 - FEE)
                    positions[coin] = {"amount": amount, "entry_price": buy_price}
                    cash -= invest_money

        # [Step 3] 일일 자산 평가
        daily_positions_value = 0
        dt_index = sorted_dates.index(dt)

        for coin, pos in positions.items():
            price = None
            if coin in daily_market:
                price = daily_market[coin]["close"]  # 오늘 종가로 평가
            else:
                # 데이터가 누락된 경우 직전 유효 종가를 찾아 평가
                for i in range(dt_index - 1, -1, -1):
                    prev_dt = sorted_dates[i]
                    if coin in market_data.get(prev_dt, {}):
                        price = market_data[prev_dt][coin]["close"]
                        break

                # 끝내 가격이 없으면 진입가로 대체하여 에러 방지
                if price is None:
                    price = pos["entry_price"]

            daily_positions_value += pos["amount"] * price

        daily_equity = cash + daily_positions_value  # 현금 + 보유 코인 가치
        equity_curve.append(daily_equity)  # 자산 곡선 업데이트

    # 최종 결과 계산 및 출력
    final_equity = equity_curve[-1]
    total_days = (sorted_dates[-1] - sorted_dates[0]).days

    cagr = calculate_cagr(Initial_Capital, final_equity, total_days) * 100
    mdd = calculate_mdd(equity_curve)

    print(
        f"[PORTFOLIO FINAL] CAGR: {cagr:.2f}% | MDD: {mdd:.2f}% | Calmar: {cagr / mdd if mdd > 0 else 0:.2f}"
    )
    print(f"Final Equity: {final_equity:.2f}")


def main():
    """테스트할 데이터 파일 리스트를 정의하고 시뮬레이션 실행"""
    upbit_files = [
        "data/raw/KRW-BTC_day.parquet",
        "data/raw/KRW-ETH_day.parquet",
        "data/raw/KRW-XRP_day.parquet",
        "data/raw/KRW-TRX_day.parquet",
    ]
    run_portfolio_simulation(upbit_files)


if __name__ == "__main__":
    main()  # 스크립트 실행 시 메인 함수 호출
