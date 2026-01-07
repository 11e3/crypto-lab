"""
================================================================================
[LEGACY / PROTOTYPE] 초기 실시간 매매 봇 (Deprecated)
================================================================================

[개요]
본 스크립트는 `bt.py`에서 검증된 '변동성 돌파 + 동적 노이즈 필터' 전략을
업비트(Upbit) 거래소에서 실제로 구동하기 위해 작성된 단일 파일 봇(V1)입니다.
`pyupbit` 라이브러리의 웹소켓(WebSocket) 기능을 활용하여 실시간 가격을 수신하고
주문을 실행합니다.

[시스템 재구축(Refactoring) 및 폐기 사유]

1. 상태 관리의 취약성 (Stateless Design)
   - 문제점: `has_bought`라는 단순 딕셔너리로 보유 상태를 관리함.
     봇이 재시작되거나 네트워크가 끊겼다 복구될 때, 실제 계좌 잔고와 봇의 메모리 상태가
     불일치(State Drift)하여 중복 매수나 매도 누락 사고 위험이 큼.
   - 개선: 현재 시스템은 데이터베이스(SQLite)와 거래소 잔고를 주기적으로 동기화(Sync)하는
     `OrderManager`를 도입하여 영속적이고 정확한 포지션 관리를 수행함.

2. 에러 핸들링 및 복구 능력 부족 (Resilience)
   - 문제점: `try-except` 블록 내에서 단순 `time.sleep`으로 재시도를 처리함.
     업비트 점검 시간(09:00 전후)이나 간헐적인 API 500 에러 발생 시
     웹소켓 연결이 영구적으로 끊어지거나 봇이 멈추는 현상(Zombie Process) 발생.
   - 개선: `Watchdog` 패턴과 지수 백오프(Exponential Backoff) 재연결 전략을 적용하여
     무중단 운영이 가능한 구조로 고도화함.

3. 하드코딩된 설정 및 확장성 한계
   - 문제점: API 키, 슬랙 토큰, 티커 목록이 코드 내에 하드코딩되거나 전역 변수로 관리됨.
     운영 환경(Testnet vs Live) 분리가 어렵고, 전략 파라미터 변경 시 코드 수정 후 재배포가 필수적임.
   - 개선: `.env` 환경 변수와 `config.yaml`을 통한 외부 설정 관리 시스템을 도입함.

4. 동기식 블로킹 구조 (Synchronous Blocking)
   - 문제점: `get_daily_metrics` 함수나 주문 요청이 네트워크 지연으로 늦어질 경우,
     웹소켓 수신 루프 전체가 멈춰(Blocking) 실시간 시세를 놓치는 현상 발생.
   - 개선: `asyncio` 기반의 비동기 아키텍처로 전환하여 시세 수신과 주문 처리를 병렬로 수행함.

[현재 상태]
본 파일의 로직은 `src/execution/` (실행 엔진) 및 `src/strategies/` (전략 로직)으로
분리되어 마이그레이션 완료되었습니다.
================================================================================
"""

import datetime  # 시간 관련 로직 처리 (장 시작/마감 시간 체크)
import time  # 지연(sleep) 처리 및 재시도 로직용

import pyupbit  # 업비트 API 래퍼 라이브러리
import requests  # 텔레그램 메시지 발송을 위한 HTTP 요청 라이브러리

# --- [CONFIGURATION] ---
# [RISK] API 키가 코드에 노출될 위험이 있음 (현재 시스템은 환경 변수로 개선됨)
ACCESS_KEY = ""
SECRET_KEY = ""

# Telegram Settings
TELEGRAM_TOKEN = ""
TELEGRAM_CHAT_ID = ""

# Trading Settings
TICKERS = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-TRX"]  # 거래 대상 코인 리스트
FEE = 0.0005  # 업비트 거래 수수료 (0.05%)
TARGET_SLOTS = 4  # 자금 관리: 전체 자산을 최대 4등분하여 분산 투자

# Strategy Parameters (Based on bt5.py)
# bt.py와 동일한 전략 파라미터를 사용하여 백테스트 결과와의 일관성 유지
SMA_PERIOD = 5  # 단기 이평선 및 K값 산출 기간
N = 2  # 장기/단기 비율 (자유도 제어)
TREND_SMA_PERIOD = SMA_PERIOD * N  # 10일 (추세 필터 기준선)
LONG_NOISE_PERIOD = SMA_PERIOD * N  # 10일 (노이즈 베이스라인)

# Initialize Upbit
upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)


def send_telegram(message):
    """텔레그램 메시지 발송 함수 (에러 발생 시 봇이 멈추지 않도록 예외 처리)"""
    if not TELEGRAM_TOKEN or "YOUR_" in TELEGRAM_TOKEN:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        requests.post(url, data=data)
    except Exception as e:
        print(f"Telegram Error: {e}")


def get_balance_safe(currency):
    """
    [안전 장치] 잔고 조회 API 호출 실패 시 0을 반환하여
    봇이 예기치 않게 종료되는 것을 방지함.
    """
    try:
        balance = upbit.get_balance(currency)
        if balance is None:
            return 0.0
        return float(balance)
    except Exception:
        return 0.0


def get_krw_balance():
    """원화(KRW) 잔고 조회 편의 함수"""
    return get_balance_safe("KRW")


def get_current_price_safe(ticker):
    """현재가 조회 실패 시 0 반환 (매도 로직 오작동 방지)"""
    try:
        price = pyupbit.get_current_price(ticker)
        if price is None:
            return 0.0
        return float(price)
    except Exception:
        return 0.0


def get_daily_metrics(ticker):
    """
    [핵심 전략 로직] 일봉 데이터를 받아 매매 지표를 계산함.

    계산 항목:
    1. Short Noise (K): 최근 5일 노이즈 평균 -> 변동성 돌파 계수
    2. Long Noise: 최근 10일 노이즈 평균 -> 시장 혼잡도 기준선
    3. SMA5: 매도(Exit) 및 기본 필터 기준
    4. SMA10: 대세 상승장(Trend) 판단 기준
    """
    try:
        # 최근 20일치 일봉 데이터 요청 (이동평균 계산을 위한 여유분 포함)
        df = pyupbit.get_ohlcv(ticker, interval="day", count=20)

        # 데이터가 부족하면 지표 계산 불가 -> None 반환
        if df is None or len(df) < TREND_SMA_PERIOD:
            return None

        # 노이즈 비율 계산: 1 - |시가-종가| / (고가-저가)
        # 0으로 나누기 에러(ZeroDivisionError) 방지를 위해 replace(0, 1) 사용
        ranges = df["high"] - df["low"]
        noise_series = 1 - abs(df["open"] - df["close"]) / ranges.replace(0, 1)
        # 변동폭이 0인 경우 노이즈를 0으로 처리 (보수적 접근)
        noise_series[ranges == 0] = 0

        # [중요] 지표 계산 시 '오늘(진행 중인 봉)'은 제외하고 '어제(확정 봉)'까지만 사용
        # iloc[-1]: 오늘, iloc[-2]: 어제

        # 1. Short Noise (최근 5일 평균 K값)
        short_noise = noise_series.iloc[-1 - SMA_PERIOD : -1].mean()

        # 2. Long Noise (최근 10일 평균 베이스라인)
        long_noise = noise_series.iloc[-1 - LONG_NOISE_PERIOD : -1].mean()

        # 3. SMA5 (5일 이동평균)
        sma5 = df["close"].iloc[-1 - SMA_PERIOD : -1].mean()

        # 4. SMA10 (10일 이동평균 - 추세선)
        sma10 = df["close"].iloc[-1 - TREND_SMA_PERIOD : -1].mean()

        # Target Price (매수 목표가) 계산
        # 목표가 = 오늘 시가 + (전일 변동폭 * K)
        today_open = df.iloc[-1]["open"]
        prev_range = df.iloc[-2]["high"] - df.iloc[-2]["low"]
        target = today_open + prev_range * short_noise

        return {
            "target": target,
            "k": short_noise,
            "long_noise": long_noise,
            "sma5": sma5,
            "sma10": sma10,
        }
    except Exception as e:
        print(f"Error metrics {ticker}: {e}")
        return None


def sell_all(ticker):
    """
    시장가 전량 매도 함수.
    최소 주문 금액(5000원) 이상일 때만 주문을 실행하여 API 에러 방지.
    """
    try:
        currency = ticker.split("-")[1]
        balance = get_balance_safe(currency)
        curr_price = get_current_price_safe(ticker)
        # 업비트 최소 주문 가능 금액: 5000 KRW
        if balance > 0 and curr_price > 0 and (balance * curr_price > 5000):
            upbit.sell_market_order(ticker, balance)
            send_telegram(f"[SELL] Sold all {ticker}")
    except Exception as e:
        print(f"Sell Error {ticker}: {e}")


def run_bot():
    print("Starting Trading Bot (Dynamic Relative Noise Strategy)...")

    # [초기 진단] API 키 유효성 검사
    try:
        upbit.get_balances()
        print("SUCCESS: API Keys are valid and working.")
    except Exception as e:
        print(f"!!! API CONNECTION FAILED: {e}")
        time.sleep(3)

    send_telegram("🚀 Bot Started: Dynamic Noise Filter (N=2) + Trend Filter")

    # [상태 관리] 메모리 상에서 매수 여부 관리 (취약점: 봇 재시작 시 초기화됨)
    has_bought = dict.fromkeys(TICKERS, False)
    target_info = {}  # 계산된 목표가 및 지표들을 저장할 딕셔너리

    print("Initializing Targets...")
    for ticker in TICKERS:
        metrics = get_daily_metrics(ticker)
        if metrics:
            target_info[ticker] = metrics
            print(
                f"[{ticker}] Tgt:{metrics['target']:.0f} | K:{metrics['k']:.2f} vs Base:{metrics['long_noise']:.2f} | SMA5:{metrics['sma5']:.0f} SMA10:{metrics['sma10']:.0f}"
            )
        else:
            print(f"{ticker}: Failed to calculate targets.")
        time.sleep(0.5)  # API 요청 제한(Rate Limit) 준수를 위한 대기

    # [복구 로직] 봇 재시작 시 이미 보유 중인 코인이 있는지 확인하여 상태 동기화
    print("Checking existing holdings...")
    for ticker in TICKERS:
        currency = ticker.split("-")[1]
        balance = get_balance_safe(currency)
        curr_price = get_current_price_safe(ticker)
        if balance > 0 and curr_price > 0 and (balance * curr_price > 5000):
            has_bought[ticker] = True
            print(f"✅ Recovered: Holding {ticker}")

    print("Entering Trading Loop...")

    # 웹소켓 연결 시도
    try:
        wm = pyupbit.WebSocketManager("ticker", TICKERS)
    except Exception as e:
        print(f"WebSocket Connection Failed: {e}")
        return

    while True:
        try:
            # 실시간 데이터 수신 (Blocking call)
            data = wm.get()
            if data["type"] == "ticker":
                ticker = data["code"]
                curr_price = data["trade_price"]

                now = datetime.datetime.now()

                # --- [Daily Reset Logic: 09:00:00] ---
                # 매일 아침 9시 장 시작 시점에 목표가 갱신 및 매도 조건 체크 수행
                if now.hour == 9 and now.minute == 0 and now.second <= 10:
                    wm.terminate()  # 웹소켓 잠시 종료 (안정성 확보)
                    print("Performing Daily Reset...")
                    time.sleep(5)  # 서버 데이터 갱신 대기

                    # 1. Exit Logic (스윙 전략: 종가가 5일 이평선 아래로 내려가면 매도)
                    for t in TICKERS:
                        try:
                            # 어제 종가 확인 (오늘 아침 9시 기준 '어제' 봉이 확정됨)
                            df = pyupbit.get_ohlcv(t, interval="day", count=10)
                            if df is not None and len(df) >= 7:
                                yesterday_close = df.iloc[-2]["close"]
                                # Exit 기준: 어제 기준의 SMA5
                                sma5_exit = df["close"].iloc[-7:-2].mean()

                                if has_bought[t]:
                                    if yesterday_close < sma5_exit:
                                        sell_all(t)
                                        has_bought[t] = False
                                        send_telegram(
                                            f"📉 [EXIT] {t} Trend Broken (Close {yesterday_close} < SMA {sma5_exit:.1f})"
                                        )
                                    else:
                                        send_telegram(f"✊ [HOLD] {t} Trend Intact")
                        except Exception as e:
                            print(f"Error daily check {t}: {e}")

                    # 2. Recalculate Targets (당일 새로운 목표가 계산)
                    target_info = {}
                    msg = "[DAILY UPDATE]\n"

                    for t in TICKERS:
                        for _ in range(3):  # API 실패 시 최대 3회 재시도 (Retry Logic)
                            metrics = get_daily_metrics(t)
                            if metrics:
                                target_info[t] = metrics
                                msg += f"{t}: Tgt {metrics['target']:.0f}, K {metrics['k']:.2f}\n"
                                break
                            time.sleep(1)
                    send_telegram(msg)
                    print(msg)

                    time.sleep(6)  # 9시 0분 10초가 지날 때까지 대기 (중복 실행 방지)
                    wm = pyupbit.WebSocketManager("ticker", TICKERS)  # 웹소켓 재연결
                    continue

                # --- [Entry Logic] ---
                # 이미 보유 중이면 매수 로직 건너뜀
                if has_bought[ticker]:
                    continue

                metrics = target_info.get(ticker)
                if not metrics:
                    continue

                # [전략 조건 검사]
                # 1. Breakout: 현재가가 목표가를 돌파했는가?
                cond_breakout = curr_price >= metrics["target"]
                # 2. Basic Filter: 타겟가가 5일 이평선 위에 있는가? (정배열)
                cond_sma5 = metrics["target"] > metrics["sma5"]
                # 3. Trend Filter: 타겟가가 10일 이평선 위에 있는가? (중기 추세 상승)
                cond_trend = metrics["target"] > metrics["sma10"]
                # 4. Relative Noise Filter: 현재 노이즈(K)가 평소(Long Noise)보다 낮은가? (변동성 안정)
                cond_noise = metrics["k"] < metrics["long_noise"]

                if cond_breakout and cond_sma5 and cond_trend and cond_noise:
                    krw_bal = get_krw_balance()
                    if krw_bal > 5000:
                        # 자금 관리: 남은 슬롯 수에 비례하여 투자금 산정 (Fixed Fractional)
                        current_held = sum(has_bought.values())
                        available_slots = TARGET_SLOTS - current_held

                        if available_slots > 0:
                            # 현재 예수금을 남은 슬롯 수로 나누어 투자
                            buy_amount = (krw_bal / available_slots) * (1 - FEE)

                            if buy_amount > 5000:
                                # 시장가 매수 주문 실행
                                res = upbit.buy_market_order(ticker, buy_amount)
                                if res and "uuid" in res:
                                    has_bought[ticker] = True
                                    msg = (
                                        f"🔥 [BUY] {ticker}\n"
                                        f"Price: {curr_price}\n"
                                        f"Target: {metrics['target']:.0f}\n"
                                        f"Noise: {metrics['k']:.2f} < {metrics['long_noise']:.2f}"
                                    )
                                    send_telegram(msg)
                                    print(msg)
                                else:
                                    print(f"Buy Failed: {res}")

        except Exception as e:
            # [복구 로직] 웹소켓 끊김 등 예외 발생 시 재연결 시도
            print(f"Loop Error: {e}")
            time.sleep(3)
            try:
                wm.terminate()
            except:
                pass
            wm = pyupbit.WebSocketManager("ticker", TICKERS)


if __name__ == "__main__":
    run_bot()
