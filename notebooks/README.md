# 📚 Jupyter Notebooks - Crypto Lab 실습 가이드

## 개요

이 디렉토리에는 Crypto Lab의 **실제 사용 사례를 분석하는 Jupyter 노트북**이 포함되어 있습니다.

각 노트북은 **구체적인 시나리오**를 통해 시스템의 역량을 보여줍니다.

---

## 📖 노트북 가이드

### 1️⃣ [01-Backtesting-Case-Study.ipynb](01-Backtesting-Case-Study.ipynb)

**변동성 돌파 전략의 백테스팅 사례 연구**

#### 학습 내용
- 백테스팅 설정 및 실행
- 자산 곡선 분석
- 성능 지표 계산 (Sharpe, Sortino, Calmar)
- 거래 통계 및 패턴 분석
- 드로우다운 관리

#### 주요 실습
```python
# 백테스트 설정
config = BacktestConfig(
    tickers=["KRW-BTC", "KRW-ETH", "KRW-XRP"],
    initial_capital=1_000_000,
    fee_rate=0.0005,
)

# 전략 실행
strategy = VanillaVBO()
result = run_backtest(config, strategy)

# 성과 분석
print(f"수익률: {result.metrics.total_return_pct:.2f}%")
print(f"Sharpe: {result.metrics.sharpe_ratio:.2f}")
```

#### 📊 생성 차트
- 자산 곡선 (Equity Curve)
- 월별 성과 히트맵
- 거래 수익률 분포
- 누적 손익
- 포트폴리오 드로우다운

#### ✅ 체크포인트
- [ ] 초기 자본 설정 확인
- [ ] 거래 비용 영향 검토
- [ ] 최대 낙폭 분석
- [ ] 거래당 평균 수익 계산

---

### 2️⃣ [02-Portfolio-Optimization.ipynb](02-Portfolio-Optimization.ipynb)

**포트폴리오 구성: MPT vs 리스크 패리티 vs 켈리 기준 비교**

#### 학습 내용
- 현대포트폴리오이론 (MPT)
- 리스크 패리티 (Equal Risk Contribution)
- 켈리 기준 (Optimal Position Sizing)
- 거래비용 모델링
- 리밸런싱 전략

#### 주요 실습
```python
# MPT 최적화
returns = pd.DataFrame({
    'BTC': [...],
    'ETH': [...],
    'STAKING': [...],
})

weights_mpt = optimizer.optimize_mpt(returns)
# → 결과: BTC 52%, ETH 35%, STAKING 13%

# 리스크 패리티
weights_rp = optimizer.optimize_risk_parity(returns)
# → 각 자산의 리스크 기여도 동일

# 켈리 기준 (거래 통계 기반)
kelly_allocation = optimizer.kelly_portfolio(trades)
```

#### 📊 생성 차트
- 효율적 변경선 (Efficient Frontier)
- 방법별 가중치 비교
- 거래비용 분석
- 리밸런싱 누적 비용

#### 💡 주요 인사이트
| 방법 | 수익률 | 위험 | 사용 시기 |
|------|-------|------|---------|
| **MPT** | 높음 | 중간 | 장기 전략적 배분 |
| **리스크 패리티** | 중간 | 낮음 | 분산 투자 우선 |
| **켈리** | 변동 | 조건부 | 거래 시스템 |

---

### 3️⃣ [03-Live-Trading-Analysis.ipynb](03-Live-Trading-Analysis.ipynb)

**실시간 거래 시뮬레이션 및 위험 관리 분석**

#### 학습 내용
- 실시간 가격 시뮬레이션
- 비동기 주문 실행
- 동적 포지션 추적
- 위험 제약 조건
- 라이브 거래 체크리스트

#### 주요 실습
```python
# 가격 시뮬레이터 (GBM)
simulator = PriceSimulator(
    initial_prices={"BTC": 50000, "ETH": 3000},
    volatility=0.02  # 2% 일일 변동성
)

# 포트폴리오 관리
portfolio = Portfolio(initial_capital=1_000_000)
portfolio.open_position("BTC", quantity=0.5, price=50000)
portfolio.close_position("BTC", quantity=0.5, price=51000)

# 신호 생성
strategy = SimpleMomentumStrategy(momentum_period=5)
signal = strategy.generate_signal(simulator, "BTC")
```

#### 📊 생성 차트
- 실시간 포트폴리오 가치 변화
- 포지션 드로우다운
- 단계별 수익률 분포
- 누적 성과

#### 🎯 위험 지표
- 최대 낙폭 (MDD)
- Value at Risk (VaR)
- Conditional VaR (CVaR)
- Sharpe 비율 (연율)

#### ✅ 라이브 거래 체크리스트
- [ ] 종이 거래로 1개월+ 검증
- [ ] 다양한 시장 환경에서 성과 확인
- [ ] 손절매/익절 수준 최적화
- [ ] 자동 알림 시스템 구성
- [ ] API 키 및 보안 재확인
- [ ] **소액부터 시작** (전체 자본의 5% 이하)

---

## 🚀 시작하기

### 필수 요구사항
```bash
# 기본 설치
pip install jupyter pandas numpy matplotlib seaborn

# 선택 사항
pip install scipy  # 최적화
pip install scikit-learn  # 머신러닝
```

### 노트북 실행
```bash
# Jupyter Lab 시작
jupyter lab

# 또는 Jupyter Notebook
jupyter notebook
```

### 실행 순서
1. **01-Backtesting-Case-Study** ← 기본 개념 이해
2. **02-Portfolio-Optimization** ← 자산 배분 전략
3. **03-Live-Trading-Analysis** ← 실거래 시뮬레이션

---

## 📊 데이터 준비

### 샘플 데이터 생성
```bash
python scripts/generate_sample_data.py
```

이 명령어는 다음 파일을 생성합니다:
```
data/raw/sample_KRW-BTC.csv
```

### 실제 거래 데이터 사용
```python
from src.data.collector import UpbitDataCollector

collector = UpbitDataCollector()
ohlcv_data = collector.get_ohlcv("KRW-BTC", "day", 365)
```

---

## 💡 활용 아이디어

### 초급 (기본 이해)
- [ ] 각 노트북의 셀을 순차 실행
- [ ] 파라미터 변경해보기
- [ ] 결과 비교

### 중급 (심화 분석)
- [ ] 다른 자산쌍으로 백테스트
- [ ] 전략 파라미터 최적화
- [ ] 위험 제약 조건 추가

### 고급 (커스터마이징)
- [ ] 새로운 전략 구현
- [ ] 포트폴리오 구성 방법 추가
- [ ] 머신러닝 기반 신호 개발

---

## 🔍 일반적인 질문

### Q: 노트북에서 오류가 발생합니다
**A:** 다음을 확인하세요:
1. 모든 필수 패키지 설치 (`pip install -e ".[dev]"`)
2. Python 버전 3.14+
3. 작업 디렉토리가 프로젝트 루트

### Q: 데이터를 로드할 수 없습니다
**A:** 
```bash
# 샘플 데이터 생성
python scripts/generate_sample_data.py

# 또는 Upbit API에서 직접 (환경변수 설정 필요)
# .env 파일에 UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY 설정
```

### Q: 실제 거래 데이터로 백테스트하려면?
**A:**
```python
from src.data.collector import UpbitDataCollector

collector = UpbitDataCollector()
ohlcv = collector.get_ohlcv("KRW-BTC", "day", periods=365)

# BacktestConfig에 직접 사용
config = BacktestConfig(ohlcv_data=ohlcv, ...)
```

---

## 📚 추가 리소스

### 관련 문서
- [Architecture Documentation](../docs/architecture.md)
- [Strategy Guide](../docs/guides/strategy_guide.md)
- [Configuration Guide](../docs/guides/configuration.md)
- [Performance Optimization](../docs/guides/performance_optimization.md)

### 예제 스크립트
- [Basic Backtest](../examples/basic_backtest.py)
- [Custom Strategy](../examples/custom_strategy.py)
- [Strategy Benchmark](../examples/strategy_benchmark.py)
- [Portfolio Optimization](../examples/portfolio_optimization.py)

### 외부 참고자료
- [Jupyter Notebook Tutorial](https://jupyter-notebook.readthedocs.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Guide](https://numpy.org/doc/)
- [Matplotlib Visualization](https://matplotlib.org/)

---

## 🎓 학습 경로

### Week 1: 기본 개념
- 백테스팅 이해 (노트북 01)
- 성능 지표 계산
- 자산 곡선 분석

### Week 2: 포트폴리오 관리
- 자산 배분 전략 (노트북 02)
- 거래비용 영향
- 리밸런싱 정책

### Week 3: 실거래 준비
- 라이브 시뮬레이션 (노트북 03)
- 위험 관리
- 종이 거래 (Paper Trading)

### Week 4+: 고급 주제
- 전략 최적화
- 머신러닝 신호
- 포트폴리오 개선

---

## ⚠️ 중요 공지

### 위험 고지
- **손실 가능성**: 암호화폐 거래는 자본 손실 위험이 있습니다
- **백테스트 한계**: 과거 성과가 미래 성과를 보장하지 않습니다
- **실거래 전 준비**: 항상 종이 거래로 검증하세요
- **소액 시작**: 전체 자본의 5% 이하로 시작하세요

자세한 내용은 [DISCLAIMER.md](../DISCLAIMER.md) 참조

---

## 🤝 기여하기

노트북 개선 제안:
1. [GitHub Issues](https://github.com/11e3/crypto-lab/issues) 제출
2. Pull Request로 개선안 제시
3. Discussions에서 질문 및 공유

---

**마지막 업데이트**: 2025년 1월
**버전**: 1.0
**작성자**: Crypto Lab 개발팀

---

**Happy Learning! 🚀**
