"""
Main entry point for Crypto Quant System.

Web UI로 모든 기능을 사용할 수 있습니다.

실행 방법:
    uv run streamlit run src/web/app.py

기능:
    - 데이터 수집: Upbit에서 OHLCV 데이터 수집
    - 백테스트: 전략 성과 시뮬레이션
    - 파라미터 최적화: Grid/Random search
    - 고급 분석: Monte Carlo, Walk-Forward
"""

if __name__ == "__main__":
    print("=" * 60)
    print("Crypto Quant System")
    print("=" * 60)
    print()
    print("Web UI를 사용하세요:")
    print()
    print("  uv run streamlit run src/web/app.py")
    print()
    print("=" * 60)
