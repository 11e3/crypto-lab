# 백테스트 스크립트

이 디렉토리에는 백테스트 관련 스크립트가 포함되어 있습니다.

## 파일

- `legacy_bt_with_trades.py`: 거래 추출 기능이 포함된 레거시 백테스트 구현

## 참고

백테스트 실행을 위해서는 CLI 명령어를 사용하세요:
```bash
upbit-quant backtest --tickers KRW-BTC KRW-ETH --interval day
```

또는 Jupyter 노트북을 사용하세요:
```bash
jupyter notebook notebooks/01_vbo.ipynb
```
