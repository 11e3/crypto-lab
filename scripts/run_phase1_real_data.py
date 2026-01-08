"""
Phase 1 재검증 (실데이터/샘플데이터): WFA, Robustness, Permutation 실행 및 리포트 저장

사용법:
- 기본: python scripts/run_phase1_real_data.py
- CSV 지정: python scripts/run_phase1_real_data.py --csv data/raw/sample_ohlcv.csv
- Parquet 지정: python scripts/run_phase1_real_data.py --parquet data/raw/KRW-BTC_day.parquet

리포트 출력:
- reports/phase1_real_wfa.html
- reports/phase1_real_robustness.html
- reports/phase1_real_permutation.html
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtester.permutation_test import PermutationTester  # noqa: E402
from src.backtester.robustness_analysis import RobustnessAnalyzer  # noqa: E402
from src.backtester.walk_forward_auto import WalkForwardAnalyzer  # noqa: E402
from src.strategies.volatility_breakout.vbo import VanillaVBO  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_input_or_generate(csv_path: Path | None, parquet_path: Path | None) -> pd.DataFrame:
    if parquet_path and parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        # 기대 컬럼: date/timestamp, open, high, low, close, volume
        date_col = (
            "date" if "date" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
        )
        if date_col is None:
            # 인덱스가 타임스탬프일 수 있으므로 인덱스를 컬럼으로 복구
            idx_name = df.index.name or "index"
            df = df.reset_index()
            date_col = (
                "date"
                if "date" in df.columns
                else ("timestamp" if "timestamp" in df.columns else idx_name)
            )
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.rename(columns={date_col: "date"})
        cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        # volume 누락 시 기본값 채움
        if "volume" not in cols:
            df["volume"] = 0.0
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"] = df[["open", "low", "close"]].min(axis=1)
        return df.set_index("date")
    if csv_path and csv_path.exists():
        df = pd.read_csv(csv_path)
        # 기대 컬럼: date, open, high, low, close, volume
        date_col = "date" if "date" in df.columns else "timestamp"
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.rename(columns={date_col: "date"})
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"] = df[["open", "low", "close"]].min(axis=1)
        return df.set_index("date")
    logger.warning("입력 미제공: 샘플 합성 데이터 생성")
    dates = pd.date_range(start="2018-01-01", periods=2929, freq="D")
    np.random.seed(17)
    trend = np.linspace(0, 0.2, len(dates))
    returns = np.random.normal(0, 0.02, len(dates)) + trend / len(dates)
    prices = 100 * np.exp(np.cumsum(returns))
    data = pd.DataFrame(
        {
            "date": dates,
            "open": prices * (1 + np.random.normal(0, 0.005, len(dates))),
            "high": prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            "close": prices,
            "volume": np.random.uniform(100, 1000, len(dates)),
        }
    )
    data["high"] = data[["open", "high", "close"]].max(axis=1)
    data["low"] = data[["open", "low", "close"]].min(axis=1)
    return data.set_index("date")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="OHLCV CSV 경로")
    parser.add_argument("--parquet", type=str, default=None, help="OHLCV Parquet 경로")
    args = parser.parse_args()
    csv_path = Path(args.csv) if args.csv else None
    parquet_path = Path(args.parquet) if args.parquet else None

    data = load_input_or_generate(csv_path, parquet_path)

    # WFA
    wfa = WalkForwardAnalyzer(data=data, train_period=504, test_period=252, step=63)
    report_wfa = wfa.run(
        strategy_factory=lambda p: VanillaVBO(
            use_improved_noise=True,
            use_adaptive_k=True,
            use_dynamic_slippage=False,
            use_cost_calculator=False,
            min_hold_periods=3,
            **p,
        ),
        param_ranges={"sma_period": [3, 4, 5], "trend_sma_period": [7, 8, 9]},
    )

    # Robustness
    robust = RobustnessAnalyzer(
        data=data,
        strategy_factory=lambda p: VanillaVBO(
            use_improved_noise=True,
            use_adaptive_k=True,
            use_dynamic_slippage=False,
            use_cost_calculator=False,
            min_hold_periods=3,
            **p,
        ),
    )
    report_robust = robust.analyze(
        optimal_params={"sma_period": 4, "trend_sma_period": 8},
        parameter_ranges={"sma_period": [3, 4, 5], "trend_sma_period": [7, 8, 9]},
    )

    # Permutation
    perm = PermutationTester(
        data=data,
        strategy_factory=lambda: VanillaVBO(
            use_improved_noise=True,
            use_adaptive_k=True,
            use_dynamic_slippage=False,
            use_cost_calculator=False,
        ),
    )
    report_perm = perm.run(num_shuffles=1000, shuffle_columns=["close", "volume"])

    # 저장
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    robust.export_report_html(report_robust, reports_dir / "phase1_real_robustness.html")
    perm.export_report_html(report_perm, str(reports_dir / "phase1_real_permutation.html"))

    # WFA HTML이 별도 메서드 없으면 간단 텍스트 저장
    wfa_summary = (
        f"WFA Summary\nIS Avg Return: {report_wfa.in_sample_avg_return:.2f}%\n"
        f"OOS Avg Return: {report_wfa.out_of_sample_avg_return:.2f}%\n"
        f"Overfitting Ratio: {report_wfa.overfitting_ratio:.2f}%\n"
    )
    (reports_dir / "phase1_real_wfa.txt").write_text(wfa_summary, encoding="utf-8")

    print("저장 완료:")
    print(f" - {reports_dir / 'phase1_real_wfa.txt'}")
    print(f" - {reports_dir / 'phase1_real_robustness.html'}")
    print(f" - {reports_dir / 'phase1_real_permutation.html'}")


if __name__ == "__main__":
    logger.info(f"실데이터 재검증 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    main()

