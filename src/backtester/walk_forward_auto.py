"""
자동화된 Walk-Forward Analysis (WFA) 엔진.

Walk-Forward Analysis는 과적합을 검증하기 위한 핵심 방법론입니다.
- Training 구간에서 파라미터 최적화
- Test 구간에서 Out-of-Sample (OOS) 성과 측정
- In-Sample (IS) vs Out-of-Sample 비교

과적합 판단 기준:
- OOS/IS 비율 > 0.3: 정상 (또는 약간 보수적)
- OOS/IS 비율 0.1-0.3: 경고 (중간 수준 과적합)
- OOS/IS 비율 < 0.1: 위험 (심각한 과적합)
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtester.engine import BacktestConfig, BacktestResult
from src.strategies.base import Strategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WFASegment:
    """Walk-Forward Analysis 한 구간 (Training + Test)."""

    period_start: pd.Timestamp
    period_end: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    in_sample_result: BacktestResult | None = None
    out_of_sample_result: BacktestResult | None = None
    optimal_params: dict[str, Any] | None = None

    @property
    def oos_is_ratio(self) -> float:
        """OOS/IS 수익률 비율 (과적합 지표)."""
        if self.in_sample_result is None or self.out_of_sample_result is None:
            return 0.0

        is_return = self.in_sample_result.total_return
        oos_return = self.out_of_sample_result.total_return

        if is_return <= 0:
            return 0.0

        return oos_return / is_return

    @property
    def overfitting_severity(self) -> str:
        """과적합 정도 판정."""
        ratio = self.oos_is_ratio
        if ratio > 0.3:
            return "정상"
        elif ratio > 0.1:
            return "경고"
        else:
            return "위험"


@dataclass
class WFAReport:
    """Walk-Forward Analysis 종합 리포트."""

    segments: list[WFASegment] = field(default_factory=list)

    # 집계 통계
    in_sample_avg_return: float = 0.0
    out_of_sample_avg_return: float = 0.0
    overfitting_ratio: float = 0.0

    in_sample_sharpe: float = 0.0
    out_of_sample_sharpe: float = 0.0

    in_sample_mdd: float = 0.0
    out_of_sample_mdd: float = 0.0

    parameter_stability: dict[str, list[float]] = field(default_factory=dict)

    generated_at: datetime = field(default_factory=datetime.now)


class WalkForwardAnalyzer:
    """
    Walk-Forward Analysis 자동 실행기.

    사용 예:
    ```python
    analyzer = WalkForwardAnalyzer(
        data=ohlcv_df,
        train_period=252 * 2,  # 2년
        test_period=252,        # 1년
        step=63                 # 3개월 롤링
    )

    report = analyzer.run(
        strategy_factory=lambda params: VanillaVBO(**params),
        param_ranges={
            'sma_period': [3, 4, 5],
            'noise_period': [7, 8, 9]
        }
    )
    ```
    """

    def __init__(
        self,
        data: pd.DataFrame,
        train_period: int = 252 * 2,
        test_period: int = 252,
        step: int = 63,
    ):
        """
        Initialize Walk-Forward Analysis.

        Args:
            data: OHLCV 데이터 (index는 datetime, columns: open, high, low, close, volume)
            train_period: Training 기간 (거래일 단위)
            test_period: Test 기간 (거래일 단위)
            step: Rolling window step (거래일 단위)
        """
        self.data = data
        self.train_period = train_period
        self.test_period = test_period
        self.step = step

        self.segments: list[WFASegment] = []

        logger.info(
            f"WalkForwardAnalyzer initialized: "
            f"train={train_period}d, test={test_period}d, step={step}d"
        )

    def run(
        self,
        strategy_factory: Callable[[dict[str, Any]], Strategy],
        param_ranges: dict[str, list[Any]],
        backtest_config: BacktestConfig | None = None,
        verbose: bool = True,
    ) -> WFAReport:
        """
        Walk-Forward Analysis 실행.

        Args:
            strategy_factory: 파라미터를 받아 Strategy를 생성하는 함수
            param_ranges: 테스트할 파라미터 범위
            backtest_config: 백테스트 설정
            verbose: 진행 상황 로깅

        Returns:
            WFAReport: 분석 결과
        """
        if backtest_config is None:
            backtest_config = BacktestConfig()

        self.initial_capital = backtest_config.initial_capital

        # 1. 모든 WFA 구간 생성
        segments = self._generate_segments()

        if verbose:
            logger.info(f"Generated {len(segments)} WFA segments")

        # 2. 각 구간 처리
        for i, segment in enumerate(segments):
            if verbose:
                logger.info(
                    f"[{i + 1}/{len(segments)}] Processing segment: "
                    f"{segment.train_start.date()} ~ {segment.test_end.date()}"
                )

            # 2.1 Training 데이터 추출
            train_data = self.data[
                (self.data.index >= segment.train_start) & (self.data.index <= segment.train_end)
            ]

            # 2.2 Test 데이터 추출
            test_data = self.data[
                (self.data.index >= segment.test_start) & (self.data.index <= segment.test_end)
            ]

            if len(train_data) < self.train_period * 0.9:
                logger.warning(f"Segment {i}: Insufficient training data, skipping")
                continue

            # 2.3 Training 구간에서 파라미터 최적화
            optimal_params = self._optimize_parameters_grid(
                train_data, strategy_factory, param_ranges
            )

            segment.optimal_params = optimal_params

            # 2.4 최적 파라미터로 Training 백테스트
            strategy_is = strategy_factory(optimal_params)
            is_result = self._simple_backtest(train_data, strategy_is)
            segment.in_sample_result = is_result

            # 2.5 최적 파라미터로 Test 백테스트 (OOS)
            strategy_oos = strategy_factory(optimal_params)
            oos_result = self._simple_backtest(test_data, strategy_oos)
            segment.out_of_sample_result = oos_result

            # 2.6 진행 현황 로깅
            if verbose:
                logger.info(
                    f"  IS Return: {is_result.total_return:.2%}, "
                    f"OOS Return: {oos_result.total_return:.2%}, "
                    f"Ratio: {segment.oos_is_ratio:.2%} ({segment.overfitting_severity})"
                )

            self.segments.append(segment)

        # 3. 결과 집계
        report = self._aggregate_results()

        if verbose:
            logger.info(
                f"WFA completed: "
                f"IS avg {report.in_sample_avg_return:.2%}, "
                f"OOS avg {report.out_of_sample_avg_return:.2%}, "
                f"Overfitting ratio: {report.overfitting_ratio:.2%}"
            )

        return report

    def _generate_segments(self) -> list[WFASegment]:
        """모든 WFA 구간 생성."""
        segments = []

        min_required_length = self.train_period + self.test_period

        if len(self.data) < min_required_length:
            raise ValueError(f"Data length ({len(self.data)}) < required ({min_required_length})")

        # Rolling window로 구간 생성
        for i in range(0, len(self.data) - min_required_length + 1, self.step):
            train_start_idx = i
            train_end_idx = i + self.train_period
            test_start_idx = train_end_idx
            test_end_idx = test_start_idx + self.test_period

            if test_end_idx > len(self.data):
                break

            segment = WFASegment(
                period_start=self.data.index[i],
                period_end=self.data.index[test_end_idx - 1],
                train_start=self.data.index[train_start_idx],
                train_end=self.data.index[train_end_idx - 1],
                test_start=self.data.index[test_start_idx],
                test_end=self.data.index[test_end_idx - 1],
            )
            segments.append(segment)

        return segments

    def _optimize_parameters_grid(
        self, data: pd.DataFrame, strategy_factory: Callable[..., Any], param_ranges: dict[str, list[Any]]
    ) -> dict[str, Any]:
        """
        Training 구간에서 파라미터 최적화 (Grid Search).

        목표: Training 수익률 최대화
        """
        from itertools import product

        # 파라미터 조합 생성
        param_keys = list(param_ranges.keys())
        param_values = [param_ranges[key] for key in param_keys]

        best_params = None
        best_return = -float("inf")

        for param_combo in product(*param_values):
            params = dict(zip(param_keys, param_combo, strict=False))

            try:
                strategy = strategy_factory(params)
                result = self._simple_backtest(data, strategy)

                if result.total_return > best_return:
                    best_return = result.total_return
                    best_params = params
            except Exception as e:
                logger.debug(f"Parameter combination {params} failed: {e}")
                continue

        if best_params is None:
            # Fallback: 첫 번째 조합 사용
            first_combo = [v[0] for v in param_values]
            best_params = dict(zip(param_keys, first_combo, strict=False))
            logger.warning("Optimization failed, using default parameters")

        return best_params

    def _simple_backtest(self, data: pd.DataFrame, strategy: Strategy) -> "BacktestResult":
        """
        간단한 벡터화 백테스트 (BacktestEngine 대신 직접 구현).

        Args:
            data: OHLCV 데이터
            strategy: 트레이딩 전략

        Returns:
            BacktestResult 객체
        """
        from src.backtester.engine import BacktestResult

        try:
            # 데이터 복사
            df = data.copy()

            # 지표 계산 및 신호 생성
            df = strategy.calculate_indicators(df)
            df = strategy.generate_signals(df)

            # 신호 확인
            if "signal" not in df.columns:
                # 신호가 없으면 성능 메트릭만 반환
                result = BacktestResult()
                result.total_return = 0.0
                result.sharpe_ratio = 0.0
                result.mdd = 0.0
                result.total_trades = 0
                result.winning_trades = 0
                result.win_rate = 0.0
                return result

            # 포지션 추적
            position = 0  # 0: 없음, 1: 롱, -1: 숏
            entry_price = 0
            trades = []
            equity = [self.initial_capital]

            for _idx, row in df.iterrows():
                signal = row.get("signal", 0)
                close = row.get("close", 0)

                if signal != 0 and position == 0:
                    # 엔트리
                    entry_price = close
                    position = signal
                elif signal * position < 0:
                    # 엑싯 (반대 신호)
                    if position != 0:
                        pnl = (close - entry_price) * position / entry_price
                        trades.append(pnl)
                        equity.append(equity[-1] * (1 + pnl))
                        position = signal
                        entry_price = close if signal != 0 else 0
                    else:
                        position = 0

                if position == 0 and len(equity) > 1:
                    equity.append(equity[-1])

            # 보유 중인 포지션 정리
            if position != 0 and len(df) > 0:
                last_close = df.iloc[-1].get("close", entry_price)
                pnl = (last_close - entry_price) * position / entry_price
                trades.append(pnl)
                equity.append(equity[-1] * (1 + pnl))

            # 메트릭 계산
            total_return = (
                (equity[-1] - self.initial_capital) / self.initial_capital if equity else 0.0
            )

            # Sharpe 비율
            if len(equity) > 1:
                returns = np.diff(equity) / equity[:-1]
                sharpe = (
                    np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
                    if np.std(returns) > 0
                    else 0.0
                )
            else:
                sharpe = 0.0

            # Max Drawdown
            if len(equity) > 1:
                cummax = np.maximum.accumulate(equity)
                dd = (np.array(equity) - cummax) / cummax
                max_drawdown = np.min(dd) if len(dd) > 0 else 0.0
            else:
                max_drawdown = 0.0

            # 승률
            if trades:
                winning_trades = sum(1 for t in trades if t > 0)
                win_rate = winning_trades / len(trades)
            else:
                win_rate = 0.0
                winning_trades = 0

            result = BacktestResult()
            result.total_return = total_return
            result.sharpe_ratio = sharpe
            result.mdd = max_drawdown
            result.total_trades = len(trades)
            result.winning_trades = winning_trades
            result.win_rate = win_rate
            result.equity_curve = np.array(equity)

            return result

        except Exception as e:
            logger.error(f"Simple backtest error: {e}")
            result = BacktestResult()
            result.total_return = 0.0
            result.sharpe_ratio = 0.0
            result.mdd = 0.0
            result.total_trades = 0
            result.winning_trades = 0
            result.win_rate = 0.0
            return result

    def _aggregate_results(self) -> WFAReport:
        """모든 세그먼트 결과 집계."""
        report = WFAReport(segments=self.segments)

        if not self.segments:
            return report

        # 수익률 집계
        is_returns = [s.in_sample_result.total_return for s in self.segments if s.in_sample_result]
        oos_returns = [
            s.out_of_sample_result.total_return for s in self.segments if s.out_of_sample_result
        ]

        report.in_sample_avg_return = float(np.mean(is_returns)) if is_returns else 0.0
        report.out_of_sample_avg_return = float(np.mean(oos_returns)) if oos_returns else 0.0

        # 과적합 비율
        if report.in_sample_avg_return > 0:
            report.overfitting_ratio = report.out_of_sample_avg_return / report.in_sample_avg_return

        # Sharpe 비율 집계
        is_sharpes = [s.in_sample_result.sharpe_ratio for s in self.segments if s.in_sample_result]
        oos_sharpes = [
            s.out_of_sample_result.sharpe_ratio for s in self.segments if s.out_of_sample_result
        ]

        report.in_sample_sharpe = float(np.mean(is_sharpes)) if is_sharpes else 0.0
        report.out_of_sample_sharpe = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0

        # MDD 집계
        is_mdds = [s.in_sample_result.mdd for s in self.segments if s.in_sample_result]
        oos_mdds = [s.out_of_sample_result.mdd for s in self.segments if s.out_of_sample_result]

        report.in_sample_mdd = float(np.mean(is_mdds)) if is_mdds else 0.0
        report.out_of_sample_mdd = float(np.mean(oos_mdds)) if oos_mdds else 0.0

        # 파라미터 안정성 (각 파라미터가 얼마나 일정한지)
        if self.segments and self.segments[0].optimal_params:
            for param_name in self.segments[0].optimal_params:
                param_values = [
                    s.optimal_params[param_name] for s in self.segments if s.optimal_params
                ]
                report.parameter_stability[param_name] = param_values

        return report

    def export_report_html(self, report: WFAReport, output_path: str | Path) -> None:
        """
        WFA 리포트를 HTML로 저장.
        """
        html = self._generate_html(report)

        Path(output_path).write_text(html, encoding="utf-8")
        logger.info(f"WFA report saved to {output_path}")

    def _generate_html(self, report: WFAReport) -> str:
        """HTML 리포트 생성."""
        html_parts = []

        # 헤더
        html_parts.append("""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Walk-Forward Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #4CAF50; color: white; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                .warning { color: #FF9800; }
                .danger { color: #F44336; }
                .success { color: #4CAF50; }
            </style>
        </head>
        <body>
            <h1>Walk-Forward Analysis Report</h1>
        """)

        # 요약
        html_parts.append(f"""
        <h2>Summary</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>In-Sample Avg Return</td>
                <td>{report.in_sample_avg_return:.2%}</td>
            </tr>
            <tr>
                <td>Out-of-Sample Avg Return</td>
                <td>{report.out_of_sample_avg_return:.2%}</td>
            </tr>
            <tr>
                <td>Overfitting Ratio (OOS/IS)</td>
                <td class="{"success" if report.overfitting_ratio > 0.3 else "warning" if report.overfitting_ratio > 0.1 else "danger"}">
                    {report.overfitting_ratio:.2%}
                </td>
            </tr>
            <tr>
                <td>In-Sample Sharpe</td>
                <td>{report.in_sample_sharpe:.2f}</td>
            </tr>
            <tr>
                <td>Out-of-Sample Sharpe</td>
                <td>{report.out_of_sample_sharpe:.2f}</td>
            </tr>
            <tr>
                <td>In-Sample MDD</td>
                <td>{report.in_sample_mdd:.2%}</td>
            </tr>
            <tr>
                <td>Out-of-Sample MDD</td>
                <td>{report.out_of_sample_mdd:.2%}</td>
            </tr>
        </table>
        """)

        # 세부 구간 테이블
        html_parts.append("""
        <h2>Detailed Results by Segment</h2>
        <table>
            <tr>
                <th>Period</th>
                <th>IS Return</th>
                <th>OOS Return</th>
                <th>OOS/IS Ratio</th>
                <th>Overfitting</th>
            </tr>
        """)

        for _i, seg in enumerate(report.segments):
            if seg.in_sample_result and seg.out_of_sample_result:
                html_parts.append(f"""
                <tr>
                    <td>{seg.train_start.date()} ~ {seg.test_end.date()}</td>
                    <td>{seg.in_sample_result.total_return:.2%}</td>
                    <td>{seg.out_of_sample_result.total_return:.2%}</td>
                    <td>{seg.oos_is_ratio:.2%}</td>
                    <td class="{"success" if seg.oos_is_ratio > 0.3 else "warning" if seg.oos_is_ratio > 0.1 else "danger"}">
                        {seg.overfitting_severity}
                    </td>
                </tr>
                """)

        html_parts.append("""
        </table>
        </body>
        </html>
        """)

        return "".join(html_parts)


if __name__ == "__main__":
    # 테스트 코드
    print("WalkForwardAnalyzer module loaded successfully")
