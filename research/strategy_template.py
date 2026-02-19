"""[전략 이름] 리서치 스크립트 (바이낸스 선물).

[전략 설명]

결과: [PENDING]

Usage:
    python research/fetch_data.py                  # 먼저 데이터 수집
    python research/strategy_template.py           # 백테스트 실행
    python research/strategy_template.py --sweep   # 파라미터 스윕
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

# --- 수수료/슬리피지 (per leg) ---
TAKER_FEE = 0.0005  # 0.05% per leg
SLIPPAGE = 0.0003  # 0.03% per leg
COST_PER_LEG = TAKER_FEE + SLIPPAGE  # 0.08% per leg


# === 데이터 로드 ===


def load_data(interval: str = "1d") -> tuple[pd.DataFrame, pd.DataFrame]:
    """research/data/ 에서 모든 parquet을 읽어 (closes, volumes) 반환."""
    files = sorted(DATA_DIR.glob(f"*_{interval}.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {DATA_DIR} for interval={interval}")

    closes: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}
    for f in files:
        symbol = f.stem.replace(f"_{interval}", "")
        df = pd.read_parquet(f)
        closes[symbol] = df["close"]
        volumes[symbol] = df["close"] * df["volume"]  # USD 거래대금

    close_df = pd.DataFrame(closes).dropna(how="all").ffill()
    vol_df = pd.DataFrame(volumes).reindex(close_df.index).fillna(0)
    print(f"Loaded {len(close_df.columns)} symbols, {len(close_df)} rows "
          f"({close_df.index[0]} ~ {close_df.index[-1]})")
    return close_df, vol_df


# === 시그널 생성 ===


def generate_signals(
    closes: pd.DataFrame,
    # TODO: 전략 파라미터 추가
    # example_param: int = 20,
) -> pd.DataFrame:
    """TODO: 시그널 로직 구현.

    Args:
        closes: 종가 DataFrame (index=dates, columns=symbols)

    Returns:
        signals: +1 (롱), -1 (숏), 0 (플랫). closes와 동일 shape.
    """
    signals = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)

    # ===================================================================
    # 여기에 시그널 로직 구현
    # 예시:
    #   ma_fast = closes.rolling(10).mean()
    #   ma_slow = closes.rolling(40).mean()
    #   signals[ma_fast > ma_slow] = 1.0
    #   signals[ma_fast < ma_slow] = -1.0
    # ===================================================================

    return signals


# === 백테스트 엔진 ===


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    trades: int
    metrics: dict[str, float]


def backtest(
    closes: pd.DataFrame,
    signals: pd.DataFrame,
    leverage: float = 1.0,
    vol_target: float | None = None,
    vol_lookback: int = 20,
) -> BacktestResult:
    """시그널 기반 포트폴리오 백테스트.

    - 균등 배분 (시그널 있는 자산끼리 1/n)
    - 거래 비용 적용
    - vol_target: 변동성 타겟팅 (e.g., 0.15 = 연 15%). None이면 비활성.
    """
    daily_returns = closes.pct_change().fillna(0)

    # 포지션 가중치: 시그널 있는 자산끼리 균등 배분
    n_positions = signals.abs().sum(axis=1).replace(0, 1)
    weights = signals.div(n_positions, axis=0) * leverage

    # 변동성 타겟팅
    if vol_target is not None:
        port_ret = (weights.shift(1) * daily_returns).sum(axis=1)
        realized_vol = port_ret.rolling(vol_lookback).std() * np.sqrt(365)
        vol_scalar = (vol_target / realized_vol.clip(lower=0.01)).clip(upper=3.0)
        weights = weights.multiply(vol_scalar, axis=0)

    # 거래 비용
    turnover = weights.diff().abs().sum(axis=1).fillna(0)
    cost = turnover * COST_PER_LEG

    # 포트폴리오 수익률
    port_returns = (weights.shift(1) * daily_returns).sum(axis=1) - cost
    port_returns = port_returns.iloc[1:]

    equity = (1 + port_returns).cumprod()

    trades = int((weights.diff().abs() > 0).sum().sum())
    metrics = compute_metrics(port_returns, equity)

    return BacktestResult(
        equity=equity,
        returns=port_returns,
        positions=weights,
        trades=trades,
        metrics=metrics,
    )


# === 메트릭 ===


def compute_metrics(returns: pd.Series, equity: pd.Series) -> dict[str, float]:
    """핵심 성과 지표."""
    days = len(returns)
    years = days / 365

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    growth = max(1 + total_return, 1e-8)
    cagr = growth ** (1 / max(years, 0.01)) - 1

    vol = returns.std() * np.sqrt(365)
    sharpe = (returns.mean() * 365) / max(vol, 1e-8)

    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    mdd = drawdown.min()

    calmar = cagr / max(abs(mdd), 1e-8)

    win_rate = (returns > 0).sum() / max((returns != 0).sum(), 1)

    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / max(gross_loss, 1e-8)

    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "MDD": mdd,
        "Calmar": calmar,
        "Win Rate": win_rate,
        "Profit Factor": profit_factor,
        "Total Return": total_return,
        "Days": days,
        "Annualized Days": 365,
    }


# === 시각화 ===


SCRIPT_NAME = Path(__file__).stem


def plot_equity(result: BacktestResult, title: str = "Strategy") -> None:
    """에퀴티 커브 + 드로다운."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1, 1]})

    # 에퀴티 커브
    ax = axes[0]
    ax.plot(result.equity.index, result.equity.values, linewidth=1.2)
    ax.set_title(f"{title} — Equity Curve", fontsize=13)
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)

    # 드로다운
    ax = axes[1]
    peak = result.equity.cummax()
    dd = (result.equity - peak) / peak
    ax.fill_between(dd.index, dd.values, 0, alpha=0.4, color="red")
    ax.set_ylabel("Drawdown")
    ax.set_title("Underwater", fontsize=11)
    ax.grid(True, alpha=0.3)

    # 포지션 수
    ax = axes[2]
    n_pos = result.positions.abs().gt(0).sum(axis=1)
    ax.bar(n_pos.index, n_pos.values, width=1.5, alpha=0.5, color="steelblue")
    ax.set_ylabel("# Positions")
    ax.set_title("Active Positions", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(__file__).parent / f"{SCRIPT_NAME}_result.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def print_metrics(result: BacktestResult, params: dict) -> None:
    """결과 출력."""
    m = result.metrics
    sharpe = m["Sharpe"]
    mdd = m["MDD"]

    print("\n" + "=" * 60)
    print(f"{SCRIPT_NAME.upper()} BACKTEST RESULT")
    print("=" * 60)
    print(f"Parameters: {params}")
    print("-" * 40)
    print(f"  CAGR   : {m['CAGR']:+.2%}")
    print(f"  Sharpe : {sharpe:.4f}  {'PASS' if sharpe >= 1.0 else 'FAIL'}")
    print(f"  MDD    : {mdd:+.2%}    {'PASS' if mdd >= -0.30 else 'FAIL'}")
    print(f"  Calmar : {m['Calmar']:.4f}")
    print(f"  Vol    : {m['Volatility']:.2%}")
    print(f"  Trades : {result.trades}")

    # 연도별 수익률
    yearly = result.returns.groupby(result.returns.index.year).apply(  # type: ignore[attr-defined]
        lambda r: (1 + r).prod() - 1
    )
    print("\n--- Yearly Returns ---")
    for year, ret in yearly.items():
        print(f"  {year}: {ret:+.2%}")

    print("=" * 60)


# === 파라미터 스윕 ===


def parameter_sweep(closes: pd.DataFrame) -> pd.DataFrame:
    """주요 파라미터 조합 스윕.

    TODO: 전략에 맞게 파라미터 조합 수정
    """
    results = []

    # TODO: 전략 파라미터 조합 정의
    param_grid = [
        {"example_param": 10},
        {"example_param": 20},
        {"example_param": 40},
    ]

    print(f"\nRunning parameter sweep: {len(param_grid)} combinations")

    for params in param_grid:
        sigs = generate_signals(closes, **params)
        res = backtest(closes, sigs)
        results.append({
            **params,
            "Sharpe": res.metrics["Sharpe"],
            "CAGR": res.metrics["CAGR"],
            "MDD": res.metrics["MDD"],
            "Calmar": res.metrics["Calmar"],
            "Trades": res.trades,
        })

    df = pd.DataFrame(results).sort_values("Sharpe", ascending=False)
    print("\n--- Top 5 by Sharpe ---")
    print(df.head(5).to_string(index=False, float_format="{:.4f}".format))
    print("\n--- Top 5 by Calmar (Sharpe >= 0.8) ---")
    filtered = df[df["Sharpe"] >= 0.8].sort_values("Calmar", ascending=False)
    print(filtered.head(5).to_string(index=False, float_format="{:.4f}".format))
    return df


# === 메인 ===


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{SCRIPT_NAME} backtest")
    # TODO: 전략 파라미터 CLI 인자 추가
    # parser.add_argument("--example-param", type=int, default=20)
    parser.add_argument("--leverage", type=float, default=1.0, help="Leverage multiplier")
    parser.add_argument("--vol-target", type=float, default=None, help="Annualized vol target")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--interval", default="1d", help="Data interval")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to include")
    parser.add_argument("--exclude", nargs="+", default=None, help="Symbols to exclude")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot")
    args = parser.parse_args()

    closes, volumes = load_data(args.interval)

    # 유니버스 필터링
    if args.symbols:
        valid = [s for s in args.symbols if s in closes.columns]
        if not valid:
            print(f"No matching symbols. Available: {list(closes.columns)}")
            return
        closes = pd.DataFrame(closes[valid])
        volumes = pd.DataFrame(volumes[valid])
    if args.exclude:
        keep = [c for c in closes.columns if c not in args.exclude]
        closes = pd.DataFrame(closes[keep])
        volumes = pd.DataFrame(volumes[keep])
    print(f"Universe: {list(closes.columns)}")

    if args.sweep:
        parameter_sweep(closes)
        return

    # TODO: CLI 인자에서 파라미터 가져오기
    params = {
        "leverage": args.leverage,
    }

    signals = generate_signals(closes)
    result = backtest(closes, signals, args.leverage, vol_target=args.vol_target)
    params["vol_target"] = args.vol_target
    print_metrics(result, params)

    if not args.no_plot:
        plot_equity(result, f"{SCRIPT_NAME}")


if __name__ == "__main__":
    main()
