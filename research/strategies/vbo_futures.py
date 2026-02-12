"""VBO Futures Strategy -- BTC Long-Only.

BULL 레짐에서만 롱 진입, BEAR 레짐에서는 현금 보유.
켈리 기준 자금관리 + 확신도 기반 레버리지.

비용 모델: taker 0.05% + slippage 0.03% = 0.08% per leg
펀딩비: 8시간마다 0.01% (포지션 보유 시 차감)

검증 결과 (2022-2024, base 설정):
  CAGR 25.8%, MDD -10.0%, Sharpe 1.09, 58 trades

Usage:
    from strategies.vbo_futures import VBOFuturesStrategy
    s = VBOFuturesStrategy()
    metrics, eq_df, trades = s.backtest()
    s.plot_equity(eq_df, trades)
    s.parameter_sweep()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


# ──────────────────────────────────────────────
# Trade record
# ──────────────────────────────────────────────
@dataclass
class Trade:
    entry_date: object  # pd.Timestamp
    exit_date: object
    direction: str  # LONG
    entry_price: float
    exit_price: float
    leverage: float
    position_frac: float  # notional / equity
    pnl_pct: float  # PnL as % of equity (after fees)
    pnl_amount: float
    exit_reason: str  # signal / stop_loss / end_of_period


# ──────────────────────────────────────────────
# Strategy
# ──────────────────────────────────────────────
class VBOFuturesStrategy:
    """BTC Long-Only VBO with Kelly position sizing + confidence leverage.

    Parameters:
    ───────────
    Signal:
      ma_short (5)       : 단기 MA 기간 -- 트렌드 판단
      btc_ma (20)        : BTC MA 기간 -- 시장 레짐 판단 (BULL/BEAR)
      noise_ratio (0.5)  : 변동성 돌파 계수 -- target = open + (prev_H - prev_L) * ratio

    Risk:
      atr_period (14)    : ATR 계산 기간
      atr_stop_mult (2.0): 스탑로스 = entry - ATR * mult
      max_leverage (3.0) : 최대 레버리지 cap

    Kelly:
      kelly_fraction (0.5)    : Kelly 안전계수 (0.5 = half-Kelly)
      kelly_lookback (30)     : Kelly 계산에 필요한 최소 과거 트레이드 수
      min_kelly (0.05)        : Kelly 하한 (최소 5% 리스크)
      default_risk_frac (0.1) : Kelly 데이터 부족 시 기본 리스크 비율

    Cost:
      fee (0.0005)        : taker 수수료 0.05%
      slippage (0.0003)   : 슬리피지 0.03%
      funding_rate (0.0001): 펀딩비 0.01% per 8h (일 3회)

    Entry:
      1. 레짐 = BULL (prev_close > prev_btc_ma)
      2. 트렌드 확인 (prev_close > prev_ma)
      3. VBO 돌파 (high >= target_long)

    Exit (any):
      1. 스탑로스 (low <= stop)
      2. 트렌드 반전 (prev_close < prev_ma)
      3. 레짐 전환 (BEAR)

    Position sizing:
      risk_budget = equity * kelly * confidence
      notional    = risk_budget / stop_distance_pct
      leverage    = notional / equity (capped)

    Confidence (0-1):
      60% 레짐 강도 + 40% 트렌드 강도, ATR 정규화
    """

    def __init__(
        self,
        # Signal
        ma_short: int = 5,
        btc_ma: int = 20,
        noise_ratio: float = 0.5,
        # Risk
        atr_period: int = 14,
        atr_stop_mult: float = 2.0,
        max_leverage: float = 3.0,
        # Kelly
        kelly_fraction: float = 0.5,
        kelly_lookback: int = 30,
        min_kelly: float = 0.05,
        default_risk_frac: float = 0.1,
        # Cost
        fee: float = 0.0005,
        slippage: float = 0.0003,
        funding_rate: float = 0.0001,
        # Capital
        initial_capital: float = 1_000_000,
    ):
        self.ma_short = ma_short
        self.btc_ma = btc_ma
        self.noise_ratio = noise_ratio
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.max_leverage = max_leverage
        self.kelly_fraction = kelly_fraction
        self.kelly_lookback = kelly_lookback
        self.min_kelly = min_kelly
        self.default_risk_frac = default_risk_frac
        self.fee = fee
        self.slippage = slippage
        self.funding_rate = funding_rate
        self.initial_capital = initial_capital

    # ── Data ──────────────────────────────────
    @staticmethod
    def load_data(symbol: str = "BTC") -> pd.DataFrame:
        path = DATA_DIR / f"{symbol}USDT_1d.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No data: {path}")
        return pd.read_parquet(path)

    def _prepare(
        self, df: pd.DataFrame, start: str | None, end: str | None
    ) -> pd.DataFrame:
        df = df.copy()
        if start:
            df = df[df.index >= start]
        if end:
            df = df[df.index <= end]

        # Moving averages
        df["ma"] = df["close"].rolling(self.ma_short).mean()
        df["btc_ma_val"] = df["close"].rolling(self.btc_ma).mean()

        # ATR
        high_low = df["high"] - df["low"]
        high_prev = (df["high"] - df["close"].shift(1)).abs()
        low_prev = (df["low"] - df["close"].shift(1)).abs()
        df["tr"] = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
        df["atr"] = df["tr"].rolling(self.atr_period).mean()

        # Previous-day values (look-ahead bias 방지)
        for col in ["close", "high", "low", "ma", "btc_ma_val", "atr"]:
            df[f"prev_{col}"] = df[col].shift(1)

        # Regime
        df["regime"] = np.where(
            df["prev_close"] > df["prev_btc_ma_val"], "BULL", "BEAR"
        )

        # VBO target
        vol = df["prev_high"] - df["prev_low"]
        df["target_long"] = df["open"] + vol * self.noise_ratio

        # Signal confidence (0-1)
        regime_str = (
            (df["prev_close"] - df["prev_btc_ma_val"]).abs()
            / df["prev_atr"].replace(0, np.nan)
        ).clip(0, 3) / 3

        trend_str = (
            (df["prev_close"] - df["prev_ma"]).abs()
            / df["prev_atr"].replace(0, np.nan)
        ).clip(0, 3) / 3

        df["confidence"] = (regime_str * 0.6 + trend_str * 0.4).clip(0, 1)
        return df

    # ── Kelly ─────────────────────────────────
    def _calc_kelly(self, trades: list[Trade]) -> float:
        """Rolling Kelly fraction from recent trades."""
        if len(trades) < self.kelly_lookback:
            return self.default_risk_frac

        recent = trades[-self.kelly_lookback :]
        wins = [t for t in recent if t.pnl_pct > 0]
        losses = [t for t in recent if t.pnl_pct <= 0]

        if not wins or not losses:
            return self.default_risk_frac

        win_rate = len(wins) / len(recent)
        avg_win = np.mean([t.pnl_pct for t in wins])
        avg_loss = abs(np.mean([t.pnl_pct for t in losses]))

        if avg_loss == 0:
            return self.default_risk_frac

        r = avg_win / avg_loss
        kelly = win_rate - (1 - win_rate) / r
        kelly = max(self.min_kelly, min(kelly * self.kelly_fraction, 1.0))
        return kelly

    # ── Position sizing ───────────────────────
    def _size_position(
        self,
        equity: float,
        kelly: float,
        confidence: float,
        entry_price: float,
        stop_price: float,
    ) -> tuple[float, float]:
        """Kelly + confidence -> position notional & leverage.

        risk_budget = equity * kelly * confidence
        notional    = risk_budget / stop_distance_pct
        leverage    = notional / equity (capped at max_leverage)
        """
        stop_pct = abs(entry_price - stop_price) / entry_price
        if stop_pct < 1e-8:
            return 0.0, 0.0

        adj_conf = max(confidence, 0.3)
        risk_budget = equity * kelly * adj_conf

        notional = risk_budget / stop_pct
        leverage = notional / equity

        if leverage > self.max_leverage:
            leverage = self.max_leverage
            notional = equity * leverage

        return round(notional, 2), round(leverage, 2)

    # ── Backtest ──────────────────────────────
    def backtest(
        self,
        start: str = "2022-01-01",
        end: str = "2024-12-31",
    ) -> tuple[dict, pd.DataFrame, list[Trade]]:
        df = self.load_data("BTC")
        df = self._prepare(df, start, end)

        equity = self.initial_capital
        pos: dict | None = None
        trades: list[Trade] = []
        eq_curve: list[dict] = []

        for date, row in df.iterrows():
            if pd.isna(row.get("prev_atr")) or pd.isna(row.get("prev_btc_ma_val")):
                eq_curve.append({"date": date, "equity": equity})
                continue

            # ── Exit ─────────────────────────
            if pos is not None:
                exit_price = None
                exit_reason = ""

                if row["low"] <= pos["stop"]:
                    exit_price = pos["stop"] * (1 - self.slippage)
                    exit_reason = "stop_loss"
                elif (
                    row["prev_close"] < row["prev_ma"]
                    or row["regime"] == "BEAR"
                ):
                    exit_price = row["open"] * (1 - self.slippage)
                    exit_reason = "signal"

                if exit_price is not None:
                    raw_pnl = pos["notional"] * (exit_price - pos["entry"]) / pos["entry"]
                    fee_cost = pos["notional"] * self.fee * 2
                    hold_days = max((date - pos["entry_date"]).days, 1)
                    funding_cost = pos["notional"] * self.funding_rate * 3 * hold_days
                    net_pnl = raw_pnl - fee_cost - funding_cost
                    pnl_pct = net_pnl / equity * 100 if equity > 0 else 0

                    equity += net_pnl
                    equity = max(equity, 0)

                    trades.append(Trade(
                        entry_date=pos["entry_date"],
                        exit_date=date,
                        direction="LONG",
                        entry_price=pos["entry"],
                        exit_price=exit_price,
                        leverage=pos["lev"],
                        position_frac=pos["notional"] / (equity + abs(net_pnl)) if equity > 0 else 0,
                        pnl_pct=pnl_pct,
                        pnl_amount=net_pnl,
                        exit_reason=exit_reason,
                    ))
                    pos = None

                    if equity <= 0:
                        eq_curve.append({"date": date, "equity": 0})
                        break

            # ── Entry (BULL only) ────────────
            if pos is None and equity > 0 and row["regime"] == "BULL":
                if (
                    row["high"] >= row["target_long"]
                    and row["prev_close"] > row["prev_ma"]
                ):
                    kelly = self._calc_kelly(trades)
                    conf = row["confidence"] if not pd.isna(row["confidence"]) else 0
                    entry = max(row["target_long"], row["open"]) * (1 + self.slippage)
                    stop = entry - row["prev_atr"] * self.atr_stop_mult
                    notional, lev = self._size_position(equity, kelly, conf, entry, stop)
                    if notional > 0:
                        pos = {
                            "entry_date": date,
                            "entry": entry,
                            "stop": stop,
                            "lev": lev,
                            "notional": notional,
                        }

            eq_curve.append({"date": date, "equity": equity})

        # ── Close open position at period end ─
        if pos is not None and equity > 0:
            last = df.iloc[-1]
            exit_price = last["close"] * (1 - self.slippage)
            raw_pnl = pos["notional"] * (exit_price - pos["entry"]) / pos["entry"]
            fee_cost = pos["notional"] * self.fee * 2
            hold_days = max((df.index[-1] - pos["entry_date"]).days, 1)
            funding_cost = pos["notional"] * self.funding_rate * 3 * hold_days
            net_pnl = raw_pnl - fee_cost - funding_cost
            pnl_pct = net_pnl / equity * 100 if equity > 0 else 0
            equity += net_pnl

            trades.append(Trade(
                entry_date=pos["entry_date"],
                exit_date=df.index[-1],
                direction="LONG",
                entry_price=pos["entry"],
                exit_price=exit_price,
                leverage=pos["lev"],
                position_frac=pos["notional"] / equity if equity > 0 else 0,
                pnl_pct=pnl_pct,
                pnl_amount=net_pnl,
                exit_reason="end_of_period",
            ))

        # ── Build results ─────────────────────
        eq_df = pd.DataFrame(eq_curve).set_index("date")
        metrics = self._calc_metrics(eq_df, trades)
        return metrics, eq_df, trades

    # ── Metrics ───────────────────────────────
    def _calc_metrics(self, eq_df: pd.DataFrame, trades: list[Trade]) -> dict:
        if len(eq_df) == 0:
            return {}

        initial = eq_df["equity"].iloc[0]
        final = eq_df["equity"].iloc[-1]
        days = (eq_df.index[-1] - eq_df.index[0]).days
        years = days / 365.25

        total_ret = (final / initial - 1) * 100
        cagr = (pow(final / initial, 1 / years) - 1) * 100 if years > 0 else 0

        running_max = eq_df["equity"].expanding().max()
        mdd = ((eq_df["equity"] / running_max - 1) * 100).min()

        daily_ret = eq_df["equity"].pct_change().dropna()
        sharpe = (
            (daily_ret.mean() / daily_ret.std()) * np.sqrt(365)
            if len(daily_ret) > 0 and daily_ret.std() > 0
            else 0.0
        )

        wins = [t for t in trades if t.pnl_pct > 0]
        losses = [t for t in trades if t.pnl_pct <= 0]
        stop_exits = [t for t in trades if t.exit_reason == "stop_loss"]
        avg_lev = np.mean([t.leverage for t in trades]) if trades else 0

        return {
            "period": f"{eq_df.index[0].date()} ~ {eq_df.index[-1].date()}",
            "total_return_pct": round(total_ret, 2),
            "cagr_pct": round(cagr, 2),
            "mdd_pct": round(mdd, 2),
            "sharpe": round(sharpe, 2),
            "final_equity": round(final),
            "total_trades": len(trades),
            "win_rate_pct": round(len(wins) / len(trades) * 100, 1) if trades else 0,
            "avg_win_pct": round(np.mean([t.pnl_pct for t in wins]), 2) if wins else 0,
            "avg_loss_pct": round(np.mean([t.pnl_pct for t in losses]), 2) if losses else 0,
            "stop_loss_exits": len(stop_exits),
            "avg_leverage": round(avg_lev, 2),
            "avg_hold_days": (
                round(np.mean([(t.exit_date - t.entry_date).days for t in trades]), 1)
                if trades else 0
            ),
        }

    # ── Visualization ─────────────────────────
    def plot_equity(
        self, eq_df: pd.DataFrame, trades: list[Trade], save_path: str | None = None
    ) -> None:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # 1) Equity curve
        ax = axes[0]
        ax.plot(eq_df.index, eq_df["equity"], "k-", linewidth=1)
        ax.set_ylabel("Equity")
        ax.set_title("VBO Futures (Long-Only) -- Equity Curve")
        ax.grid(True, alpha=0.3)

        for t in trades:
            color = "green" if t.pnl_pct > 0 else "red"
            if t.entry_date in eq_df.index:
                eq_val = eq_df.loc[t.entry_date, "equity"]
                ax.scatter(t.entry_date, eq_val, color=color, marker="^", s=30, zorder=5)

        # 2) Drawdown
        ax = axes[1]
        running_max = eq_df["equity"].expanding().max()
        dd = (eq_df["equity"] / running_max - 1) * 100
        ax.fill_between(eq_df.index, dd, 0, alpha=0.4, color="red")
        ax.set_ylabel("Drawdown %")
        ax.grid(True, alpha=0.3)

        # 3) Leverage per trade
        ax = axes[2]
        colors = ["green" if t.pnl_pct > 0 else "red" for t in trades]
        ax.bar([t.entry_date for t in trades], [t.leverage for t in trades],
               color=colors, alpha=0.6, width=2)
        ax.set_ylabel("Leverage")
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved: {save_path}")
        plt.show()

    # ── Parameter sweep ───────────────────────
    def parameter_sweep(
        self,
        start: str = "2022-01-01",
        end: str = "2024-12-31",
    ) -> pd.DataFrame:
        results = []
        for noise in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for atr_mult in [1.5, 2.0, 2.5, 3.0]:
                for max_lev in [2.0, 3.0, 5.0]:
                    s = VBOFuturesStrategy(
                        noise_ratio=noise,
                        atr_stop_mult=atr_mult,
                        max_leverage=max_lev,
                        kelly_fraction=self.kelly_fraction,
                        kelly_lookback=self.kelly_lookback,
                        default_risk_frac=self.default_risk_frac,
                        fee=self.fee,
                        slippage=self.slippage,
                        funding_rate=self.funding_rate,
                    )
                    m, _, _ = s.backtest(start, end)
                    results.append({
                        "noise_ratio": noise,
                        "atr_stop_mult": atr_mult,
                        "max_leverage": max_lev,
                        **m,
                    })

        df = pd.DataFrame(results)
        df = df.sort_values("sharpe", ascending=False)
        return df


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    s = VBOFuturesStrategy()
    metrics, eq_df, trades = s.backtest()

    print("=" * 70)
    print("VBO Futures (Long-Only) -- BTC (2022-2024)")
    print("=" * 70)
    for k, v in metrics.items():
        print(f"  {k:>25s}: {v}")
    print("=" * 70)

    print(f"\n  Last 10 trades:")
    for t in trades[-10:]:
        print(
            f"    {t.entry_date.date()} -> {t.exit_date.date()} "
            f"{t.leverage:.1f}x PnL={t.pnl_pct:+.2f}% ({t.exit_reason})"
        )

    s.plot_equity(eq_df, trades)
