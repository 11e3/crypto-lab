"""Entry processing logic for vectorized backtest engine."""

from datetime import date

import numpy as np
import pandas as pd

from src.backtester.engine.position_sizer import calculate_position_sizes_for_entries
from src.backtester.engine.trade_simulator import (
    SimulationState,
    handle_normal_entry,
    handle_whipsaw,
)
from src.backtester.models import BacktestConfig
from src.risk.position_sizing import calculate_position_size


def process_entries(
    state: SimulationState,
    config: BacktestConfig,
    d_idx: int,
    current_date: date,
    sorted_dates: np.ndarray,
    tickers: list[str],
    n_tickers: int,
    arrays: dict[str, np.ndarray],
    valid_data: np.ndarray,
    ticker_historical_data: dict[str, pd.DataFrame],
) -> None:
    """Process entry signals for the current date."""
    not_in_position = state.position_amounts == 0
    can_enter = arrays["entry_signals"][:, d_idx] & not_in_position & valid_data

    if not np.any(can_enter):
        return

    candidate_idx = _sort_candidates_by_noise(can_enter, arrays["short_noises"], d_idx)

    position_sizes = calculate_position_sizes_for_entries(
        state,
        config,
        candidate_idx,
        tickers,
        arrays["entry_prices"],
        d_idx,
        current_date,
        ticker_historical_data,
    )

    for t_idx in candidate_idx:
        current_positions = int(np.sum(state.position_amounts > 0))
        available_slots = int(config.max_slots - current_positions)

        if available_slots <= 0:
            break

        buy_price = arrays["entry_prices"][t_idx, d_idx]

        invest_amount = calculate_invest_amount(
            state,
            tickers[t_idx],
            position_sizes,
            available_slots,
            buy_price,
            d_idx,
            ticker_historical_data,
            config,
        )

        # Use pre-computed whipsaw flag (entry_signal & exit_signal on same bar)
        # This represents vectorized signal computation where both entry and exit
        # conditions are met on the same bar. In VBO strategy:
        # - Entry: high >= target (intraday breakout)
        # - Exit: close < sma (end of day below SMA)
        # Both signals being true = same-day entry and exit
        is_whipsaw = bool(arrays["whipsaws"][t_idx, d_idx])

        if is_whipsaw:
            handle_whipsaw(
                state,
                t_idx,
                d_idx,
                current_date,
                tickers,
                arrays,
                invest_amount,
                buy_price,
                config.fee_rate,
                config.slippage_rate,
            )
        else:
            handle_normal_entry(state, t_idx, d_idx, invest_amount, buy_price, config.fee_rate)


def _sort_candidates_by_noise(
    can_enter: np.ndarray, short_noises: np.ndarray, d_idx: int
) -> np.ndarray:
    """Sort candidate indices by noise (lower noise = higher priority)."""
    candidate_idx = np.where(can_enter)[0]

    if np.any(~np.isnan(short_noises[candidate_idx, d_idx])):
        noise_values = short_noises[candidate_idx, d_idx]
        noise_values = np.where(np.isnan(noise_values), np.inf, noise_values)
        candidate_idx = candidate_idx[np.argsort(noise_values)]

    return candidate_idx


def calculate_invest_amount(
    state: SimulationState,
    ticker: str,
    position_sizes: dict[str, float],
    available_slots: int,
    buy_price: float,
    d_idx: int,
    ticker_historical_data: dict[str, pd.DataFrame],
    config: BacktestConfig,
) -> float:
    """Calculate investment amount for a single entry."""
    if config.position_sizing != "equal" and ticker in position_sizes:
        return position_sizes[ticker]

    if config.position_sizing != "equal" and ticker in ticker_historical_data:
        hist_df = ticker_historical_data[ticker]
        hist_up_to_date = (
            hist_df.iloc[: d_idx + 1]
            if isinstance(hist_df.index, pd.DatetimeIndex)
            else hist_df.iloc[: d_idx + 1]
        )
        return calculate_position_size(
            method=config.position_sizing,  # type: ignore[arg-type]
            available_cash=state.cash,
            available_slots=available_slots,
            ticker=ticker,
            current_price=buy_price,
            historical_data=hist_up_to_date,
            target_risk_pct=config.position_sizing_risk_pct,
            lookback_period=config.position_sizing_lookback,
        )

    return state.cash / available_slots
