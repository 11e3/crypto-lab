-- Mart: Daily Metrics
-- ====================
-- Daily aggregated metrics for dashboard and reporting.
--
-- Includes:
-- - Price and volume metrics
-- - Feature values
-- - Trading activity

{{ config(
    materialized='table',
    tags=['marts', 'daily', 'metrics']
) }}

with features as (
    select * from {{ ref('int_daily_features') }}
),

trades as (
    select
        trade_date,
        symbol,
        count(*) as trade_count,
        sum(case when trade_action = 'BUY' then 1 else 0 end) as buy_count,
        sum(case when trade_action = 'SELL' then 1 else 0 end) as sell_count,
        sum(amount_krw) as total_volume_krw

    from {{ ref('stg_trades') }}
    group by trade_date, symbol
),

combined as (
    select
        f.symbol,
        f.trade_date,

        -- Price metrics
        f.open_price,
        f.high_price,
        f.low_price,
        f.close_price,
        f.volume,

        -- Returns
        f.return_1d,
        f.return_5d,
        f.return_20d,

        -- Technical indicators
        f.ma_5,
        f.ma_20,
        f.ma_60,
        f.ma_5_20_ratio,
        f.price_ma_20_ratio,
        f.volatility_20d,
        f.rsi_14,
        f.atr_14,
        f.atr_ratio,
        f.volume_ratio,
        f.ma_alignment,

        -- Trading activity
        coalesce(t.trade_count, 0) as trade_count,
        coalesce(t.buy_count, 0) as buy_count,
        coalesce(t.sell_count, 0) as sell_count,
        coalesce(t.total_volume_krw, 0) as trading_volume_krw,

        -- Market regime (simplified)
        case
            when f.return_20d > 0.02 and f.volatility_20d < 0.03 then 'BULL'
            when f.return_20d < -0.02 and f.volatility_20d < 0.03 then 'BEAR'
            when f.volatility_20d >= 0.03 then 'HIGH_VOL'
            else 'SIDEWAYS'
        end as market_regime,

        -- Signals
        case when f.ma_5 > f.ma_20 and f.rsi_14 < 70 then 1 else 0 end as bullish_signal,
        case when f.ma_5 < f.ma_20 and f.rsi_14 > 30 then 1 else 0 end as bearish_signal,

        f._calculated_at

    from features f
    left join trades t
        on f.symbol = t.symbol
        and f.trade_date = t.trade_date
)

select * from combined
