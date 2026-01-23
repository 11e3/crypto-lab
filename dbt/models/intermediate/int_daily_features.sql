-- Intermediate: Daily Features
-- =============================
-- Calculate technical indicators from OHLCV data.
--
-- Features calculated:
-- - Returns (1d, 5d, 20d)
-- - Moving averages (5, 20, 60)
-- - Volatility
-- - RSI
-- - ATR

{{ config(
    materialized='table',
    tags=['intermediate', 'features']
) }}

with ohlcv as (
    select * from {{ ref('stg_ohlcv') }}
),

with_returns as (
    select
        symbol,
        trade_date,
        open_price,
        high_price,
        low_price,
        close_price,
        volume,
        price_range,

        -- Returns
        (close_price / lag(close_price, 1) over w - 1) as return_1d,
        (close_price / lag(close_price, 5) over w - 1) as return_5d,
        (close_price / lag(close_price, 20) over w - 1) as return_20d,

        -- Daily return for volatility
        (close_price / lag(close_price, 1) over w - 1) as daily_return

    from ohlcv
    window w as (partition by symbol order by trade_date)
),

with_ma as (
    select
        *,

        -- Moving averages
        avg(close_price) over (partition by symbol order by trade_date rows between 4 preceding and current row) as ma_5,
        avg(close_price) over (partition by symbol order by trade_date rows between 19 preceding and current row) as ma_20,
        avg(close_price) over (partition by symbol order by trade_date rows between 59 preceding and current row) as ma_60,

        -- Volume MA
        avg(volume) over (partition by symbol order by trade_date rows between 19 preceding and current row) as volume_ma_20

    from with_returns
),

with_volatility as (
    select
        *,

        -- Volatility (20-day rolling std of returns)
        stddev(daily_return) over (partition by symbol order by trade_date rows between 19 preceding and current row) as volatility_20d,

        -- ATR (14-day)
        avg(price_range) over (partition by symbol order by trade_date rows between 13 preceding and current row) as atr_14

    from with_ma
),

with_rsi as (
    select
        *,

        -- RSI calculation
        case
            when sum(case when daily_return > 0 then daily_return else 0 end) over w14 = 0 then 50
            when sum(case when daily_return < 0 then -daily_return else 0 end) over w14 = 0 then 100
            else 100 - (100 / (1 + (
                sum(case when daily_return > 0 then daily_return else 0 end) over w14 /
                nullif(sum(case when daily_return < 0 then -daily_return else 0 end) over w14, 0)
            )))
        end as rsi_14

    from with_volatility
    window w14 as (partition by symbol order by trade_date rows between 13 preceding and current row)
),

final as (
    select
        symbol,
        trade_date,
        open_price,
        high_price,
        low_price,
        close_price,
        volume,

        -- Returns
        return_1d,
        return_5d,
        return_20d,

        -- Moving averages
        ma_5,
        ma_20,
        ma_60,

        -- MA ratios
        ma_5 / nullif(ma_20, 0) as ma_5_20_ratio,
        close_price / nullif(ma_20, 0) as price_ma_20_ratio,

        -- Volatility
        volatility_20d,
        atr_14,
        atr_14 / nullif(close_price, 0) as atr_ratio,

        -- RSI
        rsi_14,

        -- Volume
        volume / nullif(volume_ma_20, 0) as volume_ratio,

        -- MA alignment (trend strength)
        case
            when ma_5 > ma_20 and ma_20 > ma_60 then 1.0
            when ma_5 > ma_20 or ma_20 > ma_60 then 0.5
            else 0.0
        end as ma_alignment,

        current_timestamp as _calculated_at

    from with_rsi
    where ma_60 is not null  -- Ensure enough history
)

select * from final
