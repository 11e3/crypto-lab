-- Staging: Trading Logs
-- ======================
-- Clean and normalize trade logs from CSV files.
--
-- Source: logs/*/trades.csv
-- Grain: One row per trade

{{ config(
    materialized='view',
    tags=['staging', 'trades']
) }}

with source as (
    -- Read from CSV files
    select
        *,
        regexp_extract(filename, '/([^/]+)/trades\.csv$', 1) as account_name
    from read_csv_auto('../logs/*/trades.csv', filename=true, header=true)
),

cleaned as (
    select
        -- Trade identification
        account_name,
        "timestamp" as trade_timestamp,
        date as trade_date,

        -- Trade details
        upper(action) as trade_action,  -- BUY or SELL
        upper(symbol) as symbol,
        cast(price as double) as price,
        cast(quantity as double) as quantity,
        cast(amount as double) as amount_krw,

        -- P&L (only for SELL trades)
        cast(nullif(profit_pct, '') as double) as profit_pct,
        cast(nullif(profit_krw, '') as double) as profit_krw,

        -- Metadata
        current_timestamp as _loaded_at

    from source
    where "timestamp" is not null
)

select * from cleaned
