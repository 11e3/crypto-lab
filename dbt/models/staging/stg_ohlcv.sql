-- Staging: OHLCV Data
-- ====================
-- Clean and normalize raw OHLCV data from parquet files.
--
-- Source: data/raw/day/*.parquet
-- Grain: One row per symbol per day

{{ config(
    materialized='view',
    tags=['staging', 'ohlcv']
) }}

with source as (
    -- Read from parquet files using DuckDB's read_parquet
    select
        *,
        regexp_extract(filename, '([A-Z]+)\.parquet$', 1) as symbol
    from read_parquet('../data/raw/day/*.parquet', filename=true)
),

cleaned as (
    select
        -- Primary key components
        symbol,
        "timestamp" as trade_date,

        -- OHLCV fields
        cast(open as double) as open_price,
        cast(high as double) as high_price,
        cast(low as double) as low_price,
        cast(close as double) as close_price,
        cast(volume as double) as volume,

        -- Derived fields
        cast(high as double) - cast(low as double) as price_range,
        cast(close as double) - cast(open as double) as price_change,

        -- Metadata
        current_timestamp as _loaded_at

    from source
    where "timestamp" is not null
      and cast(close as double) > 0
      and cast(volume as double) >= 0
)

select * from cleaned
