-- Staging: Backtest Results
-- ==========================
-- Clean and normalize backtest result files.
--
-- Source: data/backtest_results/*.json
-- Grain: One row per backtest run per symbol

{{ config(
    materialized='view',
    tags=['staging', 'backtest']
) }}

with source as (
    -- Read from JSON files using DuckDB's read_json
    select
        *,
        regexp_extract(filename, 'backtest_results_(\d{8})\.json$', 1) as run_date
    from read_json_auto('../data/backtest_results/*.json', filename=true)
),

flattened as (
    select
        run_date,
        unnest(results) as result
    from source
),

cleaned as (
    select
        -- Run identification
        run_date,
        result.symbol as symbol,
        result.strategy as strategy,

        -- Performance metrics
        cast(result.total_return as double) as total_return,
        cast(result.buy_hold_return as double) as buy_hold_return,
        cast(result.sharpe_ratio as double) as sharpe_ratio,
        cast(result.max_drawdown as double) as max_drawdown,
        cast(result.cagr as double) as cagr,

        -- Trade statistics
        cast(result.trade_count as int) as trade_count,
        cast(result.data_rows as int) as data_rows,

        -- Metadata
        result.backtest_date as backtest_timestamp,
        current_timestamp as _loaded_at

    from flattened
    where result.status = 'success'
)

select * from cleaned
