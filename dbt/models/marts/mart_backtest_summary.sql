-- Mart: Backtest Summary
-- =======================
-- Aggregated backtest results for strategy comparison.
--
-- Use cases:
-- - Strategy selection
-- - Performance tracking over time
-- - A/B testing of strategies

{{ config(
    materialized='table',
    tags=['marts', 'backtest', 'summary']
) }}

with backtest_results as (
    select * from {{ ref('stg_backtest_results') }}
),

-- Latest results per symbol/strategy
latest_results as (
    select *
    from backtest_results
    qualify row_number() over (
        partition by symbol, strategy
        order by run_date desc
    ) = 1
),

-- Strategy performance summary
strategy_summary as (
    select
        strategy,

        -- Counts
        count(distinct symbol) as symbol_count,
        count(*) as total_runs,

        -- Performance averages
        avg(total_return) as avg_return,
        avg(sharpe_ratio) as avg_sharpe,
        avg(max_drawdown) as avg_mdd,
        avg(cagr) as avg_cagr,

        -- Best/Worst
        max(total_return) as best_return,
        min(total_return) as worst_return,
        max(sharpe_ratio) as best_sharpe,
        min(max_drawdown) as worst_mdd,

        -- Win rate (positive return)
        sum(case when total_return > 0 then 1 else 0 end)::float / count(*) as positive_return_rate,

        -- Outperformance vs buy-hold
        sum(case when total_return > buy_hold_return then 1 else 0 end)::float / count(*) as outperform_rate

    from latest_results
    group by strategy
),

-- Symbol performance by strategy
symbol_strategy as (
    select
        symbol,
        strategy,
        total_return,
        buy_hold_return,
        sharpe_ratio,
        max_drawdown,
        cagr,
        trade_count,
        run_date,

        -- Rank within strategy
        rank() over (partition by strategy order by sharpe_ratio desc) as sharpe_rank,
        rank() over (partition by strategy order by total_return desc) as return_rank

    from latest_results
),

final as (
    -- Strategy summary
    select
        'STRATEGY_SUMMARY' as record_type,
        strategy,
        null as symbol,
        null as total_return,
        null as buy_hold_return,
        avg_sharpe as sharpe_ratio,
        avg_mdd as max_drawdown,
        avg_cagr as cagr,
        null as trade_count,
        symbol_count,
        total_runs,
        avg_return,
        best_return,
        worst_return,
        positive_return_rate,
        outperform_rate,
        null as sharpe_rank,
        null as return_rank,
        current_timestamp as _updated_at

    from strategy_summary

    union all

    -- Symbol-level results
    select
        'SYMBOL_RESULT' as record_type,
        strategy,
        symbol,
        total_return,
        buy_hold_return,
        sharpe_ratio,
        max_drawdown,
        cagr,
        trade_count,
        null as symbol_count,
        null as total_runs,
        null as avg_return,
        null as best_return,
        null as worst_return,
        null as positive_return_rate,
        null as outperform_rate,
        sharpe_rank,
        return_rank,
        current_timestamp as _updated_at

    from symbol_strategy
)

select * from final
