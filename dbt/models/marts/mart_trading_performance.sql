-- Mart: Trading Performance
-- ==========================
-- Aggregated trading performance metrics by account and symbol.
--
-- Use cases:
-- - Dashboard KPIs
-- - Performance comparison
-- - Risk analysis

{{ config(
    materialized='table',
    tags=['marts', 'trading', 'performance']
) }}

with trade_pnl as (
    select * from {{ ref('int_trade_pnl') }}
    where position_status = 'CLOSED'
),

-- Per-symbol performance
symbol_performance as (
    select
        account_name,
        symbol,

        -- Trade counts
        count(*) as total_trades,
        sum(case when profit_pct > 0 then 1 else 0 end) as winning_trades,
        sum(case when profit_pct <= 0 then 1 else 0 end) as losing_trades,

        -- Win rate
        sum(case when profit_pct > 0 then 1 else 0 end)::float / nullif(count(*), 0) as win_rate,

        -- Returns
        sum(profit_krw) as total_profit_krw,
        avg(profit_pct) as avg_profit_pct,
        max(profit_pct) as best_trade_pct,
        min(profit_pct) as worst_trade_pct,

        -- Risk metrics
        stddev(profit_pct) as profit_std,
        avg(case when profit_pct > 0 then profit_pct end) as avg_win_pct,
        avg(case when profit_pct <= 0 then profit_pct end) as avg_loss_pct,

        -- Holding period
        avg(holding_days) as avg_holding_days,
        max(holding_days) as max_holding_days,

        -- Time range
        min(entry_date) as first_trade_date,
        max(exit_date) as last_trade_date

    from trade_pnl
    group by account_name, symbol
),

-- Calculate additional metrics
with_ratios as (
    select
        *,

        -- Profit factor
        abs(avg_win_pct * winning_trades) / nullif(abs(avg_loss_pct * losing_trades), 0) as profit_factor,

        -- Expectancy
        (win_rate * avg_win_pct) + ((1 - win_rate) * coalesce(avg_loss_pct, 0)) as expectancy,

        -- Sharpe-like ratio (simplified)
        avg_profit_pct / nullif(profit_std, 0) as risk_adjusted_return

    from symbol_performance
),

-- Account totals
account_totals as (
    select
        account_name,
        'ALL' as symbol,

        sum(total_trades) as total_trades,
        sum(winning_trades) as winning_trades,
        sum(losing_trades) as losing_trades,
        sum(winning_trades)::float / nullif(sum(total_trades), 0) as win_rate,

        sum(total_profit_krw) as total_profit_krw,
        avg(avg_profit_pct) as avg_profit_pct,
        max(best_trade_pct) as best_trade_pct,
        min(worst_trade_pct) as worst_trade_pct,

        avg(profit_std) as profit_std,
        avg(avg_win_pct) as avg_win_pct,
        avg(avg_loss_pct) as avg_loss_pct,

        avg(avg_holding_days) as avg_holding_days,
        max(max_holding_days) as max_holding_days,

        min(first_trade_date) as first_trade_date,
        max(last_trade_date) as last_trade_date,

        null as profit_factor,
        null as expectancy,
        null as risk_adjusted_return

    from with_ratios
    group by account_name
),

final as (
    select *, current_timestamp as _updated_at
    from with_ratios

    union all

    select *, current_timestamp as _updated_at
    from account_totals
)

select * from final
