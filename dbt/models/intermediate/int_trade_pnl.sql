-- Intermediate: Trade P&L
-- ========================
-- Calculate P&L for each trade by matching BUY and SELL.
--
-- Logic:
-- - Match BUY with subsequent SELL for same symbol
-- - Calculate holding period and returns

{{ config(
    materialized='table',
    tags=['intermediate', 'trades', 'pnl']
) }}

with trades as (
    select * from {{ ref('stg_trades') }}
),

-- Number trades for matching
numbered_trades as (
    select
        *,
        row_number() over (
            partition by account_name, symbol, trade_action
            order by trade_timestamp
        ) as trade_seq

    from trades
),

-- Match BUY with SELL
matched_trades as (
    select
        b.account_name,
        b.symbol,

        -- Entry
        b.trade_timestamp as entry_timestamp,
        b.trade_date as entry_date,
        b.price as entry_price,
        b.quantity as entry_quantity,
        b.amount_krw as entry_amount,

        -- Exit
        s.trade_timestamp as exit_timestamp,
        s.trade_date as exit_date,
        s.price as exit_price,
        s.quantity as exit_quantity,
        s.amount_krw as exit_amount,

        -- P&L from trade log
        s.profit_pct,
        s.profit_krw,

        -- Calculated P&L
        (s.price - b.price) / nullif(b.price, 0) * 100 as calc_profit_pct,
        (s.price - b.price) * least(b.quantity, s.quantity) as calc_profit_krw,

        -- Holding period
        datediff('day', b.trade_timestamp, s.trade_timestamp) as holding_days

    from numbered_trades b
    inner join numbered_trades s
        on b.account_name = s.account_name
        and b.symbol = s.symbol
        and b.trade_action = 'BUY'
        and s.trade_action = 'SELL'
        and b.trade_seq = s.trade_seq
),

-- Add unrealized positions (BUY without matching SELL)
all_positions as (
    select
        account_name,
        symbol,
        entry_timestamp,
        entry_date,
        entry_price,
        entry_quantity,
        entry_amount,
        exit_timestamp,
        exit_date,
        exit_price,
        exit_quantity,
        exit_amount,
        coalesce(profit_pct, calc_profit_pct) as profit_pct,
        coalesce(profit_krw, calc_profit_krw) as profit_krw,
        holding_days,
        'CLOSED' as position_status,
        current_timestamp as _calculated_at

    from matched_trades

    union all

    -- Open positions
    select
        b.account_name,
        b.symbol,
        b.trade_timestamp as entry_timestamp,
        b.trade_date as entry_date,
        b.price as entry_price,
        b.quantity as entry_quantity,
        b.amount_krw as entry_amount,
        null as exit_timestamp,
        null as exit_date,
        null as exit_price,
        null as exit_quantity,
        null as exit_amount,
        null as profit_pct,
        null as profit_krw,
        null as holding_days,
        'OPEN' as position_status,
        current_timestamp as _calculated_at

    from numbered_trades b
    where b.trade_action = 'BUY'
      and not exists (
          select 1 from numbered_trades s
          where s.account_name = b.account_name
            and s.symbol = b.symbol
            and s.trade_action = 'SELL'
            and s.trade_seq = b.trade_seq
      )
)

select * from all_positions
