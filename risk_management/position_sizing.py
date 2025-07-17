#!/usr/bin/env python
# coding: utf-8

# In[10]:
def apply_position_sizing(df, risk_pct = 0.2, initial_capital = 100_000):
    df.copy()
    
    # Adjusts returns by allocating only a fraction of capital per trade.
    equity = [initial_capital]

    for i in range(1, len(df)):
    
        # Uses previous equity to calculate position size
        prev_equity = equity[-1]
        position_size = prev_equity * risk_pct
        position = df['position'].shift(1).iloc[i]
        daily_return = df['returns'].iloc[i]

        pnl = position_size * position * daily_return
        new_equity = prev_equity + pnl
        equity.append(new_equity)
    
    df['equity_curve'] = equity
    return df