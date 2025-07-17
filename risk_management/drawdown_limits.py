#!/usr/bin/env python
# coding: utf-8

# In[3]:
def apply_drawdown_limit(df, max_drawdown = 0.1):
    equity = 100_000 * (1 + df['strategy']).cumprod()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak

    df['drawdown'] = drawdown

    df['strategy_dd_limited'] = df['strategy']
    df.loc[df['drawdown'] < -max_drawdown, 'strategy_dd_limited'] = 0
    df['equity_curve'] = 100_000 * (1 + df['strategy_dd_limited']).cumprod()
    return df