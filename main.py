#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Core imports
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
import matplotlib.pyplot as plt
sns.set()


# In[6]:


# Local project modules
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from strategies.pairs_trading import PairsTradingStrategy

from engine.backtest import BacktestEngine
from engine.execution import ExecutionSimulator

from analytics.performance import StrategyAnalytics
from analytics.statistical_tests import bootstrap_returns

from portfolio.optimizer import optimize_portfolio, equal_weight_portfolio

from risk_management.stop_loss import apply_stop_loss
from risk_management.position_sizing import apply_position_sizing
from risk_management.drawdown_limits import apply_drawdown_limit


# In[7]:


strategy = MeanReversionStrategy(window=20, z_entry=1.5, z_exit=0.5)
bt = BacktestEngine('AAPL', '2020-01-01', '2024-12-31', strategy)
df = bt.results
df[['equity_curve', 'buy_hold']].plot(figsize=(12, 6), title="Mean Reversion: AAPL")


# In[8]:


exec_sim = ExecutionSimulator()
df_exec = exec_sim.adjust_returns(df)

analytics = StrategyAnalytics(df_exec, return_col='net_strategy')
summary = analytics.summary()
print(summary)

df_exec[['equity_curve', 'buy_hold']].plot(figsize=(12, 6), title="Mean Reversion Strategy: AAPL")


# In[10]:


tickers = ['AAPL', 'MSFT', 'GOOGL']
returns_df = pd.DataFrame()

for ticker in tickers:
    strat = MeanReversionStrategy()
    bt = BacktestEngine(ticker, '2020-01-01', '2024-12-31', strat)
    returns_df[ticker] = bt.results['strategy']

returns_df.dropna(inplace=True)

w_opt = optimize_portfolio(returns_df, risk_aversion=0.5)
print("Optimized Weights:", w_opt)


# In[11]:


# stop loss position sizing
df_sl = apply_stop_loss(df_exec, stop_loss_pct=0.02)
df_ps = apply_position_sizing(df_sl, risk_pct=0.02)

df_ps['equity_curve'].plot(figsize=(12, 6), title="With Stop Loss + Position Sizing")


# In[12]:


# Performance Analytics and Significance Testing
analytics = StrategyAnalytics(df_exec, return_col='net_strategy')
print(analytics.summary())
print("Significant?", analytics.is_significant())


# In[13]:


# Comparison of Strategies on One Asset (i.e., Mean reversion vs. momentum on the same stock.)
mr = BacktestEngine('MSFT', '2020-01-01', '2024-12-31', MeanReversionStrategy())
mo = BacktestEngine('MSFT', '2020-01-01', '2024-12-31', MomentumStrategy(window=50))

mr_curve = mr.results['equity_curve']
mo_curve = mo.results['equity_curve']

pd.DataFrame({'Mean Reversion': mr_curve, 'Momentum': mo_curve}).plot(figsize=(12, 6), title="Strategy Comparison: MSFT")


# In[35]:


import yfinance as yf
import pandas as pd

# Download and clean KO
ko = yf.download('KO', start='2020-01-01', end='2024-12-31', auto_adjust=False)[['Adj Close']].copy()
ko.columns = ['KO']

# Download and clean PEP
pep = yf.download('PEP', start='2020-01-01', end='2024-12-31', auto_adjust=False)[['Adj Close']].copy()
pep.columns = ['PEP']

# Merge and rename to flat structure
df_pair = pd.concat([ko, pep], axis=1).dropna()
df1 = pd.DataFrame({'price': df_pair['KO']})
df2 = pd.DataFrame({'price': df_pair['PEP']})


# In[37]:


from strategies.pairs_trading import PairsTradingStrategy

strategy = PairsTradingStrategy(window=30, z_entry=1.5, z_exit=0.5)
df_pairs = strategy.generate_signals(df1, df2)

df_pairs['equity_curve'] = 100_000 * (1 + df_pairs['strategy']).cumprod()
df_pairs['equity_curve'].plot(figsize=(12, 6), title="Pairs Trading: KO vs PEP")


# In[113]:


# Portfolio Optimization Across Assets
tickers = ['AAPL', 'MSFT', 'GOOGL']
returns_df = pd.DataFrame()

for ticker in tickers:
    strat = MeanReversionStrategy()
    bt = BacktestEngine(ticker, '2020-01-01', '2024-12-31', strat)
    returns_df[ticker] = bt.results['strategy']

returns_df.dropna(inplace=True)

w_opt = optimize_portfolio(returns_df, risk_aversion=0.5)
w_eq = equal_weight_portfolio(returns_df)

opt_curve = 100_000 * (1 + (returns_df * w_opt).sum(axis=1)).cumprod()
eq_curve  = 100_000 * (1 + (returns_df * w_eq).sum(axis=1)).cumprod()

pd.DataFrame({'Optimized': opt_curve, 'Equal Weighted': eq_curve}).plot(figsize=(12, 6), title="Portfolio Comparison")


# In[115]:


# Final Summary Report
final_analytics = StrategyAnalytics(pd.DataFrame({'strategy': (returns_df * w_opt).sum(axis=1)}))
print("Optimized Portfolio Performance:")
print(final_analytics.summary())