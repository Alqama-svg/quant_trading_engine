# Core imports
import os
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
import matplotlib.pyplot as plt
sns.set()

# Create 'images' folder if not exists
os.makedirs("images", exist_ok=True)

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


# 1. Mean Reversion Strategy Backtest (AAPL)
strategy = MeanReversionStrategy(window=20, z_entry=1.5, z_exit=0.5)
bt = BacktestEngine('AAPL', '2020-01-01', '2024-12-31', strategy)
df = bt.results
df[['equity_curve', 'buy_hold']].plot(figsize=(12, 6), title="Mean Reversion: AAPL")
plt.savefig("images/mean_reversion_aapl.png")
plt.close()

# 2. Execution Simulation and Analytics
exec_sim = ExecutionSimulator()
df_exec = exec_sim.adjust_returns(df)

analytics = StrategyAnalytics(df_exec, return_col='net_strategy')
print(analytics.summary())

df_exec[['equity_curve', 'buy_hold']].plot(figsize=(12, 6), title="Execution Adjusted Strategy: AAPL")
plt.savefig("images/execution_adjusted.png")
plt.close()

# 3. Strategy Comparison: Mean Reversion vs. Momentum (MSFT)
mr = BacktestEngine('MSFT', '2020-01-01', '2024-12-31', MeanReversionStrategy())
mo = BacktestEngine('MSFT', '2020-01-01', '2024-12-31', MomentumStrategy(window=50))
mr_curve = mr.results['equity_curve']
mo_curve = mo.results['equity_curve']

pd.DataFrame({'Mean Reversion': mr_curve, 'Momentum': mo_curve}).plot(figsize=(12, 6), title="Strategy Comparison: MSFT")
plt.savefig("images/strategy_comparison.png")
plt.close()

# 4. Pairs Trading Strategy (KO vs. PEP)
ko = yf.download('KO', start='2020-01-01', end='2024-12-31')[['Adj Close']].copy()
pep = yf.download('PEP', start='2020-01-01', end='2024-12-31')[['Adj Close']].copy()
ko.columns = ['KO']
pep.columns = ['PEP']
df_pair = pd.concat([ko, pep], axis=1).dropna()
df1 = pd.DataFrame({'price': df_pair['KO']})
df2 = pd.DataFrame({'price': df_pair['PEP']})

strategy = PairsTradingStrategy(window=30, z_entry=1.5, z_exit=0.5)
df_pairs = strategy.generate_signals(df1, df2)
df_pairs['equity_curve'] = 100_000 * (1 + df_pairs['strategy']).cumprod()

df_pairs['equity_curve'].plot(figsize=(12, 6), title="Pairs Trading: KO vs PEP")
plt.savefig("images/pairs_trading.png")
plt.close()

# 5. Portfolio Optimization Across Assets
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
plt.savefig("images/portfolio_comparison.png")
plt.close()

# 6. Risk Management (Stop Loss + Position Sizing)
df_sl = apply_stop_loss(df_exec, stop_loss_pct=0.02)
df_ps = apply_position_sizing(df_sl, risk_pct=0.02)
df_ps['equity_curve'].plot(figsize=(12, 6), title="With Stop Loss + Position Sizing")
plt.savefig("images/risk_management.png")
plt.close()

# Final Portfolio Report
final_analytics = StrategyAnalytics(pd.DataFrame({'strategy': (returns_df * w_opt).sum(axis=1)}))
print("Optimized Portfolio Performance:")
print(final_analytics.summary())