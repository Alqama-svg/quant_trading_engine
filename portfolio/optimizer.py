#!/usr/bin/env python
# coding: utf-8

# In[20]:


import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


# In[21]:


tickers = ['AAPL', 'MSFT', 'GOOGl', 'NVDA']
returns_df = pd.DataFrame()

for ticker in tickers:
    df = yf.download(ticker, start = '2020-01-01', end = '2024-12-31')[['Close']]
    df.columns = ['ticker']
    df = df.pct_change()
    returns_df = pd.concat([returns_df, df], axis = 1)

returns_df.dropna(inplace=True)
returns_df.head()


# In[42]:


# mean variance optimizer using CVXPY
import cvxpy as cp

def optimize_portfolio(returns, risk_aversion = 1.0):
    mu = returns.mean().values         # Convert to NumPy array
    cov = returns.cov().values         # Convert to NumPy array
    n = len(mu)

    w = cp.Variable(n)
    objective  = cp.Maximize(mu.T @ w - risk_aversion * cp.quad_form(w, cov))
    constraints = [cp.sum(w) == 1, w >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver = cp.SCS)

    return pd.Series(w.value, index = returns.columns)

def equal_weight_portfolio(returns):
    n = returns.shape[1]
    return pd.Series([1/n] * n, index = returns.columns) 


# In[44]:


opt_weights = optimize_portfolio(returns_df, risk_aversion = 0.5)
print(opt_weights)


# In[46]:


portfolio_returns = (returns_df * opt_weights).sum(axis=1)
equity_curve = 100_000 * (1 + portfolio_returns).cumprod()

plt.figure(figsize=(12, 6))
plt.plot(equity_curve, label='Optimized Portfolio')
plt.title("Optimized Portfolio Equity Curve")
plt.legend()
plt.show()


# In[ ]:




