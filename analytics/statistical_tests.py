#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


# In[2]:


ticker = 'AAPL'
df = yf.download(ticker, start='2020-01-01', end='2024-12-31')[['Close']]
df.columns = ['price']

window = 50
df['ma'] = df['price'].rolling(50).mean()
df['position'] = np.where(df['price'] > df['ma'], 1, -1)
df['returns'] = df['price'].pct_change()
df['strategy'] = df['position'].shift(1) * df['returns']
df.dropna(inplace=True)


# In[14]:


#t-Test

from scipy import stats

class StrategyAnalytics:
    def __init__(self, df, return_col='strategy', risk_free_rate=0.0):
        self.df = df.copy()
        self.returns = self.df[return_col]
        self.rfr = risk_free_rate

    def total_return(self):
        return (1 + self.returns).prod() - 1

    def annualized_return(self):
        return (1 + self.total_return()) ** (252 / len(self.returns)) - 1

    def annualized_volatility(self):
        return self.returns.std() * np.sqrt(252)

    def sharpe_ratio(self):
        excess_return = self.returns - self.rfr / 252
        return excess_return.mean() / excess_return.std() * np.sqrt(252)

    def max_drawdown(self):
        cum_returns = (1 + self.returns).cumprod()
        peak = cum_returns.cummax()
        drawdown = (cum_returns - peak) / peak
        return drawdown.min()

    def calmar_ratio(self):
        dd = abs(self.max_drawdown())
        return self.annualized_return() / dd if dd != 0 else np.nan

    def t_test(self):
        t_stat, p_value = stats.ttest_1samp(self.returns.dropna(), 0)
        return round(t_stat, 3), round(p_value, 4)

    def is_significant(self, alpha=0.05):
        _, p = self.t_test()
        return p < alpha

    def summary(self):
        return {
            "Total Return (%)": round(100 * self.total_return(), 2),
            "Annualized Return (%)": round(100 * self.annualized_return(), 2),
            "Annualized Volatility (%)": round(100 * self.annualized_volatility(), 2),
            "Sharpe Ratio": round(self.sharpe_ratio(), 2),
            "Max Drawdown (%)": round(100 * self.max_drawdown(), 2),
            "Calmar Ratio": round(self.calmar_ratio(), 2)
        }


# In[16]:


analytics = StrategyAnalytics(df)


# In[18]:


t, p = analytics.t_test()
print(f"t-stat: {t}, p-value: {p}")
print("Statistically significant?", "Yes" if analytics.is_significant() else "No")


# In[20]:


# Bootstrapping addition
def bootstrap_returns(returns, n_bootstraps=1000):
    observed_mean = returns.mean()
    sample_means = []

    for _ in range(n_bootstraps):
        sample = returns.sample(frac=1, replace=True)
        sample_means.append(sample.mean())

    p_value = np.mean([1 if abs(m) >= abs(observed_mean) else 0 for m in sample_means])
    return round(p_value, 4)


# In[22]:


boot_p = bootstrap_returns(df['strategy'])
print(f"Bootstrap p-value: {boot_p}")


# In[24]:


report = pd.DataFrame([analytics.summary()])
report['t-stat'], report['p-value'] = analytics.t_test()
report['Significant (p<0.05)'] = analytics.is_significant()
report.T  # Transpose for easier viewing


# In[ ]:




