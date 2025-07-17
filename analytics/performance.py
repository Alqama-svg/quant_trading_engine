#!/usr/bin/env python
# coding: utf-8

# In[1]:


# analytics/performance.py

import numpy as np
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





