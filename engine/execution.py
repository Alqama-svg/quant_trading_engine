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


# engine/execution.py

class ExecutionSimulator:
    def __init__(self, slippage=0.0005, fee=0.0005):
        self.slippage = slippage
        self.fee = fee

    def adjust_returns(self, df):
        df = df.copy()
        df['trades'] = df['position'].diff().abs().fillna(0)
        df['net_strategy'] = df['strategy'] - df['trades'] * (self.slippage + self.fee)
        df['equity_curve'] = 100_000 * (1 + df['net_strategy']).cumprod()
        return df


# In[ ]:




