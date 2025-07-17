#!/usr/bin/env python
# coding: utf-8

# In[29]:


import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

from strategies.base import Strategy

class PairsTradingStrategy(Strategy): 
    def __init__(self, window = 30, z_entry = 1.0, z_exit = 0.0):
        self.window = window
        self.z_entry = z_entry
        self.z_exit = z_exit

    def generate_signals(self, df1, df2):
        spread = df1['price'] - df2['price']
        mean = spread.rolling(self.window).mean()
        std = spread.rolling(self.window).std()
        z_score = (spread - mean) / std

        position = pd.Series(0, index=df1.index)
        position[z_score > self.z_entry] = -1  # Short spread
        position[z_score < -self.z_entry] = 1  # Long spread
        position[abs(z_score) < self.z_exit] = 0  # Close position

        df = pd.DataFrame({
            'spread': spread,
            'z_score': z_score,
            'position': position,
            'returns': (df1['price'] - df1['price'].shift(1)) - (df2['price'] - df2['price'].shift(1))
        })
        df['strategy'] = df['position'].shift(1) * df['returns']
        df.dropna(inplace=True)
        return df