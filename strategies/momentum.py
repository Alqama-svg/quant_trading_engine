#!/usr/bin/env python
# coding: utf-8

# In[7]:


import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MomentumStrategy:
    def __init__(self, window=50):
        self.window = window

    def generate_signals(self, df):
        df['ma'] = df['price'].rolling(self.window).mean()
        df['position'] = np.where(df['price'] > df['ma'], 1, -1)
        return df

def run_backtest(ticker, strategy, start='2020-01-01', end='2024-12-31', initial_cash=100_000):
    df = yf.download(ticker, start=start, end=end)[['Close']].copy()
    df.columns = ['price']
    df.dropna(inplace=True)
    df = strategy.generate_signals(df)
    df['returns'] = df['price'].pct_change()
    df['strategy'] = df['position'].shift(1) * df['returns']
    df.dropna(inplace=True)
    df['equity_curve'] = initial_cash * (1 + df['strategy']).cumprod()
    df['buy_hold'] = initial_cash * (1 + df['returns']).cumprod()
    return df

def plot_performance(df, ticker, window):
    fig, ax = plt.subplots(figsize=(12, 6))
    df[['equity_curve', 'buy_hold']].plot(ax=ax)
    ax.set_title(f"Momentum Backtest: {ticker} ({window}-day MA)")
    ax.legend()
    st.pyplot(fig)


# In[ ]:




