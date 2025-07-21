#!/usr/bin/env python
# coding: utf-8

# In[7]:

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Strategy:
    def __init__(self, window=20):
        self.window = window

    def generate_signals(self, df):
        raise NotImplementedError("generate_signals() must be implemented by the subclass")

class MeanReversionStrategy(Strategy):
    def __init__(self, window=20, z_entry=1.0, z_exit=0.0):
        super().__init__(window)
        self.z_entry = z_entry
        self.z_exit = z_exit

    def generate_signals(self, df):
        df['mean'] = df['price'].rolling(self.window).mean()
        df['std'] = df['price'].rolling(self.window).std()
        df['z_score'] = (df['price'] - df['mean']) / df['std']
        df['position'] = 0
        df.loc[df['z_score'] < -self.z_entry, 'position'] = 1
        df.loc[df['z_score'] > self.z_entry, 'position'] = -1
        df.loc[abs(df['z_score']) < self.z_exit, 'position'] = 0
        return df

class BackTestEngine:
    def __init__(self, ticker, start_date, end_date, strategy, initial_cash=100_000):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.results = None
        self._prepare_data()

    def _prepare_data(self):
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date)[['Close']]
        df.columns = ['price']
        df.dropna(inplace=True)
        df = self.strategy.generate_signals(df)
        df['returns'] = df['price'].pct_change()
        df['strategy'] = df['position'].shift(1) * df['returns']
        df.dropna(inplace=True)
        df['equity_curve'] = self.initial_cash * (1 + df['strategy']).cumprod()
        df['buy_hold'] = self.initial_cash * (1 + df['returns']).cumprod()
        self.results = df

    def plot_performance(self):
        if self.results is None:
            st.write("No results to plot.")
            return
        fig, ax = plt.subplots(figsize=(12, 6))
        self.results[['equity_curve', 'buy_hold']].plot(ax=ax, title=f"{self.ticker} Strategy Backtest")
        st.pyplot(fig)


# In[ ]:




