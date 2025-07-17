#!/usr/bin/env python
# coding: utf-8

# In[25]:

import yfinance as yf
import pandas as pd

class BacktestEngine:
    def __init__(self, ticker, start_date, end_date, strategy, initial_cash=100_000):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.results = None
        self._prepare_data()  # correctly calls the method

    def _prepare_data(self):
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date, auto_adjust=False)

        if 'Adj Close' in df.columns:
            df = df[['Adj Close']].copy()
        else:
            df = df[['Close']].copy()

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
        if self.results is not None:
            self.results[['equity_curve', 'buy_hold']].plot(
                figsize=(12, 6), title=f"{self.ticker} Strategy Backtest"
            )






