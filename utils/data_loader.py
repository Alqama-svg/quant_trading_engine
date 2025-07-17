#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


# In[3]:


def load_price_data(ticker, start_date, end_date, column='Close'):
    """
    Downloads historical price data for a given ticker.

    Args:
        ticker (str): Stock or asset symbol.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        column (str): Which price column to use (default 'Adj Close').

    Returns:
        pd.DataFrame: DataFrame with a single 'price' column.
    """
    df = yf.download(ticker, start=start_date, end=end_date)[[column]].copy()
    df.columns = ['price']
    df.dropna(inplace=True)
    return df


# In[5]:


def load_multiple_prices(tickers, start_date, end_date):
    """
    Loads adjusted close prices for multiple tickers.

    Returns:
        pd.DataFrame: Combined DataFrame with tickers as columns.
    """
    data = {}
    for ticker in tickers:
        df = load_price_data(ticker, start_date, end_date)
        data[ticker] = df['price']
    return pd.DataFrame(data).dropna()

