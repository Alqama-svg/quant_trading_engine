#!/usr/bin/env python
# coding: utf-8

# In[8]:


'''Quantitative Strategy Demo : This notebook demonstrates using the Mean Reversion and Momentum strategies on historical stock data'''


# In[12]:


import sys
import os
sys.path.append(os.path.abspath(".."))  # If notebook is inside /notebooks


# In[14]:


import sys
sys.path.append(".")


# In[20]:


from strategies.mean_reversion import MeanReversionStrategy
from engine.backtest import BacktestEngine
import matplotlib.pyplot as plt

bt = BacktestEngine('AAPL', '2020-01-01', '2024-12-31', MeanReversionStrategy())
bt.results['equity_curve'].plot(title="Mean Reversion: AAPL", figsize=(10, 5))
plt.show()


# In[ ]:




