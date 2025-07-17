#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
sys.path.append(os.path.abspath(".."))


# In[3]:


import sys
sys.path.append(".")


# In[13]:


import unittest
from strategies.mean_reversion import MeanReversionStrategy
from engine.backtest import BacktestEngine

class TestBacktestEngine(unittest.TestCase):
    def test_equity_curve_positive(self):
        strat = MeanReversionStrategy()
        bt = BacktestEngine('AAPL', '2020-01-01', '2024-12-31', strat)
        equity = bt.results['equity_curve']
        self.assertGreater(equity.iloc[-1], 0)

suite = unittest.TestLoader().loadTestsFromTestCase(TestBacktestEngine)
unittest.TextTestRunner(verbosity=2).run(suite)


# In[ ]:




