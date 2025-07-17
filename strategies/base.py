#!/usr/bin/env python
# coding: utf-8

# In[1]:


class Strategy:
    def generate_signals(self, df):
        raise NotImplementedError("generate_signals must be implemented by the strategy.")

