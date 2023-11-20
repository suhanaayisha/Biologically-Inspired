#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math


# In[2]:


class Activation:
    def evaluate(self, x):
        pass

class Sigmoid(Activation):
    def evaluate(self, x):
        return 1 / (1 + np.exp(-x))
    
class Tanh(Activation):
    def evaluate(self, x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

class Relu(Activation):
    def evaluate(selx,x):
        return np.maximum(0,x)


# In[ ]:




