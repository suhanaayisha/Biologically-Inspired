import numpy as np
from .activation import Sigmoid

class Layer:
    def __init__(self, no_of_layers, nodes_per_layer, activation):
        np.random.seed(42)
        size= (nodes_per_layer,no_of_layers)
        self.activation = activation
        self._weights = np.random.uniform(-1,1,size)
        self._bias = np.random.uniform(-1,1,nodes_per_layer)

    def setWeights(self, offset, params):
        n = self._weights.shape[0] 
        m = self._weights.shape[1] 
        
        step = n*m + n
        lparams = params[offset:offset+step]

        weights = lparams[:n*m]
        self._weights = np.reshape(weights,(n,-1))
        self._bias = lparams[-n:]
        return offset+step
      
    def getDimension(self):
        return (self._weights.shape[0] * self._weights.shape[1]) + self._bias.shape[0]

    def forward(self, inp):
        self._inp = inp

        weighted_sum = np.dot(self._weights,inp)  + self._bias
        output = self.activation.evaluate(weighted_sum)
        return output