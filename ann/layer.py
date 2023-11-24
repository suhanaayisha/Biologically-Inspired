import numpy as np

class Layer:
    def __init__(self, nb_inputs, nb_nodes, activation):
        np.random.seed(42)
        size= (nb_nodes,nb_inputs)
        self._w = np.random.uniform(-1,1,size)
        self._b = np.random.uniform(-1,1,nb_nodes)
        self.activation = activation
        # self.weights = np.random.rand(no_nodes, input_size)
        # self.bias = np.random.rand(1,no_nodes)
        self.x=[]

    def setWeights(self, offset, params):
        n = self._w.shape[0] 
        m = self._w.shape[1] 
        
        step = n*m + n
        lparams = params[offset:offset+step]

        weights = lparams[:n*m]
        self._w = np.reshape(weights,(n,-1))
        self._b = lparams[-n:]
        return offset+step
      
    def getDimension(self):
        return (self._w.shape[0] * self._w.shape[1]) + self._b.shape[0]

    def forward(self, xin):
        self._xin = xin

        weighted_sum = np.dot(self._w,xin)  + self._b
        output = self.activation.evaluate(weighted_sum)
        return output