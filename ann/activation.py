import numpy as np

class Activation:
    def evaluate(self, x):
        pass

class Sigmoid(Activation):
    def evaluate(self, x):
        return 1 / (1 + np.exp(-x))
    
class Tanh(Activation):
    def evaluate(self, x):
        sigmoid = Sigmoid()
        return 2 * sigmoid.evaluate(2*x)-1

class Relu(Activation):
    def evaluate(selx,x):
        return np.maximum(0,x)