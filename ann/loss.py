import numpy as np

class Loss:
    def evaluate(self, y, t):
        pass
    
class Mse (Loss):
    def evaluate(self, y,t):
        return 2*(t-y)**2

    
class Binary_cross_entropy(Loss):
    def evaluate(self, y, t):
        term0 = (1-t) * np.log(1-y + 1e-7)
        term1 = t * np.log(y + 1e-7)
        return -(term0 + term1)
