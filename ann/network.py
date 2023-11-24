
def threshold(y):
    return 1 if y>= 0.5 else 0

class Network:
    def __init__(self): #initialise the empty list of layers
        self.layers = []
        
    def append(self,layer): #to append a layer to the network
        self.layers.append(layer)
    
    def getDimension(self):
        dim = 0
        for layer in self.layers:
            dim += layer.getDimension()
        return dim
    
    def setWeights(self, params):
        for layer in self.layers:
            offset=0
            offset += layer.setWeights(offset, params)

    def evaluate(self,x, y, loss_func):
        L=0
        acc = 0
        for i in range(x.shape[0]):
            y_hat, loss_val = self.forward(x[i],y[i], loss_func)
            L+= loss_val
            y_hat_01 = list(map(threshold, y_hat))
            acc +=1 if y_hat_01==y[i] else 0

        
        L = L / x.shape[0]
        acc = acc / x.shape[0]
        return acc, L
    

    def forward(self, x, y, loss_func):
        self.out = x
        for layer in self.layers:
            self.out = layer.forward(self.out)
        loss = loss_func.evaluate(self.out, y)
        return self.out, loss

