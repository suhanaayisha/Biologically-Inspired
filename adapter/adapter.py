from pso.fitness import Fitness

class AdapterFunction(Fitness):
    def __init__(self, ann, x, y, loss_function):
        self.ann = ann
        self.x = x
        self.y = y
        self.loss = loss_function

    def getANN(self):
        return self.ann
    
    def getDimension(self):
        dim = self.ann.getDimension()
        return dim
    
    def setVariables(self, params):
        self.ann.updateWeights(params)
        # return Fitness().setVariables(params)
    
    def evaluate(self, params):
        self.ann.updateWeights(params)
        acc, loss = self.ann.evaluate(self.x, self.y, self.loss)
        return acc, loss
    
