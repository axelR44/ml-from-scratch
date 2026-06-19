import numpy as np

class ReLU:
    def forward(self, X):
        self.X = X 
        return np.maximum(0, X)

    def backward(self, dA):
        #next gradient multiplied by the derivative of ReLU
        return dA * (self.X > 0)
    
class Sigmoid:
    def forward(self, X):
        self.X = 1 / (1 + np.exp(-X))
        return self.X

    def backward(self, dA):
        return dA * self.X * (1 - self.X)