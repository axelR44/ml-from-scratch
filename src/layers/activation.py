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
    
class LeakyReLU:
    def __init__(self, alpha = 0.01):
        self.alpha = alpha
    def forward(self, X):
        self.X = X
        return np.where(X > 0, X, self.alpha * X)
    
    def backward(self, dY):
            grad = np.where(self.X > 0, 1.0, self.alpha)
            return dY * grad
