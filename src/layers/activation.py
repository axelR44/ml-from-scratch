import numpy as np
from src.layers.layer import Layer

class ReLU(Layer):
    def forward(self, X, training=True):
        self.X = X 
        return np.maximum(0, X)

    def backward(self, dA):
        #next gradient multiplied by the derivative of ReLU
        return dA * (self.X > 0)
    
class Sigmoid(Layer):
    def forward(self, X):
        self.X = 1 / (1 + np.exp(-X))
        return self.X

    def backward(self, dA):
        return dA * self.X * (1 - self.X)
    
class LeakyReLU(Layer):
    def __init__(self, alpha = 0.01):
        super().__init__()
        self.alpha = alpha
    def forward(self, X):
        self.X = X
        return np.where(X > 0, X, self.alpha * X)
    
    def backward(self, dY):
            grad = np.where(self.X > 0, 1.0, self.alpha)
            return dY * grad

class Softmax(Layer):
    def forward(self, X):
        # stabilité numérique : on soustrait le max par ligne (évite overflow)
        exp = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.probs = exp / np.sum(exp, axis=1, keepdims=True)
        return self.probs

    def backward(self, dY):
        # somme pondérée par ligne : <dY, p>
        dot = np.sum(dY * self.probs, axis=1, keepdims=True)
        return self.probs * (dY - dot)