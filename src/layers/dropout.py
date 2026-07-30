import numpy as np
from src.layers.layer import Layer

class Dropout(Layer):
    
    #is used in tests to account for the random nature of the layer
    is_stochastic = True

    def __init__(self, p=0.5):
        super().__init__()

        
        
        self.p = p
        self.rng = None

    def set_rng(self, rng):
            self.rng = rng

    def forward(self, X):
        if not self.training:
            return X
        
        if self.rng is None:
            self.rng = np.random.default_rng()

        #garde même échelle des activations
        self.mask = (self.rng.random(X.shape) > self.p) / (1 - self.p)
        return X * self.mask

    def backward(self, dY):
        return dY * self.mask
