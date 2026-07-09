import numpy as np

class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.rng = None

    def set_rng(self, rng):
            self.rng = rng

    def forward(self, X, training=True):
        if not training:
            return X
        
        if self.rng is None:
            self.rng = np.random.default_rng()

        #garde même échelle des activations
        self.mask = (self.rng.random(*X.shape) > self.p) / (1 - self.p)
        return X * self.mask

    def backward(self, dA):
        return dA * self.mask
    
    def count_params(self):
        return 0