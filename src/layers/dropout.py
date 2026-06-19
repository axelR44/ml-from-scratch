import numpy as np

class Dropout:
    def __init__(self, p=0.5):
        self.p = p

    def forward(self, X, training=True):
        if not training:
            return X

        #garde même échelle des activations
        self.mask = (np.random.rand(*X.shape) > self.p) / (1 - self.p)
        return X * self.mask

    def backward(self, dA):
        return dA * self.mask