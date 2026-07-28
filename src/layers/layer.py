import numpy as np

class Layer:
    def __init__(self):
        self.training = True

    def forward(self, X):
        raise NotImplementedError

    def backward(self, dY):
        raise NotImplementedError

    def parameters(self):
        return []

    def count_params(self):
        return sum(p["param"].size for p in self.parameters())

    def set_rng(self, rng):
        pass

    def train(self):
        self.training = True

    def eval(self):
        self.training = False