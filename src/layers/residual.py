import numpy as np
from src.layers.layer import Layer

class Residual(Layer):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers  # le bloc F

    def set_rng(self, rng):
        # propager le RNG aux sous-couches (comme le fait ton Model)
        child = np.random.SeedSequence(rng.integers(1e9)).spawn(len(self.layers))
        for layer, seq in zip(self.layers, child):
            if hasattr(layer, "set_rng"):
                layer.set_rng(np.random.default_rng(seq))

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        self.F = out
        return X + out           # le skip

    def backward(self, dY):
        grad = dY
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return dY + grad          # dY (chemin direct) + dY·F'(x) (chemin bloc)