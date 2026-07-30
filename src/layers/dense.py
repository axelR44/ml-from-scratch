import numpy as np
from src.layers.layer import Layer

class Dense(Layer):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.initialized = False
        self.rng = None

    def set_rng(self, rng):
            self.rng = rng

    def build(self, input_size):
            
        if self.rng is None:
            self.rng = np.random.default_rng()

        self.W = (self.rng.normal(size = (input_size, self.output_size)) * np.sqrt(2 / input_size))
        self.b = np.zeros((1, self.output_size))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.initialized = True

    def forward(self, X):
        self.X = X  # cache pour backward
        
        if not self.initialized:
            self.build(X.shape[1])

        return X @ self.W + self.b
    
    def backward(self, dZ):
        
        self.dW = (self.X.T @ dZ)
        self.db = np.sum(dZ, axis=0, keepdims=True)

        return dZ @ self.W.T 
    
    def parameters(self):
        if not self.initialized:
            return []
        return [
            {"param": self.W, "grad": self.dW},
            {"param": self.b, "grad": self.db},
        ]


