import numpy as np

class Neuron:
    
    """A conceptual class that has never been used. 
    It is preferable to use dense layers to optimize computation."""

    def __init__(self, input_size, activation):
        self.w = np.random.randn(input_size) * 0.01
        self.b = 0.0
        self.activation = activation

    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return self.activation(z)
