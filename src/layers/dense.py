import numpy as np

class Dense:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size) * 0.01
        self.b = np.zeros(output_size)

    def forward(self, X):
        self.X = X  # cache pour backward
        return X @ self.W + self.b

    def backward(self, dZ):
        #weight gradient
        self.dW = (self.X.T @ dZ) / self.X.shape[0]

        #bias gradient
        self.db = np.sum(dZ, axis=0) / self.X.shape[0]

        return dZ @ self.W.T
    
    def step(self, lr):
            self.W -= lr * self.dW
            self.b -= lr * self.db
    
    

