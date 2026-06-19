import numpy as np

class Dense:
    def __init__(self, input_size, output_size):
        #initialisation HE
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.b = np.zeros((1,output_size))


    def forward(self, X):
        self.X = X  # cache pour backward
        return X @ self.W + self.b
    
    def backward(self, dZ, lambda_l2=0.0):
        batch_size = self.X.shape[0]

        self.dW = (self.X.T @ dZ) / batch_size
        self.db = np.sum(dZ, axis=0, keepdims=True) / batch_size

        if lambda_l2 > 0:
            self.dW += lambda_l2 * self.W

        return dZ @ self.W.T 
    
    def parameters(self):
            return [
                {"param": self.W, "grad": self.dW},
                {"param": self.b, "grad": self.db}
            ]

    
    

