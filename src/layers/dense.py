import numpy as np

class Dense:
    def __init__(self, output_size):
        #initialisation HE
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

        self.initialized = True

    def forward(self, X):
        self.X = X  # cache pour backward
        
        if not self.initialized:
                    self.build(X.shape[1])

        return X @ self.W + self.b
    
    def backward(self, dZ, lambda_l2=0.0):
        
        batch_size = self.X.shape[0]

        self.dW = (self.X.T @ dZ)
        self.db = np.sum(dZ, axis=0, keepdims=True)

        if lambda_l2 > 0:
            self.dW += lambda_l2 * self.W

        return dZ @ self.W.T 
    
    def parameters(self):
            return [
                {"param": self.W, "grad": self.dW},
                {"param": self.b, "grad": self.db}
            ]
