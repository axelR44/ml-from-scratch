import numpy as np

class BatchNorm2D:

    def __init__(self, eps=1e-5, momentum=0.9):
        self.eps = eps
        self.momentum = momentum
        
        self.initialized = False

    def build(self, in_channel):
        self.gamma = np.ones((1, in_channel, 1, 1))
        self.beta = np.zeros((1, in_channel, 1, 1))
        self.running_mean = np.zeros((1, in_channel, 1, 1))
        self.running_var = np.ones((1, in_channel, 1, 1))
        self.initialized = True

    
    def forward(self, X, training=True):
        if not self.initialized:
            self.build(X.shape[1])

        self.X = X

        if training:

            self.mean = np.mean(X,axis=(0,2,3), keepdims=True)

            self.var = np.var(X, axis=(0,2,3), keepdims=True)

            self.X_centered = X - self.mean
            self.std_inv = (1 / np.sqrt(self.var + self.eps))
            self.X_norm = (self.X_centered * self.std_inv)
            self.running_mean = (self.momentum * self.running_mean + (1-self.momentum) * self.mean)
            self.running_var = (self.momentum * self.running_var + (1-self.momentum) * self.var)

            return (self.gamma * self.X_norm + self.beta)
        else:

            X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.eps)
            return (self.gamma * X_norm+ self.beta)
        
    def backward(self, dY):
        N = (self.X.shape[0]*self.X.shape[2]*self.X.shape[3])

        dX_norm = dY * self.gamma

        dvar = np.sum(dX_norm * (self.X - self.mean) * -0.5 * (self.var + self.eps)**(-1.5), axis=(0,2,3), keepdims=True)
        dmean = np.sum(dX_norm * -1 / np.sqrt(self.var + self.eps), axis=(0,2,3), keepdims=True) + dvar * np.mean(-2 * (self.X - self.mean), axis=(0,2,3), keepdims=True)

        dX = dX_norm / np.sqrt(self.var + self.eps) + dvar * 2 * (self.X - self.mean) / N + dmean / N

        self.dgamma = np.sum(dY * self.X_norm, axis=(0,2,3), keepdims=True)
        self.dbeta = np.sum(dY, axis=(0,2,3), keepdims=True)

        return dX
    
    
    def count_params(self):
        if not self.initialized:
            return 0

        return self.gamma.size + self.beta.si
