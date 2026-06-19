import numpy as np

class BatchNorm:
    def __init__(self, input_dim, eps=1e-5, momentum=0.9):
        self.eps = eps
        self.momentum = momentum

        # paramètres appris
        self.gamma = np.ones((1, input_dim))
        self.beta = np.zeros((1, input_dim))

        # stats pour inference
        self.running_mean = np.zeros((1, input_dim))
        self.running_var = np.ones((1, input_dim))

    def forward(self, X, training=True):
        if training:
            self.X = X

            self.mean = np.mean(X, axis=0, keepdims=True)
            self.var = np.var(X, axis=0, keepdims=True)

            self.X_norm = (X - self.mean) / np.sqrt(self.var + self.eps)

            # update running stats
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var

            return self.gamma * self.X_norm + self.beta

        else:
            X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.eps)
            return self.gamma * X_norm + self.beta
        
    def backward(self, dY):
        N = self.X.shape[0]

        dX_norm = dY * self.gamma

        dvar = np.sum(dX_norm * (self.X - self.mean) * -0.5 * (self.var + self.eps)**(-1.5), axis=0, keepdims=True)
        dmean = np.sum(dX_norm * -1 / np.sqrt(self.var + self.eps), axis=0, keepdims=True) + dvar * np.mean(-2 * (self.X - self.mean), axis=0, keepdims=True)

        dX = dX_norm / np.sqrt(self.var + self.eps) + dvar * 2 * (self.X - self.mean) / N + dmean / N

        self.dgamma = np.sum(dY * self.X_norm, axis=0, keepdims=True)
        self.dbeta = np.sum(dY, axis=0, keepdims=True)

        return dX