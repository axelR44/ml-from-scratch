import numpy as np

class MSE:
    def forward(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        return np.mean((y_true - y_pred) ** 2)

    def backward(self):
        n = self.y_true.shape[0]
        #derived from MSE
        return -2 * (self.y_true-self.y_pred) / n