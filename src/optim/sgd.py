import numpy as np

class SGD:
    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr

    def step(self,  clip_norm = None):
        if clip_norm is not None:
            self.clip(clip_norm)
        
        for layer in self.model.layers:
            if hasattr(layer, "W") and hasattr(layer, "dW"):
                layer.W -= self.lr * layer.dW

            if hasattr(layer, "b") and hasattr(layer, "db"):
                layer.b -= self.lr * layer.db


    def zero_grad(self):
        for p in self.parameters:
            p["grad"] = 0

    def clip(self, max_norm):

        total_norm = 0

        for layer in self.model.layers:
            if hasattr(layer, "dW"):
                total_norm += np.sum(layer.dW ** 2)
                total_norm += np.sum(layer.db ** 2)

        total_norm = np.sqrt(total_norm)
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)

            for layer in self.model.layers:
                if hasattr(layer, "dW"):
                    layer.dW *= scale
                    layer.db *= scale
                if hasattr(layer, "gamma"):
                    layer.gamma -= self.lr * layer.dgamma
                    layer.beta -= self.lr * layer.dbeta