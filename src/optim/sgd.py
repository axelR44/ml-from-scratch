import numpy as np

class SGD:
    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr

    def step(self,  clip_norm = None, lambda_l2=0.0):
        params = self.model.parameters()

        if clip_norm is not None:
            self.clip(params, clip_norm)

        for p in params:
            grad = p["grad"]
            if lambda_l2 > 0:
                grad = grad + lambda_l2 * p["param"]
            p["param"] -= self.lr * grad

    def zero_grad(self):
        for p in self.parameters:
            p["grad"] = 0

    def clip(self, params, max_norm):
            total_norm = np.sqrt(sum(np.sum(p["grad"] ** 2) for p in params))
            if total_norm > max_norm:
                scale = max_norm / (total_norm + 1e-6)
                for p in params:
                    p["grad"] *= scale

                    