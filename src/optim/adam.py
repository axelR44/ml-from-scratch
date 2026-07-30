import numpy as np

class Adam:
    def __init__(self, model, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.model = model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.biggest_norm = np.inf
        self.eps = eps
        self.t = 0
        self.m = None      # init différée : les couches sont lazy
        self.v = None

    def step(self, clip_norm=None, lambda_l2=0.0):
        params = self.model.parameters()

        # init paresseuse : au premier step, les params existent enfin
        if self.m is None:
            self.m = [np.zeros_like(p["param"]) for p in params]
            self.v = [np.zeros_like(p["param"]) for p in params]

        if clip_norm is not None:
            self.clip(params, clip_norm)

        self.t += 1

        for i, p in enumerate(params):
            grad = p["grad"]
            if lambda_l2 > 0:
                grad = grad + lambda_l2 * p["param"]

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p["param"] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def clip(self, params, max_norm):
        total_norm = np.sqrt(sum(np.sum(p["grad"] ** 2) for p in params))
        if total_norm >= self.biggest_norm:
            self.biggest_norm = total_norm
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)
            for p in params:
                p["grad"] *= scale