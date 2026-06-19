import numpy as np

class Adam:
    def __init__(self, model, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.model = model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.t = 0  # timestep

        # stocker m et v pour chaque paramètre
        self.m = []
        self.v = []
        self.biggest_norm = 0

        # init
        for layer in self.model.layers:
            if hasattr(layer, "W"):
                self.m.append(np.zeros_like(layer.W))
                self.v.append(np.zeros_like(layer.W))

                self.m.append(np.zeros_like(layer.b))
                self.v.append(np.zeros_like(layer.b))

    def step(self, clip_norm = None):
        self.t += 1
        idx = 0

        if clip_norm is not None:
            self.clip(clip_norm)

        for layer in self.model.layers:
            if hasattr(layer, "W"):

                # ===== W =====
                g = layer.dW

                self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * g
                self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (g ** 2)

                # bias correction
                m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
                v_hat = self.v[idx] / (1 - self.beta2 ** self.t)

                layer.W -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

                idx += 1

                # ===== b =====
                g = layer.db

                self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * g
                self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (g ** 2)

                m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
                v_hat = self.v[idx] / (1 - self.beta2 ** self.t)

                layer.b -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

                idx += 1

    def clip(self, max_norm):

        total_norm = 0

        for layer in self.model.layers:
            if hasattr(layer, "dW"):
                total_norm += np.sum(layer.dW ** 2)
                total_norm += np.sum(layer.db ** 2)

        total_norm = np.sqrt(total_norm)
        if total_norm>= self.biggest_norm:
            self.biggest_norm = total_norm
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)

            for layer in self.model.layers:
                if hasattr(layer, "dW"):
                    layer.dW *= scale
                    layer.db *= scale