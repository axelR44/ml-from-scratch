import numpy as np

class CrossEntropy:
    def forward(self, y_true, logits):
        # softmax
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exp / np.sum(exp, axis=1, keepdims=True)

        self.y_true = y_true
        N = y_true.shape[0]

        return -np.mean(np.log(self.probs[np.arange(N), y_true]+ 1e-9))

    def backward(self):
        N = self.y_true.shape[0]
        grad = self.probs.copy()
        grad[np.arange(N), self.y_true] -= 1
        return grad / N