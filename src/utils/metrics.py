import numpy as np

def accuracy(y_true, logits):
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == y_true)
