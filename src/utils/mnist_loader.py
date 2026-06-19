import gzip
import numpy as np
import os
import gzip
import numpy as np

def load_mnist():

    def load_images(file):
        with gzip.open(file, 'rb') as f:
            f.read(16)
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(-1, 784) / 255.0

    def load_labels(file):
        with gzip.open(file, 'rb') as f:
            f.read(8)
            return np.frombuffer(f.read(), dtype=np.uint8)

    X_train = load_images("data/train-images-idx3-ubyte.gz")
    y_train = load_labels("data/train-labels-idx1-ubyte.gz")

    X_test = load_images("data/t10k-images-idx3-ubyte.gz")
    y_test = load_labels("data/t10k-labels-idx1-ubyte.gz")

    return X_train, y_train, X_test, y_test
