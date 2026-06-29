from src.models.sequential import Model
from src.layers.dense import Dense
from src.layers.activation import ReLU, Sigmoid
from src.layers.dropout import Dropout
from src.layers.batchnorm import BatchNorm
from src.losses.cross_entropy import CrossEntropy
import numpy as np
from src.utils.mnist_loader import load_mnist
import matplotlib.pyplot as plt


def accuracy(y_true, logits):
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == y_true)

model = Model([
    Dense(784, 128),
    BatchNorm(128),
    ReLU(),
    Dropout(0.3),

    Dense(128, 64),
    ReLU(),

    Dense(64, 10) 
])

model = model.load('model_mnsi')


X_train, y_train, X_test, y_test = load_mnist()



X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784) 

y_pred = model.predict(X_train)
preds = np.argmax(y_pred, axis=1)

print(np.bincount(preds))
print(model.evaluate(X_test, y_test, CrossEntropy()))
wrong = np.where(preds != y_train)[0]
print(accuracy(y_train, y_pred))
for i in range(20):
    img = X_train[i].reshape(28,28)

    plt.imshow(img, cmap="gray")
    plt.title(f"Pred: {preds[i]} / True: {y_train[i]}")
    plt.show()

for i in range(20):
    idx = wrong[i]
    img = X_train[idx].reshape(28,28)
    
    plt.imshow(img, cmap="gray")
    plt.title(f"Pred: {preds[idx]} / True: {y_train[idx]}")
    plt.show()