from src.models.sequential import Model
from src.layers.dense import Dense
from src.layers.activation import ReLU, Sigmoid
from src.layers.conv2d import Conv2D
from src.layers.flatten import Flatten
from src.layers.maxpool import MaxPool2D
from src.losses.cross_entropy import CrossEntropy
import numpy as np
from src.utils.mnist_loader import load_mnist
import matplotlib.pyplot as plt

from src.utils.metrics import accuracy


#image of 28*28, we use a padding of 3

H_conv = 28 - 3 + 1  # 26
W_conv = 28 - 3 + 1  # 26

H_pool = H_conv // 2  # 13
W_pool = W_conv // 2  # 13


model = Model([
    Conv2D(1, 8, 3),
    ReLU(),
    MaxPool2D(),
    Flatten(),
    Dense(8*H_pool*W_pool, 64),
    ReLU(),
    Dense(64, 10)
])
model.load('best_model')

X_train, y_train, X_test, y_test = load_mnist()

#image shape
X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)

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