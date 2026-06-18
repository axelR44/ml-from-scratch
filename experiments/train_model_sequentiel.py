from src.models.sequential import Model
from src.layers.dense import Dense
from src.layers.activation import ReLU, Sigmoid
from src.losses.MSE import MSE

import numpy as np
import matplotlib.pyplot as plt

# dataset
X = np.linspace(0,60,2000).reshape(-1, 1)
y = 2*X+8

# model
model = Model([
    Dense(1, 50),
    ReLU(),
    Dense(50,20),
    ReLU(),
    Dense(20,5),
    ReLU(),
    Dense(5, 1),
], verbose=100)

# loss
loss = MSE()

# train
model.fit(X, y, loss, lr=0.1, epochs=10000)
plt.plot(np.log(model.losses))
plt.show()

X_test = np.linspace(70,90,200).reshape(-1, 1)
y_test = 2*X_test+8
y_pred = model.predict(X_test)
plt.plot(X_test, y_pred, label = "prediction")
plt.plot(X_test, y_test, label = 'reel')
plt.legend()
plt.show()