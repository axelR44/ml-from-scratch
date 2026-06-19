from src.models.sequential import Model
from src.layers.dense import Dense
from src.layers.activation import ReLU, Sigmoid
from src.losses.MSE import MSE
from src.layers.dropout import Dropout


import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(0,60,2000).reshape(-1, 1)
y = 2*X+8

X_test = np.linspace(20,90,200).reshape(-1, 1)
y_test = 2*X_test+8

# model
model = Model([
    Dense(1, 50),
    ReLU(),
    Dense(50,20),
    ReLU(),
    Dense(20,5),
    ReLU(),
    Dense(5, 1),
])

loss = MSE()

model.fit(X, y, X_test, y_test, loss, lr=0.0001, epochs=500, patience=300, save_path='model1', optimizer_name="Adam", scheduler_name="WCOS")
plt.plot(np.log(model.val_losses))
plt.show()
model2 = model.load('model1')

y_pred = model2.predict(X_test)
plt.plot(X_test, y_pred, label = "prediction")
plt.plot(X_test, y_test, label = 'reel')
plt.legend()
plt.show()