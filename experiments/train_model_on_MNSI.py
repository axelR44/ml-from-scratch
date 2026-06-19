from src.models.sequential import Model
from src.layers.dense import Dense
from src.layers.activation import ReLU, Sigmoid
from src.layers.dropout import Dropout
from src.layers.batchnorm import BatchNorm
from src.losses.cross_entropy import CrossEntropy
import numpy as np
from src.utils.mnist_loader import load_mnist
import matplotlib.pyplot as plt

X_train, y_train, X_test, y_test = load_mnist()



X_train = X_train.reshape(-1, 784) 
X_test = X_test.reshape(-1, 784) 

model = Model([
    Dense(784, 128),
    BatchNorm(128),
    ReLU(),
    Dropout(0.2),

    Dense(128, 64),
    ReLU(),

    Dense(64, 10) 
])

model.fit(
    X_train,
    y_train,
    X_test,
    y_test,
    loss_fn=CrossEntropy(),
    lr=0.001,
    epochs=50,
    optimizer_name="Adam",
    scheduler_name="WCOS",
    save_path='model_mnsi'
)



y_pred = model.predict(X_test)
preds = np.argmax(y_pred, axis=1)

wrong = np.where(preds != y_test)[0]

for i in range(5):
    idx = wrong[i]
    img = X_test[idx].reshape(28,28)
    
    plt.imshow(img, cmap="gray")
    plt.title(f"Pred: {preds[idx]} / True: {y_test[idx]}")
    plt.show()