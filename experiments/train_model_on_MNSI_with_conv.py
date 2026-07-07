from src.models.sequential import Model
from src.layers.dense import Dense
from src.layers.activation import ReLU, Sigmoid
from src.layers.conv2d import Conv2D
from src.layers.flatten import Flatten
from src.layers.dropout import Dropout
from src.layers.batchnorm import BatchNorm
from src.losses.cross_entropy import CrossEntropy
import numpy as np
from src.utils.mnist_loader import load_mnist
import matplotlib.pyplot as plt
from src.callback import *
from src.utils.metrics import accuracy

X_train, y_train, X_test, y_test = load_mnist()

#image shape
X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)
#image of 28*28, we use a padding of 3
H_out = 28 - 3 + 1
W_out = 28 - 3 + 1
model = Model([
    Conv2D(1, 8, 3),
    ReLU(),
    Flatten(),
    Dense(8*H_out*W_out, 64),
    ReLU(),
    Dense(64, 10)
])

model.compile(
    loss=CrossEntropy(),
    metrics=[accuracy]
)
model.fit(X_train,y_train, X_test, y_test, lr=0.001,
        epochs=10,optimizer_name="Adam", scheduler_name="WCOS", save_path='model_mnsi',
        callbacks=[EarlyStopping(patience=20),
                    ModelCheckpoint("best_model"),
                    ReduceLROnPlateau(factor=0.5,patience=10),
                    History(),
                    CSVLogger("training_log.csv"),
                    ProgressBarCallback()
                    ])

y_pred = model.predict(X_test)
preds = np.argmax(y_pred, axis=1)

wrong = np.where(preds != y_test)[0]

"""for i in range(5):
    idx = wrong[i]
    img = X_test[idx].reshape(28,28)
    
    plt.imshow(img, cmap="gray")
    plt.title(f"Pred: {preds[idx]} / True: {y_test[idx]}")
    plt.show()"""

from src.analysis.analysis import (
    plot_history,
    confusion_matrix_from_logits,
    plot_confusion_matrix
)

# après entraînement
plot_history(model.history, save_path="analysis/data/loss.png", show=False)

y_pred = model.predict(X_test)
cm = confusion_matrix_from_logits(y_test, y_pred, num_classes=10)

plot_confusion_matrix(
    cm,
    class_names=[str(i) for i in range(10)],
    normalize=False,
    save_path="analysis/data/confusion_matrix.png",
    show=False
)

plot_confusion_matrix(
    cm,
    class_names=[str(i) for i in range(10)],
    normalize=True,
    save_path="analysis/data/confusion_matrix_normalized.png",
    show=False
)
