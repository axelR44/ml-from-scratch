import numpy as np

from src.layers.dense import Dense
from src.layers.conv2d import Conv2D
from src.layers.flatten import Flatten
from src.layers.activation import ReLU, Sigmoid
from src.layers.pooling import MaxPool2D,AvgPool2D
from src.layers.batchnorm import BatchNorm
from src.layers.batchnorm2d import BatchNorm2D
from src.models.sequential import Model

def test_save_load_same_predictions():
    X = np.random.randn(4, 1, 28, 28)

    model = Model([
        Conv2D(8, 3, padding=1),
        BatchNorm2D(),
        ReLU(),
        MaxPool2D(),
        Flatten(),
        Dense(10)
    ])

    y1 = model.predict(X)

    model.save("tmp_test_model")
    model2 = Model.load("tmp_test_model")

    y2 = model2.predict(X)

    assert np.allclose(y1, y2, atol=1e-8)
    print("les tests sur la sauvegarde du modèle sont passés")

if __name__ == "__main__":
    test_save_load_same_predictions()