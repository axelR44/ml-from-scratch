import numpy as np

from src.models.sequential import Model
from src.layers.conv2d import Conv2D
from src.layers.dense import Dense
from src.layers.activation import ReLU, Sigmoid
from src.layers.flatten import Flatten
from src.layers.pooling import MaxPool2D,AvgPool2D
from src.layers.batchnorm import BatchNorm
from src.layers.batchnorm2d import BatchNorm2D


def check_shape(name, output, expected_shape):
    actual_shape = output.shape

    if actual_shape != expected_shape:
        raise AssertionError(
            f"{name} mauvaise shape : "
            f"attendu {expected_shape}, obtenu {actual_shape}"
        )

    print(f"{name:<20} OK {actual_shape}")


def test_conv2d_same_stride1():
    print("\ntest_conv2d_same_stride1")

    X = np.random.randn(2, 1, 28, 28)

    conv = Conv2D( out_channels=8,kernel_size=3,padding=1, stride=1)

    out = conv.forward(X)

    check_shape("Conv2D same s1", out, (2, 8, 28, 28))

def test_conv2d_padding_stride2():
    print("\n=== test_conv2d_padding_stride2 ===")

    X = np.random.randn(2, 1, 28, 28)

    conv = Conv2D(out_channels=8, kernel_size=3, padding=1, stride=2)

    out = conv.forward(X)

    check_shape("Conv2D pad s2", out, (2, 8, 14, 14))


def test_maxpool2d():
    print("\n=== test_maxpool2d ===")

    X = np.random.randn(2, 8, 14, 14)

    pool = MaxPool2D(pool_size=2)

    out = pool.forward(X)

    check_shape("MaxPool2D", out, (2, 8, 7, 7))


def test_flatten():
    print("\n=== test_flatten ===")

    X = np.random.randn(2, 8, 7, 7)

    flatten = Flatten()

    out = flatten.forward(X)

    check_shape("Flatten", out,(2, 8 * 7 * 7))


def test_dense_lazy():
    print("\n=== test_dense_lazy ===")

    X = np.random.randn(2, 392)

    dense = Dense(64)

    out = dense.forward(X)

    check_shape("Dense lazy", out, (2, 64))


def test_batchnorm_1d():
    print("\n=== test_batchnorm_1d ===")

    X = np.random.randn(4, 64)

    bn = BatchNorm()

    out = bn.forward(X, training=True)

    check_shape("BatchNorm", out, (4, 64))

    dY = np.random.randn(*out.shape)
    dX = bn.backward(dY)

    check_shape("BatchNorm backward", dX, X.shape)


def test_batchnorm_2d():
    print("\n=== test_batchnorm_2d ===")

    X = np.random.randn(4, 8, 14, 14)

    bn = BatchNorm2D()

    out = bn.forward(X, training=True)

    check_shape("BatchNorm2D", out, (4, 8, 14, 14))

    dY = np.random.randn(*out.shape)
    dX = bn.backward(dY)

    check_shape("BatchNorm2D backward", dX, X.shape)


def test_full_cnn_forward_shapes():
    print("\n=== test_full_cnn_forward_shapes ===")

    X = np.random.randn(2, 1, 28, 28)

    model = Model([
        Conv2D(8, 3, padding=1, stride=2),
        ReLU(),
        MaxPool2D(),
        Flatten(),
        Dense(64),
        ReLU(),
        Dense(10)
    ])

    expected_shapes = [
        (2, 8, 14, 14),
        (2, 8, 14, 14),
        (2, 8, 7, 7),
        (2, 392),
        (2, 64),
        (2, 64),
        (2, 10),
    ]

    out = X

    for layer, expected_shape in zip(model.layers, expected_shapes):
        out = layer.forward(out)

        check_shape(layer.__class__.__name__, out, expected_shape)


def test_full_cnn_with_batchnorm2d():
    print("\n=== test_full_cnn_with_batchnorm2d ===")

    X = np.random.randn(2, 1, 28, 28)

    model = Model([
        Conv2D(8, 3, padding=1, stride=1),
        BatchNorm2D(),
        ReLU(),
        MaxPool2D(),
        Flatten(),
        Dense(64),
        ReLU(),
        Dense(10)
    ])

    expected_shapes = [
        (2, 8, 28, 28),
        (2, 8, 28, 28),
        (2, 8, 28, 28),
        (2, 8, 14, 14),
        (2, 8 * 14 * 14),
        (2, 64),
        (2, 64),
        (2, 10),
    ]

    out = X

    for layer, expected_shape in zip(model.layers, expected_shapes):

        if "training" in layer.forward.__code__.co_varnames:
            out = layer.forward(out, training=True)
        else:
            out = layer.forward(out)

        check_shape(layer.__class__.__name__, out, expected_shape)


def run_all_shape_tests():
    test_conv2d_same_stride1()
    test_conv2d_padding_stride2()
    test_maxpool2d()
    test_flatten()
    test_dense_lazy()
    test_batchnorm_1d()
    test_batchnorm_2d()
    test_full_cnn_forward_shapes()
    test_full_cnn_with_batchnorm2d()

    print("\nTous les tests de shapes sont passés")


if __name__ == "__main__":
    run_all_shape_tests()