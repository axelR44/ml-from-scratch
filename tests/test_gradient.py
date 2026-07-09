import numpy as np

from src.layers.dense import Dense
from src.layers.conv2d import Conv2D
from src.layers.flatten import Flatten
from src.layers.activation import ReLU, Sigmoid
from src.layers.pooling import MaxPool2D,AvgPool2D, GlobalAveragePooling2D
from src.layers.batchnorm import BatchNorm
from src.layers.batchnorm2d import BatchNorm2D


def relative_error(a, b, eps=1e-12):
    return np.abs(a - b) / (np.abs(a) + np.abs(b) + eps)


def forward_layer(layer, X, training=True):
    if "training" in layer.forward.__code__.co_varnames:
        return layer.forward(X, training=training)

    return layer.forward(X)


def backward_layer(layer, dY):
    try:
        return layer.backward(dY, lambda_l2=0.0)
    except TypeError:
        return layer.backward(dY)


def loss_from_output(Y, dY):
    """
    Loss simple :
        L = sum(Y * dY)
    Donc :
        dL/dY = dY
    """
    return np.sum(Y * dY)


def sample_indices(shape, n_checks=10):
    total = np.prod(shape)
    n_checks = min(n_checks, total)

    flat_indices = np.random.choice(total,size=n_checks, replace=False)

    return [np.unravel_index(idx, shape) for idx in flat_indices]


def print_result(name, errors, tolerance=1e-5):
    max_err = np.max(errors)
    mean_err = np.mean(errors)

    print(
        f"{name:<20} | "
        f"max={max_err:.3e} | "
        f"mean={mean_err:.3e} | "
        f"tol={tolerance:.1e}"
    )

    assert max_err < tolerance, (
        f"\n{name} a échoué au gradient check.\n"
        f"Erreur max      : {max_err:.3e}\n"
        f"Erreur moyenne  : {mean_err:.3e}\n"
        f"Tolérance       : {tolerance:.3e}\n"
    )

    return max_err


def check_input_gradient(layer, X, name, eps=1e-5, n_checks=10, tolerance=1e-5):
    """
    Vérifie dL/dX.
    """

    X = X.astype(np.float64)

    Y = forward_layer(layer, X, training=True)
    dY = np.random.randn(*Y.shape)

    dX = backward_layer(layer, dY)

    errors = []

    for idx in sample_indices(X.shape, n_checks):
        original = X[idx]

        X[idx] = original + eps
        loss_plus = loss_from_output(forward_layer(layer, X, training=True), dY)

        X[idx] = original - eps
        loss_minus = loss_from_output(forward_layer(layer, X, training=True), dY)

        X[idx] = original

        grad_num = (loss_plus - loss_minus) / (2 * eps)
        grad_backprop = dX[idx]

        errors.append(relative_error(grad_num, grad_backprop))

    return print_result(f"{name} dX", np.array(errors), tolerance=tolerance)


def check_parameter_gradient(
        layer,
        X,
        param_name,
        grad_name,
        name,
        eps=1e-5,
        n_checks=10,
        tolerance=1e-5,
    ):
    """
    Vérifie dL/dParam.

    Exemple :
        param_name = "W"
        grad_name  = "dW"
    """

    X = X.astype(np.float64)
    param = getattr(layer, param_name)
    setattr(layer, param_name, param.astype(np.float64))
    Y = forward_layer(layer, X, training=True)
    dY = np.random.randn(*Y.shape)

    backward_layer(layer, dY)

    param = getattr(layer, param_name)
    grad = getattr(layer, grad_name)

    errors = []

    for idx in sample_indices(param.shape, n_checks):
        original = param[idx]

        param[idx] = original + eps
        loss_plus = loss_from_output(forward_layer(layer, X, training=True), dY)

        param[idx] = original - eps
        loss_minus = loss_from_output(forward_layer(layer, X, training=True), dY)

        param[idx] = original

        grad_num = (loss_plus - loss_minus) / (2 * eps)
        grad_backprop = grad[idx]

        errors.append(relative_error(grad_num, grad_backprop))

    return print_result(f"{name} {param_name}", np.array(errors), tolerance=tolerance)


def check_layer(
        layer,
        X,
        name,
        check_input=True,
        n_checks=10,
        tolerance=1e-5
    ):
    """
    Vérifie automatiquement :
    - dX
    - W / b si présents
    - gamma / beta si présents
    """

    print(f"\n{name}")
    errors = []
    if check_input:
        errors.append(check_input_gradient(layer, X.copy(), name,n_checks=n_checks, tolerance=tolerance))

    param_pairs = [
        ("W", "dW"),
        ("b", "db"),
        ("gamma", "dgamma"),
        ("beta", "dbeta"),
    ]

    for param_name, grad_name in param_pairs:
        if hasattr(layer, param_name):
            errors.append(
                check_parameter_gradient(
                    layer,
                    X.copy(),
                    param_name,
                    grad_name,
                    name,
                    n_checks=n_checks,
                    tolerance=tolerance
                )
            )
    return errors

def run_all_tests_gradient():
    np.random.seed(42)

    all_errors = []
    all_errors += check_layer(Dense(3), np.random.randn(4, 5), "Dense", n_checks=20, tolerance=1e-6)
    all_errors += check_layer(ReLU(), np.random.randn(4, 5), "ReLU", check_input=True, n_checks=20, tolerance=1e-6)
    all_errors += check_layer(Sigmoid(), np.random.randn(4, 5), "Sigmoid", check_input=True, n_checks=20,tolerance=1e-6)
    all_errors += check_layer(Flatten(), np.random.randn(2, 3, 4, 5),"Flatten", check_input=True,n_checks=20,tolerance=1e-6)
    all_errors += check_layer(Conv2D(out_channels=4,kernel_size=3, padding=0,stride=1), np.random.randn(2, 3, 7, 7), "Conv2D valid",n_checks=20, tolerance=1e-5)
    all_errors += check_layer(Conv2D(out_channels=4, kernel_size=3, padding=1, stride=2),np.random.randn(2, 3, 8, 8),"Conv2D pad/stride",n_checks=20,tolerance=1e-5)
    all_errors += check_layer(MaxPool2D(pool_size=2),np.random.randn(2, 3, 6, 6), "MaxPool2D", check_input=True, n_checks=30, tolerance=1e-5)
    all_errors += check_layer(BatchNorm(), np.random.randn(5, 4), "BatchNorm",n_checks=20, tolerance=1e-5)
    all_errors += check_layer(BatchNorm2D(), np.random.randn(4, 3, 5, 5),"BatchNorm2D", n_checks=20, tolerance=1e-5)
    all_errors += check_layer(GlobalAveragePooling2D(), np.random.randn(2, 3, 5, 5), "GlobalAveragePooling2D", check_input=True, n_checks=20, tolerance=1e-6)
    all_errors += check_layer(AvgPool2D(pool_size=2), np.random.randn(2, 3, 6, 6), "AvgPool2D", check_input=True, n_checks=30, tolerance=1e-6)
    max_error = np.max(all_errors)

    print("\nRésumé")
    print(f"Erreur max globale : {max_error:.3e}")

    
    assert max_error < 1e-5, (f"Erreur globale trop élevée : {max_error:.3e}")

    print("les gradients semblent ok\n")


if __name__ == "__main__":
    run_all_tests_gradient()