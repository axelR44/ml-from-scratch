import numpy as np

from src.layers.layer import Layer

from src.layers.dense import Dense
from src.layers.conv2d import Conv2D
from src.layers.flatten import Flatten
from src.layers.dropout import Dropout
from src.layers.activation import ReLU, Sigmoid, LeakyReLU, Softmax
from src.layers.pooling import MaxPool2D, AvgPool2D, GlobalAveragePooling2D, Pool2DBase
from src.layers.batchnorm import BatchNorm
from src.layers.batchnorm2d import BatchNorm2D

INPUT_2D = (4, 6)          # (N, features): Dense, BatchNorm, activations...
INPUT_4D = (2, 3, 8, 8)    # (N, C, H, W): Conv2D, pooling, BatchNorm2D...

REGISTRY = {
    "Dense": (lambda: Dense(5), INPUT_2D),
    "ReLU": (lambda: ReLU(), INPUT_2D),
    "Sigmoid": (lambda: Sigmoid(), INPUT_2D),
    "LeakyReLU": (lambda: LeakyReLU(), INPUT_2D),
    "Softmax": (lambda: Softmax(), INPUT_2D),
    "Dropout": (lambda: Dropout(p=0.5), INPUT_2D),
    "BatchNorm": (lambda: BatchNorm(), INPUT_2D),
    "Flatten": (lambda: Flatten(), INPUT_4D),
    "Conv2D": (lambda: Conv2D(4, 3, padding=1), INPUT_4D),
    "MaxPool2D": (lambda: MaxPool2D(pool_size=2), INPUT_4D),
    "AvgPool2D": (lambda: AvgPool2D(pool_size=2), INPUT_4D),
    "GlobalAveragePooling2D": (lambda: GlobalAveragePooling2D(), INPUT_4D),
    "BatchNorm2D": (lambda: BatchNorm2D(), INPUT_4D),
}

# Abstract or composite classes that must not be tested as plain layers.
SKIP = {"Layer", "Pool2DBase", "Residual"}


def discover_layers():
    """Recursively discover every concrete subclass of Layer."""
    found = {}

    def recurse(cls):
        for sub in cls.__subclasses__():
            if sub.__name__ not in SKIP:
                found[sub.__name__] = sub
            recurse(sub)

    recurse(Layer)
    return found


def build_layer(name):
    """Instantiate a layer and provide its test input."""
    if name in REGISTRY:
        factory, shape = REGISTRY[name]
        return factory(), np.random.randn(*shape).astype(np.float64)

    cls = discover_layers()[name]
    try:
        layer = cls()
    except TypeError as e:
        raise AssertionError(
            f"[{name}] discovered but missing from REGISTRY, and cannot be "
            f"instantiated without arguments ({e}). Add an entry to REGISTRY."
        )
    return layer, np.random.randn(*INPUT_2D).astype(np.float64)


def check_contract(layer, name):
    """Check that the layer respects the Layer interface."""
    assert isinstance(layer, Layer), f"[{name}] does not inherit from Layer"
    assert hasattr(layer, "training"), (f"[{name}] missing training attribute")
    assert callable(getattr(layer, "forward", None)), f"[{name}] no forward"
    assert callable(getattr(layer, "backward", None)), f"[{name}] no backward"

    params = layer.parameters()
    assert isinstance(params, list), f"[{name}] parameters() does not return a list"
    for p in params:
        assert "param" in p and "grad" in p, (f"[{name}] a parameters() entry is missing the param/grad keys")


def check_forward(layer, X, name):
    """Check that forward produces a finite output."""
    Y = layer.forward(X)
    assert np.all(np.isfinite(Y)), f"[{name}] forward produces NaN/inf"
    return Y


def check_backward_shapes(layer, X, Y, name):
    """Check that dX matches X's shape and each grad matches its param."""
    dY = np.random.randn(*Y.shape)
    dX = layer.backward(dY)

    assert dX.shape == X.shape, f"[{name}] dX {dX.shape} != X {X.shape}"
    assert np.all(np.isfinite(dX)), f"[{name}] backward produces NaN/inf"

    for p in layer.parameters():
        assert p["param"].shape == p["grad"].shape, (f"[{name}] param {p['param'].shape} != grad {p['grad'].shape}")

def check_gradient_numeric(layer, X, name, eps=1e-5, n_checks=15, tol=1e-4):
    """Compare the analytic backward to a centered finite difference on dX.

    Only applies to deterministic layers.
    """
    X = X.astype(np.float64)
    Y = layer.forward(X)
    dY = np.random.randn(*Y.shape)
    dX = layer.backward(dY)

    def loss(out):
        return np.sum(out * dY)  # L = sum(Y * dY) so dL/dY = dY

    total = X.size
    idxs = np.random.choice(total, size=min(n_checks, total), replace=False)

    max_err = 0.0
    for flat in idxs:
        idx = np.unravel_index(flat, X.shape)
        orig = X[idx]

        X[idx] = orig + eps
        lp = loss(layer.forward(X))
        X[idx] = orig - eps
        lm = loss(layer.forward(X))
        X[idx] = orig

        num = (lp - lm) / (2 * eps)
        ana = dX[idx]
        err = abs(num - ana) / (abs(num) + abs(ana) + 1e-12)
        max_err = max(max_err, err)

    assert max_err < tol, (f"[{name}] incorrect dX gradient: max relative error = {max_err:.2e}")
    return max_err


def is_mode_sensitive(layer, X):
    """Detect whether the output depends on train/eval mode (dynamic test)."""
    if hasattr(layer, "set_rng"):
        layer.set_rng(np.random.default_rng(0))
    layer.train()
    y_train = layer.forward(X.copy())

    if hasattr(layer, "set_rng"):
        layer.set_rng(np.random.default_rng(0))
    layer.eval()
    y_eval = layer.forward(X.copy())
    layer.train()

    return not np.allclose(y_train, y_eval)


def check_mode_behavior(name):
    """For a mode-sensitive layer, check that the output changes and that
    eval mode is deterministic.
    """
    layer, X = build_layer(name)
    if not is_mode_sensitive(layer, X):
        return False

    layer, X = build_layer(name)
    if hasattr(layer, "set_rng"):
        layer.set_rng(np.random.default_rng(0))
    layer.eval()
    y_eval = layer.forward(X.copy())
    y_eval2 = layer.forward(X.copy())
    assert np.allclose(y_eval, y_eval2), f"[{name}] non-deterministic in eval mode"
    return True


def test_all_layers():
    np.random.seed(0)
    layers = discover_layers()
    assert layers, "No layer discovered"

    print(f"{len(layers)} layers discovered: {', '.join(sorted(layers))}\n")
    print("Layer | contract | fwd | bwd | grad | mode")

    mode_sensitive = []

    for name in sorted(layers):
        layer, X = build_layer(name)

        check_contract(layer, name)
        Y = check_forward(layer, X, name)
        check_backward_shapes(layer, X, Y, name)

        if getattr(layer, "is_stochastic", False):
            grad_str = "skip(stoch)"
        else:
            grad_err = check_gradient_numeric(layer, X, name)
            grad_str = f"{grad_err:.2e}"

        mode = check_mode_behavior(name)
        if mode:
            mode_sensitive.append(name)

        print(f"{name} | OK | OK | OK | {grad_str} | {'yes' if mode else '-'}")

    print(f"\nMode-sensitive layers (train/eval): "
          f"{', '.join(mode_sensitive) if mode_sensitive else 'none'}")
    print("All layer tests passed.\n")


if __name__ == "__main__":
    test_all_layers()