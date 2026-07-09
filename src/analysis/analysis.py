import numpy as np
import matplotlib.pyplot as plt
from src.utils.plotting import finalize_figure

def plot_history(history, save_path=None, show=True):
    plt.figure(figsize=(8, 5))

    if "train_loss" in history:
        plt.plot(history["train_loss"], label="train_loss")

    if "val_loss" in history:
        plt.plot(history["val_loss"], label="val_loss")

    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.title("loss History")
    plt.legend()
    plt.grid(True, alpha=0.3)

    finalize_figure(save_path= save_path, show=show)    

def confusion_matrix_from_logits(y_true, logits, num_classes=10):
    preds = np.argmax(logits, axis=1)
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for t, p in zip(y_true, preds):
        cm[t, p] += 1

    return cm


def plot_confusion_matrix(cm, class_names=None, normalize=False, save_path=None, show=True):
    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm / row_sums

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))

    if class_names is not None:
        ticks = np.arange(len(class_names))
        plt.xticks(ticks, class_names)
        plt.yticks(ticks, class_names)

    finalize_figure(save_path=save_path, show=show)

def plot_misclassified(X, y_true, y_pred, n=25, save_path=None, show=True):
    preds = np.argmax(y_pred, axis=1)

    errors = np.where(preds != y_true)[0]

    if len(errors) == 0:
        print("Aucune erreur de classification")
        return

    errors = errors[:n]

    cols = int(np.ceil(np.sqrt(len(errors))))
    rows = int(np.ceil(len(errors) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = np.array(axes).ravel()

    for ax in axes:
        ax.axis("off")

    for ax, idx in zip(axes, errors):
        img = X[idx]

        if img.ndim == 3:
            img = img.squeeze()

        ax.imshow(img, cmap="gray")
        ax.set_title(f"T:{y_true[idx]} P:{preds[idx]}", color="red")
        ax.axis("off")

    plt.suptitle("Misclassified Samples")

    finalize_figure(save_path=save_path, show=show)

def plot_conv_filters(conv_layer, save_path=None, show=True):
    W = conv_layer.W

    n_filters = W.shape[0]

    cols = int(np.ceil(np.sqrt(n_filters)))
    rows = int(np.ceil(n_filters / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = np.array(axes).ravel()

    for ax in axes:
        ax.axis("off")

    for i in range(n_filters):

        filt = W[i]
        if filt.shape[0] == 1:
            filt = filt[0]
        else:
            filt = np.mean(filt, axis=0)

        filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)

        axes[i].imshow(filt, cmap="gray")
        axes[i].set_title(f"F{i}")

    plt.suptitle("Conv Filters")

    finalize_figure(save_path=save_path, show=show)

def plot_feature_maps(feature_maps, save_path=None, show=True):
    """
    feature_maps shape:
    (1, C, H, W)
    """

    fmap = feature_maps[0]

    C = fmap.shape[0]

    cols = int(np.ceil(np.sqrt(C)))
    rows = int(np.ceil(C / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = np.array(axes).ravel()

    for ax in axes:
        ax.axis("off")

    for i in range(C):
        axes[i].imshow(fmap[i], cmap="viridis")
        axes[i].set_title(f"M{i}")

    plt.suptitle("Feature Maps")

    finalize_figure(save_path=save_path, show=show)