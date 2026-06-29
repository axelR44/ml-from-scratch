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