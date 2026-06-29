import os
import matplotlib.pyplot as plt

def finalize_figure(save_path=None, show=True, dpi=150):
    """
    Sauvegarde et/ou affiche la figure courante.
    """
    if save_path is not None:
        folder = os.path.dirname(save_path)
        if folder != "":
            os.makedirs(folder, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()
