from tqdm import tqdm
from src.callback.base import Callback

class ProgressBarCallback(Callback):
    def __init__(self, metrics = None):
        if metrics is None:
            self.metrics = ["train_loss", "val_loss", "lr"]
        else:
            self.metrics = metrics


    def on_train_begin(self, model):
        self.step = max(1, model.epochs // 100)

        self.pbar = tqdm(range(model.epochs), desc="Training")

    def on_epoch_end(self, model, epoch, logs):
        if epoch%self.step == 0:
            self.pbar.update(1)
            
            postfix ={ metric : f"{logs[metric]:.4f}" if metric in logs else "N/A"
                        for metric in self.metrics}

            self.pbar.set_postfix(postfix)

    def on_train_end(self, model, epochs, logs):

        self.pbar.close()