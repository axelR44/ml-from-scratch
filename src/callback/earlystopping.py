from src.callback.base import Callback
import copy 
class EarlyStopping(Callback):

    def __init__(
        self,
        monitor="val_loss",
        patience=20,
        restore_best=True
    ):
        self.monitor = monitor
        self.patience = patience
        self.restore_best = restore_best

        self.best = float("inf")
        self.counter = 0
        self.best_weights = None
    
    def on_epoch_end(self, model, epoch, logs):

        current = logs[self.monitor]

        if current < self.best:

            self.best = current
            self.counter = 0
            self.best_weights = copy.deepcopy(model.layers)
            
        else:
            self.counter += 1
            if self.counter >= self.patience:
                model.stop_training = True
                print(f"Early stopping at epoch {epoch}")
                if self.restore_best:
                    model.layers = self.best_weights

    def on_train_end(self, model, epoch, logs):
        if logs[self.monitor] is not None and self.best_weights is not None and model.use_best_model:
            model.layers = self.best_weights
