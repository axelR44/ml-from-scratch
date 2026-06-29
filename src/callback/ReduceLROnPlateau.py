from src.callback.base import Callback

class ReduceLROnPlateau(Callback): 

    def __init__(self, factor=0.5, patience=10, min_lr=1e-6):
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best = float("inf")
    
    def on_epoch_end(self, model, epoch, logs):

        val_loss = logs["val_loss"]

        if val_loss < self.best:
            self.best = val_loss
            self.counter = 0

        else:
            self.counter += 1

            if self.counter >= self.patience:

                self.model.optimizer.lr = max(
                    self.model.optimizer.lr * self.factor,
                    self.min_lr
                )

                self.counter = 0