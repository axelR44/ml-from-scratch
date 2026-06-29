from src.callback.base import Callback

class ModelCheckpoint(Callback):

    def __init__(self, path, monitor="val_loss"):
        self.path = path
        self.monitor = monitor
        self.best = float("inf")

    def on_epoch_end(self, model, epoch, logs):

        current = logs[self.monitor]

        if current < self.best:

            self.best = current
            if model.save_path is not None:
                model.save(self.path)

    
