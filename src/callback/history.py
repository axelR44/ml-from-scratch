from src.callback.base import Callback

class History(Callback):

    def on_train_begin(self, model):

        self.history = {}

    def on_epoch_end(self, model, epoch, logs):

        for k, v in logs.items():

            if k not in self.history:
                self.history[k] = []

            self.history[k].append(v)
    def on_train_end(self, model, epoch, logs):
        model.history = self.history