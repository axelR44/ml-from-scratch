
class Callback:

    def on_train_begin(self, model):
        pass

    def on_train_end(self, model, epoch, logs):
        pass

    def on_epoch_begin(self, model, epoch):
        pass

    def on_epoch_end(self, model, epoch, logs):
        pass

    def on_batch_begin(self, model, batch):
        pass

    def on_batch_end(self, model, batch, logs):
        pass