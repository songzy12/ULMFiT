import paddle

class ResetModel(paddle.callbacks.Callback):
    # This callback reset model hidden states and update learning rate before each epoch begins
    def on_epoch_begin(self, epoch=None, logs=None):
        self.model.network.reset()
