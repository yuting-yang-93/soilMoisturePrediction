import tensorflow
#
class LossHistory(tensorflow.keras.callbacks.Callback):
    """
    A Logger that log average performance per `display` steps.
    """
    def __init__(self, display, val_data):
        super(tensorflow.keras.callbacks.Callback, self).__init__()
        self.step = 0
        self.display = display
        self.metric_cache = {}
        self.val_data = val_data
        self.losses = [] # training loss
        self.val_losses = [] # validation loss

    def on_train_begin(self, logs=None):
        self.losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs=None):
        x, y = self.val_data
        val_loss = self.model.evaluate(x, y, verbose = 0)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(val_loss)
        print('\nvalidation loss: {}\n'.format(val_loss))

    def on_batch_end(self, batch, logs={}):

        x, y = self.val_data
        self.step += 1

        for k in self.params['metrics']:
            if k in logs:
                self.metric_cache[k] = self.metric_cache.get(k, 0) + logs[k]

        if self.step % self.display == 0:
            val_loss = self.model.evaluate(x, y, verbose = 0)
            self.losses.append(logs.get('loss'))

            # self.val_losses.append(logs.get('val_loss'))

            metrics_log = ''
            for (k, v) in self.metric_cache.items():
                val = v / self.display
                if abs(val) > 1e-3:
                    metrics_log += ' - %s: %.4f' % (k, val)
                else:
                    metrics_log += ' - %s: %.4e' % (k, val)
            print('step: {}/{} ... {}'.format(self.step,
                                              self.params['steps'],
                                              metrics_log))
            self.metric_cache.clear()



class LossHistory(tensorflow.keras.callbacks.Callback):
    def __init__(self, val_data):
        super(tensorflow.keras.callbacks.Callback, self).__init__()
        self.losses = []
        self.val_losses = []
        self.val_data = val_data

    def on_train_begin(self, logs=None):
        self.losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs=None):
        x, y = self.val_data
        val_loss = self.model.evaluate(x, y, verbose = 0)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(val_loss)
        print('\nvalidation loss: {}\n'.format(val_loss))


