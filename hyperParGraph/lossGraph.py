import tensorflow
import matplotlib.pyplot as plt
import os.path

class LossHistory(tensorflow.keras.callbacks.Callback):

    def __init__(self, graphName, localPath):
        super(tensorflow.keras.callbacks.Callback, self).__init__()
        self.graphName = graphName
        self.localPath = localPath

    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.val_loss['batch'].append(logs.get('val_loss'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.val_loss['epoch'].append(logs.get('val_loss'))

    def loss_plot(self, loss_type):
        # graph location
        final_path = './%s/lossPlot'%self.localPath
        if not os.path.isdir(final_path):
            os.makedirs (final_path)

        iters = range(len(self.losses[loss_type]))

        plt.figure()
        # loss
        plt.plot(iters, self.losses[loss_type], 'steelblue', label='train loss')
        if loss_type == 'epoch':
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'lightsteelblue', label='val loss')

        plt.grid(False)
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        # plt.title("train and validation loss graph --- %s"%self.graphName)
        plt.title("train and validation loss graph")
        plt.legend(loc="upper right")
        storePath = "{0}/loss_{1}.png".format(final_path, self.graphName)
        plt.savefig(storePath)
        plt.show()
