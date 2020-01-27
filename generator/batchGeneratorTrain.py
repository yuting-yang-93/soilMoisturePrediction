import numpy as np
import keras
from tensorflow.python.keras.utils.data_utils import Sequence

# This package includes two functions which are used in 'fit_generator' function, suitable for larger dataset
# 'batch_train_generator': generate one batch for 'fit_generator' function.
#                          All dataset will be used into each epoch.
# 'batch_train_generator_random': generate one batch randomly for 'fit_generator' function.
#                                 Not have to put all dataset into each epoch, by steps_per_epoch function.

class batch_train_generator(Sequence):
    '''
    Generate one batch for training dataset for 'fit_generator' function.
    All dataset will be used into each epoch.
    In fit_generator should be set the value of steps_per_epoch
    steps_per_epoch = int(int(len(self.data)/float(self.timesteps))/float(self.batch_size))
    '''

    def __init__(self, data, batch_size, timesteps, input_dim, output_dim, shuffle=True):

        """Initialization

        :param data: dataframe
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.data = data
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'

        print ('length', int(int(len(self.data)/float(self.timesteps))/float(self.batch_size)))

        return int(int(len(self.data)/float(self.timesteps))/float(self.batch_size))

    def __getitem__(self, index):
        '''
        Generate one batch of data
        The number of samples of one batch: batch_size
        :param index: ID of one batch

        '''

        # Generate indexes of the batch
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        #print('len of indexes', len(self.indexes))
        #print('len of batch_indexes', len(batch_indexes))


        # 3D: shape: (batch_size, timesteps, input_dim)

        # Allocate a new array for the batch of input-signals.
        x_shape = (self.batch_size, self.timesteps, self.input_dim)
        x_batch = np.zeros(shape=x_shape, dtype=np.float64)

        # Allocate a new array for the batch of output-signals.
        y_shape = (self.batch_size, self.timesteps, self.output_dim)
        y_batch = np.zeros(shape=y_shape, dtype=np.float64)

        # Fill the batch with random sequences of data.

        count = 0
        i = 0
        while i < self.batch_size:

            # for i in batch_indexes:

            # Get a random start-index.
            # This points somewhere into the training-data.
            start_index = batch_indexes[i]
            # print('iteration: ',i, ' ', start_index)
            stop = False
            while stop == False:

                one_sample = self.data.iloc[start_index:start_index+self.timesteps,]
                one_sample_array = one_sample.values
                one_sample_noStation = np.delete(one_sample_array, 0, 1)

                one_sample_X = one_sample.iloc[:,:-self.output_dim].values
                one_sample_y = one_sample.iloc[:,-self.output_dim:].values

                if len(set([one_sample_X[j][0] for j in range(len(one_sample_X))])) == 1 and len(
                        one_sample_noStation[~np.isnan(np.array(one_sample_noStation, dtype=np.float64)).any()]) != 0:

                    one_sample_X = np.delete(one_sample_X, 0, 1) # delete station information

                    #  Copy the sequences of data starting at this index.
                    x_batch[i] = one_sample_X
                    y_batch[i] = one_sample_y

                    i = i + 1

                    stop = True

                else:
                    start_index = np.random.randint(int(len(self.data)/float(self.timesteps))-1)


        # Generate data
        X, y = self.__data_generation(x_batch, y_batch)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # totalLen = len(self.data) - (self.timesteps - 1) * (self.numStation - 1) - self.timesteps + 1
        self.indexes = np.arange(int(len(self.data)/float(self.timesteps)))
        # self.indexes = np.arange(len(self.data) - self.batch_size + 1)

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, arr_batch_X, arr_batch_y):
        #
        # print('arr_batch_X', arr_batch_X.shape)
        # print('arr_batch_y', arr_batch_y.shape)
        return arr_batch_X, arr_batch_y



def batch_train_generator_random(data, timesteps, batch_size, input_dim, output_dim):
    '''
    Generate one batch randomly for training dataset for 'fit_generator' function.
    Do not have to use all dataset into each epoch.
    In fit_generator should be set the value of steps_per_epoch
    E.G. steps_per_epoch = 20, means 20 batches of samples to randomly yield from this generator function before declaring one epoch finished and starting the next epoch.)

    :param data: DataFrame: training dataset to be dealed.
    :param timesteps: Int. The number of data points for one sequence.
    :param batch_size: Int. The number of samples for each batch.
    :param input_dim: Int. The number of input variable.
    :param output_dim: Int. The number of output variable.
    :return: x_batch: array. one batch X of train dataset
             y_batch: array. one batch y of train dataset
    '''

    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, timesteps, input_dim)
        x_batch = np.zeros(shape=x_shape, dtype=np.float64)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, timesteps, output_dim)
        y_batch = np.zeros(shape=y_shape, dtype=np.float64)

        # Fill the batch with random sequences of data.
        i = 0
        while i < batch_size:

            # Get a random start-index.
            # This points somewhere into the training-data.

            start_index = np.random.randint(len(data) - timesteps)
            # print('trainDataIteration:', i, start_index)
            one_sample = data.iloc[start_index:start_index+timesteps,]
            one_sample_array = one_sample.values
            one_sample_noStation = np.delete(one_sample_array, 0, 1)

            one_sample_X = one_sample.iloc[:, :-output_dim].values
            one_sample_y = one_sample.iloc[:, -output_dim:].values

            if len(set([one_sample_X[j][0] for j in range(len(one_sample_X))])) == 1 and len(
                    one_sample_noStation[~np.isnan(np.array(one_sample_noStation, dtype=np.float64)).any()]) != 0:

                one_sample_X = np.delete(one_sample_X, 0, 1) # delete station information

                #  Copy the sequences of data starting at this index.
                x_batch[i] = one_sample_X
                y_batch[i] = one_sample_y

                i = i + 1

        yield (x_batch, y_batch)