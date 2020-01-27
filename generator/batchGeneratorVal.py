import numpy as np
from numpy import array
import pandas as pd

# This package includes one function which is used in 'fit_generator' function, suitable for larger dataset
# 'batch_val_generator': generate one batch of validation dataset for 'fit_generator' function.
#                          All dataset will be used into each epoch.

def batch_val_generator(valData, timesteps, batch_size, output_dim):
    '''
    Generate one batch for validation dataset for 'fit_generator' function.
    All dataset will be used to evaluate for each epoch.

    :param valData: DataFrame. Validation Dataset
    :param timesteps: Int. The number of data points for one sequence.
    :param batch_size: Int. The number of samples for each batch.
    :param output_dim: Int. The number of output variable.
    :return: x_batch: array. one batch X of validation dataset
             y_batch: array. one batch y of validation dataset
    '''


    stations = valData.station.unique()
    sample = pd.DataFrame()

    for station in stations:
        stationData = valData[valData.station == station]
        noStation = stationData.drop(['station'], axis= 1)
        sample = pd.concat([sample, noStation], axis = 0)

    arr = sample.values

    if int(arr.shape[0]/timesteps/float(batch_size)) == 0:
        sample_arr = array(arr)

    else:
        sample_arr = array(arr[0: int(arr.shape[0]/timesteps/float(batch_size))*timesteps*batch_size,])

    batch = sample_arr.reshape((int((sample_arr.shape[0])/float(timesteps)), timesteps, sample_arr.shape[1]))

    # remove nan values
    noNaNBatch = batch[[~np.isnan(batch[i]).any(axis=1).any() for i in range(batch.shape[0])]]

    x_batch = noNaNBatch[:,:,:-output_dim]
    y_batch = noNaNBatch[:,:,-output_dim:]

    yield (x_batch, y_batch)