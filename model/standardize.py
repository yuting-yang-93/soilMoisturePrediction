import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def interpolatation(data, Maxlimit):
    '''
    :param data: training or test dataset
    :param Maxlimit: Maximum number of consecutive NaNs to fill. Must be greater than 0.
    :return: preparedData:DataFrame; preprocessed dataframe
    '''

    imputatedData = pd.DataFrame()
    for station in data.station.unique():
        preparedData = data[data.station == station]
        preparedData = preparedData.drop(['station'], axis = 1)
        preparedData.interpolate(method='linear', limit=(Maxlimit))
        preparedData['station'] = station

        imputatedData = pd.concat([imputatedData, preparedData], axis = 0)
    return imputatedData

def flatten(X):
    '''
    Flatten a 3D array.
    This function will re-create the original 2D array from which the 3D arrays were created.
    It is the inverse of temporalize, meaning X = flatten(temporalize(X))

    :param X:  Input, A 3D array for lstm, where the array is [sample-size, timesteps,  features].
    :return:   Output, flattened_X  A 2D array, [sample-size+timesteps-1, features].
    '''

    flattened_X = np.empty((X.shape[0] + X.shape[1] - 1, X.shape[2]))  # sample x features array.
    for j in range(0, X.shape[1]):
        flattened_X[j] = X[0][j]

    index = X.shape[1]
    for i in range(1, X.shape[0]):
        flattened_X[index] = X[i][X.shape[1] - 1]
        index = index + 1
    return (flattened_X)


def scale(X, scaler):
    '''
    # Initialize a scaler using the training data.

    :param X:      A 3D array for lstm, where the array is sample x timesteps x features.
    :param scaler: A scaler object,
                     e.g., sklearn.preprocessing.StandardScaler, sklearn.preprocessing.normalize

    :return:       X_scaled, caled 3D array.
    '''

    X_scaled = X.copy()
    for i in range(X.shape[0]):
        X_scaled[i, :, :] = scaler.transform(X[i, :, :])

    return X_scaled