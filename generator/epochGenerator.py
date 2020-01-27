import numpy as np

# This package includes two functions which are used in 'fit' function.
# 'train_val_split_generator': split data into train and validation dataset and generate array for 'fit' function
# 'train_generator': transform train or validation dataset into array for 'fit' function.

def train_val_split_generator(data, timesteps, output_dim, validation_split = 0.1):
    '''
    Split data into train and validation dataset for each station.
    Generate the array, which can be put into fit function of keras.

    :param train_val_data: DataFrame, to be splitted into train and validation dataset
    :param timesteps: int. The number of data points for one sequence.
    :param output_dim: int. The number of output variable.
    :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data.
                      The model_all will set apart this fraction of the training data, will not train on it,
                      and will evaluate the loss and any model_all metrics on this data at the end of each epoch.
                      The validation data is selected from the previous samples in the x and y data provided for each station.
    :return: np.array(X): array. X of training dataset
             np.array(y): array. y of training dataset
             np.array(X_val): array X of validation dataset
             np.array(y_val): array y of validation dataset
    '''
    X = []
    y = []
    X_val = []
    y_val = []

    # split for each station
    for station in data.station.unique():
        station_data = data[data.station == station]
        station_data_interpolate = station_data.drop(['station'], axis = 1).interpolate(method='linear', limit = 6)
        # station_data = station_data.interpolate(method='linear', limit = 6)
        in_start = 0
        # arr = station_data.drop(['station'], axis = 1).values
        arr = station_data_interpolate.values
        in_end = in_start + timesteps

        while (in_end < len(station_data)):
            # moving by each timesteps
            # if the array between in_start and in_end contains NaN value, then ignore.
            if np.isnan(arr[in_start: in_end, ]).any() == True: # has nan value
                # in_start = in_start + 1
                in_start = in_end
                in_end = in_start + timesteps
            else:
                x_input = arr[in_start:in_end,:-output_dim]
                y_input = arr[in_start:in_end,-output_dim:]

                if in_start > int(len(station_data)*(validation_split)):
                    X.append(x_input)
                    y.append(y_input)

                else:
                    X_val.append(x_input)
                    y_val.append(y_input)

                in_start = in_end
                in_end = in_start + timesteps

    return np.array(X), np.array(y), np.array(X_val), np.array(y_val)


def train_generator(data, timesteps, output_dim):
    '''
    Generate the array, which can be put into 'fit' function of keras.
    Train or validation dataset can be put into thif function
    :param data: DataFrame.
    :param timesteps: int. The number of data points for one sequence.
    :param output_dim: int. The number of output variable.

    :return: np.array(X): array. X of training dataset
             np.array(y): array. y of training dataset
    '''

    X = []
    y = []
    for station in data.station.unique():
        station_data = data[data.station == station]
        in_start = 0
        arr = station_data.drop(['station'], axis = 1).values
        in_end = in_start + timesteps

        while (in_end < len(station_data)):
            # moving by each timesteps
            # if the array between in_start and in_end contains NaN value, then ignore.
            if np.isnan(arr[in_start: in_end, ]).any() == True: # has nan value
                # in_start = in_start + 1
                in_start = in_end
                in_end = in_start + timesteps

            else:
                x_input = arr[in_start:in_end,:-output_dim]
                X.append(x_input)
                y_input = arr[in_start:in_end,-output_dim:]
                y.append(y_input)
                in_start = in_end
                in_end = in_start + timesteps

    return np.array(X), np.array(y)
