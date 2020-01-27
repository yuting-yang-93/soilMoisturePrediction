import numpy as np
import keras

def test_generator(test, station, timesteps, output_dim):
    '''
    Generate the array for test dataset, given station
    Train or validation dataset can be put into the function
    :param test: DataFrame.
    :param station: string. target station
    :param timesteps: int. The number of data points for one sequence.
    :param output_dim: int. The number of output variable.

    :return: np.array(X): array. X of test dataset
             np.array(y): array. y of test dataset
             timeStamp: the timeStamp for test
    '''

    X = []
    y = []
    timeStamp = []

    # select test data for specific station.
    in_start = 0
    station_Test = test[test.station == station]
    test_df = station_Test.drop(['station'], axis = 1).interpolate(method='linear', limit = 12)

    # in_start = 0
    # station_Test = station_Test.interpolate(method='linear', limit = 12)
    # test_df = station_Test.drop(['station'], axis = 1)
    arr = test_df.values
    in_end = in_start + timesteps

    while (in_end < len(station_Test)):
        if np.isnan(arr[in_start: in_end, ]).any() == True: # has nan value
            in_start = in_start + 1
            in_end = in_start + timesteps

        else:

            # store timeStamp_input
            timeStamp_input = test_df[in_start:in_end].index
            timeStamp.extend(timeStamp_input)

            # store X for test dataset
            x_input = arr[in_start:in_end,:-output_dim]
            X.append(x_input)

            # store y for test dataset
            y_input = arr[in_start:in_end,-output_dim:]
            y.append(y_input)

            # moving index
            in_start = in_end
            in_end = in_start + timesteps

    return np.array(X), np.array(y), timeStamp



def test_generator_covered(test, station, timesteps, output_dim, overlappedSize):
    '''
    covedred periods = 8
    e.g. first sample: [x0,x1,x2,x3,x4,x5,....,x23]
    e.g. second sample: [x15, x16, x17, x18...., x40]

    Generate the array for test dataset, given station
    Train or validation dataset can be put into the function
    :param test: DataFrame.
    :param station: string. target station
    :param timesteps: int. The number of data points for one sequence.
    :param output_dim: int. The number of output variable.

    :return: np.array(X): array. X of test dataset
             np.array(y): array. y of test dataset
             timeStamp: the timeStamp for test
    '''
    X = []
    y = []
    timeStamp = []

    # # select test data for specific station.
    # station_Test = test[test.station == station]
    # in_start = 0

    # station_Test = station_Test.interpolate(method='linear', limit = 12)
    # test_df = station_Test.drop(['station'], axis = 1)
    # arr = test_df.values
    # in_end = in_start + timesteps

    # select test data for specific station.
    in_start = 0
    station_Test = test[test.station == station]
    test_df = station_Test.drop(['station'], axis = 1).interpolate(method='linear', limit = 24)
    arr = test_df.values
    in_end = in_start + timesteps
    if timesteps <= 24:
        coverdPeriod = int(timesteps / 3)

    else:
        coverdPeriod = overlappedSize

    while (in_end < len(station_Test)):
        if np.isnan(arr[in_start: in_end, ]).any() == True: # has nan value
            in_start = in_start + 1
            in_end = in_start + timesteps

        else:

            # store timeStamp_input
            timeStamp_input = test_df[in_start:in_end].index
            timeStamp.extend(timeStamp_input)

            # store X for test dataset
            x_input = arr[in_start:in_end,:-output_dim]
            X.append(x_input)

            # store y for test dataset
            y_input = arr[in_start:in_end,-output_dim:]
            y.append(y_input)

            # moving index
            in_start = in_end - coverdPeriod
            in_end = in_start + timesteps

    return np.array(X), np.array(y), timeStamp