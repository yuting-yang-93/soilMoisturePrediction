import pandas as pd
def flatten_test_covered(timeStamp, arr):
    '''
    :param timeStamp: timeStamp of test
    :param arr: arr needed to flatten
    :return: FlattenTimeStamp
    :return: flattenArr: flatted array
    '''
    # generate DataFrame
    TimeStampDF = pd.DataFrame(timeStamp, columns = ['timeStamp'])
    arrReshape = arr.reshape(arr.shape[0]*arr.shape[1],arr.shape[2])
    valueDF = pd.DataFrame(arrReshape)

    # concat two DataFrame
    table = pd.concat([TimeStampDF, valueDF], axis = 1)

    # keep the first duplicated timeStamp
    cleandTable = table.groupby('timeStamp').head(1).reset_index(drop = True)

    # generate flatten time Stamp
    FlattenTimeStamp = cleandTable.timeStamp

    # generate flatten array
    flattenArr = cleandTable.drop(['timeStamp'], axis =1).values

    return FlattenTimeStamp, flattenArr.reshape(-1, flattenArr.shape[1])