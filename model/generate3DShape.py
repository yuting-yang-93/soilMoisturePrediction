# OPEN PACKAGE
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from generator.epochGenerator import train_generator, train_val_split_generator
from configuration import ModelConfiguration
config = ModelConfiguration()

# ========= transform categorical to binary variables
def categoricalTransform(data, isDummy=False):
    preData = data.drop(['station'], axis = 1)
    preData = pd.get_dummies(preData, drop_first = isDummy)
    finalData = pd.concat([data.loc[:,['station']], preData], axis = 1)
    return finalData

# after one code encoding, we need add the columns which not appears in the test but exists in train
def completeColumn(targetDataset, secData):
    missingCol = list((set(secData.columns)).difference(set(targetDataset.columns)))
    for col in missingCol:
        targetDataset[col] = 0
    return targetDataset

# # ====== import dataset
# data = pd.read_csv('../data/preparedData.csv')
# targetVar = ['sm_20']
# inputCols = ['sm_5', 'p', 'st_5', 'at', 'station', 'TAXNWRB_Class1', 'TAXOUSDA_Class1']
def generated3D (data, targetVar,inputCols,isScaled=True, isDummy = False):

    # ======== split test and train stations
    testStation = config.testStations
    train_val_Station = list(set(data.station.unique()).difference(set(testStation)))
    testData = data[data['station'].isin(testStation)]
    train_val_Data = data[data['station'].isin(train_val_Station)]

    # ========= clean dataset
    to_columns = ['sm_5', 'sm_20', 'sm_50', 'sm_100', 'p', 'at', 'st_5']
    cleaned_train_val_Data = train_val_Data.copy()
    for col in to_columns:
        cleaned_train_val_Data[col] = np.where(cleaned_train_val_Data['anomaly'] == -1, np.nan, cleaned_train_val_Data[col])

    cleaned_train_val_Data.index = pd.to_datetime(cleaned_train_val_Data.timeStamp)
    testData.index = pd.to_datetime(testData.timeStamp)

    cleaned_train_val_Data = cleaned_train_val_Data.drop(['timeStamp'], axis=1)
    testData = testData.drop(['timeStamp'], axis=1)

    inputCols.extend(targetVar)
    preparing_train_val = cleaned_train_val_Data.loc[:,inputCols]
    preparing_test = testData.loc[:, inputCols]


    transformed_train_val = categoricalTransform(preparing_train_val, isDummy=isDummy)
    transformed_test = categoricalTransform(preparing_test,isDummy=isDummy)

    train_val = completeColumn(transformed_train_val, transformed_test)
    test = completeColumn(transformed_test, transformed_train_val)

    # ==========  standardization of training dataset
    notInX = targetVar.copy()
    notInX.append('station')
    X_train_val = train_val[train_val.columns[~train_val.columns.isin(notInX)]]
    y_train_val = train_val.loc[:, targetVar]

    scaler = MinMaxScaler().fit(X_train_val)
    scaler_y = MinMaxScaler().fit(y_train_val)

    X_train_val_scaled = scaler.transform(X_train_val)
    y_train_val_scaled = scaler_y.transform(y_train_val)

    X_train_val_df = pd.DataFrame(X_train_val_scaled, columns=X_train_val.columns, index=X_train_val.index)
    y_train_val_df = pd.DataFrame(y_train_val_scaled, columns=y_train_val.columns, index=y_train_val.index)
    train_val_scaled = pd.concat([X_train_val_df, y_train_val_df], axis=1)
    train_val_scaled = pd.concat([train_val_scaled, train_val.loc[:, ['station']]], axis=1)

    # put the target variable to the last position
    cols = ['station'] + [col for col in test if col not in (targetVar) and col != 'station'] + targetVar
    test = test[cols]
    train_val_scaled = train_val_scaled[cols]
    train_val = train_val[cols]

    # the number of length
    print('train - the number of stations:', len(train_val_scaled.station.unique()))
    print('test - the number of stations:', len(test.station.unique()))
    print('#columns', len(train_val.columns))

    # normal Parameters
    timesteps = config.timesteps
    output_dim = len(targetVar)
    if isScaled == 'True':
        X_train, y_train, X_val, y_val = \
            train_val_split_generator(train_val_scaled, timesteps, output_dim, validation_split=config.validation_split)

    else:
        X_train, y_train, X_val, y_val = \
            train_val_split_generator(train_val, timesteps, output_dim, validation_split=config.validation_split)

    X_test = test[test.columns[~test.columns.isin(notInX)]]
    y_test = test.loc[:, targetVar]
    X_test_scaled = scaler.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    y_test_df = pd.DataFrame(y_test_scaled, columns=y_test.columns, index=y_test.index)
    test_scaled = pd.concat([X_test_df, y_test_df], axis=1)
    test_scaled = pd.concat([test_scaled, test.loc[:, ['station']]], axis=1)
    test_scaled = test_scaled[cols]

    # return preparing_train_val, preparing_test
    return X_train, y_train, X_val, y_val, test, test_scaled, scaler_y




