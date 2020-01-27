from generator.testGenerator import test_generator_covered, test_generator
from generator.testFlatten import flatten_test_covered
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

def generateLSTMForecast(model, scaler_y, test, test_scaled, timesteps, output_dim, overlappedSize=24, isStandard = True):
    forecastTable = pd.DataFrame()

    for targetStation in test.station.unique():
        X_test, y_test, timeStamp = test_generator_covered(test, targetStation, timesteps, output_dim, overlappedSize)
        X_test_scaled, y_test_scaled, timeStamp_scaler = test_generator_covered(test_scaled, targetStation, timesteps, output_dim, overlappedSize)
        flattenTimeStamp, y_true = flatten_test_covered(timeStamp, y_test)

        if isStandard == True:
            pred_test = model.predict(X_test_scaled)
            flattenTimeStamp, pred_test = flatten_test_covered(timeStamp, pred_test)
            yhat= scaler_y.inverse_transform(pred_test)

        else:
            pred_test = model.predict(X_test)
            flattenTimeStamp, yhat= flatten_test_covered(timeStamp, pred_test)

        res_df = pd.DataFrame(list(flattenTimeStamp), columns=['timeStamp'])
        res_df['forecast'] = yhat
        res_df['actual'] = y_true
        res_df['AE'] = np.absolute((yhat - y_true))
        res_df['station'] = targetStation

        forecastTable = pd.concat([forecastTable, res_df], axis = 0)
    return forecastTable


def generateLRForecast (X_train, y_train, test, timesteps, output_dim):
    forecastTable = pd.DataFrame()

    for targetStation in test.station.unique():
        X_test, y_test, timeStamp = test_generator_covered(test, targetStation, timesteps, output_dim, overlappedSize=48)
        flattenTimeStamp, y_true = flatten_test_covered(timeStamp, y_test)

        # linear regression Method
        X_train_flatten = X_train.reshape(X_train.shape[0]*X_train.shape[1],X_train.shape[2])
        y_train_flatten = y_train.reshape(y_train.shape[0]*y_train.shape[1],y_train.shape[2])
        reg = LinearRegression().fit(X_train_flatten, y_train_flatten)
        flattenTimeStamp, x_test_faltten = flatten_test_covered(timeStamp, X_test)
        yhat = reg.predict(x_test_faltten)

        res_df = pd.DataFrame(list(flattenTimeStamp), columns=['timeStamp'])
        res_df['forecast'] = yhat
        res_df['actual'] = y_true
        res_df['AE'] = np.absolute((yhat - y_true))
        res_df['station'] = targetStation

        forecastTable = pd.concat([forecastTable, res_df], axis = 0)

    return forecastTable




def generateLSTMForecastWithoutOverlap(model, scaler_y, test, test_scaled, timesteps, output_dim, isStandard = True):
    forecastTable = pd.DataFrame()

    for targetStation in test.station.unique():
        X_test, y_test, flattenTimeStamp = test_generator(test, targetStation, timesteps, output_dim)
        X_test_scaled, y_test_scaled, timeStamp_scaler = test_generator(test_scaled, targetStation, timesteps, output_dim)
        # flattenTimeStamp, y_true = flatten_test_covered(timeStamp, y_test)
        print(y_test.shape)
        y_true = np.reshape(y_test, (y_test.shape[0]*y_test.shape[1], -1))
        print(y_true.shape)

        if isStandard == True:
            pred_test = model.predict(X_test_scaled)
            print(pred_test.shape)
            pred_test = np.reshape(pred_test, (pred_test.shape[0]*y_test.shape[1], -1))
            print(pred_test.shape)
            yhat= scaler_y.inverse_transform(pred_test)

        else:
            pred_test = model.predict(X_test)
            print(pred_test.shape)
            pred_test = np.reshape(pred_test, (pred_test.shape[0]*y_test.shape[1], -1))
            print(pred_test.shape)

        res_df = pd.DataFrame(list(flattenTimeStamp), columns=['timeStamp'])
        res_df['forecast'] = yhat
        res_df['actual'] = y_true
        res_df['AE'] = np.absolute((yhat - y_true))
        res_df['station'] = targetStation

        forecastTable = pd.concat([forecastTable, res_df], axis = 0)
    return forecastTable