import pandas as pd
from sklearn.linear_model import LinearRegression
from evaluation.evaluation import MAEScore

def generateExponentialData(dataset, variable, A, K = 24, method = 'complex'):
    '''
    generate the data for linear equation model
    Formular:
    u_Filtered,a[k] = a*u_Filtered,a [k-1] + (1-a)*u[k]  for k > 1
    u_Filtered,a[k] =  u[k]                              for k = 1

    when K = 3
    y[3] = b0 + b1 * (1-a)*u[3] + (1-a)*a*u[2] + a*a*(1-a)*u[1])

    :param dataset: DataFrame. dataset for equation model
    :param variable: list. input variables
    :param A: list. the list of parameter a
    :param K: timesteps
    :return expoData: DataFrame. prepared dataset for equation model
    '''
    expoData = pd.DataFrame()
    for station in dataset.station.unique():
        print(station)
        station_data = dataset.loc[dataset.station == station]
        dataset_ori = station_data.loc[:,variable]
        for a in A:
            exponenetialData = (1-a) * dataset_ori.copy()
            for k in range(1, K+1):
                if k != K:
                    exponenetialData = exponenetialData + (1-a)*a**k*dataset_ori.shift(periods=k)
                else:
                    if method == 'complex':
                        if (1-a) > a**k:
                            exponenetialData = exponenetialData + a**k*dataset_ori.shift(periods=k)
                        else:
                            exponenetialData = exponenetialData + (1-a)*a**k*dataset_ori.shift(periods=k)
                    else:
                        exponenetialData = exponenetialData + (1-a)*a**k*dataset_ori.shift(periods=k)

            exponenetialData.columns = [var + '_a{0}'.format(int(100*a)) for var in variable]
            station_data = pd.concat([station_data, exponenetialData], axis = 1)
        expoData = pd.concat([expoData, station_data], axis = 0)
    return expoData


def resultCompare(train, test, xVar, targetVars):
    pred_LR_DF = pd.DataFrame()
    if type(xVar) != list:
        xVarName = xVar.tolist()
        situation = '%s'%xVarName
    else:
        situation = '%s'%xVar
    for targetVar in targetVars:
        reg = LinearRegression().fit(train.loc[:,xVar], train[targetVar])
        yhat_LR = reg.predict(test.loc[:,xVar])
        pred_LR_DF = pd.concat([pred_LR_DF, pd.DataFrame(yhat_LR, columns = [targetVar])], axis = 1)

    res_lr_dic = MAEScore(pred_LR_DF.values, test.loc[:,targetVars].values, targetVars)
    res_lr = pd.DataFrame.from_dict(res_lr_dic)
    res_lr['input Variables'] = situation
    return res_lr
