import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import configuration
config = configuration.ModelConfiguration()

def singleFeatureOutlierDetect(dataset):
    '''
    Detect outlier throung the quantiles of attribute.
    If we only use Isolation Forest detection, there are still some anomaly for each single attribute
    So before Isolation Forest, we use quantiles detection for each attribute
    IF the value is outlier, then replaced by NaN value

    :param allData: DataFrame: original Dataset
    :return: preparedData: DataFrame without outlier deteced by Q1-Q3 quantile method
    '''
    data_numeric = dataset.loc[:,config.numeric_columns]
    # calculate Q1 and Q3
    Q1 = data_numeric.quantile(0.25)
    Q3 = data_numeric.quantile(0.75)
    IQR = Q3 - Q1

    # outliers for each attribute
    data_numeric_out = data_numeric[~((data_numeric < (Q1 - 1.5 * IQR)) |(data_numeric > (Q3 + 1.5 * IQR))).any(axis=1)]

    # merged dataset
    data_nonNumeric = dataset.drop(config.numeric_columns, axis = 1)
    firstCleanedData = data_nonNumeric.merge(data_numeric_out, how = 'left', left_index = True, right_index = True)

    return firstCleanedData

def isolationForestOutlierDetect(firstCleanedData):
    """
    Using isolation Forest to detect outlier for each station
    :param firstCleanedData: DataFrame, cleaned data after 'singleFeatureOutlierDetect' function
    :return: preparedData: DataFrame: original data with new column 'anomaly'

    """

    anomalyResAll = pd.DataFrame()

    # For each station using isolation Forest Outlier detection
    for cache_station in firstCleanedData.station.unique():
        data_numeric = firstCleanedData.loc[firstCleanedData.station == cache_station ,config.to_IForest_columns]
        data_numeric = data_numeric.dropna()

        # standardization
        scaler = StandardScaler()
        np_scaled = scaler.fit_transform(data_numeric)
        data_numeric_scaled = pd.DataFrame(np_scaled)

        # Isolation Forest Model
        clf = IsolationForest(max_samples='auto', random_state=2019)
        clf.fit(data_numeric_scaled)

        # Prediction.
        # Anomaly: -1, Normal: 1
        pred = clf.predict(data_numeric_scaled)

        anomalyRes = pd.DataFrame(pred, columns = ['anomaly'], index=data_numeric.index)
        anomalyResAll = pd.concat([anomalyResAll, anomalyRes], axis = 0)

    preparedData = firstCleanedData.merge(anomalyResAll, how = 'left', left_index = True, right_index = True)

    return preparedData



