import configuration
import pandas as pd

def delHighMissingValue(ismnData, startYear):
    '''
    select candidate stations with lower missing value rate
    :param ismnData: DataFrame: the ismn dataset
    :param startYear: start year of research
    :return: preparedISMNData: DataFrame with candidate stations
    '''

    config = configuration.ModelConfiguration()

    ismnData.timeStamp = pd.to_datetime(ismnData['timeStamp'])
    ismnData['year'] = ismnData['timeStamp'].dt.year
    data = ismnData[ismnData.year >= startYear]

    # information of missing values
    missingInfo = data.drop(['station', 'sensor', 'sensorCode', 'network', 'timeStamp', 'year'], 1).isna().groupby(
        [data.station, data.sensor], sort=False).sum()

    data = data.reset_index(drop=True)

    # total points for each station
    size = len(data[data.station == data.station[0]])

    # percentage of missing values for each station and each variable
    mvRate = round(missingInfo / (float(size)), 3)

    # candidates are the stations whose missing value rate is less than threshold_missingValue
    dic = mvRate[mvRate < config.threshold_missingValue].isnull().any(axis=1)
    candidates = list(dic[dic == False].index)

    candiData = pd.DataFrame(candidates, columns=['station', 'sensor'])

    preparedISMNData = candiData.merge(data, on=['station', 'sensor'], how='left')

    multiSensorsDF = candiData[candiData.station.isin(candiData[candiData.loc[:,['station']].duplicated() == True].station)]

    multiStation = multiSensorsDF.station.unique()
    preparedISMNData = preparedISMNData[~((preparedISMNData['station'].isin(multiStation) == True) & (preparedISMNData.sensorCode == 'B'))]

    preparedISMNData = preparedISMNData.drop(['sensor','sensorCode','network','year'], axis = 1)

    # candiWithMultiSensors = candiData[
    #     candiData.station.isin(candiData[candiData.loc[:, ['station']].duplicated() == True].station)]

    return preparedISMNData


# def getCandidates(data, soilData, soilClassVar):
#
#     config = configuration.ModelConfiguration()
#
#     # information of missing values
#     missingInfo = data.drop('station', 1).isna().groupby(data.station, sort=False).sum()
#
#     # total points for each station
#     size = len(data[data.station == data.station[0]])
#
#     # percentage of missing values for each station and each variable
#     missingPresent = round(missingInfo / (float(size)), 3)
#
#     # candidates are the stations whose missing value rate is less than threshold_missingValue
#     series = missingPresent[missingPresent < config.threshold_missingValue].isnull().any(axis=1)
#     candidate_stations = list(series[series == False].index)
#
#     soilData = soilData[soilData.station.isin(candidate_stations)]
#
#     soilTypeFreq = soilData.loc[:, [soilClassVar, 'station']].\
#         groupby(soilClassVar).count().sort_values(['station'], ascending=False)
#
#     soilTypeFreq = soilTypeFreq.loc[soilTypeFreq.station > config.threshold_stations].reset_index()
#
#     candidates_soilTypes = list(soilTypeFreq[soilClassVar].unique())
#
#     preparedData = data[data.station.isin(candidate_stations)]
#     preparedData = preparedData[preparedData[soilClassVar].isin(candidates_soilTypes)]
#
#     return preparedData