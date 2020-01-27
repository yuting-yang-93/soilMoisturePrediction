import pandas as pd
from importData.pipelineData import *
import ismn.metadata_collector as metadata_collector
from preprocessing.outlierDetect import *
from preprocessing.dropStationHighMVR import delHighMissingValue
from model.generateEWLRdata import generateExponentialData
import os.path
path_to_data = './data/'
import configuration
config = configuration.ModelConfiguration()


if not os.path.exists(os.path.join(path_to_data, 'python_metadata', 'metadata.npy')):
    os.mkdir(os.path.join(path_to_data, 'python_metadata'))
    metadata = metadata_collector.collect_from_folder(path_to_data)
    np.save(
        os.path.join(path_to_data, 'python_metadata', 'metadata.npy'), metadata)
else:
    metadata = np.load(
        os.path.join(path_to_data, 'python_metadata', 'metadata.npy'),
        allow_pickle=True)

# ================  ismnData
if os.path.exists('./data/ismnData.csv') == False:
    print('./data/ismnData.csv is not existed, reloading ismnData.csv...')
    ismnData = storeISMNDataintoLocal(config.date_start, config.date_end)
else: ismnData = pd.read_csv('./data/ismnData.csv')

# delete the stations with high missing value rate from ISMN Dataset
if os.path.exists('./data/ismnData_140Stations.csv') == False:
    print('./data/ismnData_140Stations.csv is not existed, reloading ismnData_140Stations.csv...')
    preparedISMNData= delHighMissingValue(ismnData, 2012)
    preparedISMNData.to_csv('./data/ismnData_140Stations.csv', index=False)
else:
    preparedISMNData = pd.read_csv('./data/ismnData_140Stations.csv')

# ================  coorData
if os.path.exists('./data/coorData.csv') == False:
    coorData = storeCoorDataintoLocal()
else: coorData = pd.read_csv('./data/coorData.csv')

# ================  soilData
if os.path.exists('./data/soilData.csv') == False:
    print('./data/soilData.csv is not existed, reloading soilData.csv...')
    soilData = storeSoilDataintoLocal(coorData, config.soilProperties, config.classProperties)
else: soilData = pd.read_csv('./data/soilData.csv')

# ================  dataset: mergedData ISMN, coordinate and soil Dataset
if os.path.exists('./data/mergedData.csv') == False:
    print('./data/mergedData.csv is not existed, reloading mergedData.csv...')
    mergedData = preparedISMNData.merge(coorData, on = ['station'], how = 'left')
    mergedData = mergedData.merge(soilData.drop(['long','lat','network'], axis = 1), on=['station'], how = 'left')
    mergedData.to_csv('./data/mergedData.csv', index=False)
else: mergedData = pd.read_csv('./data/mergedData.csv')

#================= detect outlier
if os.path.exists('./data/preparedData.csv') == False:
    print('./data/preparedData.csv is not existed, reloading preparedData.csv...')
    cleanData = singleFeatureOutlierDetect(mergedData)
    cleanData = isolationForestOutlierDetect(cleanData)
    soilTypeDF = cleanData.loc[:,['station','TAXNWRB_Class1','TAXOUSDA_Class1']].\
        drop_duplicates().\
        reset_index(drop = True)

    dataset = cleanData.filter(regex='station|sm|p|at|st_5|anomaly|timeStamp')
    variable = ['sm_5', 'p', 'at', 'st_5']
    K = config.EWLR_K
    A = config.EWLR_A
    preparedData = generateExponentialData(dataset, variable, A, K=K, method='simple').dropna(how = 'any', axis = 0)
    preparedData = preparedData.merge(soilTypeDF, on = ['station'], how = 'left')
    preparedData['timeStamp'] = pd.to_datetime(preparedData.timeStamp).dt.tz_convert(None)
    preparedData.to_csv('./data/preparedData.csv', index=False)
else: preparedData = pd.read_csv('./data/preparedData.csv')
