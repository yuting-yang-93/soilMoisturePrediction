import os.path
import matplotlib as mpl
from importData.getSoilTypeData import *

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

# import pytesmo.io.ismn.interface as ismn  #old version of pytesmo
import ismn.interface as ismn  # new version of pytesmo

# import local functions
from importData.getISMNData import *
from importData.getCoordinates import *
import configuration
config = configuration.ModelConfiguration()

# set path of ISMN data
path_to_ismn_data = './data/'
# os.remove(path_to_ismn_data + "python_metadata/metadata.npy")  # reinitialize the cache
# os.rmdir(path_to_ismn_data + "python_metadata")  # reinitialize the cache

def storeISMNDataintoLocal(date_start, date_end):
    # import and merge ISMN data
    ISMN_reader = ismn.ISMN_Interface(path_to_ismn_data)
    networks = ISMN_reader.list_networks()
    # get the soil moisture and precipitation data
    ismnData = merge_network(ISMN_reader, networks, date_start, date_end)
    print("Available Networks:")
    print(networks)
    print(ismnData.head(3))

    # store ismn Data as csv.file
    ismnData['timeStamp'] = ismnData.index
    ismnData.to_csv('./data/ismnData.csv', index=False)
    return ismnData

# get the coordination of stations
def storeCoorDataintoLocal():
    # import and merge ISMN data
    ISMN_reader = ismn.ISMN_Interface(path_to_ismn_data)
    networks = ISMN_reader.list_networks()

    coorData = merge_networks_coordinates(ISMN_reader, networks)
    coorData.to_csv('./data/coorData.csv', index=False)
    return coorData

# get the soil type of stations
def storeSoilDataintoLocal(coorData, soilProperties, classProperties):

    soilData = pd.DataFrame()
    for index in range(len(coorData)):
        station = coorData['station'].iloc[index]
        network = coorData['network'].iloc[index]
        long = coorData['long'].iloc[index]
        lat = coorData['lat'].iloc[index]
        print('getting soil information of station %s '
              'with latitude %s and longitude %s ...'%(station, str(round(lat,1)), str(round(long,1))))

        # get soil properties
        soilPropData = get_soiProperties(lat, long, soilProperties)
        # get the soil classes
        soilClassData = get_mostProbClass(lat, long, classProperties)

        soilPropData = soilPropData.merge(soilClassData, on = ['lat','long'])
        soilPropData['station'] = station
        soilPropData['network'] = network
        soilData = pd.concat([soilData, soilPropData], axis = 0)

    soilData.to_csv('./data/soilData.csv', index=False)
    return soilData














