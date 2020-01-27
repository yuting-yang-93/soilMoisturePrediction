import requests
import pandas as pd
import numpy as np


# get soil properties for each coordinates
def get_soiProperties(lat, long, soilProperties):

    webPath = "https://rest.soilgrids.org/query?lon="
    soilInfoData = pd.DataFrame()

    for property in soilProperties:

        resp = requests.get(webPath + str(round(long,2)) +
                            "&lat=" + str(round(lat,2)) + "&attributes=%s"%str(property))

        # no this property existed for the coordinates, then round 1
        if (property not in pd.DataFrame(resp.json()).index) == True:
            resp = requests.get(webPath + str(round(long, 1)) +
                                "&lat=" + str(round(lat, 1)) + "&attributes=%s" % str(property))

            # ScottBase has no information about soil types
            if resp.json()['properties']['soilmask'] == 'nodata':
                break

        # get the data with one property
        sigData = pd.DataFrame(resp.json()).loc[property]

        # only get useful column - 'properties'
        # transform the dictionay 'properties' as dataframe
        transData = pd.DataFrame.from_dict(sigData.properties)

        # no values for this specific coordinates
        if all(v is None for v in transData.M) == True:
            resp = requests.get(webPath + str(round(long, 1)) +
                                "&lat=" + str(round(lat, 1)) +
                                "&attributes=%s" % str(property))
            sigData = pd.DataFrame(resp.json()).loc[property]
            transData = pd.DataFrame.from_dict(sigData.properties)

        #  transpose
        preparedSigData = transData.T.tail(1).add_prefix('%s_'%property).reset_index(drop=True)
        soilInfoData = pd.concat([soilInfoData, preparedSigData], axis=1)

    soilInfoData['long'] = long
    soilInfoData['lat'] = lat
    return soilInfoData

# # get top three classes with high probability

def get_mostProbClass(lat, long, classProperties):

    webPath = "https://rest.soilgrids.org/query?lon="
    soilClassData = pd.DataFrame()


    for property in classProperties:
        resp = requests.get(webPath + str(round(long,2)) +
                            "&lat=" + str(round(lat,2)) + "&attributes=%s"%str(property))

        # no this property existed for the coordinates, then round 1
        if (property not in pd.DataFrame(resp.json()).index) == True:
            resp = requests.get(webPath + str(round(long, 1)) +
                                "&lat=" + str(round(lat, 1)) + "&attributes=%s" % str(property))

            # ScottBase has no information about soil types
            if resp.json()['properties']['soilmask'] == 'nodata':
                break

        sigData = pd.DataFrame(resp.json())
        classDict = sigData.loc[property].properties

        # none values of property for this specific coordinates is higher than 95%, then round1
        if sum(v is None for v in classDict.values())/len(classDict) > 0.95:
            resp = requests.get(webPath + str(round(long, 1)) +
                                "&lat=" + str(round(lat, 1)) + "&attributes=%s" % str(property))
            sigData = pd.DataFrame(resp.json())
            classDict = sigData.loc[property].properties

            # none value for round 1 is still higher than 95%, then imputation with np.nan
            if sum(v is None for v in classDict.values())/len(classDict) > 0.95:
                mostProbClassDict = {'%s_Class1' % property: np.nan, '%s_Class2' % property: np.nan,
                                     '%s_Class3' % property: np.nan}
                sigClassData = pd.DataFrame(mostProbClassDict, index=[0])
                soilClassData = pd.concat([soilClassData, sigClassData], axis=1)
                continue

        sortedDict = sorted(classDict, key=lambda k: classDict[k], reverse=True)

        # get top three classes with high probability
        mostProbClassDict = {'%s_Class1' % property: sortedDict[0], '%s_Class2' % property: sortedDict[1],
                             '%s_Class3' % property: sortedDict[2]}

        sigClassData = pd.DataFrame(mostProbClassDict, index=[0])
        soilClassData = pd.concat([soilClassData, sigClassData], axis=1)

    soilClassData['lat'] = lat
    soilClassData['long'] = long

    return soilClassData
