import pandas as pd
import numpy as np
import re
import configuration


# import and merge data from all stations in one network
def merge_stations_sm(ISMN_reader, network, date_start, date_end):
    """
    :param network: String
           date_start, date_end: date
    :return: merged_station_ts: DataFrame. Soil moisture from all stations of one network
    """

    #===== put into the configuration file

    config = configuration.ModelConfiguration()

    missingVarStation = []
    missingDepthStation = []
    notUniqueSensorStation = []

    doubtSensor = []

    stations = ISMN_reader.list_stations(network=network)
    print ("Number of stations: " + str(len(stations)))

    # create DataFrame to combine data from all stations which belong to the same network
    # col_name = ['sm_5', 'sm_10', 'sm_20', 'sm_50', 'sm_100', 'station']
    merged_station_ts = pd.DataFrame()

    i_station = 0
    for station in stations:

        missingVar = False
        notCompleteSensor = False
        i_station = i_station + 1
        print (station)

        station_obj = ISMN_reader.get_station(station)
        print(str(i_station) + ". Station: %s" % station)

        # get the variables that this station measures:
        # ['air temperature' 'precipitation' 'soil moisture' 'soil temperature']
        # variables = station_obj.get_variables()
        variables = ['soil moisture', 'precipitation', 'soil temperature', 'air temperature']


        depths_from, depths_to = station_obj.get_depths('soil moisture')
        sensors = station_obj.get_sensors('soil moisture', depths_from[0], depths_to[0])
        print('sensors', sensors)
        i = 65 # chr(65) = 'A'

        for sensor in sensors:
            sensorCode = chr(i)
            i = i + 1

            # create DataFrame to store multi-depth soil moisture for one station
            # old version
            # merge_variable = pd.DataFrame(index=pd.date_range(date_start, date_end, freq='H'))
            # new version
            merge_variable = pd.DataFrame(
                index=pd.date_range(date_start, date_end, freq=config.timeSeriesUnit, tz='UTC'))


            for variable in variables:
                # depths_from, e.g. [0.05, 0.1, 0.2, 0.51, 1.02]
                # depths _to, e.g. [0.05, 0.1, 0.2, 0.51, 1.02]

                depths_from, depths_to = station_obj.get_depths(variable)

                # ======== stations can not be considered
                # === requirement 1
                # if the variable table is missing, then we do not need this station data
                if ((depths_from is None) and (depths_to is None)):
                    missingVar = True
                    missingVarStation.append([station,variable])
                    print ('the variable %(var)s in the station %(sta)s is empty'%{'var': variable, 'sta':station})
                    break

                # === requirement 2
                # if the sm in the one depth what we need is missing, then we do not need this station data
                actual_depth = [int(i * 100 / 5) * 5 for i in depths_from]
                if ((variable == 'soil moisture') and (set(config.need_depth).issubset(set(actual_depth)) != True)):
                    missingVar = True
                    missingDepthStation.append(station)
                    print('One or more depth information is missing')
                    break

                # ========= Standard variable names
                # design the column name
                if variable == 'precipitation':
                    abbrev = 'p'

                # get the prefix, e.g. 'sm', 'at', 'st'
                else:
                    firstLetter = re.search('^(.?)', variable)
                    secLetter = re.search('\s(.+?)', variable)
                    abbrev = firstLetter.group(0) + secLetter.group(0).replace(" ", "")

                # ========= depths of each variables
                unique_depths_from = list(set(list(depths_from)))
                for depth in unique_depths_from:
                    # read the data of the variable, depth, sensor combination
                    roundDepth = int(depth * 100 / 5) * 5

                    # if there are some depths we don't need, then direct ignore
                    if ((variable in ['soil moisture', 'soil temperature']) and (roundDepth not in config.need_depth)):
                        continue

                    if variable in ['precipitation', 'air temperature']: # only one depth
                        currSensors = station_obj.get_sensors(variable, depth_from=depth, depth_to=depth)
                        # if precipitation or air temperature has more than one sensor
                        # take 'Pulse-Count' as prima sensor for precipitation, otherwise take the first one
                        if len(currSensors) > 1:
                            if 'Pulse-Count' in currSensor and variable == 'precipitation':
                                currSensor = 'Count'
                            else:
                                currSensor = currSensors[0]
                                doubtSensor.append([station, variable, currSensor, depth])
                        else:
                            currSensor = currSensors

                    # soil temperature
                    elif variable in ['soil temperature']:
                        currSensors = station_obj.get_sensors(variable, depth_from=depth, depth_to=depth)
                        if sensor not in currSensors:
                            currSensor = currSensors[0]
                            doubtSensor.append([station, variable, currSensor, depth])
                        else:
                            currSensor = sensor

                    # soil moisture
                    else:
                        currSensors = station_obj.get_sensors(variable, depth_from=depth, depth_to=depth)
                        if sensor not in currSensors:
                            doubtSensor.append([station, variable, currSensor, depth])
                            notCompleteSensor = True
                            break
                        else:
                            currSensor = sensor
                    print(variable, sensor, depth)
                    time_series = station_obj.read_variable(variable, depth_from=depth, depth_to=depth, sensor=currSensor)
                    print(time_series)
                    time_series_data = time_series.data

                    # =============
                    # add parameters, hour or day
                    time_series_hour = time_series_data.resample(config.timeSeriesUnit).mean()

                    # to make sure the depths in column name are exactly divided by 5, avoid e.g. sm_101
                    time_series_hour.columns = [abbrev + '_' + str(roundDepth)]

                    merge_variable = merge_variable.merge(time_series_hour, left_index=True, right_index=True, how='left')

                if notCompleteSensor == True:
                    break
            if notCompleteSensor == True:
                break

            if missingVar != True:
                merge_variable['station'] = station
                merge_variable['sensor'] = sensor
                merge_variable['sensorCode'] = sensorCode
                merged_station_ts = pd.concat([merged_station_ts, merge_variable], axis=0, sort=False)
                print(merged_station_ts.tail(5))
    merged_station_ts['network'] = network

    print ('stations with missing variables', missingVarStation)
    print ('stations with missing depths', missingDepthStation)
    print ('stations with more than one station', notUniqueSensorStation)
    print ('check', doubtSensor)

    doubtSensorDf = pd.DataFrame(doubtSensor)
    doubtSensorDf.to_csv('doubtSensor.csv', index=False)
    return merged_station_ts.rename(columns={'p_0':'p','at_-200':'at'})


# merge all networks
def merge_network(ISMN_reader, networks, date_start, date_end):
    """
    :param network: String
           date_start, date_end: date
    :return: merged_network_ts: DataFrame. Soil moisture from all stations and all networks
    """
    merged_network_ts = pd.DataFrame()
    for network in networks:
        merged_station_ts = merge_stations_sm(ISMN_reader, network, date_start, date_end)
        merged_network_ts = pd.concat([merged_network_ts, merged_station_ts], axis=0)
    return merged_network_ts



