import pandas as pd
# get the lat and long information of each station
def get_station_coordinates(ISMN_reader, station):
    """
    :param station: String
    :return: lat, long (both floats)
    """
    station_obj = ISMN_reader.get_station(station)
    return station_obj.latitude, station_obj.longitude

# get the lat and long information for all stations
def merge_stations_coordinates(ISMN_reader, network):
    """
    :param network: String
    :return: merged_station_ts: DataFrame. Soil moisture from all stations of one network
    """
    stations = ISMN_reader.list_stations(network=network)
    print ("Number of stations: " + str(len(stations)))

    # create DataFrame to combine data from all stations which belong to the same network

    lat_array = []
    long_array = []
    station_array = []

    for station in stations:
        print('get the coordinates of station %s'%station)
        lat, long = get_station_coordinates(ISMN_reader, station)
        lat_array.append(lat)
        long_array.append(long)
        station_array.append(station)
    merge_station_coor_dict = {'station':station_array,'lat':lat_array,'long':long_array}
    return merge_station_coor_dict

# merge all networks
def merge_networks_coordinates(ISMN_reader, networks):
    """
    :param network: String
           date_start, date_end: date
    :return: merged_network_ts: DataFrame. Soil moisture from all stations and all networks
    """
    merged_network_coor = pd.DataFrame()
    for network in networks:
        merge_station_coor_dict = merge_stations_coordinates(ISMN_reader, network)
        merge_station_coor = pd.DataFrame.from_dict(merge_station_coor_dict)
        merge_station_coor['network'] = network

        merged_network_coor = pd.concat([merged_network_coor, merge_station_coor], axis=0)
    return merged_network_coor