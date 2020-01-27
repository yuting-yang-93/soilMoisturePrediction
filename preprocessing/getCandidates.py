def candidatesOfMainSoilType(preparedData):
    '''
    combine the TAXNWRB_Class1 and TAXOUSDA_Class1.
    If the number of stations, which belongs to the same TAXNWRB_Class1 and same TAXOUSDA_Class1, is not smaller than 5,
    then set these stations as candidates
    :param preparedData:
    :return:dataInMainSoilTypes: dataFrame, whose stations belong to the same soil types
    '''
    frequencySoilType = preparedData.loc[:,['TAXNWRB_Class1','TAXOUSDA_Class1','station']].\
        drop_duplicates().groupby(['TAXNWRB_Class1','TAXOUSDA_Class1']).count().\
        sort_values('station', ascending=False)

    candidates = frequencySoilType[frequencySoilType.station >= 5]

    candidatesDF = candidates.drop(['station'], axis = 1)

    dataInMainSoilTypes = preparedData.merge(candidatesDF, how = 'inner', on = ['TAXNWRB_Class1','TAXOUSDA_Class1'])

    return dataInMainSoilTypes