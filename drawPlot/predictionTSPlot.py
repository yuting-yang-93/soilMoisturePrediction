from plotly import graph_objs as go
import plotly

# use like this
# predictionTSPlot(data, ['sm_5','EWLR','LSTM','actual'], 'Tunica')
def predictionTSPlot(dataset, targets, station):
    '''

    :param dataset: DataFrame. Predicted data set, including timeStamp, station, actual value, predicted values
    :param targets: list. The variables which will be showed in the plot
    :param station: Str. the station name
    :return: plot
    '''
    data = dataset[dataset.station == station]
    data.index = data.timeStamp

    plotly.offline.init_notebook_mode(connected=True)

    fig = go.Figure()

    colors = ['chocolate', 'lightslategrey', '#98AFC7', 'lightblue', 'darkblue', 'peachpuff', 'darkorange']
    i = 0

    for target in targets:
        if target == 'actual':
            fig.add_trace(
                go.Scatter(x=data.timeStamp, y=data[target], name=target, mode='lines', marker_color='darkred',
                           line=dict(shape='linear', color='darkred', width=1)))

        elif target == 'sm_5':
            fig.add_trace(
                go.Scatter(x=data.timeStamp, y=data[target], name=target, mode='lines', marker_color='darkred',
                           line=dict(shape='linear', color='#F0F0F0', width=1)))
        elif target == 'LSTM':
            fig.add_trace(
                go.Scatter(x=data.timeStamp, y=data[target], name=target, mode='lines', marker_color='darkred',
                           line=dict(shape='linear', color='#406883', width=1)))

        else:
            fig.add_trace(
                go.Scatter(x=data.timeStamp, y=data[target], name=target, mode='lines', marker_color=colors[i],
                           line=dict(shape='linear', color=colors[i], width=1)))
            i = i + 1

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    fig.layout.update(
        title=station,
        yaxis_title=r'$\text{Soil Moiture} (m^{3}m^{-3})$',
        autosize=False, width=1000, height=400
    )

    return plotly.offline.iplot(fig)



import pandas as pd
import os
from pyecharts import Line, Overlap
# target = 20
# localPath = 'model_LSTM_20cm'
def forecastPlot(foreData, target, localPath):
    """
    :param ismn_data: DataFrame. Merged dataset of ISMN
    :return: html file stored in paht './plot/ismn_ts'
    """
    # create a new folder: plot/ismn_ts
    path = '../{}/predictionPlot'.format(localPath)
    if not os.path.exists(path):
        os.makedirs(path)
    foreData = foreData.copy()
    foreData['timeStamp'] = pd.to_datetime(foreData['timeStamp'])

    for station in foreData.station.unique():
        line = Line("forecast of soil mosture at %i cm"%target, station)
        overlap = Overlap(width=1200, height=500)
        dataset = foreData[foreData.station == station]
        dataset.index = dataset.timeStamp

        fore_columns = list(dataset.filter(regex = 'fore|actual').columns)
        for forecastVar in fore_columns:
            x=dataset.index
            y=list(dataset.loc[:, [forecastVar]].values.reshape(-1))
            line.add(forecastVar, x, y, is_datazoom_show=True)

        overlap.add(line)
        overlap.render(path=path + '/{0}_forecast_{1}cm.html'.format(station, target))

