import pandas as pd
import numpy as np
import plotly
from plotly.subplots import make_subplots
from plotly import graph_objs as go

# use like this: rainfallPlot(dataPlot, ['p','fore_EWLR','LSTM','actual'], 'Selma')
def rainfallPlot(dataset, targets, station):
    data = dataset[dataset.station == station]
    data.index = data.timeStamp

    plotly.offline.init_notebook_mode(connected=True)

    fig = go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    colors = ['#D3D3D3', 'chocolate', '#406883', 'darkred', 'lightblue', 'darkblue', 'peachpuff', 'darkorange']
    i = 0

    for target in targets:

        if target == 'p':
            fig.add_trace(
                go.Scatter(x=data.timeStamp, y=data[target], name=target, mode='lines', marker_color=colors[i],
                           line=dict(shape='linear', color=colors[i], width=2)), secondary_y=True)

        else:
            fig.add_trace(
                go.Scatter(x=data.timeStamp, y=data[target], name=target, mode='lines', marker_color=colors[i],
                           line=dict(shape='linear', color=colors[i], width=2)), secondary_y=False)
        i = i + 1

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_yaxes(title_text=r'$\text{Soil Moiture} (m^{3}m^{-3})$', secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="Precipitation", secondary_y=True, showgrid=False, range=[0, 70])

    fig.layout.update(
        title=station,
        autosize=False, width=1000, height=400
    )
    fig.update_layout({
        "plot_bgcolor": "rgba(0, 0, 0, 0)",
        "paper_bgcolor": "rgba(0, 0, 0, 0)",
    })

    return plotly.offline.iplot(fig)
