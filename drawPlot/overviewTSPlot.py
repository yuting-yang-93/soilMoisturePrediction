from plotly import graph_objs as go
import plotly

# use like this: plotDraw(data, ['sm_20','sm_50','sm_100'], 'GrouseCreek')
def overviewPlot(dataset, targets, station):
    data = dataset[dataset.station == station]
    data.index = data.timeStamp

    plotly.offline.init_notebook_mode(connected=True)

    fig = go.Figure()

    # colors = ['#bccad6', ' #8d9db6', '#667292', '#f1e3dd']
    colors = ['#92a8d1', '#034f84', '#f7786b']
    i = 0

    for target in targets:
        fig.add_trace(go.Scatter(x=data.timeStamp, y=data[target], name=target, mode='lines', marker_color=colors[i],
                                 line=dict(shape='linear', color=colors[i], width=1)))

        i = i + 1
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    fig.layout.update(
        title=station,
        # xaxis_title="Timestamps",
        yaxis_title=r'$\text{Soil Moiture} (m^{3}m^{-3})$',
        autosize=False, width=1000, height=500
    )

    return plotly.offline.iplot(fig)