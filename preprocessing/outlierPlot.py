import pandas as pd
dataset = pd.read_csv('../data/cleanDataEqu.csv')
dataset.station.unique()

station = 'UAPBMarianna'
target = 'sm_20'
data = dataset.loc[dataset.station == station,['sm_5','sm_20','sm_50','sm_100','p','st_5','at','anomaly','timeStamp']]

print(len(data[data.anomaly == 1]))
print(len(data[data.anomaly == -1]))
import numpy as np
data.index = data.timeStamp
data['outlier'] = np.where(data['anomaly'] == -1, data[target], np.nan)
data['normal'] = np.where(data['anomaly'] != -1, data[target], np.nan)

from plotly import graph_objs as go
import plotly
plotly.offline.init_notebook_mode(connected=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.timeStamp, y=data.outlier, name="outlier", mode = 'markers',marker_size=2, marker_color = 'darkred'))

fig.add_trace(go.Scatter(x=data.timeStamp, y=data.normal, name="normal",mode = 'markers',marker_size=2,marker_color = 'lightgrey'))

# plotly.offline.iplot(fig)
fig.show()