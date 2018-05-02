# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.decomposition import PCA


app = dash.Dash("T-SNE")
server = app.server

tsne_df = pd.read_csv("tsne_3d.csv", index_col=0)

data = []

color_list = [
    'rgb(0,0,255)',
    'rgb(0, 255, 0)',
    'rgb(255, 0, 0)',
    'rgb(0, 0, 0)',
    'rgb(165,42,42)',
    'rgb(255,222,173)',
    'rgb(255,105,180)',
    'rgb(75,0,130)',
    'rgb(255,215,0)',
    'rgb(255,165,0)'
]

for idx, val in tsne_df.groupby(tsne_df.index):
    idx = int(idx)

    scatter = go.Scatter3d(
            name=idx,
            x=val['x'],
            y=val['y'],
            z=val['z'],
            mode='markers',
            marker=dict(
                color=color_list[idx],
                size=2,
                symbol='circle-dot'
            )
        )
    data.append(scatter)


app.layout = html.Div([
    html.H1(
                'T-SNE Explorer',
                className='tsne_h1',
                style={'text-align': 'center'}
            ),
    dcc.Graph(
            id='scatter_3d',
            figure={
                'data': data,
            },
            style={'height': '75vh',
                   'width': '75vw',
                   'border': '3px solid green',
                   'margin':'auto'
                   }
        )
]
)

if __name__ == '__main__':
    app.run_server(debug=True)
