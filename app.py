# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import numpy as np
import plotly.graph_objs as go

app = dash.Dash("T-SNE")
server = app.server


x, y, z = np.random.multivariate_normal(np.array([0,0,0]), np.eye(3), 200).transpose()
trace1 = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=3,
        line=dict(
            color='rgb(217, 217, 217)'
        )
    )
)

x2, y2, z2 = np.random.multivariate_normal(np.array([0,0,0]), np.eye(3), 200).transpose()
trace2 = go.Scatter3d(
    x=x2,
    y=y2,
    z=z2,
    mode='markers',
    marker=dict(
        color='rgb(127, 127, 127)',
        size=3,
        symbol='circle-dot'
    )
)

data = [trace1, trace2]


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
                   'border': '3px solid green'
                   }
        )
])

if __name__ == '__main__':
    app.run_server(debug=True)
