# -*- coding: utf-8 -*-
import base64
import io
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

# Initialize the data and label global variable
data_df, label_df = None, None

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

data = []

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


def input_field(title, state_id, state_value, state_max, state_min):
    """Takes as parameter the title, state, default value and range of an input field, and output a Div object with
    the given specifications."""
    return html.Div([
        html.P(title,
               style={
                   'display': 'inline-block',
                   'verticalAlign': 'mid',
                   'marginRight': '5px',
                   'margin-bottom': '0px',
                   'margin-top': '0px'
               }),

        html.Div([
            dcc.Input(
                id=state_id,
                type='number',
                value=state_value,
                max=state_max,
                min=state_min,
                size=7
            )
        ],
            style={
                'display': 'inline-block',
                'margin-top': '0px',
                'margin-bottom': '0px'
            }
        )
    ]
    )


# Layout for the t-SNE graph
tsne_layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)

# App
app.layout = html.Div([
    html.H1(
        'T-SNE Explorer',
        id='title',
        style={'text-align': 'center'}
    ),

    html.Div([
        html.Div([
            dcc.Graph(
                id='tsne-3d-plot',
                figure={
                    'data': data,
                    'layout': tsne_layout
                },
                style={
                    'height': '80vh',
                    # 'width': '60vw',
                },
            )
        ],
            className="seven columns offset-by-one"
        ),

        html.Div([

            html.H4(
                'T-SNE Parameters',
                id='tsne_h4'
            ),

            input_field("Perplexity:", "perplexity-state", 30, 50, 5),

            input_field("Number of Iterations:", "n-iter-state", 300, 1000, 250),

            # input_field("Iterations without Progress:", "iter-wp-state", 300, 1000, 50),

            input_field("Learning Rate:", "lr-state", 200, 1000, 10),

            # TODO: Change the max value to be the dimension of the input csv file
            input_field("PCA dimensions:", "pca-state", 50, 100, 3),

            html.Button(
                id='tsne-train-button',
                n_clicks=0,
                children='Start Training t-SNE'
            ),

            dcc.Upload(
                id='upload-data',
                children=html.A('Upload your main data here.'),
                style={
                    'height': '45px',
                    'line-height': '45px',
                    'border-width': '1px',
                    'border-style': 'dashed',
                    'border-radius': '5px',
                    'text-align': 'center',
                    'margin-top': '5px',
                    'margin-bottom': '5 px'
                },
                multiple=False
            ),

            dcc.Upload(
                id='upload-label',
                children=html.A('Upload your labels here.'),
                style={
                    'height': '45px',
                    'line-height': '45px',
                    'border-width': '1px',
                    'border-style': 'dashed',
                    'border-radius': '5px',
                    'text-align': 'center',
                    'margin-top': '5px',
                    'margin-bottom': '5px'
                },
                multiple=False
            ),

            html.P(id='upload-data-message',
                   style={
                       'margin-bottom': '0px'
                   }),
            html.P(id='upload-label-message',
                   style={
                       'margin-bottom': '0px'
                   })

        ],
            className="three columns"
        )
    ],
        className="row"
    )
]
)


def parse_content(contents, filename):
    """This function parses the raw content and the file names, and returns the dataframe containing the data, as well
    as the message displaying whether it was successfully parsed or not."""

    if contents is None:
        return None, ""

    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)

    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))

        else:
            return None, 'The file uploaded is invalid.'
    except Exception as e:
        print(e)
        return None, 'There was an error processing this file.'

    return df, f'{filename} successfully processed.'


@app.callback(Output('upload-data-message', 'children'),
              [Input('upload-data', 'contents'),
               Input('upload-data', 'filename')
               ])
def parse_data(contents, filename):
    global data_df

    data_df, message = parse_content(contents, filename)
    return message


@app.callback(Output('upload-label-message', 'children'),
              [Input('upload-label', 'contents'),
               Input('upload-label', 'filename')
               ])
def parse_label(contents, filename):
    global label_df  # Modify the global label dataframe

    label_df, message = parse_content(contents, filename)
    return message


@app.callback(Output('tsne-3d-plot', 'figure'),
              [Input('tsne-train-button', 'n_clicks')],  # TODO: n_clicks is uneeded here, find the right way to use button as input
              [State('perplexity-state', 'value'),
               State('n-iter-state', 'value'),
               State('lr-state', 'value'),
               State('pca-state', 'value')])
def update_graph(n_clicks, perplexity, n_iter, learning_rate, pca_dim):
    """When the button is clicked, the t-SNE algorithm is run, and the graph is updated when it finishes running"""

    print(perplexity, n_iter, learning_rate, pca_dim)

    # TODO: This is a temporary fix to the null error thrown. Need to find more reasonable solution.
    if data_df is None or label_df is None:
        global data
        return {'data': data, 'layout': tsne_layout}  # Return the default values

    pca = PCA(n_components=3)

    # Combine the reduced data with its label
    reduced_df = pd.DataFrame(pca.fit_transform(data_df), columns=['x', 'y', 'z'])

    label_df.columns = ['label']

    combined_df = reduced_df.join(label_df)

    data = []

    # Group by the values of the label
    for idx, val in combined_df.groupby('label'):
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

    return {'data': data, 'layout': tsne_layout}


# Load external CSS
external_css = [
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
    "//fonts.googleapis.com/css?family=Raleway:400,300,600",
    "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css"]

for css in external_css:
    app.css.append_css({"external_url": css})

# Running the server
if __name__ == '__main__':
    app.run_server(debug=True)
