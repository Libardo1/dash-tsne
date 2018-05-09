# -*- coding: utf-8 -*-
import base64
import io
import os
import dash
import time
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import numpy as np
import flask
from flask_cors import CORS
import pandas as pd
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

app = dash.Dash(__name__)
server = app.server
CORS(server)

if 'DYNO' in os.environ:
    app.scripts.append_script({
        'external_url': 'https://cdn.rawgit.com/chriddyp/ca0d8f02a1659981a0ea7f013a378bbd/raw/e79f3f789517deec58f41251f7dbb6bee72c44ab/plotly_ga.js'
    })

# Generate the default scatter plot
tsne_df = pd.read_csv("data/tsne_3d.csv", index_col=0)

data = []

for idx, val in tsne_df.groupby(tsne_df.index):
    idx = int(idx)

    scatter = go.Scatter3d(
        name=f"Digit {idx}",
        x=val['x'],
        y=val['y'],
        z=val['z'],
        mode='markers',
        marker=dict(
            size=2.5,
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
    # In-browser storage of global variables
    html.Div(
        id="data-df-and-message",
        style={'display': 'none'}
    ),

    html.Div(
        id="label-df-and-message",
        style={'display': 'none'}
    ),

    # Main app
    html.Div([
        html.H2(
            't-SNE Explorer',
            id='title',
            style={
                'float': 'left',
                'margin-top': '20px',
                'margin-bottom': '0',
                'margin-left': '7px'
            }
        ),
        html.Img(
            src="https://s3-us-west-1.amazonaws.com/plotly-tutorials/logo/new-branding/dash-logo-by-plotly-stripe.png",
            style={
                'height': '100px',
                'float': 'right'
            }
        )
    ],
        className="row"
    ),

    html.Div([
        html.Div([
            # Data about the graph
            html.Div(
                id="kl-divergence",
                style={'display': 'none'}
            ),

            html.Div(
                id="end-time",
                style={'display': 'none'}
            ),

            html.Div(
                id="error-message",
                style={'display': 'none'}
            ),

            # The graph
            dcc.Graph(
                id='tsne-3d-plot',
                figure={
                    'data': data,
                    'layout': tsne_layout
                },
                style={
                    'height': '80vh',
                },
            )
        ],
            id="plot-div",
            className="eight columns"
        ),

        html.Div([

            html.H4(
                't-SNE Parameters',
                id='tsne_h4'
            ),

            input_field("Perplexity:", "perplexity-state", 30, 50, 5),

            input_field("Number of Iterations:", "n-iter-state", 300, 1000, 250),

            input_field("Learning Rate:", "lr-state", 200, 1000, 10),

            input_field("Initial PCA dimensions:", "pca-state", 30, 10000, 3),

            html.Button(
                id='tsne-train-button',
                n_clicks=0,
                children='Start Training t-SNE'
            ),

            dcc.Upload(
                id='upload-data',
                children=html.A('Upload your input data here.'),
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
                multiple=False,
                max_size=-1
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
                multiple=False,
                max_size=-1
            ),

            html.Div([
                html.P(id='upload-data-message',
                       style={
                           'margin-bottom': '0px'
                       }),

                html.P(id='upload-label-message',
                       style={
                           'margin-bottom': '0px'
                       }),

                html.Div(id='training-status-message',
                         style={
                             'margin-bottom': '0px',
                             'margin-top': '0px'
                         }),

                html.P(id='error-status-message')
            ],
                id='output-messages',
                style={
                    'margin-bottom': '2px',
                    'margin-top': '2px'
                }
            )
        ],
            className="four columns"
        )
    ],
        className="row"
    ),

    html.Div([
        dcc.Markdown('''
**What is t-SNE?**
t-distributed stochastic neighbor embedding, created by van der Maaten and Hinton in 2008, is a visualization algorithm that reduce a high-dimensional space (e.g. an image or a word embedding) into two or three dimensions, so we can visualize how the data is distributed. A classical example is MNIST, a dataset of 60,000 handwritten digits of size 28x28 in black and white. When you reduce the MNIST dataset using t-SNE, you can clearly see all the digit clustered together, with the exception of a few that might have been poorly written. [You can read a detailed explanation of the algorithm on van der Maaten's personal blog.](https://lvdmaaten.github.io/tsne/)

**How to use the app**
To train your own t-SNE, you can input your own high-dimensional dataset and the corresponding labels inside the upload fields. For convenience, small sample datasets are included inside the data folder. The training can take a lot of time depending on the size of the dataset (the complete MNIST dataset could take 15-30 min), so it is recommended to clone the repo and run the app locally if you want to use bigger datasets. [You can find the repository containing this model here.](https://github.com/plotly/dash-tsne)''')
    ],
        style={
            'margin-top': '15px'
        },
        className="row"
    )
],
    className="container",
    style={
        'width': '90%',
        'max-width': 'none',
        'font-size': '1.5rem'
    }
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


# Uploaded data --> Hidden Data Div
@app.callback(Output('data-df-and-message', 'children'),
              [Input('upload-data', 'contents'),
               Input('upload-data', 'filename')])
def parse_data(contents, filename):
    data_df, message = parse_content(contents, filename)

    if data_df is None:
        return [None, message]

    elif data_df.shape[1] < 3:
        return [None, f'The dimensions of {filename} are invalid.']

    return [data_df.to_json(orient="split"), message]


# Uploaded labels --> Hidden Label div
@app.callback(Output('label-df-and-message', 'children'),
              [Input('upload-label', 'contents'),
               Input('upload-label', 'filename')])
def parse_label(contents, filename):
    label_df, message = parse_content(contents, filename)

    if label_df is None:
        return [None, message]

    elif label_df.shape[1] != 1:
        return [None, f'The dimensions of {filename} are invalid.']

    return [label_df.to_json(orient="split"), message]


# Hidden Data Div --> Display upload status message (Data)
@app.callback(Output('upload-data-message', 'children'),
              [Input('data-df-and-message', 'children')])
def output_upload_status_data(data):
    return data[1]


# Hidden Label Div --> Display upload status message (Labels)
@app.callback(Output('upload-label-message', 'children'),
              [Input('label-df-and-message', 'children')])
def output_upload_status_label(data):
    return data[1]


# Button Click --> Update graph with states
@app.callback(Output('plot-div', 'children'),
              [Input('tsne-train-button', 'n_clicks')],
              [State('perplexity-state', 'value'),
               State('n-iter-state', 'value'),
               State('lr-state', 'value'),
               State('pca-state', 'value'),
               State('data-df-and-message', 'children'),
               State('label-df-and-message', 'children')
               ])
def update_graph(n_clicks, perplexity, n_iter, learning_rate, pca_dim, data_div, label_div):
    """Run the t-SNE algorithm upon clicking the training button"""

    error_message = None  # No error message at the beginning

    # Fix for startup POST
    if n_clicks <= 0 or data_div is None or label_div is None:
        global data
        kl_divergence, end_time = None, None

    else:
        # Extract the data dataframe and the labels dataframe from the divs. they are both the first child of the div,
        # and are serialized in json
        data_df = pd.read_json(data_div[0], orient="split")
        label_df = pd.read_json(label_div[0], orient="split")

        # Fix the range of possible values
        if n_iter > 1000:
            n_iter = 1000
        elif n_iter < 10:
            n_iter = 10

        if perplexity > 50:
            perplexity = 50
        elif perplexity < 5:
            perplexity = 5

        if learning_rate > 1000:
            learning_rate = 1000
        elif learning_rate < 10:
            learning_rate = 10

        if pca_dim > data_df.shape[1]:  # We limit the pca_dim to the dimensionality of the dataset
            pca_dim = data_df.shape[1]
        elif pca_dim < 3:
            pca_dim = 3

        # Start timer
        start_time = time.time()

        # Apply PCA on the data first
        pca = PCA(n_components=pca_dim)
        data_pca = pca.fit_transform(data_df)

        # Then, apply t-SNE with the input parameters
        tsne = TSNE(n_components=3,
                    perplexity=perplexity,
                    learning_rate=learning_rate,
                    n_iter=n_iter)

        try:
            data_tsne = tsne.fit_transform(data_pca)
            kl_divergence = tsne.kl_divergence_

            # Combine the reduced t-sne data with its label
            tsne_data_df = pd.DataFrame(data_tsne, columns=['x', 'y', 'z'])

            label_df.columns = ['label']

            combined_df = tsne_data_df.join(label_df)

            data = []

            # Group by the values of the label
            for idx, val in combined_df.groupby('label'):
                scatter = go.Scatter3d(
                    name=idx,
                    x=val['x'],
                    y=val['y'],
                    z=val['z'],
                    mode='markers',
                    marker=dict(
                        size=2.5,
                        symbol='circle-dot'
                    )
                )
                data.append(scatter)

            end_time = time.time() - start_time

        # Catches Heroku server timeout
        except:
            error_message = "We were unable to train the t-SNE model due to timeout. Try to clone the repo and run the program locally."
            kl_divergence, end_time = None, None
            # return [
            #     html.Div(
            #         id="kl-divergence",
            #         style={'display': 'none'}
            #     ),
            #
            #     html.Div(
            #         id="end-time",
            #         style={'display': 'none'}
            #     ),
            #
            #     html.Div([
            #         "We were unable to train the t-SNE model due to timeout. Try to clone the repo and run the program locally."
            #     ],
            #         id="error-message",
            #         style={'display': 'none'}
            #     ),
            #
            #     # The graph
            #     dcc.Graph(
            #         id='tsne-3d-plot',
            #         figure={
            #             'data': data,
            #             'layout': tsne_layout
            #         },
            #         style={
            #             'height': '80vh',
            #         },
            #     )
            # ]

    return [
        # Data about the graph
        html.Div([
            kl_divergence
        ],
            id="kl-divergence",
            style={'display': 'none'}
        ),

        html.Div([
            end_time
        ],
            id="end-time",
            style={'display': 'none'}
        ),

        html.Div([
            error_message
        ],
            id="error-message",
            style={'display': 'none'}
        ),

        # The graph
        dcc.Graph(
            id='tsne-3d-plot',
            figure={
                'data': data,
                'layout': tsne_layout
            },
            style={
                'height': '80vh',
            },
        )
    ]


# Updated graph --> Training status message
@app.callback(Output('training-status-message', 'children'),
              [Input('end-time', 'children'),
               Input('kl-divergence', 'children')])
def update_training_info(end_time, kl_divergence):
    # If an error message was output during the training.

    if end_time is None or kl_divergence is None or end_time[0] is None or kl_divergence[0] is None:
        return None
    else:
        end_time = end_time[0]
        kl_divergence = kl_divergence[0]

        return [
            html.P(f"t-SNE trained in {end_time:.2f} seconds.",
                   style={'margin-bottom': '0px'}),
            html.P(f"Final KL-Divergence: {kl_divergence:.2f}",
                   style={'margin-bottom': '0px'})
        ]


@app.callback(Output('error-status-message', 'children'),
              [Input('error-message', 'children')])
def show_error_message(error_message):
    if error_message is not None:
        return [
            html.P(error_message[0])
        ]

    else:
        return []


# Load external CSS
external_css = [
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
    "//fonts.googleapis.com/css?family=Raleway:400,300,600",
    "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
    # "https://cdn.rawgit.com/plotly/dash-tsne/92bb6e64/loading_screen.css",
    "https://raw.githubusercontent.com/plotly/dash-tsne/master/loading_screen.css",
    "https://cdn.rawgit.com/plotly/dash-app-stylesheets/2cc54b8c03f4126569a3440aae611bbef1d7a5dd/stylesheet.css"
]

for css in external_css:
    app.css.append_css({"external_url": css})

# Running the server
if __name__ == '__main__':
    app.run_server(debug=True)
