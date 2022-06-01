import numpy as np
from skimage import io
import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from dash import html, dcc
from dash_canvas import DashCanvas
from dash_canvas.utils import  parse_jsonstring
# from dash_canvas.utils import array_to_data_url
# from PIL import Image
# from io import BytesIO
# import base64
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
import pickle


# Load pkl files
with open('model_outputs/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model_outputs/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# with open('model_outputs/xgb_model_updated.pkl', 'rb') as f:
#     xgb_model = pickle.load(f)

with open('analysis/optimization/cahn/xgb_model6.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

# Global variables
tabtitle ='Digits classifier'
sourceurl = 'https://scikit-learn.org/stable/auto_examples/classification/' \
            'plot_digits_classification.html'
githublink = 'https://github.com/cahn1/506-digit-classifier-xgboost/tree' \
             '/update1'
canvas_size = 200

# Figure templates
templates=['plotly', 'ggplot2', 'seaborn', 'simple_white', 'plotly_white',
           'plotly_dark', 'presentation', 'xgridoff', 'ygridoff', 'gridon', 'none']
data = []
layout= go.Layout(
    xaxis={
        'showgrid': True,
        'visible': True,
        'showticklabels': False,
        'showline': True,
        'zeroline': False,
        'mirror': True,
        'ticks': None},
    yaxis={
        'showgrid': True,
        'visible': True,
        'showticklabels':False,
        'showline': True,
        'zeroline': False,
        'mirror': True,
        'ticks': None},
    newshape={
        'line_color': None,
        'fillcolor': None,
        # 'opacity':0.8,
        # 'line':{'width':30}
    },
    template=templates[6],
    font_size=12,
    dragmode='drawopenpath',
    width=580,
    height=630
)
blank_fig = go.Figure(data, layout)


# Helper function
def squash_matrix(df, cols, rows):
    x = 0
    col_cut = df.shape[1] //cols
    row_cut = df.shape[0] //rows
    df2 = pd.DataFrame()
    for segment in range(cols):
        df2[segment] = df.iloc[:, x:x+col_cut].mean(axis=1).astype(int)
        x += col_cut
    df3 = df2.groupby(np.arange(len(df))//row_cut).mean().astype(int)
    if len(df3) == rows:
        return df3
    return df3.iloc[:rows]


# Helper function
def array_to_data_url(img, dtype=None):
    """
    Converts numpy array to data string, using Pillow.
    The returned image string has the right format for the ``image_content``
    property of DashCanvas.
    Parameters
    ==========
    img : numpy array
    Returns
    =======
    image_string: str
    """
    if dtype is not None:
        img = img.astype(dtype)
    df = pd.DataFrame(img)
    # reduce the number of columns to 28
    return squash_matrix(df, cols=28, rows=28)

# app server config
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config['suppress_callback_exceptions'] = True
app.title=tabtitle


# Set up Dash layout
app.layout = html.Div(children=[
        html.H1('Handwritten Digit Classifier'),

        html.Div(id='reset-page', key='page', children=[

        html.Div([
            html.Div([
                html.H3('Draw & Submit'),
                html.Br(),
                html.Br(),
                html.Br(),
                DashCanvas(
                    id='canvas',
                    lineWidth=5,
                    lineColor='rgba(255, 0, 0, 0.5)',
                    width=canvas_size,
                    height=canvas_size,
                    hide_buttons=["zoom", "pan", "line", "pencil",
                                  "rectangle", "undo", "select"],
                    goButtonTitle='Submit',
                ),
                html.A(html.Button('Reset'), href='/'),
            ], style={"padding-left": "20px", "align": "left"},
                className="three columns"),

            html.Div([
                html.H3('Image converted to Dataframe', style={
                    "padding-left": "65px", "align": "left"}),
                dcc.Graph(id='output-figure', figure=blank_fig,
                style={'width': '100%', 'height': '100%', "padding-left":
                    "1px", "align": "left"}
                ),
            ], style={"padding-left": "0px", "align":"left"},
                className='six columns'),

            html.Div([
                html.H3('Predicted Digit'),
                html.Br(),
                html.H4('Random Forest Model:'),
                html.H6(id='rf-prediction', children='...'),
                html.H6(id='rf-probability', children='waiting for inputs'),
                html.Br(),
                html.H4('XGBoost Model:'),
                html.H6(id='xgb-prediction', children='...'),
                html.H6(id='xgb-probability', children='waiting for inputs'),
            ], className='three columns'),
        ], className="twelve columns"),]),
        html.Div(children=[
            html.Div('Updated'),
            html.Div(children=[
                html.Li('Added Reset button'),
                html.Li('Decreased drawing line width.'),
                html.Li('Adjusted test_size=0.1'),
                html.Li('Augmented X_train from 7000 to 315000 using 1up, '
                        '1down, 1right, 1left'),
                html.Li('Tried to use GridSearchCV, RandomizedSearchCV but '
                        'didn\'t complete within 8 hours with augmented data.'),
                html.Li('Therefore used static params with GridSearchCV.'),
                html.Li('Accuracy: 0.9851428571428571 '
                        'Precision: 0.9849865193732226 '
                        'Recall: 0.9852401268810149 '
                        'F1 Score: 0.9850931222167884'),
            ],),
        ],),
        html.Br(),
        html.A('Code on Github', href=githublink),
        html.Br(),
        html.A("Data Source", href=sourceurl),
], className="twelve columns")


# callback
@app.callback(
    Output('output-figure', 'figure'),
    Output('rf-prediction', 'children'),
    Output('rf-probability', 'children'),
    Output('xgb-prediction', 'children'),
    Output('xgb-probability', 'children'),
    Input('canvas', 'json_data'))
def update_data(string):
    if string:
        data = json.loads(string)
        print('9999999999999')
        print(f'{string}')
        print('9999999999999')
        print('----------------------------\n')
        # shape file
        # explore the contents of the shape file
        print(data['objects'][0]['path'])
        print('----------------------------\n')
        mask = parse_jsonstring(string, shape=(canvas_size, canvas_size))
        img = (255 * mask).astype(np.uint8) # transform the data
        print('============================\n')
        # explore the transformed data
        print(img)
        print('============================\n')
        array_to_data_output = array_to_data_url(img)
        print('*****************************')
        print(array_to_data_output)
        print('*****************************')

        # display as heatmap
        fig = px.imshow(array_to_data_output, text_auto=True,
                        color_continuous_scale='Purples')
        fig.layout.height = 600
        fig.layout.width = 600
        fig.update(layout_coloraxis_showscale=False)
        fig.update(layout_showlegend=False)

        # pickle the user input
        filename = open('user-input-digit.pkl', 'wb')
        pickle.dump(array_to_data_output, filename)
        filename.close()

        # convert the user input to the format expected by the model
        some_digit_array = np.reshape(array_to_data_output.values, -1)
        print('some_digit_array', [some_digit_array])

        # standardize
        some_digit_scaled = scaler.transform([some_digit_array])

        # make a prediction: Random Forest
        rf_pred = rf_model.predict(some_digit_scaled)
        rf_prob_array = rf_model.predict_proba(some_digit_scaled)
        rf_prob = max(rf_prob_array[0])
        rf_prob = round(rf_prob*100,2)

        # make a prediction: XG Boost
        xgb_pred = xgb_model.predict(some_digit_scaled)
        xgb_prob_array = xgb_model.predict_proba(some_digit_scaled)
        xgb_prob = max(xgb_prob_array[0])
        xgb_prob = round(xgb_prob*100,2)

    else:
        raise PreventUpdate
    return fig, f'Digit: {rf_pred[0]}', f'Probability: {rf_prob}%', \
           f'Digit: {xgb_pred[0]}', f'Probability: {xgb_prob}%'


if __name__ == '__main__':
    app.run_server(debug=True)
