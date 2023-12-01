# -*- coding: utf-8 -*-
"""
targets: 
    set camera and 
    set rotation of body 
    
refs:
https://dash.plotly.com/dash-core-components/slider
https://community.plotly.com/t/slider-with-play-button-for-animations-independent-of-plotly/53188/2
https://www.youtube.com/watch?v=d9SmpNfMg7U
https://towardsdatascience.com/how-to-create-animated-scatter-maps-with-plotly-and-dash-f10bb82d357a play button
https://stackoverflow.com/questions/71906091/python-plotly-dash-automatically-iterate-through-slider-play-button
"""
#******* CLEANED AND CLOSED *******
from dash import Dash, dcc, html, Input, Output, callback, dash_table
import glob, os, yaml, sys
sys.path.append("..") 
sys.path.append("../..") 
from dabra import insight
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
emt_file_path = '../../dabra/data/0409/3D point tracks.emt' # :dart:: putit to conf
df0 = pd.read_csv(emt_file_path, skiprows=9, delimiter=r"\s+")
# df = df0.iloc[:,2:] # puvodni
df = df0.iloc[300,2:]
df = pd.DataFrame(df.values.reshape(89,3), columns = ['X', 'Y','Z'])

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

# %% slider  popopo5ll


initial_value = df0.index.values[0]
last_value = df0.index.values[-1]
incrementationing = 1
fig = px.scatter_3d(df, x = 'X', y = 'Y', z = 'Z', color = df.index * 0.01)
fig.add_trace(
    go.Scatter3d(
        x=df['X'],
        y=df['Y'],
        z=df['Z'], 
        mode='lines',
    ))
# exir
#%% layout 
app.layout = html.Div([
            dcc.Slider( initial_value, last_value, 
               step=None,value=0,id='my-slider'),
            html.Div(id='slider-output-container'),
            dcc.Graph(id='dagrap', ), 
            ])

@callback(
    Output("dagrap", "figure"),
    Input('my-slider', 'value'))
def updateGraph(value):
    df_ = df0.iloc[value,2:]
    df_ = pd.DataFrame(df_.values.reshape(89,3), columns = ['X', 'Y','Z'])
    fig = px.scatter_3d(df_, x = 'X', y = 'Y', z = 'Z', color = df.index )
    trace = go.Scatter3d(
            x=df_['X'],
            y=df_['Y'],
            z=df_['Z'], 
            # mode='lines',
            marker=dict(
            size=7,
            color=df_.index,  # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8
            )
        )
    return {"data": [trace]}
    # return fig
if __name__ == "__main__":    app.run_server(debug=True, port=8058)

