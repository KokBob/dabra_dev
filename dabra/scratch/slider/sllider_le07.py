# -*- coding: utf-8 -*-
"""
stav: nejaky erro v dashi, neprehava tak jak je treba
https://www.youtube.com/playlist?list=PLh3I780jNsiTXlWYiNWjq2rBgg3UsL1Ub
https://community.plotly.com/t/slider-with-play-button-for-animations-independent-of-plotly/53188
https://dash.plotly.com/dash-core-components/slider
https://community.plotly.com/t/slider-with-play-button-for-animations-independent-of-plotly/53188/2
https://www.youtube.com/watch?v=d9SmpNfMg7U
https://towardsdatascience.com/how-to-create-animated-scatter-maps-with-plotly-and-dash-f10bb82d357a play button
https://stackoverflow.com/questions/71906091/python-plotly-dash-automatically-iterate-through-slider-play-button
"""
#******* CLEANED AND CLOSED *******
from dash import Dash, dcc, html, Input, Output, callback, dash_table, State
import glob, os, yaml, sys
# sys.path.append("..") 
sys.path.append("../..") 
sys.path.append("../../..") 
from dabra import insight
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
emt_file_path = '../../../dabra/data/0409/3D point tracks.emt' # :dart:: putit to conf
df0 = pd.read_csv(emt_file_path, skiprows=9, delimiter=r"\s+")
# df = df0.iloc[:,2:] # puvodni
df = df0.iloc[300,2:]
df = pd.DataFrame(df.values.reshape(89,3), columns = ['X', 'Y','Z'])
frames = list(set(df0["Frame"]))
frames.sort()
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)
# %% slider  
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
#%% 
def make_fig(frame_):
    df_ = df0.iloc[frame_,2:] 
    df_ = pd.DataFrame(df_.values.reshape(89,3), columns = ['X', 'Y','Z'])
    fig = px.scatter_3d(df_, x = 'X', y = 'Y', z = 'Z', color = df.index)
    fig.update_layout(
        title_text = f'{frame_}',
        # duration =.1,
        # updatemenusbuttons[0].args[1]['frame']
        title_x = 0.5,)
    # fig.layout.updatemenus.frame.duration = 30
    # fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 5
    return fig


app.layout = html.Div([
    dcc.Interval(id="animate", disabled=True),
    dcc.Slider(
        id="frame-slider",
        min=df0["Frame"].min(),
        max=df0["Frame"].max(),
        value=df0["Frame"].min(),
        # marks={str(frame): str(frame) for frame in df0["Frame"].unique()},
        # step=None,
        step=100,
               ),
    
    html.Div(id='slider-output-container'),
    
    dcc.Graph(id='graph-with-slider', figure = make_fig(1)),
    
    html.Button("Play", id="play")
        ]
    )

@app.callback(
    Output("graph-with-slider", "figure"),
    Output("frame-slider", "value"),
    Input("animate", "n_intervals"),
    State("frame-slider", "value"),
    prevent_initial_call=True,
)
def update_figure(n, selected_frame):
    index = frames.index(selected_frame)
    index = (index + 1) % len(frames)
    frame = frames[index]
    return make_fig(frame), frame


@app.callback(
    Output("animate", "disabled"),
    Input("play", "n_clicks"),
    State("animate", "disabled"),
)
def toggle(n, playing):
    if n:
        return not playing
    return playing
if __name__ == "__main__":   app.run_server(debug=True, port=8058)

