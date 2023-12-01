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
# emt_file_path = '../../dabra/data/0409/3D point tracks.emt' # :dart:: putit to conf
# emt_file_path = '../../dabra/data/1201/1201_Loose/9.min/3D point tracks.emt' # :dart:: putit to conf
emt_file_path = '../../dabra/data/1201/1201_normal/15. min [without first second]/3D Point Tracks.emt' # :dart:: putit to conf
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

# %%
# initial_camera_orientation = dict(
#     up=dict(x=0, y=1, z=0),
#     center=dict(x=0, y=0, z=1),
#     eye=dict(x= 0.25, y= 1.25, z=3.25)
#     )
# %%
# https://en.wikipedia.org/wiki/Transformation_matrix
import numpy as np
alpha = 12

rotation_angle_degrees = -60 # -30,-40,-50, -70
rotation_angle_radians = np.radians(rotation_angle_degrees)

# rotation_matrix = np.array([
#     [np.cos(np.pi / alpha), 0, np.sin(np.pi / alpha)],
#     [0, 1, 0],
#     [-np.sin(np.pi / alpha), 0, np.cos(np.pi / alpha)],
# ])

rotation_matrix = np.array([
    [np.cos(rotation_angle_radians ), 0, np.sin(rotation_angle_radians )],
    [0, 1, 0],
    [-np.sin(rotation_angle_radians ), 0, np.cos(rotation_angle_radians )],
])

initial_camera_position = np.array([-1.25, -1.25, 1.25])
rotated_camera_position = np.dot(rotation_matrix, initial_camera_position)

# Set up the layout with the rotated camera position
# layout = go.Layout(
initial_camera_orientation =dict(
            up=dict(x=0, y=1, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=rotated_camera_position[0], y=rotated_camera_position[1], z=rotated_camera_position[2]),
        )


#%% layout 

app.layout = html.Div([
    
    dcc.Slider( initial_value, last_value, 
               # incrementationing,
               step=None,
               value=0,
               id='my-slider'
    ),
    html.Div(id='slider-output-container'),
    # my_table := dash_table.DataTable(
    #     id = 'table-xyz',
    #     data = df.to_dict('records'),
    #     columns=[{"name": i, "id": i} for i in df.columns], 
    #     page_size = 5
    #     ),
    dcc.Graph(id='dagrap', 
               # animate = True, Tohle nefunguje se sliderem

             #*** No arguments for those spces
              # cameraPosition = [0,0,0] , 
              # cameraViewUp = [0,0,0]    
              
              ),
    

    ])

# @callback(
#     # Output('slider-output-container', 'children'),
#     Output('table-xyz', 'data'),
#     Input('my-slider', 'value'))
# def updateTable(value):
#     df = df0.iloc[value,2:]
#     df = pd.DataFrame(df.values.reshape(89,3), columns = ['X', 'Y','Z'])
#     return df.to_dict('records')
# @callback(
#     Output('slider-output-container', 'children'),
#     Input('my-slider', 'value'))
# def update_output(value):
    # return 'You have selected "{}"'.format(value)

@callback(
    Output("dagrap", "figure"),
    Input('my-slider', 'value'))
def updateGraph(value):
    df_ = df0.iloc[value,2:]
    df_ = pd.DataFrame(df_.values.reshape(89,3), columns = ['X', 'Y','Z'])
    fig = px.scatter_3d(df_, x = 'X', y = 'Y', z = 'Z', 
                        # color = df.index * 0.01
                        color = df.index 
                        )
    trace = go.Scatter3d(
            x=df_['X'],
            y=df_['Y'],
            z=df_['Z'], 
            # mode='lines',
            marker=dict(
            size=7,
            color=df_.index,  # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8, 
            
            )
        )
    # Create the graph layout
    # layout = go.Layout(
    #    title="Live Graph",
    #    xaxis=dict(range=[min(x_data), max(x_data)]),
    #    yaxis=dict(range=[min(y_data), max(y_data)]),
    # )
    # return {"data": [trace], "layout": layout}
    return {"data": [trace],
            'layout': go.Layout(
                scene=dict(
                    camera=initial_camera_orientation
                            )
                )
            }
    # return fig
if __name__ == "__main__":    app.run_server(debug=True, port=8057)

