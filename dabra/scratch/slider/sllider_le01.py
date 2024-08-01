# -*- coding: utf-8 -*-
# simple example: by sliding show the selected number
"""
https://dash.plotly.com/dash-core-components/slider
https://community.plotly.com/t/slider-with-play-button-for-animations-independent-of-plotly/53188/2
https://www.youtube.com/watch?v=d9SmpNfMg7U
https://towardsdatascience.com/how-to-create-animated-scatter-maps-with-plotly-and-dash-f10bb82d357a play button
"""

from dash import Dash, dcc, html, Input, Output, callback
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    dcc.Slider(0, 20, 5,
               value=10,
               id='my-slider'
    ),
    html.Div(id='slider-output-container')
])

@callback(
    Output('slider-output-container', 'children'),
    Input('my-slider', 'value'))
def update_output(value):
    return 'You have selected "{}"'.format(value)

if __name__ == "__main__":    app.run_server(debug=True, port=8057)
