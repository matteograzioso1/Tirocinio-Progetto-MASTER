from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import dash # pip install dash
from dash import dcc, html, Input, Output
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy
import plotly.io as pio

pio.templates.default = "plotly_white" 
#pio.templates.default = "plotly_dark" 

# Data 
directory = '/Users/giuliarovinelli/Desktop/Università/PhD/actv/actvData/data/transformData/'
df = pd.read_csv(directory+'valWithDateTime.csv')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    html.Div(children=[
        html.H3("ACTV validation", style={'textAlign': 'left'}),
        
        #html.Br(),
        html.Label('Date'),
        dcc.DatePickerSingle(
            id='my-date-picker',
            min_date_allowed=df['date'].min(),
            max_date_allowed=df['date'].max(),
            initial_visible_month=df['date'].min(),
            date=df['date'].min()
        ),
        
        html.Br(),
        html.Label('Ticket'),
        dcc.Dropdown(
            id='my-dynamic-dropdown',
            options=[{'label': '24 hours', 'value': 0},
                    {'label': '48 hours', 'value': 1},
                    {'label': '72 hours', 'value': 2},
                    {'label': '7 days', 'value': 3}],
            multi=True,
            value=[0],
            placeholder="Select a ticket type",
            style={'width': 400, 'align-items': 'left', 'justify-content': 'left'}
        ),
    ],style={'padding': 10, 'flex': 1}),

    html.Div(children=[
        dcc.Graph(id='mymap'),
    ]#, style={'padding': 10, 'flex': 1}
    )
]) 

@app.callback(
    Output(component_id='mymap', component_property='figure'),
    [Input(component_id='my-date-picker', component_property='date'),
    Input(component_id='my-dynamic-dropdown', component_property='value')]#, 
)

def update_output(date,value):
    len_choices = len(value)
    if(len_choices == 0) : 
        #print("raise PreventUpdate")
        raise PreventUpdate

    dff = df[df['travel_type'].isin(value)]
    dff.reset_index(drop=True)

    mask = (dff['date'] == date)
    dff = dff.loc[mask]

    if(len_choices > 1) :
        dff = dff.groupby(['stop_id','name_stop','lat','lon','time_slot'])['counts'].agg(['sum']).reset_index()
        dff.rename(columns={'sum': 'counts'}, inplace=True)
        dff.sort_values(by=['stop_id','time_slot'], inplace=True)

    dff.rename(columns={'time_slot': 'date_time'}, inplace=True)
    dff.sort_values(by=['date_time','stop_id'], inplace=True)
 
    fig = px.scatter_mapbox(dff, lat='lat', lon='lon', color ='counts', size = 'counts', 
                            mapbox_style="carto-positron", 
                            width = 1600, height = 720, zoom=11, 
                            animation_frame = 'date_time',#animation_group='travel_type',
                            color_continuous_scale='viridis',
                            range_color=[dff['counts'].min(),dff['counts'].max()],#size_max=50,
                            center = {'lon': 12.337817, 'lat': 45.44},hover_data={'lat': False, 'lon': False,'name_stop':True,'counts': True})
    
    return fig


if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_ui=False)

