from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash  # pip install dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

from dash.exceptions import PreventUpdate
import pandas as pd
import numpy
import plotly.io as pio

import myfunctions_md as f_md

pio.templates.default = "plotly_white" 
# pio.templates.default = "plotly_dark" 

# Import the datasets
directory = '../../../transformData/'
df = pd.read_csv(directory+'valTimeSlotsHour.csv')
df_validation = pd.read_csv(directory+'validation.csv')

# Disable the days that are not in the dataset: disabled_days_md
disabled_days = f_md.disabled_days_md(df)

# Print the disabled days
print("disabled_days: ", disabled_days)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.UNITED, dbc.icons.BOOTSTRAP]
MASTER_LOGO = "http://www.master-project-h2020.eu/wp-content/uploads/2018/02/cropped-Master-logo-1.png"

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# make a reuseable navitem for the different examples
nav_item = dbc.NavItem(dbc.NavLink("Link", href="#"))

# Navbar 
navbar = f_md.navbar()

# Callback to toggle collapse on small screens
@dash.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)

# Toggle the collapse and change icon on toggle
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# Header of the page
header = f_md.header_md(analysis_type='Multiple Date Analysis')

# Insert a paragraph describing the page
# Notice that the map is a scatter describing the density of validations for each stop
descr_list = ['This page allows to analyze the validation data for multiple dates.',
              'The user can select the dates of interest and the type of ticket to analyze.']
descr_list2 = ['The map shows the density of validations for each stop while the bar chart shows the total number of validations in different time slots.']
description = f_md.description(descr_list)
description2 = f_md.description(descr_list2)

# Layout of the page 
app.layout = html.Div([
    html.Div(children=[
        navbar, header, description, description2,
        html.Br(),
        html.Label(children = ['Date'],
                    style={'margin-right': '10px', 'font-weight': 'bold', 'font-size': '12px', 
                           'padding-top': '15px', 'color':'#3A6BAC', 'padding-left': '20px'}),
        dcc.DatePickerRange(
            id='my-date-picker-range',
            min_date_allowed=df['date'].min(),
            max_date_allowed=df['date'].max(),
            initial_visible_month=df['date'].min(),
            disabled_days=disabled_days,
            start_date=df['date'].min()#,
            # end_date=df['date'].max()
        ),
        
        html.Br(),
        html.Label(children = ['Ticket type'],
                    style={'font-weight': 'bold', 'font-size': '12px', 'padding-top': '15px', 
                           'color':'#3A6BAC', 'padding-left': '20px'}),
        dcc.Dropdown(
            id='my-dynamic-dropdown',
            options=[{'label': '24 hours', 'value': 0},
                    {'label': '48 hours', 'value': 1},
                    {'label': '72 hours', 'value': 2},
                    {'label': '7 days', 'value': 3}],
            multi=True,
            value=[0],
            placeholder="Select a ticket type",
            style={'width': 323, 'align-items': 'left', 'justify-content': 'left', 'padding-left': '20px'}
        ),
    ],style={'padding': 10, 'flex': 1, 'background-color': '#F0F0F0'}),

    html.Div(children=[
        dcc.Graph(id='mymap'),#,style={'width': '20%','display': 'inline-block'}),#style={'width': '90vh', 'height': '90vh'}),
                                    #'lineHeight': '1px',
                                    #'borderWidth': '1px',
                                    #'borderStyle': 'dashed',
                                    #'borderRadius': '1px'}),
        dcc.Graph(id='bar-chart',clickData=None)#,style={'width': '20%','display': 'inline-block'})
    ], style={'padding': 10, 'flex': 1}),

], style={'display': 'flex', 'flex-direction': 'row'})

# Callback for the map
@app.callback(
    Output(component_id='mymap', component_property='figure'),
    [Input(component_id='my-date-picker-range', component_property='start_date'),
    Input(component_id='my-date-picker-range', component_property='end_date'),
    Input(component_id='my-dynamic-dropdown', component_property='value')]#, 
)

# Function to update the map
def update_output(start_date,end_date,value):
    len_choices = len(value)
    if(len_choices == 0 or start_date > end_date or start_date == end_date) : 
        #print("raise PreventUpdate")
        raise PreventUpdate

    dff = df[df['travel_type'].isin(value)]
    dff.reset_index(drop=True)

    mask = (dff['date'] >= start_date) & (dff['date'] <= end_date)
    dff = dff.loc[mask]

    if(start_date != end_date) :
        dff = dff.groupby(['stop_id','name_stop','lat','lon','travel_type'])['counts'].agg(['sum']).reset_index()
        dff.rename(columns={'sum': 'counts'}, inplace=True)
        dff.sort_values(by=['stop_id','travel_type'], inplace=True)

    if(len_choices > 1) :
        dff = dff.groupby(['stop_id','name_stop','lat','lon'])['counts'].agg(['sum']).reset_index()
        dff.rename(columns={'sum': 'counts'}, inplace=True)
        dff.sort_values(by=['stop_id'], inplace=True)

    fig = px.scatter_mapbox(dff, lat='lat', lon='lon', color ='counts', size = 'counts', 
                            mapbox_style="carto-positron", width=1300, height=500, zoom=11, 
                            #animation_frame = 'date_time',#animation_group='travel_type',
                            color_continuous_scale='ice',
                            range_color=[dff['counts'].min(),dff['counts'].max()],#size_max=50,
                            center = {'lon': 12.337817, 'lat': 45.44},hover_data={'lat': False, 'lon': False,'name_stop':True,'counts': True,})
    
    fig.update_layout(
        margin={'t': 0,'l':0,'b':0,'r':10}
    )

    return fig

# Callback for the bar chart
@app.callback(
    Output(component_id='bar-chart', component_property='figure'),
    [Input(component_id='my-date-picker-range', component_property='start_date'),
    Input(component_id='my-date-picker-range', component_property='end_date'),
    Input(component_id='my-dynamic-dropdown', component_property='value')]#, 
)

# Function to update the bar chart
def update_output_sec(start_date,end_date,value):

    len_choices = len(value)
    if(len_choices == 0 or start_date > end_date or start_date == end_date) : 
        #print("raise PreventUpdate")
        raise PreventUpdate

    # datetime object containing current date and time
    dff2 = df[df['travel_type'].isin(value)]
    dff2.reset_index(drop=True)

    mask = (dff2['date'] >= start_date) & (dff2['date'] <= end_date)
    dff2 = dff2.loc[mask]

    dff2 = dff2.groupby(['time'])['counts'].agg(['sum']).reset_index()
    dff2.rename(columns={'sum': 'counts'}, inplace=True)
    dff2.sort_values(by=['time'], inplace=True)

    if dff2.empty:
        #print(f'Dataframe empty')
        raise PreventUpdate

    dff2['time'] = dff2['time'].map(lambda t: t[:-3])
    my_time_min = dff2['time'].min()
    my_time_max = dff2['time'].max()

    fig2 = px.bar(dff2,x='time',y='counts',hover_data=['counts'], color='counts',color_continuous_scale='ice',text_auto=True,
        height=400,width=1300,labels={'time':'Time slots'})
        #labels={'counts':'Number of tickets'})
    fig2.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig2.update_layout(
        margin={'t': 15,'l':0,'b':0,'r':12,'pad':7},
        xaxis=dict(range=[my_time_min,my_time_max],showticklabels=True),
    )
    fig2.update_yaxes(title_text='Number of tickets')
    return fig2

if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_ui=False)

