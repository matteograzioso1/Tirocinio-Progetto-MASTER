from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash  # pip install dash
from dash import dcc, html, Input, Output
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy
import plotly.io as pio

pio.templates.default = "plotly_white" 
#pio.templates.default = "plotly_dark" 

# Data 
directory = '../../../transformData/'
df = pd.read_csv(directory+'valTimeSlotsHour.csv')
df_validation = pd.read_csv(directory+'validation.csv')

# Se ci sono delle date comprese tra 'start_date' and 'end_date' che non contengono timbrature non vogliamo 
# che siano visualizzate quindi andiamo a creare una lista di 'disabled_days' con le date che devono essere 
# disabilitiate per questo problema 
start_date = df['date'].min()
end_date = df['date'].max() 
start_date = datetime.strptime(start_date, '%Y-%m-%d')
end_date = datetime.strptime(end_date, '%Y-%m-%d')
delta = end_date - start_date 
all_date = []
for i in range(delta.days + 1):
    day = start_date + timedelta(days=i)
    #day.strftime('%Y-%m-%d')
    day = str(day)[:10]
    all_date.append(day)
all_date = set(all_date)  

date = df['date'].unique()
date = set(date)
disabled_days = all_date - date
disabled_days = list(disabled_days)


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    html.Div(children=[
        html.H3("ACTV validation", style={'textAlign': 'left'}),

        #html.Br(),
        html.Label('Date'),
        dcc.DatePickerRange(
            id='my-date-picker-range',
            min_date_allowed=df['date'].min(),
            max_date_allowed=df['date'].max(),
            initial_visible_month=df['date'].min(),
            start_date=df['date'].min()#,
            #end_date=df['date'].max()
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
        dcc.Graph(id='mymap'),#,style={'width': '20%','display': 'inline-block'}),#style={'width': '90vh', 'height': '90vh'}),
                                    #'lineHeight': '1px',
                                    #'borderWidth': '1px',
                                    #'borderStyle': 'dashed',
                                    #'borderRadius': '1px'}),
        dcc.Graph(id='bar-chart',clickData=None)#,style={'width': '20%','display': 'inline-block'})
    ], style={'padding': 10, 'flex': 1})

], style={'display': 'flex', 'flex-direction': 'row'})


@app.callback(
    Output(component_id='mymap', component_property='figure'),
    [Input(component_id='my-date-picker-range', component_property='start_date'),
    Input(component_id='my-date-picker-range', component_property='end_date'),
    Input(component_id='my-dynamic-dropdown', component_property='value')]#, 
)

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
                            mapbox_style="carto-positron", width=1300, height=500, zoom=11.1, 
                            #animation_frame = 'date_time',#animation_group='travel_type',
                            color_continuous_scale='viridis',
                            range_color=[dff['counts'].min(),dff['counts'].max()],#size_max=50,
                            center = {'lon': 12.337817, 'lat': 45.44},hover_data={'lat': False, 'lon': False,'name_stop':True,'counts': True,})
    
    fig.update_layout(
        margin={'t': 0,'l':0,'b':0,'r':10}
    )

    return fig


@app.callback(
    Output(component_id='bar-chart', component_property='figure'),
    [Input(component_id='my-date-picker-range', component_property='start_date'),
    Input(component_id='my-date-picker-range', component_property='end_date'),
    Input(component_id='my-dynamic-dropdown', component_property='value')]#, 
)

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

    fig2 = px.bar(dff2,x='time',y='counts',hover_data=['counts'], color='counts',color_continuous_scale='viridis',text_auto='.2s',
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

