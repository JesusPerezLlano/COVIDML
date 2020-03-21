import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import os
import flask
from random import randint

external_stylesheets = ['https://codepen.io/anon/pen/mardKv.css']


# Setup the app
# Make sure not to change this file name or the variable names below,
# the template is configured to execute 'server' on 'app.py'
server = flask.Flask(__name__)
server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, server=server)

#app = dash.Dash()
#server = app.server

df = pd.read_csv("./spain-covid19-official-data-from-health-ministry/coronavirus_dataset_spain_by_regions.csv")
#df2 = pd.read_csv("dataset_Facebook.csv",";")
#print(df)

region = df["region"].unique()

#app = dash.Dash()
theme =  {
    'dark': True,
    'detail': '#007439',
    'primary': '#00EA64',
    'secondary': '#6E6E6E',
}

colors = {
    'background': '#111111',
    'text': '#7FDBFF',
    'foreground': '#FFA500',
    'background2': '#C0C0C0',
    'none': '#FFFFFF'
}
import base64


image_filename = 'TedCas.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read()).decode('ascii')


app.layout = html.Div([
    
    html.Div([
        html.Div([
            html.Img(src='data:image/png;base64,{}'.format(encoded_image), className='three columns', style={'height':'10%', 'width':'10%'})#, style={'width': '16%', 'display': 'inline-block'})
            ],   style={'textAlign': 'center','backgroundColor': colors['foreground']},),    
        html.Div([
            html.H2("COVID in Spain", className='three columns')#, style={'width': '16%', 'display': 'inline-block'})   
           ],   style={'textAlign': 'center'}),
            ],
        style={'width': '100%', 'display': 'inline-block'}),
    
    
    #html.H2("COVID in Spain"),
    html.Div(
        [
            dcc.Dropdown(
                id="Region",
                options=[{
                    'label': i,
                    'value': i
                } for i in region],
                value='All Regions'),
        ],
        style={'width': '25%',
               'display': 'inline-block'}),
    dcc.Graph(id='funnel-graph'),
])


@app.callback(
    dash.dependencies.Output('funnel-graph', 'figure'),
    [dash.dependencies.Input('Region', 'value')])
def update_graph(Region):
    if Region == "All Regions":
        df_plot = df.copy()
    else:
        df_plot = df[df['region'] == Region]
    #print(df_plot)
    pv = pd.pivot_table(
        df_plot,
        index=['date'],
        #columns=['cases'],
        values=['cases'],
        aggfunc=sum,
        fill_value=0)
    pv2 = pd.pivot_table(
        df_plot,
        index=['date'],
        #columns=['cases'],
        values=['severes'],
        aggfunc=sum,
        fill_value=0)

    print(pv2.index)
    print(pv.index)
    def RegionNumber(i):
        switcher={
                'Andalucí­a': 0,
                'Aragón': 1,
                'Asturias': 2,
                'Baleares': 3,
                'Canarias': 4,
                'Cantabria': 5,
                'Castilla La Mancha': 6,
                'Castilla y Leon': 7,
                'Cataluña': 8,
                'Ceuta': 9,
                'C.Valenciana': 10,
                'Extremadura': 11,
                'Galicia': 12,
                'Madrid': 13,
                'Melilla': 14,
                'Murcia': 15,
                'Navarra': 16,
                'País Vasco': 17,
                'La Rioja': 18
             }
        return switcher.get(i,0)


    IndexRegion=RegionNumber(Region)
    y = [] 
    NUCIS2=[642,157,112,120,229,60,170,284,1034,7,394,116,274,700,5,145,69,236,31,4730]
    
    NUCIS=1000
    day = df["day"].unique()
    #print(day)
    #for elemen in day:
    #    print(elemen)
    #    y[int(elemen)]=NUCIS
    y_uci = [NUCIS2[int(IndexRegion)] for i in range(len(day))]

    trace2 = go.Bar(x=pv.index, y=pv[('cases')], name='Infected')
    trace1 = go.Bar(x=pv2.index, y=pv2[('severes')], name='Severes')
    trace3 = {'x': pv.index, 'y': y_uci, 'type': 'scatter', 'name': 'número de UCIS'}
   
    #trace2 = go.Bar(x=pv.index, y=pv[('cases', 'pending')], name='Pending')
    #trace3 = go.Bar(x=pv.index, y=pv[('cases', 'presented')], name='Presented')
    #trace4 = go.Bar(x=pv.index, y=pv[('cases', 'won')], name='Won')

    return {
        'data': [trace1, trace3],#, trace3, trace4],
        'layout':
        go.Layout(
            title='COVID19 in {}'.format(Region),
            barmode='stack')
    }


if __name__ == '__main__':
    app.run_server(debug=True)
