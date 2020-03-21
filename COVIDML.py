import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import os
import flask
from random import randint



# Setup the app
# Make sure not to change this file name or the variable names below,
# the template is configured to execute 'server' on 'app.py'
server = flask.Flask(__name__)
server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))

#app = dash.Dash()
#server = app.server

df = pd.read_csv("./spain-covid19-official-data-from-health-ministry/coronavirus_dataset_spain_by_regions.csv")
#df2 = pd.read_csv("dataset_Facebook.csv",";")
#print(df)

region = df["region"].unique()

app = dash.Dash()

app.layout = html.Div([
    html.H2("COVID in Spain"),
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

    pv = pd.pivot_table(
        df_plot,
        index=['day'],
        #columns=['cases'],
        values=['cases'],
        aggfunc=sum,
        fill_value=0)

    trace1 = go.Bar(x=pv.index, y=pv[('cases')], name='Declined')
    #trace2 = go.Bar(x=pv.index, y=pv[('cases', 'pending')], name='Pending')
    #trace3 = go.Bar(x=pv.index, y=pv[('cases', 'presented')], name='Presented')
    #trace4 = go.Bar(x=pv.index, y=pv[('cases', 'won')], name='Won')

    return {
        'data': [trace1],#, trace2, trace3, trace4],
        'layout':
        go.Layout(
            title='COVID19 in {}'.format(Region),
            barmode='stack')
    }


if __name__ == '__main__':
    app.run_server(debug=True)
