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



app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server
app.config["suppress_callback_exceptions"] = True
# Setup the app
# Make sure not to change this file name or the variable names below,
# the template is configured to execute 'server' on 'app.py'

#server = flask.Flask(__name__)
#server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))
#app = dash.Dash(__name__, external_stylesheets=external_stylesheets, server=server)

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
                searchable=False,
                value='All Regions',style={'height': '55px', 'width': '500px','font-size': "155%",'display': 'inline-block','vertical-align': 'middle'}),
        ],
        style={'width': '50%',
               'display': 'inline-block','vertical-align': 'middle'}
        ),
    dcc.Graph(id='funnel-graph'),
])

#region PREDICT_DATA
#DataFlair - Make necessary imports
#import quandl
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dfdates  = []
dfprices = []

###Use this function for pandas dataframes
def gd(dataframe):
    i = 1.0
    dataframe.reset_index(drop=True, inplace=True)
    dataframe_close=dataframe['Close']
    #reverse dataframe (newest values more recent)
    reversed_df = dataframe_close.iloc[::-1]
    #We get the first values of reversed dataframe
    ndataframe2=reversed_df[0:200]
    ndataframe2.reset_index(drop=True, inplace=True)
    #We get the first values of normal dataframe
    ndataframe=dataframe_close[0:140]
    ndataframe.reset_index(drop=True, inplace=True)
    while i <= len(ndataframe2):
        a = ndataframe2#.xs(i)
        #print(a)
        dfdates.append(i)
        dfprices.append(a[i-1])
        i += 1
    
    return

def predict_covid(dates, prices, x):
    #print(dates)
    dates2 = dates
    #dates2 = np.reshape(dates,(len(dates2), 1)) # converting to matrix of n X 1
    #print(dates2)
    #print(prices)
    sc_x=StandardScaler()
    sc_y=StandardScaler()
    dates3=sc_x.fit_transform(dates2)
    prices2=sc_y.fit_transform(prices[:,np.newaxis]).flatten()
    
    print(prices2)

    print(dates3)
    svr_lin = SVR(kernel= 'linear', C= 1e3)
    svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2)
    svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) # defining the support vector regression models
    svr_rbf.fit(dates3, prices2) # fitting the data points in the models
    svr_lin.fit(dates3, prices2)
    svr_poly.fit(dates3, prices2)
    
    
    #import matplotlib.pyplot as pp

    #pp.scatter(dates2, prices, color= 'black', label= 'Data') # plotting the initial datapoints
    #pp.plot(dates2, svr_rbf.predict(dates), color= 'red', label= 'RBF model') # plotting the line made by the RBF kernel
    #pp.plot(dates2,svr_lin.predict(dates), color= 'green', label= 'Linear model') # plotting the line made by linear kernel
    #pp.plot(dates2,svr_poly.predict(dates), color= 'blue', label= 'Polynomial model') # plotting the line made by polynomial kernel
    #pp.xlabel('Date')
    #pp.ylabel('Price')
    #pp.title('Support Vector Regression')
    #pp.legend()
    #print(svr_rbf.predict(np.array(x).reshape(-1,1))[0], svr_lin.predict(np.array(x).reshape(-1,1))[0], svr_poly.predict(np.array(x).reshape(-1,1))[0])
    #pp.show()

    return svr_rbf.predict(np.array(x).reshape(-1,1))[0], svr_lin.predict(np.array(x).reshape(-1,1))[0], svr_poly.predict(np.array(x).reshape(-1,1))[0]

#endregion





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

    #print(pv2.index)
    #print(pv.index)
    
    #print(df_plot)
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

    x=2
    Xpredictor=df_plot[['day']].values
    Ypredictor=df_plot['cases'].values
    #print(Xpredictor)
    #print(Ypredictor)
    #Predict new cases
    #print(pv.index)
    predicted_covid = predict_covid(Xpredictor,  Ypredictor, x)

    IndexRegion=RegionNumber(Region)
    y = [] 
    #Orden Regiones: AndalucÃ­a. AragÃ³n, , Asturias, Baleares, Canarias, Cantabria, Castilla La Mancha, Castilla y LeÃ³n, CataluÃ±a, Ceuta, C. Valenciana, Extremadura, Galicia, Madrid, Melilla, Murcia, Navarra, PaÃ­s Vasco, La Rioja

    NUCIS2=[642,157,112,120,229,60,170,284,1034,7,394,116,274,700,5,145,69,236,31,4730]
    NHABITANTES=[8426405,1320794, 1022293,1187808,2207225,581684,2035505,2408083,7565099,84843,4974475,1065371,2700330,6640705,7060,1487698,649966,2178048,313582]




    
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
    #trace3 = {'x': pv.index, 'y': y_uci, 'type': 'scatter', 'name': 'número de UCIS'}

    #print(predict_covid[0])
    #trace4 = go.Bar(x=pv.index, y=predict_covid[0], name='Predicted')
    #trace2 = go.Bar(x=pv.index, y=pv[('cases', 'pending')], name='Pending')
    #trace3 = go.Bar(x=pv.index, y=pv[('cases', 'presented')], name='Presented')
    #trace4 = go.Bar(x=pv.index, y=pv[('cases', 'won')], name='Won')
    PI=df_plot.iloc[-1,4]/NHABITANTES[int(IndexRegion)]*100000
    return {
        'data': [trace1, trace3, trace2],#, trace4],#, trace3, trace4],
        'layout':
        go.Layout(
            title='COVID19 in {}'.format(Region) + '(Number of infected/100000 hab='+str(int(PI)) + ')'
            #,barmode='stack'
            )
    }


if __name__ == '__main__':
    app.run_server(debug=True)
