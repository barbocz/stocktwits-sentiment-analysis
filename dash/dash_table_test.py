import dash
import dash_table as dt
import pandas as pd
from joblib import dump, load
import time
import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
start_time = time.time()

# df = pd.read_csv('../processed_sentiment.csv')
df=load('../data/processed_sentiment_AAPL.pkl')
df['Date']=pd.to_datetime(df['Date'])

df=df[df['Date']=='2021-02-11']

value_counts = df['Combined sentiment'].value_counts()


data = df.to_dict('rows')


columns = [{
    'id': df.columns[2],
    'name': 'Time',
    'type': 'text'
}, {
    'id': df.columns[0],
    'name': 'Message',
    'type': 'text'
}, {
    'id': df.columns[3],
    'name': 'Sentiment',
    'type': 'text'
}]

app = dash.Dash(__name__)


app.layout = html.Div([
    html.Div([
    html.Div(style={'width': '31%', 'border': '5px outset red','display': 'inline-block'},children=[dt.DataTable(data=data, columns=columns, id='table',


                                             style_header={'fontWeight': 'bold', 'backgroundColor': 'black',
                                                           'color': 'white','textAlign': 'left'},
                                            style_cell={
                                                 'whiteSpace': 'normal', 'textAlign': 'left',
                                                 'height': 'auto'},
                                            page_size=10,

                                             style_data_conditional=[
                                                 {
                                                     'if': {
                                                         'filter_query': '{Combined sentiment} eq "Bullish"'

                                                     },
                                                     'backgroundColor': 'rgb(50, 205, 50)'
                                                 },
{
                                                     'if': {
                                                         'filter_query': '{Combined sentiment} eq "Bearish"'

                                                     },
                                                     'backgroundColor': 'rgb(255, 69, 0)'
                                                 }
                                             ]
                                             )
                     ]),
            html.Div(
                    children=[dcc.Graph(id="pie-chart",
                                        figure=px.pie(names=value_counts.index,

                                                      values=value_counts.values,
                                                      color=value_counts.index, color_discrete_map={'Bullish':'rgb(50, 205, 50)',
                                                        'Bearish':'rgb(255, 69, 0)'},

                                                      )
             )],
                style={'width': '31%','border': '5px outset red',  'float': 'right','display': 'inline-block'}),
        html.Div(
            children=['INFO'],
            style={'width': '31%', 'border': '5px outset red', 'display': 'inline-block'})
    ])
    ])
#                 ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
# )

if __name__ == '__main__':
    app.run_server(debug=True)