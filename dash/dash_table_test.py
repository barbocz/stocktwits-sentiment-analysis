import dash
import dash_table as dt
import pandas as pd
from joblib import dump, load
import time
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
start_time = time.time()

# df = pd.read_csv('../processed_sentiment.csv')
df=load('../data/processed_sentiment_AAPL.pkl')
df['Date']=pd.to_datetime(df['Date'])

current_day='2021-02-11'
ticker='AAPL'
df=df[df['Date']==current_day]

total_sentiment=len(df)

value_counts = df['Combined sentiment'].value_counts()

bearish_bullish_ratio=(value_counts['Bearish']/value_counts['Bullish'])

ema_value=2.023

signal='BUY'

row1 = html.Tr([html.Td("Date:"), html.Td(current_day)])
row2 = html.Tr([html.Td("Total sentiment:"), html.Td(total_sentiment)])
row3 = html.Tr([html.Td("Bearish / Bullish ratio:"), html.Td('{:.3f}'.format(bearish_bullish_ratio))])
row4 = html.Tr([html.Td("EMA value:"), html.Td(ema_value)])
row5 = html.Tr([html.Td("Signal:"), html.Td(signal)])
row6 = html.Tr([html.Td(""), html.Td("")])


table_body = [html.Tbody([row1, row2, row3, row4, row5,row6])]

table = dbc.Table( table_body, bordered=True)



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

@app.callback(
    Output('refresh_button', 'active'),
    Input('refresh_button', 'n_clicks'),
          )
def update_output(n_clicks):
    print( 'the button has been clicked {} times'.format(      n_clicks    ))
    return True




app.layout = html.Div([
    html.Div([
    html.Div(style={'width': '32%', 'border': '5px outset red','display': 'inline-block'},children=[dt.DataTable(data=data, columns=columns, id='table',


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
                style={'width': '32%','border': '5px outset red',  'float': 'right','display': 'inline-block'}),
        html.Div(children=[
                        html.Div(className="card-body", children=[table,dbc.Button("Refresh", id="refresh_button", n_clicks=0)

            ])
        ],

            style={'width': '32%', 'border': '5px outset red', 'float': 'right','display': 'inline-block'})
        ])
    ])
#                 ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
# )

if __name__ == '__main__':
    app.run_server(debug=True)