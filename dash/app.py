#!/usr/bin/env python
# coding: utf-8

# In[4]:
import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_table as dt
import dash_table.FormatTemplate as FormatTemplate
from datetime import date
import json
import requests
from datetime import datetime
from nltk.corpus import stopwords
import re
import string
from nltk.tokenize import word_tokenize
from joblib import dump, load
import plotly.express as px
from dash.exceptions import PreventUpdate

# In[2]:

# In[9]:
# Read Data
df_tesla_merged = pd.read_csv("../data/stock_price_merge_TSLA.csv")
df_aapl_merged = pd.read_csv("../data/stock_price_merge_AAPL.csv")
aapl_strat = pd.read_csv("../data/strategy_AAPL.csv")
tsla_strat = pd.read_csv("../data/strategy_TSLA.csv")


def remove_stopwords(row):
    stopword_list = stopwords.words('english')
    words = []
    for word in row:
        if word not in stopword_list:
            words.append(word)
    return words


# Function to remove emojis
def remove_emoji(tweets):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', tweets)

def tweets_preprocessing(raw_df):

    # Removing all tickers from comments
    raw_df['message'] = raw_df['message'].str.replace(r'([$][a-zA-z]{1,5})', '')

    # Make all sentences small letters
    # raw_df['message'] = raw_df['message'].str.lower()

    # Converting HTML to UTF-8
    # raw_df["message"] = raw_df["message"].apply(html.unescape)

    # Removing hastags, mentions, pagebreaks, handles
    # Keeping the words behind hashtags as they may provide useful information about the comments e.g. #Bullish #Lambo
    raw_df["message"] = raw_df["message"].str.replace(r'(@[^\s]+|[#]|[$])', ' ')  # Replace '@', '$' and '#...'
    raw_df["message"] = raw_df["message"].str.replace(r'(\n|\r)', ' ')  # Replace page breaks

    # Removing https, www., any links etc
    raw_df["message"] = raw_df["message"].str.replace(r'((https:|http:)[^\s]+|(www\.)[^\s]+)', ' ')

    # Removing all numbers
    # raw_df["message"] = raw_df["message"].str.replace(r'[\d]', '')

    # Remove emoji
    raw_df["message"] = raw_df["message"].apply(lambda row: remove_emoji(row))

    # Tokenization
    raw_df['message'] = raw_df['message'].apply(word_tokenize)

    # Remove Stopwords
    # raw_df['message'] = raw_df['message'].apply(remove_stopwords)

    # Remove Punctuation
    raw_df['message'] = raw_df['message'].apply(lambda row: [word for word in row if word not in string.punctuation])

    # Combining back to full sentences
    raw_df['message'] = raw_df['message'].apply(lambda row: ' '.join(row))

    # Remove special punctuation not in string.punctuation
    raw_df['message'] = raw_df['message'].str.replace(r"\“|\”|\‘|\’|\.\.\.|\/\/|\.\.|\.|\"|\'", '')

    # Remove all empty rows
    processed_df = raw_df[raw_df['message'].str.contains(r'^\s*$') == False]

    return processed_df


def get_today_sentiment_dataframe(ticker):
    format = "%Y-%m-%dT%H:%M:%SZ"
    df = pd.DataFrame(columns=['message', 'Date', 'Time', 'Combined sentiment'])
    stocktwit_url = "https://api.stocktwits.com/api/2/streams/symbol/" + ticker + ".json?"
    response = requests.get(stocktwit_url)
    response = json.loads(response.text)

    if response['response']['status'] == 429:
        print('Rate limit exceeded. Client may not make more than 400 requests an hour.')
        return df

    last_message_id = response['cursor']['max']
    done = False
    while True:
        stocktwit_url = "https://api.stocktwits.com/api/2/streams/symbol/" + ticker + ".json?" + "max=" + str(
            last_message_id)
        try:
            response = requests.get(stocktwit_url)
        except Exception:
            response = None

        response = json.loads(response.text)
        if response['response']['status'] != 200:
            print(response)
            return df

        last_message_id = response['cursor']['max']

        for message in response['messages']:
            temp = message['entities']['sentiment']
            if temp is not None and temp['basic']:
                obj = {}
                obj['message'] = message['body']

                obj["Date"] = message["created_at"].split("T")[0]
                obj["Time"] = message["created_at"].split("T")[1].split("Z")[0]
                obj['Combined sentiment'] = temp['basic']
                df = df.append(obj, ignore_index=True)

                datetime_obj = datetime.strptime(message['created_at'], format)
                if datetime_obj.hour < 1:
                    done = True
                # print(datetime_obj.hour, datetime_obj.minute)
        #
        if done:
            break
    # print(df)
    df = tweets_preprocessing(df)
    return df

def get_sentiment_property_table(ticker,current_day,ema,total_sentiment,bullish_bearish_ratio):


    if ticker == 'AAPL':
        df = df_aapl_merged
    elif ticker == 'TSLA':
        df = df_tesla_merged

    df_for_selected_day = df[df['Date'] == current_day]
    if len(df_for_selected_day) == 0:
        df_for_selected_day = df.iloc[-1:]
    else:
        bullish_bearish_ratio = df_for_selected_day['Bull/Bear Ratio'].values[0]
        total_sentiment = df_for_selected_day['Bullish'].values[0] + df_for_selected_day['Bearish'].values[0]

    calculated_ema = df_for_selected_day[f'Bull/Bear Ratio EMA {ema}'].values[0]

    if (bullish_bearish_ratio>calculated_ema):
        signal = 'BUY'
    else:
        signal='SELL'

    row1 = html.Tr([html.Td("Date:"), html.Td(current_day)])
    row2 = html.Tr([html.Td("Total sentiment:"), html.Td(total_sentiment)])
    row3 = html.Tr([html.Td("Bullish / Bearish daily ratio:"), html.Td('{:.3f}'.format(bullish_bearish_ratio))])
    row4 = html.Tr([html.Td("Bullish / Bearish  EMA ratio:"), html.Td('{:.3f}'.format(calculated_ema))])
    row5 = html.Tr([html.Td("Signal:"), html.Td(signal)])
    row6 = html.Tr([html.Td(""), html.Td("")])
    table_body = [html.Tbody([row1, row2, row3, row4, row5,row6])]
    table = dbc.Table( table_body, bordered=True)
    return table


# Functions
# days = 253-1 since day 000 has no returns
def sharpe(ema, ticker, rf=0, days=450):
    if ticker == '$AAPL':
        df = aapl_strat[f'Portfolio Value EMA {ema}'].pct_change()
        sharpe_ratio = np.sqrt(days) * (df.mean() - rf) / df.std()
        return '{:.2f}'.format(sharpe_ratio)
    elif ticker == '$TSLA':
        df = tsla_strat[f'Portfolio Value EMA {ema}'].pct_change()
        sharpe_ratio = np.sqrt(days) * (df.mean() - rf) / df.std()
        return '{:.2f}'.format(sharpe_ratio)


def total_returns(ema, ticker):
    if ticker == '$AAPL':
        df = aapl_strat[f'Portfolio Value EMA {ema}']
        tot_returns = (df.iloc[-1] - df.iloc[0]) / df.iloc[0]
        return '{:.2f}%'.format(tot_returns * 100)
    elif ticker == '$TSLA':
        df = tsla_strat[f'Portfolio Value EMA {ema}']
        tot_returns = (df.iloc[-1] - df.iloc[0]) / df.iloc[0]
        return '{:.2f}%'.format(tot_returns * 100)


def max_drawdown(ema, ticker):
    if ticker == '$AAPL':
        df = aapl_strat[[f'Portfolio Value EMA {ema}']]
        df["peak"] = df[f'Portfolio Value EMA {ema}'].cummax()
        df['drawdown'] = df[f'Portfolio Value EMA {ema}'] - df['peak']
        df['drawdown_percentage'] = (df['drawdown'] / df["peak"])
        mdd_per = df['drawdown_percentage'].min()
        return '{:.2f}%'.format(mdd_per * 100)
    elif ticker == '$TSLA':
        df = tsla_strat[[f'Portfolio Value EMA {ema}']]
        df["peak"] = df[f'Portfolio Value EMA {ema}'].cummax()
        df['drawdown'] = df[f'Portfolio Value EMA {ema}'] - df['peak']
        df['drawdown_percentage'] = (df['drawdown'] / df["peak"])
        mdd_per = df['drawdown_percentage'].min()
        return '{:.2f}%'.format(mdd_per * 100)


def num_buy_trades(ema, ticker):
    if ticker == '$AAPL':
        return len(aapl_strat.loc[aapl_strat[f'Action EMA {ema}'].str.contains('BUY')])
    elif ticker == '$TSLA':
        return len(tsla_strat.loc[tsla_strat[f'Action EMA {ema}'].str.contains('BUY')])


def num_sell_trades(ema, ticker):
    if ticker == '$AAPL':
        return len(aapl_strat.loc[aapl_strat[f'Action EMA {ema}'].str.contains('SELL')])
    elif ticker == '$TSLA':
        return len(tsla_strat.loc[tsla_strat[f'Action EMA {ema}'].str.contains('SELL')])


def volatility(ema, ticker, days=450):
    if ticker == '$AAPL':
        df = aapl_strat[f'Portfolio Value EMA {ema}'].pct_change()
        daily_vol = np.log(1 + df).std()
        annualised_vol = daily_vol * np.sqrt(days)
        return '{:.2f}%'.format(annualised_vol * 100)
    elif ticker == '$TSLA':
        df = tsla_strat[f'Portfolio Value EMA {ema}'].pct_change()
        daily_vol = np.log(1 + df).std()
        annualised_vol = daily_vol * np.sqrt(days)
        return '{:.2f}%'.format(annualised_vol * 100)


def get_table(df, ema, ticker):
    df.columns = ['Date', 'Portfolio Value ($USD)', 'Net Shares', 'Net Cash', 'Trade Signal',
                  'Pre-market P/L ($USD)', 'Trade Session P/L ($USD)', 'Total Day P/L ($USD)',
                  'Cumulative P/L ($USD)']
    df.drop(['Pre-market P/L ($USD)', 'Trade Session P/L ($USD)'], axis=1)
    # df["Date"] = df["Date"].apply(lambda x: x.date())
    data = df.to_dict('rows')

    columns = [{
        'id': df.columns[0],
        'name': 'Date',
        'type': 'datetime'
    }, {
        'id': df.columns[4],
        'name': 'Trade Signal',
        'type': 'text'
    }, {
        'id': df.columns[1],
        'name': 'Portfolio Value ($USD)',
        'type': 'numeric',
        'format': FormatTemplate.money(0)
    }, {
        'id': df.columns[2],
        'name': 'Net Shares',
        'type': 'numeric',
    }, {
        'id': df.columns[3],
        'name': 'Net Cash',
        'type': 'numeric',
        'format': FormatTemplate.money(0)
    }, {
        'id': df.columns[7],
        'name': 'Daily P/L ($USD)',
        'type': 'numeric',
        'format': FormatTemplate.money(0)
    }, {
        'id': df.columns[8],
        'name': 'Cumulative P/L ($USD)',
        'type': 'numeric',
        'format': FormatTemplate.money(0)
    }]

    package = html.Div([

                        dbc.Col(dt.DataTable(data=data, columns=columns, id='table',
                                             fixed_rows={'headers': True},
                                             style_table={'height': 350},
                                             style_header={'fontWeight': 'bold', 'backgroundColor': 'black',
                                                           'color': 'white'},

                                             style_cell_conditional=[
                                                 {
                                                     'if': {'column_id': c},
                                                     'textAlign': 'left'
                                                 } for c in ['Date', 'Region']
                                             ],

                                             style_data_conditional=[
                                                 {
                                                     'if': {'row_index': 'odd'},
                                                     'backgroundColor': 'rgb(248, 248, 248)'
                                                 }
                                             ],
                                             style_cell={
                                                 'whiteSpace': 'normal', 'textAlign': 'left',
                                                 'height': 'auto'}))
                        ])

    return package


external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/lux/bootstrap.min.css',
                        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css']

navbar = dbc.Nav(className="navbar-nav mr-auto", children=[
    dbc.DropdownMenu(label="Links", nav=True, children=[
        dbc.DropdownMenuItem([html.I(className="fa fa-apple"), "   $AAPL Stocktwits"],
                             href="https://stocktwits.com/symbol/AAPL", target="_blank"),
        dbc.DropdownMenuItem([html.I(className="fa fa-car"), "  $TSLA Stocktwits"],
                             href="https://stocktwits.com/symbol/TSLA", target="_blank"),

    ])
])

# Dash App
# app = JupyterDash(__name__, external_stylesheets=external_stylesheets)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = dbc.Container(fluid=True, children=[
    dbc.Row([
        dbc.Col([html.H2(children="StockTwits Sentiments Momentum Trading Dashboard",
                         style={'textAlign': 'centre',
                                'color': 'white'})], md=10),
        dbc.Col(md=2, children=[navbar])
    ], className='navbar navbar-expand-lg navbar-dark bg-primary'),
    html.Br(), html.Br(),
    dbc.Row([
        dbc.Col(xs=12, sm=12, md=12, lg=3, xl=3, children=[
            dbc.Col(html.H3(children='Select Bull/Bear Ratio EMA Days: ▼'),
                    style={'textAlign': 'left', 'font-weight': 'bold'}),
            dbc.Col(dcc.Dropdown(
                id='slider',
                options=[{'label': f'{i}-Days EMA Strategy', 'value': i} for i in [5, 6, 7, 8, 9, 10, 15, 20]],
                value=5)),
            html.Br(),
            dbc.Col(html.Div(id="output-panel")),
            html.Br(), html.Br(),
            dbc.Col(dbc.Card(body=True, className="card bg-light mb-3", children=[
                html.Div("Feladat leírása", className="card-header"),
                html.Div(className="card-body", children=[
                    html.P(children=["""Input adatként vesszük a Stocktwits.com Apple-l és Tesla-val foglalkozó tweet-jeit. 
                                        A Stocktwits-es tweet-ek tulajdonsága, hogy nagyobb részük Bearish/Bullish tag-gal van ellátva.
                                        Ez alapján egyszerű logisztikus regresszió használatával modell-t hozunk létre ami a nem felcímkézett tweet-ek
                                        besorolását végzi.""",

                                     html.Br(),
                                     html.Br(),
                                     """Minden naphoz képzünk egy arányt a Bullish és Bearish tweet-ek számából majd ezekből számoljuk az exponenciális mozgó átlagokat.""",
                                     html.Br(), html.Br(),
                                     """Trading stratégia:""",
                                    html.Br(),
                                    """BUY: ha az adott napra vonatkozó Bullish/Bearish ratio magasabb a választott exponenciális mozgóátlagnál""",
                                    html.Br(),
                                    """SELL: ha a Bullish/Bearish ratio alacsonyabb""",


                                     ], className="card-text")
                ])
            ])
                    )
        ]),

        dbc.Col(lg=9, xl=9, children=[
            dbc.Col(html.H3("Portfolio Backtest Performance and Signals"), width={"size": 7, "offset": 3},
                    style={'textAlign': 'center'}),
            dbc.Tabs(className="nav nav-pills", id='yaxis-column',
                     children=[
                         dbc.Tab(label='$TSLA', tab_id='$TSLA'),
                         dbc.Tab(label='$AAPL', tab_id='$AAPL')
                     ],
                     active_tab="$AAPL"),
            dcc.Graph(id='Portfolio-chart'),
            dcc.Graph(id='buy-sell-chart', clickData={'points': [{'customdata': date.today().strftime("%Y-%m-%d")}]}),
            dcc.Graph(id='bull-bear-chart'),

            dbc.Tabs(className="nav nav-pills", id='my-tab',
                     children=[
                         dbc.Tab(label='Trade Log', tab_id='t1'),
                         dbc.Tab(label=f'Sentiments on {date.today().strftime("%Y-%m-%d")}', tab_id='t2',id='sentiment_tab')
                     ],
                     active_tab="t1"),
            html.Div(id="trade-log-content",children=[dbc.Col(html.Div(id="data-table"))]),
            html.Div(id='sentiment-content',children=[dbc.Button("Refresh", id="refresh_button", n_clicks=0)] )



        ])
    ])
])




@app.callback(
    [Output('trade-log-content', 'style'),Output('sentiment-content', 'style'),Output('sentiment-content','children')],
    [Input('slider', 'value'),Input('yaxis-column', 'active_tab'),Input('sentiment_tab','label'),Input('my-tab','active_tab')])
def update_y_timeseries(ema,ticker,selected_day,active_tab):

    # ctx = dash.callback_context
    # print(ctx)
    ticker=ticker.replace('$', '')
    # print('selected_day ',selected_day)
    selected_day=selected_day.split("on ")[1]
    # if (len(clickData['points'])==3):
    #     selected_day=clickData['points'][1]['x']
    # else:
    #     selected_day=date.today().strftime("%Y-%m-%d")

    if active_tab == "t1":
        # dbc.Col(html.H4(f'EMA {ema} Trade Log ({ticker})')),
        return {'display': 'inline'},{'display': 'none'},dbc.Button("Get Today's Sentiment", id="refresh_button", n_clicks=0)

    elif active_tab == "t2":
        if selected_day!=date.today().strftime("%Y-%m-%d"):
            df = load(f'../data/processed_sentiment_{ticker}.pkl')
            df['Date'] = pd.to_datetime(df['Date'])
            df = df[df['Date'] == selected_day]
        else:
            df = get_today_sentiment_dataframe(ticker)


        value_counts = df['Combined sentiment'].value_counts()
        bullish_bearish_ratio = (value_counts['Bullish']/value_counts['Bearish'])
        total_sentiment = len(df)


        data = df.to_dict('rows')

        columns = [{
            'id': 'Time',
            'name': 'Time',
            'type': 'text'
        }, {
            'id': 'message',
            'name': 'Message',
            'type': 'text'
        }, {
            'id': 'Combined sentiment',
            'name': 'Sentiment',
            'type': 'text'
        }]

        data_table = dt.DataTable(data=data, columns=columns, id='table',

                                  style_header={'fontWeight': 'bold', 'backgroundColor': 'black',
                                                'color': 'white', 'textAlign': 'left'},
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
        property_table = get_sentiment_property_table(ticker,selected_day, ema,total_sentiment,bullish_bearish_ratio)

        pie_chart = dcc.Graph(id='pie-chart', figure=px.pie(names=value_counts.index, values=value_counts.values,
                                                            color=value_counts.index,
                                                            color_discrete_map={'Bullish': 'rgb(50, 205, 50)',
                                                                                'Bearish': 'rgb(255, 69, 0)'}, ))

        tab_content=html.Div(id='sentiment_div',  children=[
            html.Div(id='sentiment_table', style={'width': '32%', 'display': 'inline-block'}, children=[data_table]),
            html.Div(children=[pie_chart],
                     style={'width': '32%', 'float': 'right', 'display': 'inline-block'}),
            html.Div(children=[
                html.Div(className="card-body", id='sentiment_property', children=[property_table]),
                dbc.Button("Get Today's Sentiment", id="refresh_button", n_clicks=0)
            ], style={'width': '32%', 'float': 'right', 'display': 'inline-block'})
        ])

        return {'display': 'none'}, {'display': 'inline'}, tab_content

# @app.callback(Output('sentiment_tab','label'),
#                 Input('buy-sell-chart', 'clickData'))
# def day_was_selected_from_chart(clickData):
#     selected_day = clickData['points'][1]['x']
#     return f'Sentiment on {selected_day}'

@app.callback([Output('sentiment_tab','label'),Output('my-tab','active_tab')],
              [ Input('buy-sell-chart', 'clickData'), Input('refresh_button', 'n_clicks'),State('buy-sell-chart', 'clickData')],prevent_initial_call=True)
def get_today_sentiment(clickData,n_clicks,old):
    ctx = dash.callback_context
    if ctx.triggered:
        triggered_by = ctx.triggered[0]["prop_id"].split(".")[0]
        if triggered_by=='buy-sell-chart' and len(clickData['points'])==3:
            selected_day = clickData['points'][1]['x']
            return f'Sentiment on {selected_day}', 't2'
        elif triggered_by=='refresh_button' and n_clicks>0:
            return f'Sentiment on {date.today().strftime("%Y-%m-%d")}', 't2'
    raise PreventUpdate


    # # print('label changed ',n_clicks,clickData,old)
    # if n_clicks==0:
    #     raise PreventUpdate
    # selected_day = date.today().strftime("%Y-%m-%d")
    # if n_clicks>0:
    #     return f'Sentiment on {selected_day}', 't2'
    # elif len(clickData['points'])==3:
    #     selected_day = clickData['points'][1]['x']
    #     return f'Sentiment on {selected_day}', 't2'
    # else:
    #     return f'Sentiment on {selected_day}', 't1'
    # return 'Sentiment'

# @app.callback(Output('tabs-example-content', 'children'),
#               Input('tabs-example', 'value'))
# def render_content(tab):
#     if tab == 'tab-1':
#         return dcc.Graph(id='bull-bear-chart')
#     elif tab == 'tab-2':
#         return html.Div([
#             html.H3('Tab content 2')
#         ])


# Function to render Portfolio Chart
@app.callback(
    Output('Portfolio-chart', 'figure'),
    Input('yaxis-column', 'active_tab'),
    Input('slider', 'value'))  # Slider value will be EMA 5,6,7,8,9,10,15,20
def update_graph(yaxis_column_name, ema):
    # TSLA / Portfolio

    if yaxis_column_name == "$TSLA":
        data = [
            go.Scatter(x=tsla_strat[f'Date EMA {ema}'], y=(tsla_strat[f"Portfolio Value EMA {ema}"]),
                       mode='lines', name="Portfolio Value",
                       line=dict(color='red', width=0.5), fill='tozeroy')
        ]
        fig = go.Figure(data=data)


        fig.update_layout(
            title=f"<b>Portfolio Performance ($TSLA) - EMA {ema}</b>",
            template="plotly_white",
            hovermode="x",
            showlegend=True,
            legend=dict(
                x=0.4,
                y=1.15,
                orientation='h'),
            xaxis=dict(
                showgrid=True,
                showline=True,
                tickmode='auto',
                fixedrange=True,
                range=['2020-01-01', '2021-04-01']),
            yaxis=dict(
                type='linear',
                showline=False,
                showgrid=False,
                fixedrange=True,
                ticksuffix=' USD'
            ))
        fig.update_xaxes(showspikes=True, spikecolor="red", spikesnap="cursor", spikemode="across",
                         tickformat='%d %b %y')
        fig.update_yaxes(showspikes=True, spikecolor="grey", spikethickness=2)
        return fig

    # AAPL / Porfolio
    elif yaxis_column_name == "$AAPL":
        data = [
            go.Scatter(x=aapl_strat[f'Date EMA {ema}'], y=(aapl_strat[f"Portfolio Value EMA {ema}"]),
                       mode='lines', name="Portfolio Value",
                       line=dict(color='black', width=0.5), fill='tozeroy')
        ]
        fig = go.Figure(data=data)


        fig.update_layout(
            title=f"<b>Portfolio Performance ($AAPL) - EMA {ema}</b>",
            template="plotly_white",
            hovermode="x",
            showlegend=True,
            legend=dict(
                x=0.4,
                y=1.15,
                orientation='h'),
            xaxis=dict(
                showgrid=True,
                showline=True,
                tickmode='auto',
                fixedrange=True,
                ),
            yaxis=dict(
                type='linear',
                showline=False,
                showgrid=False,
                ticksuffix=' USD',
                fixedrange=True,
                range=[7000, 13000]
            ))
        fig.update_xaxes(showspikes=True, spikecolor="black", spikesnap="cursor", spikemode="across",
                         tickformat='%d %b %y')
        fig.update_yaxes(showspikes=True, spikecolor="grey", spikethickness=2)
        return fig


# Function to render Buy/ Sell Chart Callback
@app.callback(
    Output('buy-sell-chart', 'figure'),
    Input('yaxis-column', 'active_tab'),
    Input('slider', 'value'))  # Slider value will be EMA 5,6,7,8,9,10,15,20
def update_graph(yaxis_column_name, ema):
    # TSLA / Signal
    if yaxis_column_name == "$TSLA":
        data = [
            go.Scatter(x=tsla_strat[f'Date EMA {ema}'], y=(tsla_strat[f"Adjusted Close EMA {ema}"]),
                       mode='lines', name="TSLA Closing Price",

                       line=dict(color='#86d3e3', width=2)),
            go.Scatter(x=tsla_strat.loc[tsla_strat[f"Action EMA {ema}"].str.contains("BUY"), f"Date EMA {ema}"],
                       y=(tsla_strat.loc[
                           tsla_strat[f"Action EMA {ema}"].str.contains("BUY"), f"Adjusted Close EMA {ema}"]),
                       mode='markers', name="Buy",
                       marker=dict(symbol='triangle-up', color="green", size=7)),
            go.Scatter(x=tsla_strat.loc[tsla_strat[f"Action EMA {ema}"].str.contains("SELL"), f"Date EMA {ema}"],
                       y=(tsla_strat.loc[
                           tsla_strat[f"Action EMA {ema}"].str.contains("SELL"), f"Adjusted Close EMA {ema}"]),
                       mode='markers', name="Sell",
                       marker=dict(symbol='triangle-down', color="red", size=7))
        ]
        fig = go.Figure(data=data)


        fig.update_layout(
            title=f"<b>Long/ Short Signal ($TSLA) - EMA {ema}</b>",
            template="plotly_white",
            hovermode="x",
            showlegend=True,
            legend=dict(
                x=0.4,
                y=1.15,
                orientation='h'),
            xaxis=dict(
                showgrid=True,
                showline=True,
                tickmode='auto',
                fixedrange=True,
                range=['2020-01-01', '2021-04-01']),
            yaxis=dict(
                type='linear',
                showline=False,
                showgrid=False,
                fixedrange=True,
                ticksuffix=' USD'
            ))
        fig.update_xaxes(showspikes=True, spikecolor="red", spikesnap="cursor", spikemode="across",
                         tickformat='%d %b %y')
        fig.update_yaxes(showspikes=True, spikecolor="grey", spikethickness=2)
        return fig

        # AAPL / Signal
    elif yaxis_column_name == "$AAPL":
        data = [
            go.Scatter(x=aapl_strat[f'Date EMA {ema}'], y=(aapl_strat[f"Adjusted Close EMA {ema}"]),
                       mode='lines', name="AAPL Closing Price",
                       customdata=tsla_strat[f'Date EMA {ema}'],
                       line=dict(color='#86d3e3', width=2)),
            go.Scatter(x=aapl_strat.loc[aapl_strat[f"Action EMA {ema}"].str.contains("BUY"), f"Date EMA {ema}"],
                       y=(aapl_strat.loc[
                           aapl_strat[f"Action EMA {ema}"].str.contains("BUY"), f"Adjusted Close EMA {ema}"]),
                       mode='markers', name="Buy",
                       marker=dict(symbol='triangle-up', color="green", size=7)),
            go.Scatter(x=aapl_strat.loc[aapl_strat[f"Action EMA {ema}"].str.contains("SELL"), f"Date EMA {ema}"],
                       y=(aapl_strat.loc[
                           aapl_strat[f"Action EMA {ema}"].str.contains("SELL"), f"Adjusted Close EMA {ema}"]),
                       mode='markers', name="Sell",
                       marker=dict(symbol='triangle-down', color="red", size=7))
        ]
        fig = go.Figure(data=data)


        fig.update_layout(
            title=f"<b>Long/ Short Signal ($AAPL) - EMA {ema}</b>",
            template="plotly_white",
            hovermode="x",
            showlegend=True,
            legend=dict(
                x=0.4,
                y=1.15,
                orientation='h'),
            xaxis=dict(
                showgrid=True,
                showline=True,
                tickmode='auto',
                fixedrange=True,
                range=['2020-01-01', '2021-04-01']),
            yaxis=dict(
                type='linear',
                showline=False,
                showgrid=False,
                fixedrange=True,
                ticksuffix=' USD'
            ))
        fig.update_xaxes(showspikes=True, spikecolor="black", spikesnap="cursor", spikemode="across",
                         tickformat='%d %b %y')
        fig.update_yaxes(showspikes=True, spikecolor="grey", spikethickness=2)
        return fig


# Function to render stats
@app.callback(
    Output("output-panel", "children"),
    Input("slider", "value"),  # ema value
    Input('yaxis-column', 'active_tab'))  # tsla aapl
def render_output_panel(ema, ticker):
    panel = html.Div([
        html.H5(f'{ema}-Days EMA Strategy Results:'),
        dbc.Card(body=True, className="text-white bg-primary", children=[
            #             html.H3(f'EMA {ema} Strategy', className='card-title'),

            html.H6("Total Returns:", style={"color": "white"}),
            html.H3(total_returns(ema, ticker), style={"color": "white"}),

            html.H6("Sharpe Ratio:", className="text-success"),
            html.H3(sharpe(ema, ticker), className="text-success"),

            html.H6("Volatility:", style={"color": "white"}),
            html.H3(volatility(ema, ticker), style={"color": "white"}),

            html.H6("Max Drawdown:", className="text-danger"),
            html.H3(max_drawdown(ema, ticker), className="text-danger"),

            html.H6("Number of Long / Short Trades:", style={"color": "white"}),
            html.H3(f"Long: {num_buy_trades(ema, ticker)}", style={"color": "white"}),
            html.H3(f"Short: {num_sell_trades(ema, ticker)}", style={"color": "white"}),

        ])
    ])
    return panel


# Function for bull/bear graph
@app.callback(
    Output('bull-bear-chart', 'figure'),
    Input('yaxis-column', 'active_tab'))
def update_graph_2(yaxis_column_name):
    # TSLA / Signal
    if yaxis_column_name == "$TSLA":

        fig = make_subplots(specs=[[{'secondary_y': True}]])

        # TSLA Bullish Area (Visible when TSLA Dropdown)
        fig.add_trace(go.Scatter(
            x=df_tesla_merged["Date"], y=df_tesla_merged["% of Bullish"],
            mode='lines',

            name="Bullish",
            line=dict(width=1, color='rgba(0,102,0,0.3)'),
            stackgroup='one',
            groupnorm='percent'), secondary_y=False)

        # TSLA Bearish Area (Visible when TSLA Dropdown)
        fig.add_trace(go.Scatter(
            x=df_tesla_merged["Date"], y=df_tesla_merged["% of Bearish"],
            mode='lines',
            name="Bearish",
            line=dict(width=1, color='rgba(190,23,23,0.3)'),
            stackgroup='one', fill='tonexty'), secondary_y=False)

        # TSLA Chart (Visible when TSLA Dropdown)
        fig.add_trace(go.Scatter(
            x=df_tesla_merged["Date"], y=df_tesla_merged["Adjusted Close"],
            mode='lines',
            name='TSLA Closing Price',
            line=dict(width=2, color='black', dash='solid')),
            secondary_y=True)

        # 50% median line (Always visible)
        fig.add_trace(go.Scatter(
            x=df_tesla_merged["Date"], y=df_tesla_merged["Middle line"],
            mode='lines',
            name="50%",
            line=dict(width=0.5, color='black', dash='dot'),
            showlegend=False,
            hoverinfo='x'), secondary_y=False)

        # Add the top few buttons


        # Configure x & y axis & hovermode
        fig.update_layout(
            margin=dict(pad=4.5),
            title=dict(
                text="<b>Stocktwits Pre-Market Sentiments vs $TSLA Performance</b>",
            ),
            template="plotly_white",
            hovermode="x",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1),
            xaxis=dict(
                showgrid=True,
                showline=False,
                tickmode='auto',
                nticks=7,
                fixedrange=True,
                range=['2020-01-01', '2021-04-01']),
            yaxis=dict(
                # automargin=True,
                type='linear',
                range=[0, 100],
                showgrid=False,
                ticksuffix='%',
                fixedrange=True,
                tickvals=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
            # Converted into the dropdown menu option
            yaxis2=dict(
                # automargin=True,
                type='linear',
                showgrid=False,
                fixedrange=True,
                ticksuffix=' USD')
            #         range=[0, 200])
        )

        # Configure spikes
        fig.update_xaxes(showspikes=True, spikecolor="black", spikesnap="cursor", spikedash='dot', spikemode="across",
                         tickformat='%d %b %y', spikethickness=1)
        fig.update_yaxes(title_text="<b>% of Bulls vs Bears</b>", showspikes=False, spikecolor="grey",
                         spikethickness=0.25)
        fig.update_yaxes(title_text="<b>Closing Price</b>", secondary_y=True)

        return fig

        # AAPL / Signal
    elif yaxis_column_name == "$AAPL":
        fig = make_subplots(specs=[[{'secondary_y': True}]])
        # AAPL
        # AAPL Bullish Area (Visible when AAPL Dropdown)
        fig.add_trace(go.Scatter(
            x=df_aapl_merged["Date"], y=df_aapl_merged["% of Bullish"],
            mode='lines',
            name="Bullish",
            line=dict(width=1, color='rgba(0,102,0,0.3)'),
            stackgroup='one',
            groupnorm='percent'), secondary_y=False)

        # AAPL Bearish Area (Visible when AAPL Dropdown)
        fig.add_trace(go.Scatter(
            x=df_aapl_merged["Date"], y=df_aapl_merged["% of Bearish"],
            mode='lines',
            name="Bearish",
            line=dict(width=1, color='rgba(190,23,23,0.3)'),
            stackgroup='one', fill='tonexty'), secondary_y=False)

        # AAPL Chart (Visible when AAPL Dropdown)
        fig.add_trace(go.Scatter(
            x=df_aapl_merged["Date"], y=df_aapl_merged["Adjusted Close"],
            mode='lines',
            name='AAPL Closing Price',
            line=dict(width=2, color='black', dash='solid')),
            secondary_y=True)

        # 50% median line (Always visible)
        fig.add_trace(go.Scatter(
            x=df_aapl_merged["Date"], y=df_aapl_merged["Middle line"],
            mode='lines',
            name="50%",
            line=dict(width=0.5, color='black', dash='dot'),
            showlegend=False,
            hoverinfo='x'), secondary_y=False)

        # Add the top few buttons


        # Configure x & y axis & hovermode
        fig.update_layout(
            margin=dict(pad=4.5),
            title="<b>Stocktwits Pre-Market Sentiments vs $AAPL Performance </b>",
            template="plotly_white",
            hovermode="x",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1),
            xaxis=dict(
                showgrid=True,
                showline=False,
                tickmode='auto',
                nticks=7,
                fixedrange=True,
                range=['2020-01-01', '2021-04-01']),
            yaxis=dict(
                type='linear',
                range=[0, 100],
                showgrid=False,
                ticksuffix='%',
                fixedrange=True,
                tickvals=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
            # Converted into the dropdown menu option
            yaxis2=dict(
                type='linear',
                showgrid=False,
                fixedrange=True,
                ticksuffix=' USD')
            #         range=[0, 200])
        )

        # Configure spikes
        fig.update_xaxes(showspikes=True, spikecolor="black", spikesnap="cursor", spikedash='dot', spikemode="across",
                         tickformat='%d %b %y', spikethickness=1)
        fig.update_yaxes(title_text="<b>% of Bulls vs Bears</b>", showspikes=False, spikecolor="grey",
                         spikethickness=0.25)
        fig.update_yaxes(title_text="<b>Closing Price</b>", secondary_y=True)

        return fig


@app.callback(
    Output('data-table', 'children'),
    Input('yaxis-column', 'active_tab'),
    Input("slider", "value"))
def update_graph(yaxis_column_name, ema):
    # TSLA / Signal
    if (yaxis_column_name == "$TSLA") & (ema == 5):
        df = tsla_strat.iloc[:, 1:10]
        return get_table(df, ema, ticker=yaxis_column_name)
    elif (yaxis_column_name == "$TSLA") & (ema == 6):
        df = tsla_strat.iloc[:, 11:20]
        return get_table(df, ema, ticker=yaxis_column_name)
    elif (yaxis_column_name == "$TSLA") & (ema == 7):
        df = tsla_strat.iloc[:, 21:30]
        return get_table(df, ema, ticker=yaxis_column_name)
    elif (yaxis_column_name == "$TSLA") & (ema == 8):
        df = tsla_strat.iloc[:, 31:40]
        return get_table(df, ema, ticker=yaxis_column_name)
    elif (yaxis_column_name == "$TSLA") & (ema == 9):
        df = tsla_strat.iloc[:, 41:50]
        return get_table(df, ema, ticker=yaxis_column_name)
    elif (yaxis_column_name == "$TSLA") & (ema == 10):
        df = tsla_strat.iloc[:, 51:60]
        return get_table(df, ema, ticker=yaxis_column_name)
    elif (yaxis_column_name == "$TSLA") & (ema == 15):
        df = tsla_strat.iloc[:, 61:70]
        return get_table(df, ema, ticker=yaxis_column_name)
    elif (yaxis_column_name == "$TSLA") & (ema == 20):
        df = tsla_strat.iloc[:, 71:80]
        return get_table(df, ema, ticker=yaxis_column_name)

    # AAPL / Signal
    elif (yaxis_column_name == "$AAPL") & (ema == 5):
        df = aapl_strat.iloc[:, 1:10]
        return get_table(df, ema, ticker=yaxis_column_name)
    elif (yaxis_column_name == "$AAPL") & (ema == 6):
        df = aapl_strat.iloc[:, 11:20]
        return get_table(df, ema, ticker=yaxis_column_name)
    elif (yaxis_column_name == "$AAPL") & (ema == 7):
        df = aapl_strat.iloc[:, 21:30]
        return get_table(df, ema, ticker=yaxis_column_name)
    elif (yaxis_column_name == "$AAPL") & (ema == 8):
        df = aapl_strat.iloc[:, 31:40]
        return get_table(df, ema, ticker=yaxis_column_name)
    elif (yaxis_column_name == "$AAPL") & (ema == 9):
        df = aapl_strat.iloc[:, 41:50]
        return get_table(df, ema, ticker=yaxis_column_name)
    elif (yaxis_column_name == "$AAPL") & (ema == 10):
        df = aapl_strat.iloc[:, 51:60]
        return get_table(df, ema, ticker=yaxis_column_name)
    elif (yaxis_column_name == "$AAPL") & (ema == 15):
        df = aapl_strat.iloc[:, 61:70]
        return get_table(df, ema, ticker=yaxis_column_name)
    elif (yaxis_column_name == "$AAPL") & (ema == 20):
        df = aapl_strat.iloc[:, 71:80]
        return get_table(df, ema, ticker=yaxis_column_name)




# this is needed for the procfile to deploy to heroku
server = app.server

if __name__ == '__main__':
    app.run_server(debug=True, port=3336)
