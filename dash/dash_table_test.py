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
from datetime import date
import json
import requests
from datetime import datetime
from nltk.corpus import stopwords
import re
import string
from nltk.tokenize import word_tokenize

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

selected_day='2021-02-11'
ticker='AAPL'
total_sentiment = 0
pie_chart_dataframe = pd.DataFrame()
bearish_bullish_ratio = 0.0
ema_value=3.0

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
    raw_df['message'] = raw_df['message'].str.lower()

    # Converting HTML to UTF-8
    # raw_df["message"] = raw_df["message"].apply(html.unescape)

    # Removing hastags, mentions, pagebreaks, handles
    # Keeping the words behind hashtags as they may provide useful information about the comments e.g. #Bullish #Lambo
    raw_df["message"] = raw_df["message"].str.replace(r'(@[^\s]+|[#]|[$])', ' ')  # Replace '@', '$' and '#...'
    raw_df["message"] = raw_df["message"].str.replace(r'(\n|\r)', ' ')  # Replace page breaks

    # Removing https, www., any links etc
    raw_df["message"] = raw_df["message"].str.replace(r'((https:|http:)[^\s]+|(www\.)[^\s]+)', ' ')

    # Removing all numbers
    raw_df["message"] = raw_df["message"].str.replace(r'[\d]', '')

    # Remove emoji
    raw_df["message"] = raw_df["message"].apply(lambda row: remove_emoji(row))

    # Tokenization
    raw_df['message'] = raw_df['message'].apply(word_tokenize)

    # Remove Stopwords
    raw_df['message'] = raw_df['message'].apply(remove_stopwords)

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
                if datetime_obj.hour < datetime.now().hour - 4:
                    done = True
                # print(datetime_obj.hour, datetime_obj.minute)
        #
        if done:
            break
    # print(df)
    df = tweets_preprocessing(df)
    return df

def get_sentiment_property_table(current_day,total_sentiment,bearish_bullish_ratio):

    if (bearish_bullish_ratio>ema_value):
        signal = 'SELL'
    else:
        signal='BUY'
    row1 = html.Tr([html.Td("Date:"), html.Td(current_day)])
    row2 = html.Tr([html.Td("Total sentiment:"), html.Td(total_sentiment)])
    row3 = html.Tr([html.Td("Bearish / Bullish ratio:"), html.Td('{:.3f}'.format(bearish_bullish_ratio))])
    row4 = html.Tr([html.Td("EMA value:"), html.Td(ema_value)])
    row5 = html.Tr([html.Td("Signal:"), html.Td(signal)])
    row6 = html.Tr([html.Td(""), html.Td("")])
    table_body = [html.Tbody([row1, row2, row3, row4, row5,row6])]
    table = dbc.Table( table_body, bordered=True)
    return table




app = dash.Dash(__name__)

@app.callback(
    Output('sentiment_table', 'children'),
    Output('sentiment_property', 'children'),
    Output('pie-div', 'children'),
    Input('refresh_button', 'n_clicks'),
          )
def update_output(n_clicks):
    if n_clicks==0:
        df = load(f'../data/processed_sentiment_{ticker}.pkl')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[df['Date'] == selected_day]
        current_day=selected_day
    else:
        current_day=date.today().strftime("%Y-%m-%d")
        # df = pd.DataFrame({
        #     "message": ["m1", "m2"],
        #     "Date": ["2021-01-02", "2021-02-02"],
        #     "Time": ["20:00:01", "10:30:01"],
        #     "Combined sentiment": ["Bearish", "Bullish"]
        # })
        df=get_today_sentiment_dataframe(ticker)

    total_sentiment = len(df)
    value_counts = df['Combined sentiment'].value_counts()
    bearish_bullish_ratio = (value_counts['Bearish'] / value_counts['Bullish'])

    data = df.to_dict('rows')

    data_table=dt.DataTable(data=data, columns=columns, id='table',

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
    property_table=get_sentiment_property_table(current_day,total_sentiment,bearish_bullish_ratio)
    pie_chart=dcc.Graph(id='pie-chart',figure=px.pie(names=value_counts.index, values=value_counts.values,
                            color=value_counts.index, color_discrete_map={'Bullish': 'rgb(50, 205, 50)',
                                                                          'Bearish': 'rgb(255, 69, 0)'}, ))



    return data_table,property_table,pie_chart



app.layout = html.Div([
    html.Div([
    html.Div(id='sentiment_table',style={'width': '32%','display': 'inline-block'},children=[]),
            html.Div( id='pie-div',children=[],
                style={'width': '32%', 'float': 'right','display': 'inline-block'}),
        html.Div(children=[
                        html.Div(className="card-body", id='sentiment_property',children=[]),dbc.Button("Refresh", id="refresh_button", n_clicks=0)
        ],

            style={'width': '32%',  'float': 'right','display': 'inline-block'})
        ])
    ])
#                 ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
# )

if __name__ == '__main__':
    app.run_server(debug=True)