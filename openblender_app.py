#!/usr/bin/env python
# coding: utf-8

# In[4]:
import pandas as pd
import numpy as np
import dash
import OpenBlender
import warnings
warnings.filterwarnings('ignore')
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_table as dt
import dash_table.FormatTemplate as FormatTemplate
from joblib import dump, load


cnbc = load('openblender_data\\cnbc.pkl')
cnbc = cnbc.loc[cnbc['CNBC_INTER.text_last1days:apple'].str.len() > 0]
wall_street = load('openblender_data\\wall_street.pkl')
wall_street = wall_street.loc[wall_street['WALL_STREE.text_last1days:apple'].str.len() > 0]
df1 = pd.DataFrame()
df1['date'] = wall_street['date']
df1['tweets from cnbc tweet & wall street journal'] = [''.join(map(str, l)) for l in wall_street['WALL_STREE.text_last1days:apple']]

# df = df.rename(columns={0: 'lists'})
df2 = pd.DataFrame()
df2['date'] = cnbc['date']
df2['tweets from cnbc tweet & wall street journal'] = [''.join(map(str, l)) for l in cnbc['CNBC_INTER.text_last1days:apple']]
tweets_concated = pd.concat([df1, df2])

df_anchor=load( 'openblender_data\\df_final_anchor.pkl')
# print(df_anchor['date,log_diff,negative_poc,positive_poc,target'])

df_anchor = df_anchor.drop('date', axis = 1)
df_anchor['date'] = [OpenBlender.unixToDate(ts, timezone='GMT',date_format = "%Y-%m-%d",) for ts in df_anchor.timestamp]
df_anchor['direction'] = df_anchor['target'].apply(lambda x: 'Up' if x == 1 else 'Down')
df_anchor['...']='...'
word_vector=df_anchor[['date','direction','scandal','reforms','warns','project','renewed','extra','pandemic','longterm','president','jailed','...']]



def get_bw_content():
    business_source = load('openblender_data\\business_source.pkl')
    link = (html.A(html.P(item['name']), href=item['url'], target="_blank") for item in business_source)
    observation= list(item['num_observations'] for item in business_source)
    dictionary = {"Source": link,"Observation": observation}
    df = pd.DataFrame(dictionary)
    return dbc.Table.from_dataframe(df,
                                    bordered=True,
                                    hover=True,
                                    responsive=True,
                                    striped=True,
                                    style={'fontWeight': 'bold',
                                            'textAlign': 'left','width':'80%','height:':'20px'},
                              )

    # business_source = load('openblender_data\\business_source.pkl')
    # # names = list(html.A(html.P(+item['name']+), href="'+item['url']+'", target="_blank"' for item in business_source)
    #
    # # names = list(item['name'] for item in business_source)
    # # html.A(html.P('Link'), href="https://www.yahoo.com", target='_blank')
    #
    # urls = list(item['url'] for item in business_source)
    # observations = list(item['num_observations'] for item in business_source)
    # df = pd.DataFrame(list(zip(names, urls, observations)),
    #                   columns=['name', 'url', 'observation'])
    #
    # return dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True,
    #                                 style_header={'fontWeight': 'bold', 'backgroundColor': 'black',
    #                                         'color': 'white', 'textAlign': 'left'},
    #                           style_cell={
    #                               'whiteSpace': 'normal', 'textAlign': 'left',
    #                               'height': '10px'},
    #                           page_size=10,)

# def get_tw_content():
#
#     global tweets_concated
#
#
#     return dbc.Table.from_dataframe(tweets_concated,
#                                     bordered=True,
#                                     hover=True,
#                                     responsive=True,
#                                     striped=True,
#                                     style={'fontWeight': 'bold',
#                                             'textAlign': 'left','width':'80%'},
#                               )

def get_tw_content():
    data = tweets_concated.to_dict('rows')

    columns = [{
        'id': 'date',
        'name': 'date',
        'type': 'text'
    }, {
        'id': 'tweets from cnbc tweet & wall street journal',
        'name': 'tweets from cnbc tweet & wall street journal',
        'type': 'text'
    }]

    data_table = dt.DataTable(data=data, columns=columns, id='tweet_table',
                                  style_header={'fontWeight': 'bold', 'backgroundColor': 'black',
                                                'color': 'white', 'textAlign': 'left'},
                                  style_cell={
                                      'whiteSpace': 'normal', 'textAlign': 'left',
                                      'height': 'auto'},
                              style_cell_conditional=[
                                  {'if': {'column_id': 'date'},
                                   'width': '30px'},
                                  {'if': {'column_id': 'tweets from cnbc tweet & wall street journal'},
                                   'width': '400px'},
                              ],
                                  page_size=25,
                                  )



    return data_table

# def get_word_vector_content():
#
#     global word_vector
#
#
#     return dbc.Table.from_dataframe(word_vector,
#                                     bordered=True,
#                                     hover=True,
#                                     responsive=True,
#                                     striped=True,
#                                     style={'fontWeight': 'bold',
#                                             'textAlign': 'left','width':'80%'},
#                               )

def get_word_vector_content():
    data = word_vector.to_dict('rows')

    columns = [{
        'id': 'date',
        'name': 'date',
        'type': 'text'
    }, {
        'id': 'direction',
        'name': 'direction',
        'type': 'text'
    },{
        'id': 'scandal',
        'name': 'scandal',
        'type': 'text'
    },{
        'id': 'reforms',
        'name': 'reforms',
        'type': 'text'
    },{
        'id': 'warns',
        'name': 'warns',
        'type': 'text'
    },{
        'id': 'project',
        'name': 'project',
        'type': 'text'
    },{
        'id': 'renewed',
        'name': 'renewed',
        'type': 'text'
    },{
        'id': 'extra',
        'name': 'extra',
        'type': 'text'
    },{
        'id': 'pandemic',
        'name': 'pandemic',
        'type': 'text'
    },{
        'id': 'longterm',
        'name': 'longterm',
        'type': 'text'
    },{
        'id': 'president',
        'name': 'president',
        'type': 'text'
    },{
        'id': 'jailed',
        'name': 'jailed',
        'type': 'text'
    },{
        'id': '...',
        'name': '...',
        'type': 'text'
    }]

    data_table = dt.DataTable(data=data, columns=columns, id='tweet_table',
                                  style_header={'fontWeight': 'bold', 'backgroundColor': 'black',
                                                'color': 'white', 'textAlign': 'left'},
                                  style_cell={
                                      'whiteSpace': 'normal', 'textAlign': 'left',
                                      'height': 'auto','width':'100px'},

                                  page_size=25,
                                  )



    return data_table

external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/lux/bootstrap.min.css',
                        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

navbar = dbc.Nav(className="navbar-nav mr-auto", children=[
    dbc.DropdownMenu(label="Links", nav=True, children=[
        dbc.DropdownMenuItem([html.I(className="fa fa-apple"), "OpenBlender"],
                             href="https://www.openblender.io/", target="_blank"),

    ])
])

bw_content=get_bw_content()
tw_content=get_tw_content()
wv_content=get_word_vector_content()







tabs = html.Div(
    [
        dbc.Tabs(
            [
                dbc.Tab(label="BUSINESS DATASETS (69)", tab_id="tab-1"),
                dbc.Tab(label="TWEETS", tab_id="tab-2"),
                dbc.Tab(label="WORD VECTORS", tab_id="tab-3"),
                dbc.Tab(label="CONFUSION MATRIX", tab_id="tab-4"),
                dbc.Tab(label="WORD CLOUD", tab_id="tab-5"),

            ],
            id="tabs",
            active_tab="tab-1",
        ),
        html.Div(id="content"),
    ]
)

# Dash App
# app = JupyterDash(__name__, external_stylesheets=external_stylesheets)

app.layout = dbc.Container(fluid=True, children=[
    dbc.Row([
        dbc.Col([html.H2(children="Predicting Apple stock price with Continual ML using Global News (by OpenBlender.io)",
                         style={'textAlign': 'centre',
                                'color': 'white'})], md=10),
        dbc.Col(md=2, children=[navbar])
    ], className='navbar navbar-expand-lg navbar-dark bg-primary'),

    dbc.Row([
        dbc.Col(xs=12, sm=12, md=12, lg=3, xl=3, children=[

            dbc.Col(html.Div(id="output-panel")),
            html.Br(),
            dbc.Col(dbc.Card(body=True, className="card bg-light mb-3", children=[
                html.Div("Feladat leírása", className="card-header"),
                html.Div(className="card-body", children=[
                    html.P(children=["""Input adat:""",
                                     html.Br(),
                                     """* Apple historikus ársor 3 évre visszamenőlegesen""",
                                     html.Br(),
                                     """* Apple-hez kapcsolódó gazdasági hírek""",
                                     html.Br(),
                                    html.Br(),
                                     """Az OpenBlinder-en található 69 gazdasági hírforrás közül kiválasztásra került két releváns forrás""",
                                     html.Br(), html.Br(),
                                     """Vesszük a CNBC és Wall Street Journal adott időszakhoz kapcsolódó híreit""",
                                    html.Br(),
                                    """-Feautures: a cikkekben előforduló 5000 leggyakoribb szóból képzett tokenek""",
                                    html.Br(),
                                    """-Labels: ha az árfolyam a következő nap feljebb megy: Up egyébként Down""",
                                     html.Br(), html.Br(), html.Br(),
                                     """ML model típusa: RandomForestRegressor""",
                                     html.Br(),
                                     """Kapott model pontossága: 67%""",


                                     ], className="card-text")
                ])
            ])
                    )
        ]),

        dbc.Col(lg=9, xl=9, children=[
html.Br(),tabs
            # dbc.Tabs(className="nav nav-pills", id='my-tab',
            #          children=[
            #              dbc.Tab(label='BUSINESS DATASETS (69)', tab_id='bd'),
            #              dbc.Tab(label='TWEETS', tab_id='tw'),
            #              dbc.Tab(label='WORLDCLOUD', tab_id='wc'),
            #             dbc.Tab(label='TEXT VECTORIZER', tab_id='tv'),
            #             dbc.Tab(label='RESULT', tab_id='rs'),
            #
            #          ],
            #          active_tab="bd"),
            #             html.Div(id="bd-content",children=[get_bw_content()]),
            #             html.Div(id="tw-content", children=[get_tw_content()]),
            #             html.Div(id="wc-content", children=[html.Img(src=app.get_asset_url('worldcloud.png'))]),
            #             html.Div(id="tv-content", children=["tv-content"]),
            #             html.Div(id="rs-content", children=["rs-content"]),



        ])
    ])
])

@app.callback(Output("content", "children"), [Input("tabs", "active_tab")])
def switch_tab(at):
    if at == "tab-1":
        return dbc.Card(
                dbc.CardBody(
                    [
                        html.Br(),
                        bw_content
                    ]
                ),
                className="mt-3",
                )
    elif at == "tab-2":
        return dbc.Card(
                dbc.CardBody(
                    [
                        html.Br(),
                        tw_content
                    ]
                ),
                className="mt-3",
            )

    elif at == "tab-3":
        return  dbc.Card(
                dbc.CardBody(
                    [
                        html.Br(),
                        wv_content
                    ]
                ),
                className="mt-3",
            )

    elif at == "tab-4":
        return  dbc.Card(
                dbc.CardBody(
                    [
                        html.Br(),
                        html.Img(src=app.get_asset_url('conf_matrix.png'))
                    ]
                )
            )
    elif at == "tab-5":
        return  dbc.Card(
                dbc.CardBody(
                    [
                        html.Br(),
                        html.Img(src=app.get_asset_url('worldcloud.png'))
                    ]
                )
            )


    return html.P("This shouldn't ever be displayed...")

# @app.callback(
#     [Output('bd-content', 'style'),Output('tw-content', 'style'),Output('wc-content', 'style'),Output('tv-content', 'style'),Output('rs-content', 'style')],
#     [Input('my-tab','active_tab')])
# def change_tab(active_tab):
#     if (active_tab=='bd'):
#         return {'display': 'inline'}, {'display': 'none'},{'display': 'none'}, {'display': 'none'},{'display': 'none'}
#     if (active_tab=='tw'):
#         return {'display': 'none'}, {'display': 'inline'},{'display': 'none'}, {'display': 'none'},{'display': 'none'}
#     if (active_tab=='wc'):
#         return {'display': 'none'}, {'display': 'none'},{'display': 'inline'}, {'display': 'none'},{'display': 'none'}




# this is needed for the procfile to deploy to heroku
server = app.server

if __name__ == '__main__':
    app.run_server(debug=True, port=3337)
