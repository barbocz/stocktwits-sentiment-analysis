import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_table as dt
import dash_table.FormatTemplate as FormatTemplate

merge_df_tsla = pd.read_csv("TSLA_price_merge_df.csv")
merge_df = pd.read_csv("AAPL_price_merge_df.csv")
aapl_strat = pd.read_csv("aapl_strat.csv")
tsla_strat = pd.read_csv("tsla_strat.csv")

external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/lux/bootstrap.min.css',
                        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css']

navbar = dbc.Nav(className="navbar-nav mr-auto", children=[
    dbc.DropdownMenu(label="Links", nav=True, children=[
        dbc.DropdownMenuItem([html.I(className="fa fa-apple"), "   $AAPL Stocktwits"],
                             href="https://stocktwits.com/symbol/AAPL", target="_blank"),
        dbc.DropdownMenuItem([html.I(className="fa fa-car"), "  $TSLA Stocktwits"],
                             href="https://stocktwits.com/symbol/TSLA", target="_blank"),
        dbc.DropdownMenuItem([html.I(className="fa fa-linkedin"), "  Linkedin"],
                             href="https://www.linkedin.com/in/jay-lin-jiele-02a5a114a/", target="_blank"),
    ])
])

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
                html.Div("About This Dashboard", className="card-header"),
                html.Div(className="card-body", children=[
                    html.P(children=["""This experiment aims to mine $TSLA and $AAPL sentiments 
                    on StockTwits and derive trading signals based on their pre-market sentiments moving 
                    averages. StockTwits is a social media platform for retail traders to share their 
                    speculations and sentiments regarding any stock.""",
                                     html.Br(),
                                     html.Br(),
                                     """This dashboard was built on Plotly Dash to make it interactive. You can
                                     tweak the EMAs from the slider bar above to view the different backtest
                                      performances.""",
                                     html.Br(), html.Br(),
                                     """Thanks for viewing, you may find out more about the project through my 
                                     article on """,
                                     html.A("Medium.",
                                            href='https://jayljl.medium.com/mining-stocktwits-retail-sentiments-'
                                                 'for-momentum-trading-4594a91833b4',
                                            target="_blank",
                                            style={'color': 'blue'})

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
            dcc.Graph(id='buy-sell-chart'),
            dcc.Graph(id='bull-bear-chart'),
            dbc.Col(html.Div(id="data-table"))
        ])
    ])
])

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
        fig.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    #             dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )

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
                range=['2020-01-01', '2020-12-31']),
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
        fig.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    #             dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )

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
                range=['2020-01-01', '2020-12-31']),
            yaxis=dict(
                type='linear',
                showline=False,
                showgrid=False,
                ticksuffix=' USD',
                fixedrange=True,
                range=[6000, 18000]
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
        fig.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    #             dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )

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
                range=['2020-01-01', '2020-12-31']),
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
        fig.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    #             dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )

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
                range=['2020-01-01', '2020-12-31']),
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



server = app.server

if __name__ == '__main__':
    app.run_server(debug=True, port=3336)
