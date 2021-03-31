from alpha_vantage.timeseries import TimeSeries

import pandas as pd
from datetime import datetime
from joblib import dump, load



ts = TimeSeries(key='T5DNTM8CM4WR9DSB', output_format='pandas')
data, meta_data = ts.get_daily_adjusted(symbol='AAPL' ,outputsize='full')

data=data.rename(columns={"1. open": "Open", "2. high": "High","3. low": "Low","4. close": "Close","5. adjusted close": "Adjusted Close","6. volume": "Volume",})
data=data.drop(['7. dividend amount','8. split coefficient'], axis = 1)

data['PM_change'] = data.Open - data.Open.shift(-1)
data['Day_change'] = data.Close - data.Open
data['%_Change'] = data['PM_change'] / data.Open.shift(-1)
data['Date']=data.index

print(data.info())
print(data[['Open','Close','Adjusted Close','PM_change','Day_change','%_Change']])
# data.to_csv('AAPL_pricesss.csv')
dump(data, 'price_df.pkl')
quit()

# sentiment_df = load('sentiment_df.pkl')
# print(sentiment_df.columns)
# print(sentiment_df[(sentiment_df['Date']=='2021-02-01') & (sentiment_df['Combined sentiment']=='Bullish')])

# price_df = load('price_df.pkl')
# bull_bear_df = load('bb_df.pkl')

# pd.set_option("max_rows", None)
# pd.set_option('display.max_columns', None)

print(69 * 143.43 + 103)
quit()

trade_hrs_df = load('filtered_df.pkl')
trade_hrs_df['Pre-Market Date'] = pd.to_datetime(trade_hrs_df["Pre-Market Date"], format='%Y-%m-%d')

bull_bear_df = trade_hrs_df.groupby(
     ["Pre-Market Date", 'Combined sentiment']).agg({"message": "count"}).unstack().reset_index()
bull_bear_df.sort_values("Pre-Market Date", inplace=True)
bull_bear_df.set_index('Pre-Market Date', inplace=True)
bull_bear_df['Bull/Bear Ratio'] = (bull_bear_df[('message', 'Bullish')]) / (bull_bear_df[('message', 'Bearish')])
bull_bear_df.columns = ["Bearish", "Bullish", "Bull/Bear Ratio"]

# print(bull_bear_df)
#
# print(bull_bear_df[(bull_bear_df.index=='2021-02-01')])

price_df = load('price_df.pkl')
price_df['Date']=price_df.index

print(price_df[(price_df.index=='2021-02-01')])


# ema_list = [5, 6, 7, 8, 9, 10, 15, 20]
combined_df = bull_bear_df.merge(price_df, how="left", left_index=True, right_index=True)

print(bull_bear_df.info())
print(price_df.info())

combined_df.columns = ["Bearish", "Bullish", "Bull/Bear Ratio",
                       "Open", "High", "Low", "Close", "Adjusted Close","Volume",
                       "PM_change", "Day_change", "%_Change","Date"]

print(combined_df)

#
# for ema in ema_list:
#     exp_ema = combined_df['Bull/Bear Ratio'].ewm(span=ema, min_periods=ema, adjust=False).mean()
#     combined_df[f"Bull/Bear Ratio EMA {ema}"] = exp_ema
#
# combined_df["% of Bullish"] = round((combined_df['Bullish'] * 100) /
#                                     (combined_df['Bullish'] + combined_df['Bearish']), 2)
# combined_df["% of Bearish"] = round((combined_df['Bearish'] * 100) /
#                                     (combined_df['Bullish'] + combined_df['Bearish']), 2)
# combined_df['Middle line'] = 50
#
# print(data[['Close','PM_change','%_Change']])

