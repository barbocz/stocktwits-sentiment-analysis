from alpha_vantage.timeseries import TimeSeries
from joblib import dump, load

ts = TimeSeries(key='T5DNTM8CM4WR9DSB', output_format='pandas')
data, meta_data = ts.get_daily_adjusted(symbol='AAPL' ,outputsize='full')

data=data.rename(columns={"1. open": "Open", "2. high": "High","3. low": "Low","4. close": "Close","5. adjusted close": "Adjusted Close","6. volume": "Volume",})
data=data.drop(['7. dividend amount','8. split coefficient'], axis = 1)

data['PM_change'] = data.Open - data['Adjusted Close'].shift(-1)
data['Day_change'] = data['Adjusted Close'] - data.Open
data['%_Change'] = data['PM_change'] / data.Open.shift(-1)
data['Date']=data.index

print(data.info())
print(data[['Open','Close','Adjusted Close','PM_change','Day_change','%_Change']])
data.to_csv('AAPL_prices.csv')
dump(data, 'data\stock_price.pkl')