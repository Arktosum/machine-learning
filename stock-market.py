API_KEY = '9096QUVHAKSGXPDD'

import requests

SYMBOL = 'IBM'
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={SYMBOL}&interval=5min&apikey={API_KEY}'
r = requests.get(url)
data = r.json()

for point in data['Time Series (5min)']:
    this = data['Time Series (5min)'][point]
    open = this['1. open']
    high = this['2. high']
    low = this['3. low']
    close = this['4. close']
    volume = this['5. volume']
    print(open, high, low, close, volume)