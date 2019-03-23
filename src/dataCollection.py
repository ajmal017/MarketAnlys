import pyEX
import pandas as pd

f = open("../res/stocks.csv", "r")
heading = f.readline()
data = f.readlines()
f.close()

tickers = []

for i in range(len(data)):
	data[i] = data[i].split(",")

for each in data[:505]:
	if each[0] not in tickers:
		tickers.append(each[0])

df = pd.DataFrame()
print("Starting conversion")
for symbol in tickers:
	tempDF = pyEX.chartDF(symbol, timeframe='1y')
	tempDF['ticker'] = symbol
	df = df.append(tempDF)
cols = list(df) 
cols.insert(0, cols.pop(cols.index('ticker')))
df = df.ix[:, cols]
df.to_csv('../res/compiledStocks.csv')