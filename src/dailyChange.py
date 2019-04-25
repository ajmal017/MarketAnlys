import pandas as pd

# Get tickers
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
###########################################
for ticker in tickers:
    df = pd.read_csv('../res/originalCSVs/' + ticker + '.csv')
    df = df.drop('target', axis=1)
    df['change'] = 0
    for i in df.index:
        if i != 0:
            currentClose = df.loc[i, "close"]
            previousClose = df.loc[i - 1 , 'close']
            df.iloc[i, df.columns.get_loc('change')] = currentClose - previousClose
    df.to_csv('../res/fullCSVs/' + ticker + ".csv")