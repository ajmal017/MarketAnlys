import pandas as pd
import numpy as np
import requests

api_key = "5c9443483231d0.26933748"
base_url = "https://eodhistoricaldata.com/api/fundamentals/AAPL.US?api_token="

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



def doMACD(df):
    df['5-MA'] = df['close'].rolling(5).mean()
    df['12-EMA'] = np.where(df['index'] < 12, df['close'].rolling(12).mean(), pd.Series.ewm(df['close'], span=12, adjust=False).mean())
    df['26-EMA'] = np.where(df['index'] < 26, df['close'].rolling(26).mean(), pd.Series.ewm(df['close'], span=26, adjust=False).mean())
    df['MACD'] = df['26-EMA'] - df['12-EMA']

def doRSI(df):
    time_period = 14

    # df = pd.read_csv("A.csv")

    # filling up "gain" and "loss" columns with absolute value of daily change
    for i in df.index:
        if i != 0:
            if df.loc[i, "change"] > 0:
                df.loc[i, "gain"] = df.loc[i, "change"]
                df.loc[i, "loss"] = 0
            elif df.loc[i, "change"] < 0:
                df.loc[i, "gain"] = 0
                df.loc[i, "loss"] = abs(df.loc[i, "change"])
            else:
                df.loc[i, "gain"] = 0
                df.loc[i, "loss"] = 0

    # calulating average gain and average loss for the first RSI instance (14th)
    total_gain = 0
    total_loss = 0
    for i in range(1, time_period + 1):
        total_gain += df.loc[i, "gain"]
        total_loss += df.loc[i, "loss"]

    df.loc[time_period, "avg_gain"] = total_gain / time_period
    df.loc[time_period, "avg_loss"] = total_loss / time_period

    # calculating the average gain and average loss for the rest of the data
    for i in range(time_period + 1, len(df.index)):
        df.loc[i, "avg_gain"] = ((df.loc[i - 1, "avg_gain"] * (time_period - 1)) + df.loc[i, "gain"]) / time_period
        df.loc[i, "avg_loss"] = ((df.loc[i - 1, "avg_loss"] * (time_period - 1)) + df.loc[i, "loss"]) / time_period

    # calculating the RSI
    # rsi = 100 - (100 / (1 + (avg_gain / avg_loss)))
    for i in range(time_period, len(df.index)):
        rs = df.loc[i, "avg_gain"] / df.loc[i, "avg_loss"]
        df.loc[i, "rsi"] = 100 - (100 / (1 + rs))

    # exporting out to csv
    # df.to_csv('A.csv')

def doStochasticOscillator(df):
    time_period = 14

    # df = pd.read_csv("A.csv")
    print(df.head())
    for i in range(time_period, len(df.index)):
        df.loc[i, "numerator"] = df.loc[i, "close"] - df.iloc[i - 14:i, df.columns.get_loc('low')].min()
        df.loc[i, "denominator"] = df.iloc[i - 14 : i, df.columns.get_loc('high')].max() - df.iloc[i - 14 : i, df.columns.get_loc('low')].min()

    for i in range(time_period, len(df.index)):
        df.loc[i, "K"] = (df.loc[i, "numerator"] / df.loc[i, "denominator"]) * 100

    for i in range(time_period + 2, len(df.index)):
        df.loc[i, "D"] = df.iloc[i - 2 : i, df.columns.get_loc("K")].sum() / 3

    # exporting out to csv
    # df.to_csv('A.csv')


def doFundamentalDate(df, ticker):
    ticker_url = base_url.replace('AAPL', ticker)
    webdata = requests.get("https://eodhistoricaldata.com/api/fundamentals/A.US?api_token=5c9443483231d0.26933748")
    data = webdata.json()

    temp_df = pd.DataFrame()

    dateList = ["2018-04-30", "2018-07-31", "2018-10-31", "2019-01-31"]

    dateList = [item for item in data["Earnings"]["History"] if item in dateList]
    dateList.reverse()
    temp_df['Date'] = dateList

    epsActualList = [data["Earnings"]["History"][item]["epsActual"] for item in data["Earnings"]["History"] if item in dateList]
    epsActualList.reverse()
    temp_df['epsActual'] = epsActualList



    growthList = [data["Earnings"]["Trend"][item]["growth"] for item in data['Earnings']['Trend'] if item in dateList]
    growthList.reverse()
    temp_df['growth'] = growthList

    dateDict = {
        "4/30/2018": ["2018-04-30"], 
        "7/31/2018": ["2018-07-31"], 
        "10/31/2018": ["2018-10-31"], 
        "1/31/2019": ["2019-01-31"]}
    for item in dateDict:
        dateDict[item].append((temp_df.loc[temp_df['Date'] == dateDict[item][0], 'growth']).values[0])
        dateDict[item].append((temp_df.loc[temp_df['Date'] == dateDict[item][0], 'epsActual']).values[0])



    df["P/E"] = 0
    df["EPS"] = 0
    df['PEG'] = 0

    for date in dateDict.keys():
        if df.loc[df['date'] == date, 'close'].size != 0:
            df.loc[df['date'] == date, 'P/E'] = df.loc[df['date']==date, 'close'].values[0] / float(temp_df.loc[temp_df['Date'] == dateDict[date][0], 'epsActual'].values[0])
            df.loc[df['date'] == date, 'EPS'] = temp_df.loc[temp_df['Date'] == dateDict[date][0], 'epsActual'].values[0]
            df.loc[df['date'] == date, 'PEG'] = (df.loc[df['date'] == date, 'P/E'].values[0]) / float(temp_df.loc[temp_df['Date'] == dateDict[item][0], 'growth'].values[0])


for ticker in tickers:

    df = pd.read_csv("../res/fullCSVs/" + ticker + ".csv")

    # doMACD(df)
    # doRSI(df)
    #doStochasticOscillator(df)
    doFundamentalDate(df, ticker)
    df.to_csv("../res/finalCSVs/" + ticker + ".csv")


