import requests
import pandas as pd
from pprint import pprint
api_key = "5c9443483231d0.26933748"
base_url = "https://eodhistoricaldata.com/api/fundamentals/AAPL.US?api_token="

webdata = requests.get("https://eodhistoricaldata.com/api/fundamentals/A.US?api_token=5c9443483231d0.26933748")
data = webdata.json()

df = pd.DataFrame()

dateList = ["2017-04-30", "2017-07-31", "2017-10-31", "2018-01-31", "2018-04-30", "2018-07-31", "2018-10-31", "2019-01-31"]

dateList = [item for item in data["Earnings"]["History"] if item in dateList]
dateList.reverse()
df['Date'] = dateList

epsActualList = [data["Earnings"]["History"][item]["epsActual"] for item in data["Earnings"]["History"] if item in dateList]
epsActualList.reverse()
df['epsActual'] = epsActualList



growthList = [data["Earnings"]["Trend"][item]["growth"] for item in data['Earnings']['Trend'] if item in dateList]
growthList.reverse()
df['growth'] = growthList

price_df = pd.read_csv("../A.csv")

dateDict = {
    "4/30/17": ["2017-04-30"], 
    "7/31/17": ["2017-07-31"], 
    "10/31/17": ["2017-10-31"], 
    "1/31/18": ["2018-01-31"], 
    "4/30/18": ["2018-04-30"], 
    "7/31/18": ["2018-07-31"], 
    "10/31/18": ["2018-10-31"], 
    "1/31/19": ["2019-01-31"]}
for item in dateDict:
    dateDict[item].append((df.loc[df['Date'] == dateDict[item][0], 'growth']).values[0])


price_list = price_df.loc[price_df['date'].isin(dateDict.keys())]["close"].tolist()

df['close'] = ["N/A", "N/A", "N/A", "N/A"] + price_list

price_df["P/E"] = 0

for date in dateDict.keys():
    price_df.loc[price_df['date']==date, 'P/E'] = df.loc[df['Date'] == dateDict[date][0], 'growth'].values[0]

price_df.to_csv("newA.csv")



# def get_new_url(ticker, country):
# 	return base_url.replace("AAPLE.US", ticker, country)