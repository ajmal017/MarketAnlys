import pandas as pd
import csv

df = pd.read_csv('../res/compiledStocks.csv')
df = df.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1'], axis = 1)

l = [x for _,x in df.groupby('ticker')]

for each in l:
	ticker = each.iloc[0,3]
	f = open("../res/" + ticker + ".csv", "w")
	each.to_csv("../res/" + ticker + '.csv', sep=',')
	f.close()