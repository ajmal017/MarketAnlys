import pandas as pd

df = pd.read_csv('A.csv')

for i in df.index:
	if i != 0:
		currentIndex = df.loc[i, "index"]
		currentClose = df.loc[i, "close"]
		previousClose = df.loc[i - 1 , 'close']
		df.iloc[currentIndex, df.columns.get_loc('target')] = currentClose - previousClose

df.to_csv('A.csv')