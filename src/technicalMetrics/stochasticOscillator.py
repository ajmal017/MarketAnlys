import pandas as pd 

time_period = 14

df = pd.read_csv("A.csv")

for i in range(time_period, len(df.index)):
	df.loc[i, "numerator"] = df.loc[i, "close"] - df.iloc[i - 14:i, df.columns.get_loc('low')].min()
	df.loc[i, "denominator"] = df.iloc[i - 14 : i, df.columns.get_loc('high')].max() - df.iloc[i - 14 : i, df.columns.get_loc('low')].min()

for i in range(time_period, len(df.index)):
	df.loc[i, "K"] = (df.loc[i, "numerator"] / df.loc[i, "denominator"]) * 100

for i in range(time_period + 2, len(df.index)):
	df.loc[i, "D"] = df.iloc[i - 2 : i, df.columns.get_loc("K")].sum() / 3

# exporting out to csv
df.to_csv('A.csv')
