import pandas as pd 

time_period = 14

df = pd.read_csv("A.csv")

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
df.to_csv('A.csv')

