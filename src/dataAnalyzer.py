import pandas as pd
import time

		# README: Added time elapsed to this file, and changed the df loc to iloc, which makes it a bit faster.
		# My progress with fixing the day_index issue is in another file dateIndexCalc.py, you can take a look there. 
		# It looks at compiledStocks and spits out the result at CompiledStocks.csv

# start

df = pd.read_csv('../res/compiledStocks.csv')
start = time.time()
index = 0
for row in df.iterrows():
	currentIndex = row[1]['index']
	currentDayIndex = row[1]['day_index']
	currentPrice = row[1]['close']
	if currentDayIndex < 225 and index < 128482:
		priceIn30Days = df.at[currentIndex + 20, 'close']
		# df.loc[(df['index'] == currentIndex), 'target'] = ((priceIn30Days - currentPrice) / currentPrice) * 100
		df.iloc[currentIndex - 1, df.columns.get_loc('target')] = ((priceIn30Days - currentPrice) / currentPrice) * 100
	index += 1
	if index%10000 == 0:
		print("Current Index: ", index)
print('Last Index', index)
df = df.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1'], axis = 1)
end = time.time()
# end

print(df.head())
print('Time Elapsed: ', end - start)
df.to_csv('../res/withTarget.csv')