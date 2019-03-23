import pandas as pd
import time
import numpy as np

# open csv
df = pd.read_csv('../res/compiledStocks.csv')
tempdf = pd.DataFrame()

# offset index to 0, was originally 1. I think this is best. 
# We should also offset day_index to start at 0 and got to 255, as 0 indicates "0 days from origin"
df['index'] = df['index'] - 1

# np.where takes 3 commands (condition, case true, case false) 
# It is very similar to a ternary conditional operator in java, but handles series of booleans.
# I would start here, and use df.ticker == df.ticker.shift() to find where tickers change. 

#### shift() gives a new dataframe with values shifted backward 1 by default.
#### Look it up if you need, but essentially you pass parameters such as shift(1) -> shift forward 1, 
#### shift(-1) -> shift backward 1 (also the default case)
df['day_index'] = np.where(df.ticker == df.ticker.shift(), 0, 'SHIFT')

print (df.head())
df.to_csv('../res/compiledStocks2.csv')

