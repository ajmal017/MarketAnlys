import pandas as pd
import numpy as np

def doMACD(df):
    df['5-MA'] = df['close'].rolling(5).mean()
    df['12-EMA'] = np.where(df['index'] < 12, df['close'].rolling(12).mean(), pd.Series.ewm(df['close'], span=12, adjust=False).mean())
    df['26-EMA'] = np.where(df['index'] < 26, df['close'].rolling(26).mean(), pd.Series.ewm(df['close'], span=26, adjust=False).mean())
    df['MACD'] = df['26-EMA'] - df['12-EMA']

