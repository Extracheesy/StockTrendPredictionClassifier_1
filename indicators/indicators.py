
import pandas  as pd
import numpy as np

import talib

def add_RSI(df, n=14):

    df['rsi'] = talib.RSI(df.close.values, timeperiod=n)

    return df

def add_SMA(df):
    df['SMA_5'] = df['close'].rolling(window = 5).mean()
    df['SMA_10'] = df['close'].rolling(window = 10).mean()
    #df['SMA_10'] = df['close'].rolling(10).mean().shift()
    df['SMA_15'] = df['close'].rolling(window = 15).mean()
    df['SMA_20'] = df['close'].rolling(window = 20).mean()
    df['SMA_30'] = df['close'].rolling(window = 30).mean()
    df['SMA_50'] = df['close'].rolling(window = 50).mean()
    df['SMA_100'] = df['close'].rolling(window = 100).mean()
    df['SMA_200'] = df['close'].rolling(window = 200).mean()

    return df

def add_EMA(df):
    df['EMA_10'] = df.close.ewm(span=10).mean().fillna(0)
    df['EMA_20'] = df.close.ewm(span=20).mean().fillna(0)
    df['EMA_50'] = df.close.ewm(span=50).mean().fillna(0)
    df['EMA_100'] = df.close.ewm(span=100).mean().fillna(0)
    df['EMA_200'] = df.close.ewm(span=200).mean().fillna(0)

    return df

def add_MACD(df):
    EMA_12 = pd.Series(df['close'].ewm(span=12, min_periods=12).mean())
    EMA_26 = pd.Series(df['close'].ewm(span=26, min_periods=26).mean())

    df['MACD'] = pd.Series(EMA_12 - EMA_26)
    df['MACD_signal'] = pd.Series(df.MACD.ewm(span=9, min_periods=9).mean())

    return df

def add_ATR(df):
    df['ATR'] = talib.ATR(df['high'].values,
                          df['low'].values,
                          df['close'].values,
                          timeperiod=14)
    return df

def add_ADX(df):
    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

    return df

def add_CCI(df, ndays):

    TP = (df['high'] + df['low'] + df['close']) / 3
    # CCI = pd.Series((TP - pd.rolling_mean(TP, ndays)) / (0.015*pd.rolling_std(TP, ndays)), name = 'CCI')

    rolling_mean = pd.Series(TP).rolling(window=ndays).mean()
    rolling_std = pd.Series(TP).rolling(window=ndays).std()
    CCI = pd.Series((TP - rolling_mean) / (0.015*rolling_std), name = 'CCI')

    df = df.join(CCI)

    return df

def add_ROC(df):
    df['ROC'] = ((df['close'] - df['close'].shift(12)) / (df['close'].shift(12)))*100

    return df

def add_WILLIAMS(df, n=14):
    df['Williams %R'] = talib.WILLR(df.high.values,
                                    df.low.values,
                                    df.close.values, n)
    return df

def add_STOCHASTIC(df, n=14):

    df['14-high'] = df['high'].rolling(n).max()
    df['14-low'] = df['low'].rolling(n).min()
    df['SO%K'] = (df['close'] - df['14-low']) * 100 / (df['14-high'] - df['14-low'])
    df['SO%D'] = df['SO%K'].rolling(3).mean()

    df = df.drop(columns=['14-high', '14-low'])

    return df

def add_indicator(df):

    df = add_RSI(df, 14)
    df = add_SMA(df)
    df = add_EMA(df)
    df = add_MACD(df)
    df = add_ATR(df)
    df = add_ADX(df)
    df = add_CCI(df, 20)
    df = add_ROC(df)
    df = add_WILLIAMS(df, 14)
    df = add_STOCHASTIC(df, 14)

    return df