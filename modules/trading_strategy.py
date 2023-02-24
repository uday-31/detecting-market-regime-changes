import pandas as pd
import numpy as np

def sharpe(df):
    return (df.mean()*252/(df.std()*np.sqrt(252)))

def sortino(df):
    sd = np.sqrt(np.mean(df[df<0].values*df[df<0].values))
    return (df.mean()*252/(sd*np.sqrt(252)))

def mdd(df):
    return ((df.cumsum()-df.cumsum().cummax()).min())

def get_metrics(df):
    metrics = {'pnl':round((1+df).product() - 1,2),
              'sharpe':sharpe(df).round(2),
              'sortino':sortino(df).round(2),
              'volatility':round(df.std(),4),
              'mdd':mdd(df)}
    return pd.DataFrame(metrics, index=[''])

def RSI(price, lookback = 14):
    # based on: https://blog.quantinsti.com/build-technical-indicators-in-python/
    ret = price.diff()

    high = ret.clip(lower=0)
    low = -ret.clip(upper=0)
    
    avg_up = high.ewm(alpha = 1/lookback, min_periods = lookback).mean()
    avg_down = low.ewm(alpha = 1/lookback, min_periods = lookback).mean()

    return (100 - (100/(1 + (avg_up / avg_down))).dropna()).shift(1).dropna()