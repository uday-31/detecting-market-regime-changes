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
    metrics = {'pnl':round(df.sum(),2),
              'sharpe':sharpe(df).round(2),
              'sortino':sortino(df).round(2),
              'volatility':round(df.std(),4),
              'mdd':mdd(df)}
    return pd.DataFrame(metrics, index=[''])