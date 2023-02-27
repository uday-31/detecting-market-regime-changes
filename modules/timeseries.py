import pandas as pd
import numpy as np

def get_vols(df_ts: pd.DataFrame):
    """
    Returns daily volatility from price data
    
    Args:
        df_ts: output of directional_change.get_data()
    
    Returns:
        DataFrame containing daily volatility for each ticker in df_ts
    """
    
    unique_dates = sorted(set(df_ts.index.date))
    tickers = list(df_ts.columns)
    
    vols = dict()
    
    for ticker in tickers:
        px = df_ts[ticker]
        logret_sq = (np.log(px/px.shift(1))) ** 2
        vols[ticker] = list((logret_sq + logret_sq.shift(1)).dropna())[::2]
    
    return pd.DataFrame.from_dict(vols).set_index(df_ts.index[::2][:-1])

