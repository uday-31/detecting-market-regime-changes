################################################################################
# Compute directional change as per definitions in Appendix A from
# Detecting Regime Change in Computational Finance by Jun Chen, Edward P K Tsang
# Rohan Prasad
##############################################################################
import pandas as pd
import yfinance as yf
import numpy as np

def get_data(tickers, start_date, delta):
    data = yf.download(tickers, start=start_date, end = '2022-12-31') # time series data
    df_ts_close = data['Adj Close'].dropna()
    df_ts_open = data['Open'].dropna()
    df_ts_open.index = df_ts_open.index + pd.Timedelta(f'{delta}h') # adjust time
    df_ts = pd.concat([df_ts_close,df_ts_open]).sort_index()

    return df_ts

def get_pct_change(start, end):
    return (end - start) / start


def get_dc_data_v2(prices: pd.Series, theta: float) -> list[tuple]:
    """

    :param prices: prices
    :param theta: threshold
    :return: Returns a list of tuples. Each tuple is of the form
             (Directional Change Confirmation timestamp, Directional Change Confirmation price,
             Downturn/Upturn time, Downturn/Upturn price)
    """

    last_high = last_low = prices[0]
    last_low_time = last_high_time = prices.index[0]
    is_upward_run = is_downward_run = is_downward_overshoot = is_upward_overshoot = False
    ret_val = []

    for timestamp, current_price in prices[1:].items():
        if get_pct_change(last_high, current_price) <= -theta:
            is_downward_run = True
            is_upward_run = False
            is_upward_overshoot = False
            if is_downward_overshoot:
                last_high = current_price
                last_high_time = timestamp
            else:
                # reached a DC confirmation point
                ret_val.append((timestamp, current_price, last_high_time, last_high))
                is_downward_overshoot = True
        elif get_pct_change(last_low, current_price) >= theta:
            is_upward_run = True
            is_downward_run = False
            is_downward_overshoot = False
            if is_upward_overshoot:
                last_low = current_price
                last_low_time = timestamp
            else:
                # reached a DC confirmation point
                ret_val.append((timestamp, current_price, last_low_time, last_low))
                is_upward_overshoot = True
        if last_low > current_price:
            last_low = current_price
            last_low_time = timestamp
        if last_high < current_price:
            last_high = current_price
            last_high_time = timestamp
    return ret_val

def get_DC_data(data: pd.Series, theta: float) -> tuple[pd.Series]:
    """Returns the Directional Change (DC) data for a given price series.

    Args:
        data (pd.Series): price
        theta (float): threshold

    Returns:
        tuple[pd.Series]: Directional Change Confirmation and Extreme Points (DCC,EXT)
    """

    rets = data.pct_change().dropna().to_numpy() # pct change returns
    DCC = [] # idx for directional change confirmations
    EXT = [] # idx for extreme points

    prev_sign = np.sign(rets[0]).astype(int) # store sign(return) from the previous time step
    accumulated = rets[0] # accumulated % return
    idx_change = 0 # index the direction changes, candidate for EXT
    sign_already_flagged = 0

    for idx,ret in zip(range(1,len(rets)),rets[1:]):
        
        ret_sign = np.sign(ret).astype(int)

        if (ret_sign!=prev_sign):
            # sign is different from previous time step, trend ends
            idx_change = idx-1 # previous price point is a candidate for EXT
            accumulated = ret # reset accumulated sum
            
            
        # same sign
        elif ret_sign != sign_already_flagged:
            # once we flag a threshold, we don't flag it again for the same trend
            # a peak has to be followed by a trough and vice versa
            accumulated += ret
            if np.abs(accumulated) > theta:
                # we cross the threshold
                DCC.append(idx)
                EXT.append(idx_change)
                sign_already_flagged = ret_sign
        
        prev_sign = ret_sign # set the last seen sign to the current sign

    DCC = data.iloc[1:].iloc[DCC]
    EXT = data.iloc[1:].iloc[EXT]

    return (DCC,EXT)

def get_TMV(ext: pd.Series, theta: float) -> pd.Series:
    """Gets the total price movement (TMV), which is the absolute percentage of the price change in a trend, normalized by the threshold.

    Args:
        ext (pd.Series): extreme points
        theta (float): threshold

    Returns:
        pd.Series: total price movement at respective timestamps
    """
    return ext.pct_change().dropna()/theta

def get_T(ext: pd.Series) -> pd.Series:
    """Gets the time for completion of a TMV trend, in days.

    Args:
        ext (pd.Series): extreme points

    Returns:
        pd.Series: time for completion of trends at respective timestamps
    """
    # extract number of days and hours between extreme points
    t_ext = pd.Series(ext.index).diff().dropna().apply(lambda x: x.days + (x.seconds//3600)/24)
    t_ext.index = ext.index[1:]
    return t_ext

def get_R(tmv: pd.Series, T: pd.Series, theta: float) -> pd.Series:
    """Gets the absolute return (R), which is the time-adjusted return of DC.

    Args:
        tmv (pd.Series): total price movement
        T (pd.Series): time for completion of a trend
        theta (float): threshold

    Returns:
        pd.Series: time-adjusted return of DC
    """
    return tmv*theta/T


if __name__ == '__main__':
    print('Please import this file as a module.')  # %%