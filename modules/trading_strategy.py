################################################################################
# Different Trading Strategies
# Will try to implement what the paper mentions and minimise on the drawdown
##############################################################################
import pandas as pd
import numpy as np
import modules.generate_data as gd

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

#####################################################################################
#### Trading strategy as explained in the book
#####################################################################################

def regime_to_sign(val):
    if( val == 1):
        return 1
    elif( val == 0 ):
        return -1
    else:
        '''Never happens'''
        return 0
    
def strategy_regime_dependent(data, init_cap = 1, strat = "JC1", threshold = 0.5):
    """
    JC1 strategy as explained in the book Chapter 6 -> Mean reversion during normal time, momentum during abnormal time.
    It returns a dataframe after implementing JC1 strategy which is based on regime change
    We get new columns at every time which are: position, asset capital, bank capital, total capital and returns
    

    TODO: We assume that open to close and close to open times are same, useful in sharpe calculation

    params-> data: Output of gd.generate_dataset_with_columns
             init_cap: Inital capital in the strategy
             strat = Name of strat for adding columns
             threshold: When to take buy/sell decisions
    
    returns-> pd.DataFrame which has columns appended to data
                'daily_ret_strat', 'position_strat', 'asset_cap_strat', 'bank_cap_strat', 'total_cap_strat'
    """

    '''All values are after trading'''
    position = 'position_'+strat # Position in asset at this time
    daily_ret = 'daily_ret_'+strat # Return on total_cap
    asset_cap = 'asset_cap_'+strat # Price of assets I have
    bank_cap = 'bank_cap_'+strat # Capital which I don't have invested
    total_cap = 'total_cap_'+strat # Sum of above two

    d = data.copy()

    d[daily_ret] = 0
    d[position] = 0 
    d[asset_cap] = 0
    d[bank_cap] = 0
    d[bank_cap][0] = init_cap
    d[total_cap] = d[bank_cap]

    for i in range( 1,len(d) ):
        if( d['regime'][i] == -1 ):
            '''Don't do anything, regime detection has not started'''
            d[position][i] = d[position][i-1]
            d[asset_cap][i] = d[asset_cap][i-1]
            d[bank_cap][i] = d[bank_cap][i-1]
        
        else:
            if( (d[position][i-1] == 0) and (abs(d['TMV'][i]) >= threshold ) ):
                '''I go against the market with all my money'''
                d[position][i] = regime_to_sign(d['regime'][i]) * np.sign(d['TMV'][i]) * d[total_cap][i-1] / d['price'][i]
                d[asset_cap][i] = d[position][i] * d['price'][i]
                d[bank_cap][i] = d[total_cap][i-1] - d[asset_cap][i]
            elif( (d[position][i-1] == 0) and (abs(d['TMV'][i]) < threshold ) ):
                '''No position and threshold not crossed'''
                d[position][i] = d[position][i-1]
                d[asset_cap][i] = d[position][i] * d['price'][i]
                d[bank_cap][i] = d[bank_cap][i-1]
            elif( ( abs(d[position][i-1]) > 0) ):
                if( (d['regime'][i-1] == d['regime'][i]) and (d['type'][i] not in ['DCC','EXT_DCC'] ) ):
                    '''No Action to be taken'''
                    d[position][i] = d[position][i-1]
                    d[asset_cap][i] = d[position][i] * d['price'][i]
                    d[bank_cap][i] = d[bank_cap][i-1]
                else:
                    '''I take the opposite position here'''
    #                 d['debug'][i] = 'Hello'
                    d[position][i] = 0
                    d[asset_cap][i] = 0
                    d[bank_cap][i] = abs(d[position][i-1]) * d['price'][i]
        d[total_cap][i] = d[bank_cap][i] + d[asset_cap][i]
        d[daily_ret][i] = (d[total_cap][i] - d[total_cap][i-1])/d[total_cap][i-1]
        
    return d
        
    #     if( d['regime'][i] == 0 ):
    #         '''Normal Regime - Mean Reverting'''
    #         if( (d[position][i-1] == 0) and (abs(d['TMV'][i]) >= threshold ) ):
    #             '''I go against the market with all my money'''
    #             d[position][i] = -1 * np.sign(d['TMV'][i]) * d[total_cap][i-1] / d['price'][i]
    #             d[asset_cap][i] = d[position][i] * d['price'][i]
    #             d[bank_cap][i] = d[total_cap][i-1] - d[asset_cap][i]
    #         elif( (d[position][i-1] == 0) and (abs(d['TMV'][i]) < threshold ) ):
    #             '''No position and threshold not crossed'''
    #             d[position][i] = d[position][i-1]
    #             d[asset_cap][i] = d[position][i] * d['price'][i]
    #             d[bank_cap][i] = d[bank_cap][i-1]
    #         elif( ( abs(d[position][i-1]) > 0) ):
    #             if( (d['regime'][i-1] == 0) and (d['type'][i] not in ['DCC','EXT_DCC'] ) ):
    #                 '''No Action to be taken'''
    #                 d[position][i] = d[position][i-1]
    #                 d[asset_cap][i] = d[position][i] * d['price'][i]
    #                 d[bank_cap][i] = d[bank_cap][i-1]
    #             else:
    #                 '''I take the opposite position here'''
    # #                 d['debug'][i] = 'Hello'
    #                 d[position][i] = 0
    #                 d[asset_cap][i] = 0
    #                 d[bank_cap][i] = abs(d[position][i-1]) * d['price'][i]
                    
    #     elif( d['regime'][i] == 1 ):
    #         '''Abnormal regime, momentum trading'''
    #         if( (d[position][i-1] == 0) and (abs(d['TMV'][i]) >= threshold ) ):
    #             '''I follow the market with all my money'''
    #             d[position][i] = np.sign(d['TMV'][i]) * d[total_cap][i-1] / d['price'][i]
    #             d[asset_cap][i] = d[position][i] * d['price'][i]
    #             d[bank_cap][i] = d[total_cap][i-1] - d[asset_cap][i]
    #         elif( (d[position][i-1] == 0) and (abs(d['TMV'][i]) < threshold ) ):
    #             '''No position and threshold not crossed'''
    #             d[position][i] = d[position][i-1]
    #             d[asset_cap][i] = d[position][i] * d['price'][i]
    #             d[bank_cap][i] = d[bank_cap][i-1]
    #         elif( ( abs(d[position][i-1]) > 0) ):
    #             if( (d['regime'][i-1] == 1) and (d['type'][i] not in ['DCC','EXT_DCC'] ) ):
    #                 '''No Action to be taken'''
    #                 d[position][i] = d[position][i-1]
    #                 d[asset_cap][i] = d[position][i] * d['price'][i]
    #                 d[bank_cap][i] = d[bank_cap][i-1]
    #             else:
    #                 '''I take the opposite position here'''
    #                 d[position][i] = 0
    #                 d[asset_cap][i] = 0
    #                 d[bank_cap][i] = abs(d[position][i-1]) * d['price'][i]
    #     else:
    #         d[position][i] = d[position][i-1]
    #         d[asset_cap][i] = d[asset_cap][i-1]
    #         d[bank_cap][i] = d[bank_cap][i-1]
                    

def strategy_control(data, init_cap = 1, strat = "JC1", threshold = 0.5):
    """
    CT1 strategy as explained in the book Chapter 6 -> Mean reversion all the time.
    It returns a dataframe after implementing JC1 strategy which is based on regime change
    We get new columns at every time which are: position, asset capital, bank capital, total capital and returns

    TODO: We assume that open to close and close to open times are same, useful in sharpe calculation

    params-> data: Output of gd.generate_dataset_with_columns or any other strategy
             init_cap: Inital capital in the strategy
             strat = Name of strat for adding columns
             threshold: When to take buy/sell decisions
    
    returns-> pd.DataFrame which has columns appended to data
                'daily_ret_strat', 'position_strat', 'asset_cap_strat', 'bank_cap_strat', 'total_cap_strat'
    """

    '''All values are after trading'''
    position = 'position_'+strat # Position in asset at this time
    daily_ret = 'daily_ret_'+strat # Return on total_cap
    asset_cap = 'asset_cap_'+strat # Price of assets I have
    bank_cap = 'bank_cap_'+strat # Capital which I don't have invested
    total_cap = 'total_cap_'+strat # Sum of above two

    d = data.copy()

    d[daily_ret] = 0
    d[position] = 0 
    d[asset_cap] = 0
    d[bank_cap] = 0
    d[bank_cap][0] = init_cap
    d[total_cap] = d[bank_cap]

    for i in range( 1,len(d) ):
        if( d['regime'][i] == -1 ):
            '''Don't do anything, regime detection has not started'''
            d[position][i] = d[position][i-1]
            d[asset_cap][i] = d[asset_cap][i-1]
            d[bank_cap][i] = d[bank_cap][i-1]
        
        else:
            if( (d[position][i-1] == 0) and (abs(d['TMV'][i]) >= threshold ) ):
                '''I go against the market with all my money'''
                d[position][i] = -1 * np.sign(d['TMV'][i]) * d[total_cap][i-1] / d['price'][i]
                d[asset_cap][i] = d[position][i] * d['price'][i]
                d[bank_cap][i] = d[total_cap][i-1] - d[asset_cap][i]
            elif( (d[position][i-1] == 0) and (abs(d['TMV'][i]) < threshold ) ):
                '''No position and threshold not crossed'''
                d[position][i] = d[position][i-1]
                d[asset_cap][i] = d[position][i] * d['price'][i]
                d[bank_cap][i] = d[bank_cap][i-1]
            elif( ( abs(d[position][i-1]) > 0) ):
                if( (d['type'][i] not in ['DCC','EXT_DCC'] ) ):
                    '''No Action to be taken'''
                    d[position][i] = d[position][i-1]
                    d[asset_cap][i] = d[position][i] * d['price'][i]
                    d[bank_cap][i] = d[bank_cap][i-1]
                else:
                    '''I take the opposite position here'''
    #                 d['debug'][i] = 'Hello'
                    d[position][i] = 0
                    d[asset_cap][i] = 0
                    d[bank_cap][i] = abs(d[position][i-1]) * d['price'][i]
        d[total_cap][i] = d[bank_cap][i] + d[asset_cap][i]
        d[daily_ret][i] = (d[total_cap][i] - d[total_cap][i-1])/d[total_cap][i-1]
        
    return d     

def strategy_control2(data, init_cap = 1, strat = "control2", threshold = 0.5):
    """
    Momentum trading all the time.
    It returns a dataframe after implementing JC1 strategy which is based on regime change
    We get new columns at every time which are: position, asset capital, bank capital, total capital and returns

    TODO: We assume that open to close and close to open times are same, useful in sharpe calculation

    params-> data: Output of gd.generate_dataset_with_columns or any other strategy
             init_cap: Inital capital in the strategy
             strat = Name of strat for adding columns
             threshold: When to take buy/sell decisions
    
    returns-> pd.DataFrame which has columns appended to data
                'daily_ret_strat', 'position_strat', 'asset_cap_strat', 'bank_cap_strat', 'total_cap_strat'
    """

    '''All values are after trading'''
    position = 'position_'+strat # Position in asset at this time
    daily_ret = 'daily_ret_'+strat # Return on total_cap
    asset_cap = 'asset_cap_'+strat # Price of assets I have
    bank_cap = 'bank_cap_'+strat # Capital which I don't have invested
    total_cap = 'total_cap_'+strat # Sum of above two

    d = data.copy()

    d[daily_ret] = 0
    d[position] = 0 
    d[asset_cap] = 0
    d[bank_cap] = 0
    d[bank_cap][0] = init_cap
    d[total_cap] = d[bank_cap]

    for i in range( 1,len(d) ):
        if( d['regime'][i] == -1 ):
            '''Don't do anything, regime detection has not started'''
            d[position][i] = d[position][i-1]
            d[asset_cap][i] = d[asset_cap][i-1]
            d[bank_cap][i] = d[bank_cap][i-1]
        
        else:
            if( (d[position][i-1] == 0) and (abs(d['TMV'][i]) >= threshold ) ):
                '''I go against the market with all my money'''
                d[position][i] = np.sign(d['TMV'][i]) * d[total_cap][i-1] / d['price'][i]
                d[asset_cap][i] = d[position][i] * d['price'][i]
                d[bank_cap][i] = d[total_cap][i-1] - d[asset_cap][i]
            elif( (d[position][i-1] == 0) and (abs(d['TMV'][i]) < threshold ) ):
                '''No position and threshold not crossed'''
                d[position][i] = d[position][i-1]
                d[asset_cap][i] = d[position][i] * d['price'][i]
                d[bank_cap][i] = d[bank_cap][i-1]
            elif( ( abs(d[position][i-1]) > 0) ):
                if( (d['type'][i] not in ['DCC','EXT_DCC'] ) ):
                    '''No Action to be taken'''
                    d[position][i] = d[position][i-1]
                    d[asset_cap][i] = d[position][i] * d['price'][i]
                    d[bank_cap][i] = d[bank_cap][i-1]
                else:
                    '''I take the opposite position here'''
    #                 d['debug'][i] = 'Hello'
                    d[position][i] = 0
                    d[asset_cap][i] = 0
                    d[bank_cap][i] = abs(d[position][i-1]) * d['price'][i]
        d[total_cap][i] = d[bank_cap][i] + d[asset_cap][i]
        d[daily_ret][i] = (d[total_cap][i] - d[total_cap][i-1])/d[total_cap][i-1]
        
    return d     

    #     init_cap = 1
    # strat = 'JC2'



    # '''All values are after trading'''
    # position = 'position_'+strat # Position in asset at this time
    # daily_ret = 'daily_ret_'+strat # Return on total_cap
    # asset_cap = 'asset_cap_'+strat # Price of assets I have
    # bank_cap = 'bank_cap_'+strat # Capital which I don't have invested
    # total_cap = 'total_cap_'+strat # Sum of above two


    # d[daily_ret] = 0
    # d[position] = 0 # This is after trading on this event (time point)
    # d[asset_cap] = 0
    # d[bank_cap] = 0
    # d[bank_cap][0] = init_cap
    # d[total_cap] = d[bank_cap]

    # '''Threshold for TMV'''
    # thresh = 0.5

    # for i in range( 1,len(d) ):
    #     if( d['regime'][i] == 0 ):
    #         '''Normal Regime - Mean Reverting'''
    #         if( (d[position][i-1] == 0) and (abs(d['TMV'][i]) >= thresh ) ):
    #             '''I go against the market with all my money'''
    #             d[position][i] = -1 * np.sign(d['TMV'][i]) * d[total_cap][i-1] / d['price'][i]
    #             d[asset_cap][i] = d[position][i] * d['price'][i]
    #             d[bank_cap][i] = d[total_cap][i-1] - d[asset_cap][i]
    #         elif( (d[position][i-1] == 0) and (abs(d['TMV'][i]) < thresh ) ):
    #             '''No position and threshold not crossed'''
    #             d[position][i] = d[position][i-1]
    #             d[asset_cap][i] = d[position][i] * d['price'][i]
    #             d[bank_cap][i] = d[bank_cap][i-1]
    #         elif( ( abs(d[position][i-1]) > 0) ):
    #             if( (d['regime'][i-1] == 0) and (d['type'][i] not in ['DCC','EXT_DCC'] ) ):
    #                 '''No Action to be taken'''
    #                 d[position][i] = d[position][i-1]
    #                 d[asset_cap][i] = d[position][i] * d['price'][i]
    #                 d[bank_cap][i] = d[bank_cap][i-1]
    #             else:
    #                 '''I take the opposite position here'''
    # #                 d['debug'][i] = 'Hello'
    #                 d[position][i] = 0
    #                 d[asset_cap][i] = 0
    #                 d[bank_cap][i] = abs(d[position][i-1]) * d['price'][i]
                    
    #     elif( d['regime'][i] == 1 ):
    #         if( (d[position][i-1] == 0) and (abs(d['TMV'][i]) >= thresh ) ):
    #             '''I go against the market with all my money'''
    #             d[position][i] = -1 * np.sign(d['TMV'][i]) * d[total_cap][i-1] / d['price'][i]
    #             d[asset_cap][i] = d[position][i] * d['price'][i]
    #             d[bank_cap][i] = d[total_cap][i-1] - d[asset_cap][i]
    #         elif( (d[position][i-1] == 0) and (abs(d['TMV'][i]) < thresh ) ):
    #             '''No position and threshold not crossed'''
    #             d[position][i] = d[position][i-1]
    #             d[asset_cap][i] = d[position][i] * d['price'][i]
    #             d[bank_cap][i] = d[bank_cap][i-1]
    #         elif( ( abs(d[position][i-1]) > 0) ):
    #             if( (d['regime'][i-1] == 0) and (d['type'][i] not in ['DCC','EXT_DCC'] ) ):
    #                 '''No Action to be taken'''
    #                 d[position][i] = d[position][i-1]
    #                 d[asset_cap][i] = d[position][i] * d['price'][i]
    #                 d[bank_cap][i] = d[bank_cap][i-1]
    #             else:
    #                 '''I take the opposite position here'''
    # #                 d['debug'][i] = 'Hello'
    #                 d[position][i] = 0
    #                 d[asset_cap][i] = 0
    #                 d[bank_cap][i] = abs(d[position][i-1]) * d['price'][i]
    #     else:
    #         d[position][i] = d[position][i-1]
    #         d[asset_cap][i] = d[asset_cap][i-1]
    #         d[bank_cap][i] = d[bank_cap][i-1]
                    
                
                
                
    #     d[total_cap][i] = d[bank_cap][i] + d[asset_cap][i]
    #     d[daily_ret][i] = (d[total_cap][i] - d[total_cap][i-1])/d[total_cap][i-1]      

def get_sharpe(data, column):
    """
    params -> data: an ouput from one of the strategies
                column: Sharpe of which column
    returns -> sharpe for that column from the data
    """

    df = data.copy()
    df = df[ df['regime'] >= 0 ]

    '''We use half days thats why to annualize sharpe we get this'''
    return np.sqrt(2*252) * (df[column].mean()/(df[column].std( ddof = 1 ) ) )

def get_profit( data, column):
    """
    params -> data: an ouput from one of the strategies
                column: profit of which column
    returns -> Profit from this column in percent
    """

    return ( data[column].iloc[-1] - data[column].iloc[0] ) / data[column].iloc[0]

def get_drawdown( data, column ):
    """
    params -> data: an ouput from one of the strategies
                column: drawdown of which column
    TODO: Need to check this function, might be incorrect
    returns -> Max negative sum of return ( ie, minimum sum of continuous return (daily_ret) )
    """



    arr = np.array( data[column] )
    curr_min = arr.copy()
    curr = curr_min[0]

    for i in range(1, len(arr) ):
        curr_min[i] = np.minimum( arr[i], curr_min[i-1] + arr[i] )
        curr = min( curr, curr_min[i] )
    
    '''Very good strategy, never loses money'''
    if( curr > 0 ):
        return -10000
    
    '''We would like to minimize drawdown'''
    return -1 * curr

def get_metrics_trading_strategy( data, strategies ):
    """
    params -> data: an ouput from one of the strategies
              strategies: a list of strategy names

    returns -> Dict of dict of metric
    """

    ans = {}

    for strategy in strategies:
        ans[strategy] = {}
        ans[strategy]['drawdown'] = get_drawdown( data, 'daily_ret_'+strategy)
        ans[strategy]['profit'] = get_profit( data, 'total_cap_'+strategy)
        ans[strategy]['sharpe'] = get_sharpe( data, 'daily_ret_'+strategy)
    
    return ans


def get_loss_function_for_pipeline( data, DC, regimes, theta, init_cap = 1, strat = 'JC1', threshold = 1):
    """
    Function to get the loss 
    Params-> data: Time series( pd.Series ) for the full thing
             DC: List of tuples for DC indicators ( output for get_DC_data_v2 )
             regimes: Filtered regimes, output from hmm model (for train, if used) /Naive Bayes Classifier(valid)
             theta: Theta value for TMV Calculation
             init_cap: Initial Capital
             strat: Name of strat, "control" for control strategy and any other string for regime dependent strategy  
             threshold: threshold for Trading on TMV
            
    Returns: A dict of dict with keys (strat, {drawdown, profit, sharpe})
    """
    df = gd.generate_dataset_with_columns( data, DC, regimes, theta )
    if( strat == "control" ):
        df1 = strategy_control(df, init_cap=init_cap, strat=strat, threshold = threshold)
    elif( strat == "control2"):
        df1 = strategy_control2(df, init_cap=init_cap, strat=strat, threshold = threshold)
    else:
        df1 = strategy_regime_dependent(df, init_cap=init_cap, strat=strat, threshold = threshold)
    
    return get_metrics_trading_strategy( df1, [strat]  )
