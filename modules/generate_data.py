############################################################################################
### Generate data set from time series data with all relevant columns for trading strategy
### Sarthak Vishnoi
############################################################################################
import numpy as np
import pandas as pd
import modules.directional_change as dc

def generate_dataset_with_columns( data, DC, regimes, theta ):
    """
    params: data-> Time series data for ticker,
            DC -> DC List ( output of dc.get_DC_data_v2 )
            regimes -> output of fit_hmm
            theta -> Theta value for the DC Indicator calculation

    return: df -> data frame with the following columns
                  'time', 'price', 'type', 'trend', 'TMV', 'regime'
    """

    d = pd.DataFrame( data )
    d['time'] = d.index
    d.set_index( np.arange(len(d)), inplace = True)
    d.columns = ['price', 'time']
    d = d[['time', 'price']]

    '''DC'''
    DCC, DCC_idx, EXT, EXT_idx = dc.get_DCC_EXT( DC )

    '''Point type'''
    d['type'] = 'Other'

    for i in range( len(d) ):
        # Try to vectorize this code
        if( d['time'][i] in DCC_idx ):
            d['type'][i] = 'DCC'
        if( d['time'][i] in EXT_idx ):
            d['type'][i] = 'EXT'
        if( (d['time'][i] in DCC_idx) and (d['time'][i] in EXT_idx) ):
            d['type'][i] = 'EXT_DCC'
    #     d['type'][(d['time'] in DCC_idx) ] = 'DCC'
    #     d['type'][d['time'] in EXT_idx ] = 'EXT'

    '''Setting the trend style - up(1)/down(-1)/none(0) and TMV values from last ext'''    
    d['trend'] = 0 # first is always none
    d['TMV'] = 0
    LAST_EXT = -1
    LAST_EXT_idx = -1
    for i in range(len(d)):
        if(d['type'][i] in ['EXT', 'EXT_DCC']):
            LAST_EXT = d['price'][i]
            LAST_EXT_idx = d['time'][i]
        if(LAST_EXT > 0):
            d['trend'][i] = np.sign( d['price'][i] - LAST_EXT )
            d['TMV'][i] = ( d['price'][i] - LAST_EXT ) / ( LAST_EXT * theta )
            
            
    '''Adding Regime - Forward Fill'''
    d['regime'] = -1 #For initial points
    LAST_REGIME = regimes[0]
    LAST_REGIME_idx = regimes.index[0]
    for i in range( len(d) ):
        if( d['time'][i] in regimes.index ):
            d['regime'][i] = regimes[d['time'][i]]
            LAST_REGIME_idx = d['time'][i]
            LAST_REGIME = regimes[d['time'][i]]
            
        elif( d['time'][i] >= LAST_REGIME_idx ):
            d['regime'][i] = LAST_REGIME

    return d

