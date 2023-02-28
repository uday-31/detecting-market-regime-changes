import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
plt.style.use('seaborn')

def fit_hmm(n_components: int, price: pd.Series, indicator: pd.Series, ticker: str, plot: bool =False, verbose: bool = False) -> tuple[pd.Series, hmm.GaussianHMM]:
    """Fits a Hidden Markov model to the data and predicts regimes on it. Optionally makes a plot.

    Args:
        n_components (int): number of regimes
        price (pd.Series): price series of the instrument
        indicator (pd.Series): indicator series we wish to fit the model on
        ticker (str): ticker of the instrument
        plot (bool, optional): whether the regimes need to be plotted. Defaults to False.
        verbose (bool, optional): whether debugging output needs to be printed. Defaults to False.

    Returns:
        tuple[pd.Series,hmm.GaussianHMM]: the predicted regimes and the HMM model
    """
    
    X = indicator.to_numpy().reshape(-1,1)

    models, scores = [], []
    for idx in range(10):
        model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", n_iter=1000,
            random_state=idx)
  
        model.fit(X)
        models.append(model)
        scores.append(model.score(X))

    model = models[np.argmax(scores)]

    regimes = pd.Series(model.predict(X))
    regimes.index = indicator.index

    regimes = standardize_regime_labels(regimes,verbose=verbose)

    if plot:
        fig, ax = plt.subplots()
        price.plot(ax=ax, color='black')
        clr = {0:'grey',1:'red',2:'green'}

        for time_start, time_end, regime in zip(regimes.index[:-1], regimes.index[1:], regimes.values[:-1]):
            ax.axvspan(time_start,time_end, alpha=0.8, color=clr[regime])
        ax.set_title(f"regimes for {ticker}")
        ax.set_ylabel("price")
        plt.show()

    return regimes, model


def standardize_regime_labels(regimes: pd.Series, verbose: bool = True) -> pd.Series:
    """
    This is helper function to standardize regime labels. It is based on the assumption
    that regime 1 (index 0) is the normal regime and in the long term, the market is mostly in the
    normal regime.
    :param regimes: A series indicating the regimes and indexed by a datetime
    :param verbose:
    :return:
    """
    start = regimes.index[0]
    initial_regime = regimes[0]
    prev_regime = regimes[0] 
    prev_time = regimes.index[0]
    total_duration_in_initial_regime = 0

    if len(np.unique(regimes)) == 1:
        total_duration_in_initial_regime = (regimes.index[-1] - regimes.index[0]).total_seconds()
    else:
        for time, regime in regimes[1:].items():
            if regime == initial_regime:
                total_duration_in_initial_regime += (time - prev_time).total_seconds()
            prev_time = time
            prev_regime = regime

    total_duration = (regimes.index[-1] - regimes.index[0]).total_seconds()

    if verbose:
        print('Total duration of time: {}'.format(total_duration))
        print('Total duration spent in Regime {}: {}'.format(initial_regime, total_duration_in_initial_regime))
        print('Proportion of time spent in Regime {}: {}'.format(initial_regime, total_duration_in_initial_regime / total_duration))

   # if (initial_regime == 0) and ((total_duration_in_initial_regime / total_duration) <= 0.5):
    if ((initial_regime == 0) and ((total_duration_in_initial_regime / total_duration) <= 0.5)) or ((initial_regime == 1) and ((total_duration_in_initial_regime / total_duration) >= 0.5)):
        if verbose:
            print('Flipping labels between regimes.')
        regimes = 1 - regimes
    return regimes

def make_regime_plots(regimes: pd.Series, tmv: pd.Series, T: pd.Series, ticker: str, set_: str = 'test'):
    """Makes the normalized TMV versus normalized T plots, separated by regime.

    Args:
        regimes (pd.Series): regimes
        tmv (pd.Series): tmv
        T (pd.Series): T
        ticker (str): ticker
    """
    
    regime_df = (pd.DataFrame([regimes, tmv, T]).T)
    regime_df.columns = ['Regime','TMV','T']
    regime_df.Regime = regime_df.Regime.astype('category')

    # normalize
    regime_df[['TMV','T']] = (regime_df[['TMV','T']] - regime_df[['TMV','T']].min())/(regime_df[['TMV','T']].max() - regime_df[['TMV','T']].min())
    
    fig, ax = plt.subplots(figsize=(10,5))
    colors = {0:'grey', 1:'red'}
    for c in colors:
        ax.scatter(regime_df[regime_df.Regime == c]['T'], regime_df[regime_df.Regime == c]['TMV'], c=colors[c],label=f'regime {c}')
    
    ax.set_title(f"Regimes for {ticker} on the {set_} set")
    ax.set_xlabel("normalized T")
    ax.set_ylabel("normalized TMV")
    plt.legend()
    plt.show()

#%%
