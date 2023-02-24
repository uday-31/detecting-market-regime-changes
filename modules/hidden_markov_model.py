import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
plt.style.use('seaborn')

def fit_hmm(n_components: int, price: pd.Series, indicator: pd.Series, ticker: str, plot: bool =False) -> tuple[pd.Series, hmm.GaussianHMM]:
    """Fits a Hidden Markov model to the data and predicts regimes on it. Optionally makes a plot.

    Args:
        n_components (int): number of regimes
        price (pd.Series): price series of the instrument
        indicator (pd.Series): indicator series we wish to fit the model on
        ticker (str): ticker of the instrument
        plot (bool, optional): whether the regimes need to be plotted. Defaults to False.

    Returns:
        tuple[pd.Series,hmm.GaussianHMM]: the predicted regimes and the HMM model
    """
    
    X = indicator.to_numpy().reshape(-1,1)
    X_train = X[:int(0.7*len(X))] # TODO remove hardcode

    models, scores = [], []
    for idx in range(10):
        model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=1000,
            random_state=idx)
  
        model.fit(X_train)
        models.append(model)
        scores.append(model.score(X))

    model = models[np.argmax(scores)]

    regimes = pd.Series(model.predict(X))
    regimes.index = indicator.index

    if plot:
        fig, ax = plt.subplots()
        price.plot(ax=ax, color='black')
        clr = {0:'grey',1:'red',2:'green'}

        for time_start, time_end, regime in zip(regimes.index[:-1], regimes.index[1:], regimes.values[:-1]):
            ax.axvspan(time_start,time_end, alpha=0.8, color=clr[regime])
        ax.vlines(price.index[int(0.7*len(price))],0,price.max(),color='blue')
        ax.set_title(f"regimes for {ticker}")
        ax.set_ylabel("price")
        plt.show()

    return regimes, model