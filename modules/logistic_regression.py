import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def train_LR(X, y):
    """
    params: X-> X is a pandas dataframe of features
            y-> y is a list of labels

    return: returns a learned Naive Bayes Classifier

    comments: Right now we are only considering gaussian kernels
    """
    model = LogisticRegression(penalty='none')
    model.fit( X, y )
    return model

def get_predict_probs_LR( model, X ):
    """
    params: model-> A fitted Naive Bayes Model
            X -> A pd dataframe of test variables

    return: Returns a 2-D matrix of n * classes size which has predict_probabilities
    """

    return model.predict_proba(X)

def predict_LR( model, X, epsilon = 0.5 ):
    """
    params: model-> A fitted Naive Bayes Model
            X -> A pd dataframe of test variables

    return: Returns a list of predictions (highest probability) for each test point
    """
    

    if( epsilon != 0.5 ):
        ans = np.zeros(len(X))
        probs = get_predict_probs_LR( model, X )
        return np.where(probs[:, 1] >= epsilon, 1, 0)

    return model.predict(X)

def do_all_LR(X_train, y_train, X_valid, epsilon = 0.5):
    """
    Trains a naive bayes model with y_train as regimes after filteration and X_train as one of the three DC indicators

    params-> X_train: One of the three DC indicators, or a time series
             y_train: Regimes output from hmm model on the training data after filteration
             X_valid: Validation test for X, same DC indicator as X_train
    """

    model = train_LR( X_train, y_train )
    preds = predict_LR( model, X_valid, epsilon )
    return preds
