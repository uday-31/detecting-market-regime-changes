################################################################################
# Naive Bayes Classifier
# Basic Naive Bayes Classifier to classify the data based on prior probabilities, etc
# Sarthak Vishnoi
##############################################################################
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB


def train_NBC(X, y):
    """
    params: X-> X is a pandas dataframe of features
            y-> y is a list of labels

    return: returns a learned Naive Bayes Classifier

    comments: Right now we are only considering gaussian kernels
    """
    model = GaussianNB()
    model.fit( X, y )
    return model

def get_predict_probs_NBC( model, X ):
    """
    params: model-> A fitted Naive Bayes Model
            X -> A pd dataframe of test variables

    return: Returns a 2-D matrix of n * classes size which has predict_probabilities
    """

    return model.predict_proba(X)

def predict_NBC( model, X, epsilon = 0.5 ):
    """
    params: model-> A fitted Naive Bayes Model
            X -> A pd dataframe of test variables

    return: Returns a list of predictions (highest probability) for each test point
    """
    

    if( epsilon != 0.5 ):
        ans = np.zeros(len(X))
        probs = get_predict_probs_NBC( model, X )
        return np.where(probs[:, 1] >= epsilon, 1, 0)
        # for i in range(len(probs)):
        #     if probs[i][1] >= epsilon:
        #         ans[i] = 1
        # return ans

    return model.predict(X)

def do_all_NBC(X_train, y_train, X_valid, epsilon = 0.5):
    """
    Trains a naive bayes model with y_train as regimes after filteration and X_train as one of the three DC indicators

    params-> X_train: One of the three DC indicators, or a time series
             y_train: Regimes output from hmm model on the training data after filteration
             X_valid: Validation test for X, same DC indicator as X_train
    """

    model = train_NBC( X_train, y_train )
    preds = predict_NBC( model, X_valid, epsilon )
    return preds
