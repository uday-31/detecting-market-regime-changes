################################################################################
# Naive Bayes Classifier
# Basic Naive Bayes Classifier to classify the data based on prior probabilities, etc
# Sarthak Vishnoi
##############################################################################
import pandas as pd
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

def predict_NBC( model, X ):
    """
    params: model-> A fitted Naive Bayes Model
            X -> A pd dataframe of test variables

    return: Returns a list of predictions (highest probability) for each test point
    """

    return model.predict(X)
