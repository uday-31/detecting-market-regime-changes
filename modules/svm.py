################################################################################
# Support Vector Machine Classifier
# Basic SVM Classifier to classify input data
# Dhruv Baid
##############################################################################

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def train_SVM(X, y):
    """
    params: X-> X is a pandas dataframe of features
            y-> y is a list of labels

    return: returns a learned SVM Classifier
    """
    model = make_pipeline(StandardScaler(), SVC(probability=True))
    model.fit(X, y)
    return model


def get_predict_probs_SVM(model, X):
    """
    params: model-> A fitted SVM Model
            X -> A pd dataframe of test variables

    return: Returns a 2-D matrix of n * classes size which has predict_probabilities
    """
    return model.predict_proba(X)


def predict_SVM(model, X, epsilon=0.5):
    """
    params: model-> A fitted Naive Bayes Model
            X -> A pd dataframe of test variables

    return: Returns a list of predictions (highest probability) for each test point
    """
    
    if (epsilon != 0.5):
        ans = np.zeros(len(X))
        probs = get_predict_probs_SVM(model, X)
        return np.where(probs[:, 1] >= epsilon, 1, 0)
    
    return model.predict(X)


def do_all_SVM(X_train, y_train, X_valid, epsilon=0.5):
    """
    Trains a SVM model with y_train as regimes after filteration and X_train as one of the three DC indicators

    params-> X_train: One of the three DC indicators, or a time series
             y_train: Regimes output from hmm model on the training data after filteration
             X_valid: Validation test for X, same DC indicator as X_train
    """
    model = train_SVM(X_train, y_train)
    preds = predict_SVM(model, X_valid, epsilon)
    return preds
