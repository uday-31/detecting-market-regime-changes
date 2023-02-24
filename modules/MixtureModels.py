################################################################################
# Mixture Models
# Gives separate weightings to different models based on a vector q which sums to 1
# Sarthak Vishnoi
##############################################################################

import pandas as pd
import numpy as np

def get_probabilities_mixture_models( probs, q ):
    """
    params: probs-> A list of predicted probabilities from each model for each class
            q-> Weights to  be given to each model ( Need not sum to 1 ) (len(q) = M)

    returns: probabilities from the ensemble model ( linear combination of all the models ) 
    """
    M = len(probs) # Number of models
    n = probs[0].shape[0] # Number of data points in X
    K = probs[0].shape[1] # Number of classes

    assert( M == len(q) )

    q = q / np.sum(q)

    ans = np.zeros(shape=(n,K))
    
    for i in range(M):
        ans += q[i] * probs[i]

    return ans

def predict_mixture_models( probs, q ):
    """
    params: probs-> A list of predicted probabilities from each model for each class
            q-> Weights to  be given to each model ( Need not sum to 1 ) (len(q) = M)

    returns: highest predicted class based on ensemble model 
    """

    p = get_probabilities_mixture_models(probs, q)
    classes = np.argmax(p, axis = 1)
    return classes