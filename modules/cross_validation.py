################################################################################
# Implementation of 'sort-of' Cross Validation to tune the parameters of our
# model.
# Author: Rohan Prasad
################################################################################
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid


def _initialize_loss(minimize: bool):
    if minimize:
        return np.inf
    else:
        return -np.inf


class CustomCrossValidation:
    def __init__(self, pipeline: Callable, parameter_grid: dict, loss_function: Callable, verbose: bool = False):
        self.optimal_parameters = None
        self.losses = None
        self.pipeline = pipeline
        self.parameter_grid = parameter_grid
        self.loss_function = loss_function
        self.is_verbose = verbose

    def fit(self, x_train: [pd.DataFrame, pd.Series, np.ndarray], x_valid: [pd.DataFrame, pd.Series, np.ndarray],
            y_train: [pd.Series, np.ndarray] = None, y_valid: [pd.Series, np.ndarray] = None,
            metric: str = None, minimize: bool = True):

        self.losses = []
        self.optimal_parameters = None

        optimum = _initialize_loss(minimize)

        for idx, params in enumerate(ParameterGrid(self.parameter_grid)):
            self._pprint(idx, "Parameters: {}".format(idx, params))
            pipeline_out = self.pipeline(x_train, y_train, **params)
            self._pprint(idx, "Training Complete. Evaluating on validation set.")
            loss = self.loss_function(x_valid, pipeline_out)
            self._pprint(idx, "Loss: {}".format(loss))
            self.losses.append(loss)
            if metric is not None:
                optimum = self._find_optimum_value(loss, metric, minimize, optimum, params)

    def get_optimal_parameters(self):
        return self.optimal_parameters

    def get_losses(self):
        return self.losses

    def _find_optimum_value(self, loss: dict, metric: str, minimize: bool, optimum: np.float, parameters: dict):
        if minimize:
            if loss[metric] < optimum:
                self.optimal_parameters = parameters
                return loss[metric]
        else:
            if loss[metric] > optimum:
                self.optimal_parameters = parameters
                return loss[metric]
        return optimum

    def _pprint(self, idx, out):
        if self.is_verbose:
            print("Iteration: {}: {}".format(idx, out))

# %%
