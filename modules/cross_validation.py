################################################################################
# Implementation of 'sort-of' Cross Validation to tune the parameters of our
# model.
# Author: Rohan Prasad
################################################################################
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from directional_change import get_data, get_DC_data, get_DC_data_v2, get_TMV, get_T, get_R
from hidden_markov_model import fit_hmm


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

class Pipeline:

    def __init__(self, type_: str = 'equity', type_mapper: dict = {'equity':['^DJI','^GSPC'],'bond':['^TNX', '^IRX'],
                'fx':['RUB=X','GBP=X']}, start_date: str = "2005-01-01",
                train_end: str = "2017-12-31", valid_start: str = "2018-01-01", 
                valid_end:str = "2019-12-31", test_start:str = "2020-01-01",
                theta: float = 0.025, num_regimes: int = 2, trading_day: dict = {'equity':6.5, 'fx':12,'bond':9}):
        """Initializes the pipeline parameters.

        Args:
            type_ (str, optional): Asset class. Defaults to 'equity'. 'equity' or 'fx' or 'bond'
            type_mapper (_type_, optional): Maps the type to tickers. Defaults to {'equity':['^DJI','^GSPC'],'bond':['^TNX', '^IRX'], 'fx':['RUB=X','GBP=X']}.
            start_date (str, optional): Start date for training set. Defaults to "2005-01-01".
            train_end (str, optional): End date for training set. Defaults to "2017-12-31".
            valid_start (str, optional): Start date for validation set. Defaults to "2018-01-01".
            valid_end (str, optional): End date for validation set. Defaults to "2019-12-31".
            test_start (str, optional): Start date for test set. Defaults to "2020-01-01".
            theta (float, optional): Threshold for DC. Defaults to 0.025.
            num_regimes (int, optional): Number of regimes. Defaults to 2.
            trading_day (_type_, optional): Offset between open and close data. Defaults to {'equity':6.5, 'fx':12,'bond':9}.
        """

        self.type_ = type_
        self.type_mapper = type_mapper
        self.tickers = type_mapper[type_]
        self.start_date = start_date
        self.train_end = train_end
        self.valid_start = valid_start
        self.valid_end = valid_end
        self.test_start = test_start
        self.theta = theta
        self.num_regimes = num_regimes
        self.trading_day = trading_day

    def fit(self, plot: bool = False, verbose: bool = False):
        """Fits the pipeline

        Args:
            plot (bool, optional): Plot the regimes. Defaults to False.
            verbose (bool, optional): Whether debug output has to be printed. Defaults to False.
        """

        df_ts = get_data(self.tickers, self.start_date, self.trading_day[self.type_]/2)

        self.ts = {}
        self.ts['train'] = df_ts.loc[:self.train_end,:]
        self.ts['valid'] = df_ts.loc[self.valid_start:self.valid_end,:]
        self.ts['test'] = df_ts.loc[self.test_start:]

        self.dc = {}
        for cat in ['train','valid','test']:
            self.dc[cat] = {}
            for ticker in self.tickers:
                self.dc[cat][ticker] = get_DC_data(self.ts[cat][ticker], self.theta)

        self.tmv = {}
        self.T = {}
        self.R = {}
        for cat in ['train','valid','test']:
            self.tmv[cat], self.T[cat], self.R[cat] = {}, {}, {}
            for ticker in self.tickers:
                self.tmv[cat][ticker] = get_TMV(self.dc[cat][ticker],self.theta)
                self.T[cat][ticker] = get_T(self.dc[cat][ticker])
                self.R[cat][ticker] = get_R(self.tmv[cat][ticker],self.T[cat][ticker],self.theta)

        self.regimes = {}
        for ticker in self.tickers:
            reg, _ = fit_hmm(self.num_regimes, self.ts['train'][ticker], self.R['train'][ticker], ticker, plot = plot, verbose = verbose)
            self.regimes[ticker] = reg