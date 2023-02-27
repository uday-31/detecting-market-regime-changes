################################################################################
# Implementation of 'sort-of' Cross Validation to tune the parameters of our
# model.
# Author: Rohan Prasad
################################################################################
from typing import Type, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

import directional_change as dc
import hidden_markov_model as hmm
import generate_data as gd
import NaiveBayesClassifier as nbc
import trading_strategy as ts

from directional_change import get_data, get_DC_data, get_DC_data_v2, get_TMV, get_T, get_R
from hidden_markov_model import fit_hmm


def _initialize_loss(minimize: bool):
    if minimize:
        return np.inf
    else:
        return -np.inf


class CustomCrossValidation:
    def __init__(self, pipeline_class: Type, parameter_grid: dict, verbose: bool = False):
        self.optimal_parameters = None
        self.losses = None
        self.pipeline_class = pipeline_class
        self.parameter_grid = parameter_grid
        self.is_verbose = verbose

    def fit(self, data: pd.DataFrame, metric: str = None, minimize: bool = True):

        self.losses = []
        self.optimal_parameters = None

        optimum = _initialize_loss(minimize)

        for idx, params in enumerate(ParameterGrid(self.parameter_grid)):
            pipeline = self.pipeline_class(df_ts=data, **params)
            self._pprint(idx, "Parameters: {}".format(params))
            loss = pipeline.fit()
            # self._pprint(idx, "Training Complete. Evaluating on validation set.")
            # loss = self.loss_function(x_valid, pipeline_out)
            self._pprint(idx, "Loss: {}".format(loss.values[0]))
            self.losses.append(loss.values[0])
            if metric is not None:
                optimum = self._find_optimum_value(loss.values[0], metric, minimize, optimum, params)

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

    def __init__(self, df_ts: pd.DataFrame,type_: str = 'equity', type_mapper: dict = {'equity':['^GSPC'],'bond':['^IRX'],
                'fx':['GBP=X']}, start_date: str = "2005-01-01",
                train_end: str = "2017-12-31", valid_start: str = "2018-01-01", 
                valid_end:str = "2019-12-31", test_start:str = "2020-01-01",
                theta: float = 0.025, num_regimes: int = 2, trading_day: dict = {'equity':6.5, 'fx':12,'bond':9},
                DC_indicator: str = "R", threshold: float = 0.5, strat: str = "JC1", init_cap: int = 1):

        """Initializes the pipeline parameters.

        Args:
            df_ts (pd.DataFrame): price dataframe. Pulled from Yahoo Finance.
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
            DC_indicator (str, options): Which indicator to use
            threshold (float, optional): What threshold for TMV do we trade
            strat (str, optional): Name of strategy
            init_cap (int, optional): Starting capital for the strategy
        """
        self.df_ts = df_ts
        self.type_ = type_
        self.type_mapper = type_mapper
        self.tickers = type_mapper[type_]
        self.ticker = self.tickers[0]
        self.start_date = start_date
        self.train_end = train_end
        self.valid_start = valid_start
        self.valid_end = valid_end
        self.test_start = test_start
        self.theta = theta
        self.num_regimes = num_regimes
        self.trading_day = trading_day
        self.DC_indicator = DC_indicator
        self.dict_indicators = {}

        self.regimes_valid = {} # Regimes predicted on validation set
        self.trading_metrics = {} # Metrics for trading strategy
        self.threshold = threshold # Threshold for TMV for strategy
        self.strat = strat # Name for strategy ( 'control' for control strategy, anything else for test strategy)
        self.init_cap = init_cap

    def fit(self, plot: bool = False, verbose: bool = False):
        """Fits the pipeline

        Args:
            plot (bool, optional): Plot the regimes. Defaults to False.
            verbose (bool, optional): Whether debug output has to be printed. Defaults to False.
        """

        #df_ts = dc.get_data(self.tickers, self.start_date, self.trading_day[self.type_]/2)

        self.ts = {}
        self.ts['train'] = self.df_ts.loc[:self.train_end]
        self.ts['valid'] = self.df_ts.loc[self.valid_start:self.valid_end]
        self.ts['test'] = self.df_ts.loc[self.test_start:]


        self.dc = {}
        for cat in ['train','valid','test']:
            self.dc[cat] = dc.get_DC_data_v2(self.ts[cat], self.theta)

        self.tmv = {}
        self.T = {}
        self.R = {}
        for cat in ['train','valid','test']:
            self.tmv[cat], self.T[cat], self.R[cat] = {}, {}, {}
            self.tmv[cat] = dc.get_TMV(self.dc[cat],self.theta)
            self.T[cat] = dc.get_T(self.dc[cat])
            self.R[cat] = dc.get_R(self.tmv[cat],self.T[cat],self.theta)
        
        self.dict_indicators['R'] = self.R
        self.dict_indicators['T'] = self.T
        self.dict_indicators['TMV'] = self.tmv
        
        self.regimes = {}

        reg, _ = hmm.fit_hmm(self.num_regimes, self.ts['train'], self.dict_indicators[self.DC_indicator]['train'], self.ticker, plot = plot, verbose = verbose)
        self.regimes = reg

        '''Creating labels for validation set using the Naive Bayes Classifier'''
        self.regimes_valid = nbc.do_all_NBC(self.dict_indicators[self.DC_indicator]['train'].values.reshape(-1, 1), self.regimes, self.dict_indicators[self.DC_indicator]['valid'].values.reshape(-1, 1))
        print( len( self.regimes_valid) )
        print( len(self.dict_indicators[self.DC_indicator]['valid'].index) )
        
        
        self.regimes_valid = pd.Series( self.regimes_valid, index = self.dict_indicators[self.DC_indicator]['valid'].index )
        self.trading_metrics = ts.get_loss_function_for_pipeline( self.ts['valid'], self.dc['valid'], self.regimes_valid, self.theta, init_cap = self.init_cap, strat = self.strat, threshold = self.threshold)


        