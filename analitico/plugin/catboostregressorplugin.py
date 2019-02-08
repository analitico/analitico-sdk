""" Regression plugin based on CatBoost """

import analitico
import catboost
import collections
import numpy as np
import pandas as pd
import requests
import sklearn
import os.path

from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.metrics import precision_score, recall_score, accuracy_score

from analitico.utilities import analitico_to_pandas_type, get_dict_dot
from analitico.utilities import time_ms, save_json

from .plugin import IAlgorithmPlugin, PluginError
from .catboostplugin import CatBoostPlugin


##
## CatBoostRegressorPlugin
##


class CatBoostRegressorPlugin(CatBoostPlugin):
    """ A tabular data regressor based on CatBoost library """

    class Meta(CatBoostPlugin.Meta):
        name = "analitico.plugin.CatBoostRegressorPlugin"

    def create_model(self, results):
        """ Creates a CatBoostRegressor configured as requested """
        iterations = self.get_attribute("parameters.iterations", 50)
        learning_rate = self.get_attribute("parameters.learning_rate", 1)
        depth = self.get_attribute("parameters.depth", 8)
        results["parameters"]["iterations"] = iterations
        results["parameters"]["learning_rate"] = learning_rate
        results["parameters"]["depth"] = depth
        return catboost.CatBoostRegressor(iterations=iterations, learning_rate=learning_rate, depth=8)

    def score_training(self, model, test_df, test_pool, test_labels, results):
        """ Runs predictions on test set then stores metrics in results["scores"] """
        test_preds = model.predict(test_pool)
        results["scores"]["median_abs_error"] = round(median_absolute_error(test_preds, test_labels), 5)
        results["scores"]["mean_abs_error"] = round(mean_absolute_error(test_preds, test_labels), 5)
        results["scores"]["sqrt_mean_squared_error"] = round(np.sqrt(mean_squared_error(test_preds, test_labels)), 5)
        return super().score_training(model, test_df, test_pool, test_labels, results)
