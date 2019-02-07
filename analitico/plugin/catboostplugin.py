""" Regression and classification plugins based on CatBoost """

import requests
import pandas

from analitico.utilities import analitico_to_pandas_type, get_dict_dot
from .plugin import IAlgorithmPlugin, PluginError

##
## CatBoostPlugin
##


class CatBoostPlugin(IAlgorithmPlugin):
    """ Base class for CatBoost regressor and classifier plugins """

    class Meta(IAlgorithmPlugin.Meta):
        name = "analitico.plugin.CatBoostPlugin"

    def train(self, training, validation, *args, **kwargs):
        """ Train with algorithm and given data to produce a trained model """
        model_info = {"plugin": self.Meta.name, "stats": {"training_rows": len(training)}}
        return model_info


##
## CatBoostRegressorPlugin
##


class CatBoostRegressorPlugin(CatBoostPlugin):
    """ A tabular data regressor based on CatBoost library """

    class Meta(CatBoostPlugin.Meta):
        name = "analitico.plugin.CatBoostRegressorPlugin"


##
## CatBoostClassifierPlugin
##


class CatBoostClassifierPlugin(CatBoostPlugin):
    """ A tabular data classifier based on CatBoost library """

    class Meta(CatBoostPlugin.Meta):
        name = "analitico.plugin.CatBoostClassifierPlugin"
