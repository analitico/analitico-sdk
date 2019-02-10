ANALITICO_SCHEMA = "analitico"
ANALITICO_PREFIX = "analitico://"

ANALITICO_STAGING_API_ENDPOINT = "https://staging.analitico.ai/api/"
ANALITICO_API_ENDPOINT = "https://analitico.ai/api/"

ACTION_PROCESS = "process"  # process a dataframe to retrieve data
ACTION_TRAIN = "train"  # train a recipe to procude a model
ACTION_PREDICT = "predict"  # run a model to generate a prediction

import analitico.utilities
import analitico.plugin
import analitico.dataset
import analitico.mixin
import analitico.manager


def authorize(token=None, endpoint=ANALITICO_STAGING_API_ENDPOINT) -> analitico.plugin.IPluginManager:
    """ Returns a factory which can create datasets, models, plugins, etc """
    return analitico.manager.PluginManager(token, endpoint)
