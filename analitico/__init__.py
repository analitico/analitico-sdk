# schemas
ANALITICO_SCHEMA = "analitico"
ANALITICO_URL_PREFIX = "analitico://"

# API endpoints
ANALITICO_STAGING_API_ENDPOINT = "https://staging.analitico.ai/api/"
ANALITICO_API_ENDPOINT = "https://analitico.ai/api/"

# actions
ACTION_PROCESS = "process"  # process a dataframe to retrieve data
ACTION_TRAIN = "train"  # train a recipe to procude a model
ACTION_PREDICT = "predict"  # run a model to generate a prediction

# types
TYPE_PREFIX = "analitico/"
DATASET_TYPE = "dataset"
ENDPOINT_TYPE = "endpoint"
JOB_TYPE = "job"
MODEL_TYPE = "model"
RECIPE_TYPE = "recipe"
TOKEN_TYPE = "token"
USER_TYPE = "user"
WORKSPACE_TYPE = "workspace"

# IDs
DATASET_PREFIX = "ds_"  # dataset source, filters, etc
ENDPOINT_PREFIX = "ep_"  # inference endpoint configuration
JOB_PREFIX = "jb_"  # sync or async job
MODEL_PREFIX = "ml_"  # trained machine learning model (not a django model)
RECIPE_PREFIX = "rx_"  # machine learning recipe (an experiment with modules, code, etc)
TOKEN_PREFIX = "tok_"  # authorization token
USER_PREFIX = "id_"  # an identity profile
WORKSPACE_PREFIX = "ws_"  # workspace with rights and one or more projects and other resources


import analitico.utilities
import analitico.plugin
import analitico.dataset
import analitico.mixin
import analitico.manager


def authorize(token=None, endpoint=ANALITICO_STAGING_API_ENDPOINT) -> analitico.plugin.IPluginManager:
    """ Returns a factory which can create datasets, models, plugins, etc """
    return analitico.manager.PluginManager(token, endpoint)
