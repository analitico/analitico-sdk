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
LOG_TYPE = "log"
MODEL_TYPE = "model"
PLUGIN_TYPE = "plugin"
RECIPE_TYPE = "recipe"
TOKEN_TYPE = "token"
USER_TYPE = "user"
WORKSPACE_TYPE = "workspace"

# IDs
DATASET_PREFIX = "ds_"  # dataset source, filters, etc
ENDPOINT_PREFIX = "ep_"  # inference endpoint configuration
JOB_PREFIX = "jb_"  # sync or async job
LOG_PREFIX = "lg_"  # log record
MODEL_PREFIX = "ml_"  # trained machine learning model (not a django model)
RECIPE_PREFIX = "rx_"  # machine learning recipe (an experiment with modules, code, etc)
TOKEN_PREFIX = "tok_"  # authorization token
PLUGIN_PREFIX = "pl_"  # plugin instance (not plugin description or metadata)
USER_PREFIX = "id_"  # an identity profile
WORKSPACE_PREFIX = "ws_"  # workspace with rights and one or more projects and other resources
