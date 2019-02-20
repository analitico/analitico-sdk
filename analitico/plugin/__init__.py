# plugin base classes
from .interfaces import IPlugin  # NOQA: F401
from .interfaces import IDataframeSourcePlugin  # NOQA: F401
from .interfaces import IDataframePlugin  # NOQA: F401
from .interfaces import IAlgorithmPlugin  # NOQA: F401
from .interfaces import IGroupPlugin  # NOQA: F401

# plugins use the same specialized class for exceptions
from .interfaces import PluginError  # NOQA: F401

# plugins to generate dataframes from sources
from .csvdataframesourceplugin import CsvDataframeSourcePlugin

CSV_DATAFRAME_SOURCE_PLUGIN = CsvDataframeSourcePlugin.Meta.name

# plugins to tranform dataframes
from .transforms import CodeDataframePlugin
from .augmentdatesdataframeplugin import AugmentDatesDataframePlugin

CODE_DATAFRAME_PLUGIN = CodeDataframePlugin.Meta.name
AUGMENT_DATES_DATAFRAME_PLUGIN = AugmentDatesDataframePlugin.Meta.name

# machine learning algorithms
from .catboostplugin import CatBoostPlugin  # abstract
from .catboostregressorplugin import CatBoostRegressorPlugin
from .catboostclassifierplugin import CatBoostClassifierPlugin

CATBOOST_REGRESSOR_PLUGIN = CatBoostRegressorPlugin.Meta.name
CATBOOST_CLASSIFIER_PLUGIN = CatBoostClassifierPlugin.Meta.name

# plugin workflows
from .pipelineplugin import PipelinePlugin
from .dataframepipelineplugin import DataframePipelinePlugin
from .recipepipelineplugin import RecipePipelinePlugin
from .endpointpipelineplugin import EndpointPipelinePlugin

PIPELINE_PLUGIN = PipelinePlugin.Meta.name
DATAFRAME_PIPELINE_PLUGIN = DataframePipelinePlugin.Meta.name
RECIPE_PIPELINE_PLUGIN = RecipePipelinePlugin.Meta.name
ENDPOINT_PIPELINE_PLUGIN = EndpointPipelinePlugin.Meta.name

# analitico type for plugins
PLUGIN_TYPE = "analitico/plugin"

# NOQA: F401 prospector complains that these imports
# are unused but they are here to define the module
