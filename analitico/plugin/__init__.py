# plugin base classes
from .plugin import IPlugin  # NOQA: F401
from .plugin import IDataframeSourcePlugin  # NOQA: F401
from .plugin import IDataframePlugin  # NOQA: F401
from .plugin import IAlgorithmPlugin  # NOQA: F401
from .plugin import IGroupPlugin  # NOQA: F401
from .plugin import IPluginManager  # NOQA: F401
from .plugin import PluginError  # NOQA: F401

# plugins to generate dataframes from sources
from .csvdataframesourceplugin import CsvDataframeSourcePlugin

# plugins to tranform dataframes
from .transforms import CodeDataframePlugin

# machine learning algorithms
from .catboostplugin import CatBoostPlugin
from .catboostregressorplugin import CatBoostRegressorPlugin
from .catboostclassifierplugin import CatBoostClassifierPlugin

# plugin workflows
from .pipelineplugin import PipelinePlugin
from .dataframepipelineplugin import DataframePipelinePlugin
from .recipepipelineplugin import RecipePipelinePlugin
from .graphplugin import GraphPlugin

# plugin names
CATBOOST_REGRESSOR_PLUGIN = CatBoostRegressorPlugin.Meta.name
CATBOOST_CLASSIFIER_PLUGIN = CatBoostClassifierPlugin.Meta.name
CSV_DATAFRAME_SOURCE_PLUGIN = CsvDataframeSourcePlugin.Meta.name
CODE_DATAFRAME_PLUGIN = CodeDataframePlugin.Meta.name
PIPELINE_PLUGIN = PipelinePlugin.Meta.name
DATAFRAME_PIPELINE_PLUGIN = DataframePipelinePlugin.Meta.name
GRAPH_PLUGIN = GraphPlugin.Meta.name

# analitico type for plugins
PLUGIN_TYPE = "analitico/plugin"

# NOQA: F401 prospector complains that these imports
# are unused but they are here to define the module
