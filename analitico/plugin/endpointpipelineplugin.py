import analitico.utilities
import pandas
import os
import os.path

from analitico.utilities import read_json, get_dict_dot

from .plugin import PluginError
from .pipelineplugin import PipelinePlugin

##
## EndpointPipelinePlugin
##


class EndpointPipelinePlugin(PipelinePlugin):
    """
    EndpointPipelinePlugin is a base class for endpoints that take trained machine
    learning models to deliver inferences. An endpoint subclass could implement 
    inference APIs by taking a web request and returning predictions, etc.
    """

    class Meta(PipelinePlugin.Meta):
        name = "analitico.plugin.EndpointPipelinePlugin"
        inputs = [{"data": "pandas.DataFrame"}]
        outputs = [{"predictions": "pandas.DataFrame"}]

    def run(self, action=None, *args, **kwargs):
        """ Process the plugins in sequence to run predictions """
        assert args[0] and isinstance(args[0], pandas.DataFrame)

        # read training information from disk
        artifacts_path = self.manager.get_artifacts_directory()
        training_path = os.path.join(artifacts_path, "training.json")
        training = read_json(training_path)
        assert training

        # if no plugins have been configured for the pipeline,
        # create the plugin suggested by the training algorithm
        if not self.plugins:
            self.set_attribute("plugins", [{"name": get_dict_dot(training, "plugins.prediction")}])

        # run the pipeline, return predictions
        predictions = super().run(action, *args, **kwargs)
        assert predictions and isinstance(predictions, pandas.DataFrame)
        return predictions
