import analitico.utilities
import pandas as pd
import os
import os.path

from analitico.utilities import read_json, get_dict_dot

from .interfaces import PluginError
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

    def run(self, *args, **kwargs):
        """ Process the plugins in sequence to run predictions """
        try:
            assert isinstance(args[0], pd.DataFrame)
            data_df = args[0]

            # read training information from disk
            artifacts_path = self.factory.get_artifacts_directory()
            training_path = os.path.join(artifacts_path, "training.json")
            training = read_json(training_path)
            assert training

            # if no plugins have been configured for the pipeline,
            # create the plugin suggested by the training algorithm
            if not self.plugins:
                self.set_attribute("plugins", [{"name": get_dict_dot(training, "plugins.prediction")}])

            # run the pipeline, return predictions
            predictions = super().run(data_df, **kwargs)
            return predictions

        except Exception as exc:
            self.error("Error while processing prediction pipeline")
            self.logger.exception(exc)
            raise exc
