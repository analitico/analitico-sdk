import os
import os.path

from .interfaces import PluginError
from .pipelineplugin import PipelinePlugin

##
## RecipePipelinePlugin
##


class RecipePipelinePlugin(PipelinePlugin):
    """
    A recipe pipeline contains:
    - a dataframe source plugin that gathers the training data
    - a miminal (or empty) set of plugins to do some final filtering (eg. remove columns)
    - an algorithm used for training (eg. classifier, neural net, etc)
    Running the pipeline produces a trained model which can be used later for predictions.
    """

    class Meta(PipelinePlugin.Meta):
        name = "analitico.plugin.RecipePipelinePlugin"
        inputs = None
        outputs = [{"model": "dict"}]

    def run(self, *args, **kwargs):
        """ Process the plugins in sequence to create trained model artifacts """
        results = super().run(*args, **kwargs)
        if not isinstance(results, dict):
            msg = "Pipeline didn't produce a dictionary with training results"
            self.error(msg)
            raise PluginError(msg, self)

        # training.json, trained models and other artifacts should
        # now be in the artifacts directory. depending on the environment
        # these may be left on disk (SDK) or stored in cloud (APIs)
        artifacts_path = self.factory.get_artifacts_directory()
        assert len(os.listdir(artifacts_path)) >= 1
        return results
