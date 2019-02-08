import analitico.utilities
import pandas as pd
import os

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
        """ Process the plugins in sequence then create trained model """
        model_info = super().run(*args, **kwargs)
        if not isinstance(model_info, dict):
            self.logger.warn("RecipePipelinePlugin.run - pipeline didn't produce a dictionary with training results")
            return None

        # create model object
        # scan artifacts
        # upload artifacts to model
        # delete artifacts so they are not loaded to the recipe
        artifacts_path = self.manager.get_artifacts_directory()
        # ...

        return model_info
