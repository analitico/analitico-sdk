import os
import os.path
import pandas as pd

from .interfaces import PluginError
from .pipelineplugin import PipelinePlugin, plugin

import analitico.constants
import analitico.pandas
from analitico.utilities import read_json

##
## RecipePipelinePlugin
##


@plugin
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

    def run(self, *args, action=None, **kwargs):
        """ Process the plugins in sequence to create trained model artifacts """
        try:

            # when training run the recipe which will produce the training artifacts
            if analitico.constants.ACTION_TRAIN in action:
                results = super().run(*args, action=action, **kwargs)
                if not isinstance(results, dict):
                    msg = "Pipeline didn't produce a dictionary with training results"
                    self.error(msg)
                    raise PluginError(msg, self)

                # training.json, trained models and other artifacts should
                # now be in the artifacts directory. depending on the environment
                # these may be left on disk (SDK) or stored in cloud (APIs)
                artifacts_path = self.factory.get_artifacts_directory()
                assert (
                    len(os.listdir(artifacts_path)) >= 1
                ), "RecipePipelinePlugin - should produce at least one file artifact when training"
                return results

            # when predicting pass the data to the recipe for prediction
            if analitico.constants.ACTION_PREDICT in action:

                # read training information from disk
                artifacts_path = self.factory.get_artifacts_directory()
                training_path = os.path.join(artifacts_path, "training.json")
                assert os.path.isfile(training_path)

                # normally prediction input is a pd.DataFrame
                if isinstance(args[0], pd.DataFrame):
                    df_original = args[0]  # original data
                    df = df_original.copy()  # df processed inplace

                    predictions = super().run(df, action=action, **kwargs)
                    predictions["records"] = analitico.pandas.pd_to_dict(df_original)
                    predictions["processed"] = analitico.pandas.pd_to_dict(df)
                else:
                    # run the pipeline with generic input return predictions
                    predictions = super().run(*args, action=action, **kwargs)

                return predictions

        except Exception as exc:
            self.exception("RecipePipelinePlugin - error during %s", action, exception=exc)
