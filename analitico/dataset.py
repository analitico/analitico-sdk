import pandas as pd
import numpy as np

import analitico.mixin
import analitico.plugin

##
## Dataset
##


class Dataset(analitico.mixin.AttributeMixin):
    """ A dataset can retrieve data from a source and process it through a pipeline to generate a dataframe """

    plugin: analitico.plugin.IPlugin = None

    def __init__(self, manager=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "plugin" in kwargs:
            if not manager:
                raise analitico.plugin.PluginError(
                    "Dataset should be initialized with a plugin manager so it can create plugins"
                )
            self.plugin = kwargs["plugin"]
            if isinstance(self.plugin, dict):
                self.plugin = manager.create_plugin(**self.plugin)

    def get_dataframe(self, **kwargs):
        """ Creates a pandas dataframe from the plugin of this dataset (usually a source or pipeline) """
        if self.plugin:
            df = self.plugin.run(**kwargs)
            assert isinstance(df, pd.DataFrame)
            return df
        return None
