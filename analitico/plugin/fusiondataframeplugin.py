
import pandas as pd
from .interfaces import PluginError, IDataframePlugin, PluginPipelineMixin

##
## FusionDataframePlugin
##


class FusionDataframePlugin(IDataframePlugin, PluginPipelineMixin):
    """ 
    A plugin used to merge two datasources using specific merge rules.
    This plugin is a IDataframePlugin because it takes a dataframe
    (the main table or left table), modifies it (join) and returns it.
    It is also a DataframePipelinePlugin because it can embed a second
    pipeline which generates the secondary, or right, table which is
    merged with the main. Merging is performed based on rules described 
    in the "merge" attribute, which is a dictionary that closely maps
    pandas' merge parameters.
    """

    class Meta(IDataframePlugin.Meta):
        name = "analitico.plugin.FusionDataframePlugin"

    def run(self, *args, action=None, **kwargs) -> pd.DataFrame:
        """ Merge two pipelines into a single dataframe """
        try:
            df_left = args[0]
            if not isinstance(df_left, pd.DataFrame):
                self.exception("Should receive as input a single pd.DataFrame, received: %s", df_left)

            # run the other pipeline to obtain the second table that we're joining on 
            df_right = self.run_pipeline(action=action, **kwargs)
            if not isinstance(df_right, pd.DataFrame):
                self.exception("Plugins pipeline should produce a pd.DataFrame to be merged with main input, instead received: %s", df_right)

            # "merge" attribute contains a dictionary of settings that closely match those of pandas.merge:
            # https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html#database-style-dataframe-or-named-series-joining-merging

            merge = self.get_attribute("merge")
            if not merge:
                self.exception("Attribute 'merge' with merging details is required")

            # "how" determines how we merge
            how = merge.get("how", "inner")
            how_options = ['left', 'right', 'outer', 'inner']
            if how not in how_options:
                self.exception("Attribute how: %s is unknown, should be one of %s", how, str(how_options))            

            on = merge.get("on", None)
            if on:
                self.info("Merge on: %s", on)
                df_fusion = pd.merge(df_left, df_right, on=on, how=how)
            else:
                left_on = merge.get("left_on", None)
                right_on = merge.get("right_on", None)
                if left_on and right_on:
                    self.info("Merge left_on: %s, right_on: %s", left_on, right_on)
                    df_fusion = pd.merge(df_left, df_right, left_on=left_on, right_on=right_on, how=how)
                else:
                    self.exception("You need to specify how to merge dataframes either with the 'on' attribute or with the 'left_on' and 'right_on' attributes indicating the column names")

            self.info("Left columns: %s", df_left.columns)
            self.info("Left has %d rows", len(df_left))
            self.info("Right columns: %s", df_right.columns)
            self.info("Right has %d rows", len(df_right))
            self.info("Fusion columns: %s", df_fusion.columns)
            self.info("Fusion has %d rows", len(df_fusion))

            return df_fusion
        
        except Exception as exc:
            self.exception("Exception while merging dataframes", exception=exc)
