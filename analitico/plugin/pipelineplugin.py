"""
Plugins that group other plugins into logical groups like
ETL (extract, transform, load) pipeline or a graph used to
process data and create a machine learning model.
"""

import pandas as pd
from analitico.utilities import time_ms
from analitico.schema import pandas_to_analitico_type

from .interfaces import IGroupPlugin, plugin

##
## PipelinePlugin
##


@plugin
class PipelinePlugin(IGroupPlugin):
    """ 
    A plugin that creates a linear workflow by chaining together other plugins.
    Plugins that are chained in a pipeline need to take a single input and have
    a single output of the same kind so they same object can be processed from 
    the first, to the next and down to the last, then returned to caller as if
    the process was just one logical operation. PipelinePlugin can be used to 
    for example to construct ETL (extract, transform, load) workflows.
    """

    class Meta(IGroupPlugin.Meta):
        name = "analitico.plugin.PipelinePlugin"

    def run(self, *args, action=None, **kwargs):
        """ Process plugins in sequence, return combinined chained result """

        pipeline_on = time_ms()
        self.info("%s - processing...", self.Meta.name)
        for p, plugin in enumerate(self.plugins):
            plugin_on = time_ms()
            self.info("%s[%d] - processing...", plugin.Meta.name, p)
            # a plugin can have one or more input parameters and one or more
            # output parameters. results from a call to the next in the chain
            # are passed as tuples. when we finally return, if we have a single
            # result we unpackit, otherwise we return as tuple. this allows
            # a pipeline of plugins to chain plugins with a variable number of
            # parameters. each plugin is responsible for validating the type of
            # its input positional parameters and named parameters.
            args = plugin.run(*args, action=action, **kwargs)
            if not isinstance(args, tuple):
                args = (args,)

            # print out diagnostics showing outputs of pipeline plugin
            self.info("%s[%d] - done in %d ms", plugin.Meta.name, p, time_ms(plugin_on))
            for i, arg in enumerate(args):
                if isinstance(arg, pd.DataFrame):
                    self.info("output[%d]: pd.DataFrame", i)
                    self.info("  rows: %d", len(arg))
                    self.info("  columns: %d", len(arg.columns))
                    for j, column in enumerate(arg.columns):
                        self.info("  %3d %s (%s/%s)", j, column, arg.dtypes[j], pandas_to_analitico_type(arg.dtypes[j]))
                else:
                    self.info("output[%d]: %s", i, str(type(arg)))

        self.info("%s - done in %d ms", self.Meta.name, time_ms(pipeline_on))
        return args if len(args) > 1 else args[0]
