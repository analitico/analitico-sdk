import logging
import collections
import pandas
import tempfile
import os.path
import shutil
import urllib.request
import re
import requests
import json
import tempfile
import multiprocessing
import urllib.parse
import os

from urllib.parse import urlparse
from abc import ABC, abstractmethod

# Design patterns:
# https://github.com/faif/python-patterns

import analitico

from analitico.mixin import AttributeMixin
from analitico.interfaces import IFactory
from analitico.utilities import time_ms, save_json, read_json
from analitico.schema import apply_schema

##
## IPlugin - base class for all plugins
##


class IPlugin(ABC, AttributeMixin):
    """ Abstract base class for Analitico plugins """

    class Meta:
        """ Plugin metadata is exposed in its inner class """

        name = None

    # IFactory that provides runtime services to the plugin (eg: loading assets, etc)
    factory: IFactory = None

    @property
    def name(self):
        assert self.Meta.name
        return self.Meta.name

    def __init__(self, factory: IFactory, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factory = factory

    def activate(self, *args, **kwargs):
        """ Called when the plugin is initially activated """
        pass

    @abstractmethod
    def run(self, action=None, *args, **kwargs):
        """ Run will do in the subclass whatever the plugin does """
        pass

    def deactivate(self, *args, **kwargs):
        """ Called before the plugin is deactivated and finalized """
        pass

    def __str__(self):
        return self.name

    # Logging

    @property
    def logger(self):
        """ Logger that can be used by the plugin to communicate errors, etc with host """
        return self.factory.logger

    def info(self, format, *args):
        self.logger.info(format, *args)

    def warning(self, format, *args):
        self.logger.warning(format, *args)

    def error(self, format, *args):
        self.logger.error(format, *args)


##
## IDataframeSourcePlugin - base class for plugins that create dataframes
##


class IDataframeSourcePlugin(IPlugin):
    """ A plugin that creates a pandas dataframe from a source (eg: csv file, sql query, etc) """

    class Meta(IPlugin.Meta):
        inputs = None
        outputs = [{"name": "dataframe", "type": "pandas.DataFrame"}]

    @abstractmethod
    def run(self, *args, action=None, **kwargs):
        """ Run creates a dataset from the source and returns it """
        pass


##
## IDataframePlugin - base class for plugins that manipulate pandas dataframes
##


class IDataframePlugin(IPlugin):
    """
    A plugin that takes a pandas dataframe as input,
    manipulates it and returns a pandas dataframe
    """

    class Meta(IPlugin.Meta):
        inputs = [{"name": "dataframe", "type": "pandas.DataFrame"}]
        outputs = [{"name": "dataframe", "type": "pandas.DataFrame"}]

    def run(self, *args, action=None, **kwargs) -> pandas.DataFrame:
        assert isinstance(args[0], pandas.DataFrame)
        return args[0]


##
## IAlgorithmPlugin - base class for machine learning algorithms that produce trained models
##


class IAlgorithmPlugin(IPlugin):
    """ An algorithm used to create machine learning models from training data """

    class Meta(IPlugin.Meta):
        inputs = [{"name": "train", "type": "pandas.DataFrame"}, {"name": "test", "type": "pandas.DataFrame|none"}]
        outputs = [{"name": "model", "type": "dict"}]

    def _run_train(self, *args, **kwargs):
        """ 
        When an algorithm runs it always takes in a dataframe with training data,
        it may optionally have a dataframe of validation data and will return a dictionary
        with information on the trained model plus a number of artifacts.
        """
        assert isinstance(args[0], pandas.DataFrame)
        started_on = time_ms()
        results = collections.OrderedDict(
            {
                "type": "analitico/training",
                "plugins": {
                    "training": self.Meta.name,  # plugin used to train model
                    "prediction": self.Meta.name,  # plugin to be used for predictions (usually the same)
                },
                "data": {},  # number of records, etc
                "parameters": {},  # model parameters, hyperparameters
                "scores": {},  # training scores
                "performance": {"cpu_count": multiprocessing.cpu_count()},  # time elapsed, cpu, gpu, memory, disk, etc
            }
        )

        train = args[0]
        test = args[1] if len(args) > 1 else None
        results = self.train(train, test, results, *args, **kwargs)

        # finalize results and save as training.json
        results["performance"]["total_ms"] = time_ms(started_on)
        artifacts_path = self.factory.get_artifacts_directory()
        results_path = os.path.join(artifacts_path, "training.json")
        save_json(results, results_path)
        self.info("saved %s (%d bytes)", results_path, os.path.getsize(results_path))
        return results

    def _run_predict(self, *args, **kwargs):
        """ 
        When an algorithm runs it always takes in a dataframe with training data,
        it may optionally have a dataframe of validation data and will return a dictionary
        with information on the trained model plus a number of artifacts.
        """
        assert isinstance(args[0], pandas.DataFrame)
        data = args[0]

        artifacts_path = self.factory.get_artifacts_directory()
        training = read_json(os.path.join(artifacts_path, "training.json"))
        assert training

        started_on = time_ms()
        results = collections.OrderedDict(
            {
                "type": "analitico/prediction",
                "records": None,  # data may be returned along with predictions
                "predictions": None,  # predictions
                "performance": {"cpu_count": multiprocessing.cpu_count()},  # time elapsed, cpu, gpu, memory, disk, etc
            }
        )

        # force schema like in training data
        schema = training["data"]["schema"]
        data = apply_schema(data, schema)

        # load model, calculate predictions
        results = self.predict(data, training, results, *args, **kwargs)
        results["performance"]["total_ms"] = time_ms(started_on)
        return results

    def run(self, *args, action=None, **kwargs):
        """ Algorithm can run to train a model or to predict from a trained model """
        if action.endswith("/train"):
            return self._run_train(*args, **kwargs)
        if action.endswith("/predict"):
            return self._run_predict(*args, **kwargs)
        self.error("unknown action: %s", action)
        raise PluginError("IAlgorithmPlugin - action should be /train or /predict")

    @abstractmethod
    def train(self, train, test, results, *args, **kwargs):
        """ Train with algorithm and given data to produce a trained model """
        pass

    @abstractmethod
    def predict(self, data, training, results, *args, **kwargs):
        """ Return predictions from trained model """
        pass


##
## IGroupPlugin
##


class IGroupPlugin(IPlugin):
    """ 
    A composite plugin that joins multiple plugins into a functional block,
    for example a processing pipeline made of plugins or a graph workflow. 
    
    *References:
    https://en.wikipedia.org/wiki/Composite_pattern
    https://infinitescript.com/2014/10/the-23-gang-of-three-design-patterns/
    """

    plugins = []

    def __init__(self, factory: IFactory, plugins, *args, **kwargs):
        """ Initialize group and create all this plugin's children """
        super().__init__(factory=factory, *args, **kwargs)
        self.plugins = []
        for plugin in plugins:
            if isinstance(plugin, dict):
                plugin = self.factory.get_plugin(**plugin)
            self.plugins.append(plugin)


##
## PluginError
##


class PluginError(Exception):
    """ Exception generated by a plugin; carries plugin info, inner exception """

    # Plugin error message
    message: str = None

    # Plugin that generated this error (may not be defined)
    plugin: IPlugin = None

    def __init__(self, message, plugin: IPlugin = None, exception: Exception = None):
        super().__init__(message, exception)
        self.message = message
        if plugin:
            self.plugin = plugin
            plugin.logger.error(message)

    def __str__(self):
        if self.plugin:
            return self.plugin.name + ": " + self.message
        return self.message
