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

from analitico.mixin import AttributeMixin
from analitico.utilities import time_ms, save_json, read_json
from analitico.schema import apply_schema

##
## IPluginManager
##


class IPluginManager(ABC, AttributeMixin):
    """ A base abstract class for a plugin lifecycle manager and runtime environment """

    # Authorization token to be used when calling analitico APIs
    token = None

    # APIs endpoint, eg: https://analitico.ai/api/
    endpoint = None

    # Temporary directory used during plugin execution
    _temporary_directory = None

    def __init__(self, token=None, endpoint=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if token:
            assert token.startswith("tok_")
            self.token = token
        if endpoint:
            assert endpoint.startswith("http")
            self.endpoint = endpoint

    ##
    ## Plugins
    ##

    @abstractmethod
    def create_plugin(self, name: str, **kwargs):
        """ A factory method that creates a plugin from its name and settings (builder pattern) """
        pass

    def get_temporary_directory(self):
        """ Temporary directory that can be used while a plugin runs and is deleted afterwards """
        if self._temporary_directory is None:
            self._temporary_directory = tempfile.mkdtemp()
        return self._temporary_directory

    def get_artifacts_directory(self):
        """ 
        An plugin can produce various file artifacts during execution and place
        them in this directory (datasets, statistics, models, etc). If the execution 
        is completed succesfully, a subclass of IPluginManager may persist this 
        information to storage, etc. A file, eg: data.csv, can have a "sister" file
        data.csv.info that contains json metadata (eg: a model may have a sister
        file containing the model's training time, stats, etc).
        """
        artifacts = os.path.join(self.get_temporary_directory(), "artifacts")
        if not os.path.isdir(artifacts):
            os.mkdir(artifacts)
        return artifacts

    def get_cache_directory(self):
        """ Returns directory to be used for caches """
        cache_path = os.path.join(tempfile.tempdir, "analitico_cache")
        if not os.path.isdir(cache_path):
            os.mkdir(cache_path)
        return cache_path

    ##
    ## URL retrieval, authorization and caching
    ##

    # regular expression used to detect assets using analitico:// scheme
    ANALITICO_ASSET_RE = r"(analitico://workspaces/(?P<workspace_id>[-\w.]{4,256})/)"

    def get_url(self, url) -> str:
        """
        If the url uses the analitico:// scheme for assets stored on the cloud
        service, it will convert the url to a regular https:// scheme.
        If the url points to an analitico API call, the request will have the
        ?token= authorization token header added to it.
        """
        # temporarily while all internal urls are updated to analitico://
        if url.startswith("workspaces/ws_"):
            url = ANALITICO_PREFIX + url

        # see if assets uses analitico://workspaces/... scheme
        if url.startswith("analitico://"):
            if not self.endpoint:
                raise PluginError(
                    "Plugin manager was not been configured with an API endpoint therefore it cannot process: " + url
                )
            url = self.endpoint + url[len("analitico://") :]
        return url

    def get_url_stream(self, url):
        """
        Returns a stream to the given url. This works for regular http:// or https://
        and also works for analitico:// assets which are converted to calls to the given
        endpoint with proper authorization tokens. The stream is returned as an iterator.
        """
        url = self.get_url(url)
        try:
            url_parse = urlparse(url)
        except Exception as exc:
            pass
        if url_parse and url_parse.scheme in ("http", "https"):
            headers = {}
            if url_parse.hostname and url_parse.hostname.endswith("analitico.ai") and self.token:
                # if url is connecting to analitico.ai add token
                headers = {"Authorization": "Bearer " + self.token}
            response = requests.get(url, stream=True, headers=headers)
            return response.raw
        return open(url, "rb")

    def get_url_json(self, url):
        url_stream = self.get_url_stream(url)
        with tempfile.NamedTemporaryFile() as tf:
            for b in url_stream:
                tf.write(b)
            tf.seek(0)
            return json.load(tf)

    ##
    ## Factory methods
    ##

    @abstractmethod
    def get_dataset(self, dataset_id):
        return None

    ##
    ## with IPluginManager as lifecycle methods
    ##

    def __enter__(self):
        # setup
        return self

    def __exit__(self, type, value, traceback):
        """ Delete any temporary files upon exiting """
        if self._temporary_directory:
            shutil.rmtree(self._temporary_directory, ignore_errors=True)


##
## IPlugin - base class for all plugins
##


class IPlugin(ABC, AttributeMixin):
    """ Abstract base class for Analitico plugins """

    class Meta:
        """ Plugin metadata is exposed in its inner class """

        name = None

    # Manager that provides environment and lifecycle services
    manager: IPluginManager = None

    @property
    def name(self):
        assert self.Meta.name
        return self.Meta.name

    def __init__(self, manager: IPluginManager, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.manager = manager

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
        return logging.getLogger(self.name)

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
    def run(self, action=None, *args, **kwargs):
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

    def run(self, action=None, *args, **kwargs) -> pandas.DataFrame:
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
        artifacts_path = self.manager.get_artifacts_directory()
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

        artifacts_path = self.manager.get_artifacts_directory()
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

    def run(self, action, *args, **kwargs):
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

    def __init__(self, manager: IPluginManager, plugins, *args, **kwargs):
        """ Initialize group and create all this plugin's children """
        super().__init__(manager=manager, *args, **kwargs)
        self.plugins = []
        for plugin in plugins:
            if isinstance(plugin, dict):
                plugin = self.manager.create_plugin(**plugin)
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
