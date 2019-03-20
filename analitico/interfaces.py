import os
import os.path
import hashlib
import inspect
import logging

from abc import abstractmethod
from .mixin import AttributeMixin
from .exceptions import AnaliticoException
from .status import STATUS_ALL, STATUS_FAILED


class IFactory(AttributeMixin):
    """ A base abstract class providing runtime services like items and plugin creation, storage, network, etc """

    # dictionary of registered plugins name:class
    __plugins = {}

    ##
    ## Factory context
    ##

    @property
    def workspace(self):
        """ Workspace context in which this factory runs (optional) """
        return self.get_attribute("workspace")

    @property
    def token(self):
        """ API token used to call endpoint (optional) """
        return self.get_attribute("token")

    @property
    def endpoint(self):
        """ Endpoint used to call analitico APIs """
        return self.get_attribute("endpoint")

    @property
    def request(self):
        """ Request used as context when running on the server or running async jobs (optional) """
        return self.get_attribute("request")

    @property
    def job(self):
        """ Job running on the server (optional) """
        return self.get_attribute("job")

    ##
    ## Temp and cache directories
    ##

    @abstractmethod
    def get_temporary_directory(self, prefix=None):
        """ Temporary directory (concrete class may handle cleanups, etc) """
        pass

    @abstractmethod
    def get_artifacts_directory(self):
        """ 
        A plugin can produce various file artifacts during execution and place
        them in this directory (datasets, statistics, models, etc). 
        If the execution is completed succesfully, a subclass may persist this 
        information to storage, etc. A file, eg: data.csv, can have a "sister" file
        data.csv.info that contains json metadata (eg: a model may have a sister
        file containing the model's training time, stats, etc).
        """
        pass

    @abstractmethod
    def get_cache_directory(self):
        """ Returns directory to be used for caches """
        pass

    def get_cache_filename(self, unique_id):
        """ Returns the fullpath in cache for an item with the given unique_id (eg: a unique url, an md5 or etag, etc) """
        return os.path.join(self.get_cache_directory(), "cache_" + hashlib.sha256(unique_id.encode()).hexdigest())

    ##
    ## URL retrieval, authorization and caching
    ##

    @abstractmethod
    def get_url_stream(self, url, binary=False):
        """
        Returns a stream to the given url. This works for regular http:// or https://
        and also works for analitico:// assets which are converted to calls to the given
        endpoint with proper authorization tokens. The stream is returned as an iterator.
        """
        pass

    @abstractmethod
    def get_url_json(self, url):
        """ Returns json content of given URL """
        pass

    ##
    ## Plugins
    ##

    def get_plugin(self, name: str, **kwargs):
        """
        Create a plugin given its name and the environment it will run in.
        Any additional parameters passed to this method will be passed to the
        plugin initialization code and will be stored as a plugin setting.
        """
        try:
            # deprecated, temporary retrocompatibility 2019-02-24
            if name == "analitico.plugin.AugmentDatesDataframePlugin":
                name = "analitico.plugin.AugmentDatesPlugin"
            if name not in IFactory.__plugins:
                self.exception("IFactory.get_plugin - %s is not a registered plugin", name)
            return (IFactory.__plugins[name])(factory=self, **kwargs)
        except Exception as exc:
            self.exception("IFactory.get_plugin - error while creating " + name, exception=exc)

    def run_plugin(self, *args, settings, **kwargs):
        """ 
        Runs a plugin and returns its results. Takes a number of positional and named arguments
        which are passed to the plugin for execution and a dictionary of settings used to create
        the plugin. If settings are passed as an array, the method will create a pipeline plugin
        which will execute the plugins in a chain.
        """
        if isinstance(settings, list):
            settings = {"name": "analitico.plugin.PipelinePlugin", "plugins": settings}
        plugin = self.get_plugin(**settings)
        return plugin.run(*args, **kwargs)

    def get_plugins(self):
        """ Returns a list of registered plugin classes """
        return IFactory.__plugins

    @staticmethod
    def register_plugin(plugin):
        if inspect.isabstract(plugin):
            print("IFactory.register_plugin: %s is abstract and cannot be registered" % plugin.Meta.name)
            return
        if not plugin.Meta.name in IFactory.__plugins:
            IFactory.__plugins[plugin.Meta.name] = plugin
            # print("Plugin: %s registered" % plugin.Meta.name)

    def DRAFTrun_plugin(self, *args, action=None, **kwargs):
        try:
            plugin = self.get_plugin(**kwargs)
            return plugin.run(*args, action)
        except Exception as e:
            self.exception("An error occoured while running plugin")

    ##
    ## Factory methods
    ##

    @abstractmethod
    def get_item_type(self, item_id):
        """ Returns type of item from its id, eg: ds_xxx returns 'dataset', rx_xxx returns 'recipe', etc """
        pass

    @abstractmethod
    def get_item(self, item_id):
        pass

    def get_dataset(self, dataset_id):
        """ Returns dataset object """
        return self.get_item(dataset_id)

    ##
    ## Logging
    ##

    class LogAdapter(logging.LoggerAdapter):
        """ A simple adapter which will call "process" on every log record to enrich it with contextual information from the IFactory. """

        factory = None

        def __init__(self, logger, factory):
            super().__init__(logger, {})
            self.factory = factory

        def process(self, msg, kwargs):
            return self.factory.process_log(msg, kwargs)

    def process_log(self, msg, kwargs):
        """ Moves any kwargs other than 'exc_info' and 'extra' to 'extra' dictionary. """
        if "extra" not in kwargs:
            kwargs["extra"] = {}

        extra = kwargs["extra"]
        for key, value in kwargs.copy().items():
            if key not in ("exc_info", "extra"):
                extra[key] = kwargs.pop(key)

        for attr_name in ("workspace", "token", "endpoint", "request", "job"):
            attr = self.get_attribute(attr_name, None)
            if attr:
                extra[attr_name] = attr
        return msg, kwargs

    def set_logger_level(self, level):
        """ Sets logger level to given level for all future log calls make through the factory logger """
        self.set_attribute("logger_level", level)

    def get_logger(self, name="analitico"):
        """ Returns logger wrapped into an adapter that adds contextual information from the IFactory """
        logger_level = self.get_attribute("logger_level", logging.INFO)
        logger = IFactory.LogAdapter(logging.getLogger("analitico"), self)
        logger.setLevel(logger_level)
        return logger

    @property
    def logger(self):
        """ Returns logger wrapped into an adapter that adds contextual information from the IFactory """
        return self.get_logger()

    def debug(self, msg, *args, **kwargs):
        self.logger.log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.log(logging.ERROR, msg, *args, **kwargs)

    def status(self, item, status, **kwargs):
        """ Updates on the status of an item. Status is one of: created, running, canceled, completed or failed. """
        level = logging.ERROR if status == STATUS_FAILED else logging.INFO
        self.logger.log(level, "status: %s, name: %s", status, type(item).__name__, item=item, status=status, **kwargs)

    def exception(self, msg, *args, **kwargs):
        message = msg % (args)
        self.error(msg, *args, **kwargs)
        exception = kwargs.get("exception", None)
        if exception:
            raise AnaliticoException(message, **kwargs) from exception
        raise AnaliticoException(message, **kwargs)

    ##
    ## with xxx as: lifecycle methods
    ##

    def __enter__(self):
        """ Handle factory setup, eg: create temps, etc """
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """ Handles shutdown of the factory, eg: cleanups """
        pass
