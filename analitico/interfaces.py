import os
import os.path
import hashlib
import inspect

from abc import abstractmethod
from .mixin import AttributeMixin


class IFactory(AttributeMixin):
    """ A base abstract class providing runtime services like items and plugin creation, storage, network, etc """

    # dictionary of registered plugins name:class
    __plugins = {}

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

    def get_plugin(self, name: str, scope=None, **kwargs):
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
            print("Plugin: %s registered" % plugin.Meta.name)

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

    @property
    @abstractmethod
    def logger(self):
        """ A logger that should be used for tracing """
        pass

    def info(self, msg, *args, plugin=None, **kwargs):
        if self.logger:
            self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, plugin=None, **kwargs):
        if self.logger:
            self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, plugin=None, exception=None):
        if self.logger:
            self.logger.error(msg, *args, exc_info=exception)

    def exception(self, msg, *args, plugin=None, exception=None):
        msg = msg % (args)
        self.error(msg)
        from analitico.plugin import PluginError

        raise PluginError(msg, plugin=plugin, exception=exception)

    ##
    ## with xxx as: lifecycle methods
    ##

    def __enter__(self):
        """ Handle factory setup, eg: create temps, etc """
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """ Handles shutdown of the factory, eg: cleanups """
        pass
