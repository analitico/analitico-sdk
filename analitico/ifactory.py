from abc import abstractmethod

from analitico.mixin import AttributeMixin

import analitico


class IFactory(AttributeMixin):
    """ A base abstract class providing runtime services like items and plugin creation, etc """

    @property
    def token(self):
        """ API token used to call endpoint (optional) """
        return self.get_attribute("token")

    @property
    def endpoint(self):
        """ Endpoint used to call analitico APIs """
        return self.get_attribute("endpoint", analitico.ANALITICO_API_ENDPOINT)

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

    ##
    ## URL retrieval, authorization and caching
    ##

    @abstractmethod
    def get_url_stream(self, url):
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

    @abstractmethod
    def get_plugin(self, name: str, **kwargs):
        """ A factory method that creates a plugin from its name and settings (builder pattern) """
        pass

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

    def info(self, format, *args, **kwargs):
        self.logger.info(format, *args, **kwargs)

    def warning(self, format, *args, **kwargs):
        self.logger.warning(format, *args, **kwargs)

    def error(self, format, *args, **kwargs):
        self.logger.error(format, *args, **kwargs)

    ##
    ## with xxx as: lifecycle methods
    ##

    def __enter__(self):
        """ Handle factory setup, eg: create temps, etc """
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """ Handles shutdown of the factory, eg: cleanups """
        pass
