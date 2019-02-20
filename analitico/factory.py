import tempfile
import shutil
import os.path
import urllib.parse
import requests
import json
import re

from abc import ABC, abstractmethod
from urllib.parse import urlparse

import analitico.utilities
from analitico.ifactory import IFactory
from analitico.mixin import AttributeMixin
from analitico.plugin import PluginError
from analitico.dataset import Dataset

# We need to import this library here even though we don't
# use it directly below because we are instantiating the
# plugins by name from globals() and they won't be found if
# this import is not here.
import analitico.plugin


class Factory(IFactory):
    """ A factory for analitico objects implemented via API endpoint calls """

    # Authorization token to be used when calling analitico APIs
    token = None

    # APIs endpoint, eg: https://analitico.ai/api/
    endpoint = None

    def __init__(self, token=None, endpoint=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if token:
            assert token.startswith("tok_")
            self.token = token
        if endpoint:
            assert endpoint.startswith("http")
            self.endpoint = endpoint

    ##
    ## Temp and cache directories
    ##

    # Temporary directory used during plugin execution
    _temporary_directory = None

    def get_temporary_directory(self, prefix=None):
        """ Temporary directory that can be used while a plugin runs and is deleted afterwards """
        if self._temporary_directory is None:
            self._temporary_directory = tempfile.mkdtemp(prefix=prefix)
        return self._temporary_directory

    def get_artifacts_directory(self):
        """ Plugins save artifacts in subdirectory of temp """
        artifacts = os.path.join(self.get_temporary_directory(), "artifacts")
        if not os.path.isdir(artifacts):
            os.mkdir(artifacts)
        return artifacts

    def get_cache_directory(self):
        """ Returns directory to be used for caches """
        cache_path = os.path.join(tempfile.gettempdir(), "analitico_cache")
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
        assert url and isinstance(url, str)
        # temporarily while all internal urls are updated to analitico://
        if url.startswith("workspaces/ws_"):
            url = analitico.ANALITICO_URL_PREFIX + url

        # see if assets uses analitico://workspaces/... scheme
        if url.startswith("analitico://"):
            if not self.endpoint:
                raise PluginError("Factory is not configured with a valid API endpoint and cannot get: " + url)
            url = self.endpoint + url[len("analitico://") :]
        return url

    def get_url_stream(self, url):
        """
        Returns a stream to the given url. This works for regular http:// or https://
        and also works for analitico:// assets which are converted to calls to the given
        endpoint with proper authorization tokens. The stream is returned as an iterator.
        """
        assert url and isinstance(url, str)
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
        assert url and isinstance(url, str)
        url_stream = self.get_url_stream(url)
        with tempfile.NamedTemporaryFile(encoding="utf-8") as tf:
            for b in url_stream:
                tf.write(b)
            tf.seek(0)
            return json.load(tf)

    ##
    ## Plugins
    ##

    def _get_class_from_fully_qualified_name(self, name, module=None, globals=globals()):
        """ Gets a class from its fully qualified name, eg: package.module.Class """
        assert name and isinstance(name, str)
        try:
            if name:
                split = name.split(".")
                if len(split) > 1:
                    prefix = split[0]
                    name = name[len(split[0]) + 1 :]
                    module = getattr(module, prefix) if module else globals[prefix]
                    return self._get_class_from_fully_qualified_name(name, module)
                return getattr(module, split[0])
        except Exception as exc:
            pass
        return None

    def get_plugin(self, name: str, globals=globals(), **kwargs):
        """
        Create a plugin given its name and the environment it will run in.
        Any additional parameters passed to this method will be passed to the
        plugin initialization code and will be stored as a plugin setting.
        """
        try:
            assert name and isinstance(name, str)
            klass = self._get_class_from_fully_qualified_name(name, globals=globals)
            if not klass:
                raise analitico.plugin.PluginError("Factory - can't find plugin: " + name)
            return (klass)(factory=self, **kwargs)
        except Exception as exc:
            message = "Factory.get_plugin - can't create " + name
            self.warning(message, exc_info=exc)
            raise PluginError(message)

    ##
    ## Factory methods
    ##

    EMAIL_RE = "[^@]+@[^@]+\.[^@]+"  # very rough check

    def get_item_type(self, item_id):
        """ Returns item class from item id, eg: returns 'dataset' from ds_xxx """
        assert item_id and isinstance(item_id, str)
        if item_id.startswith(analitico.DATASET_PREFIX):
            return analitico.DATASET_TYPE
        if item_id.startswith(analitico.ENDPOINT_PREFIX):
            return analitico.ENDPOINT_TYPE
        if item_id.startswith(analitico.JOB_PREFIX):
            return analitico.JOB_TYPE
        if item_id.startswith(analitico.MODEL_PREFIX):
            return analitico.MODEL_TYPE
        if item_id.startswith(analitico.RECIPE_PREFIX):
            return analitico.RECIPE_TYPE
        if item_id.startswith(analitico.WORKSPACE_PREFIX):
            return analitico.WORKSPACE_TYPE
        if re.match(self.EMAIL_RE, item_id):
            return analitico.USER_TYPE
        self.warning("Factory.get_item_type - couldn't find type for: " + item_id)
        return None

    def get_item(self, item_id):
        """ Retrieves item from the server by item_id """
        assert item_id and isinstance(item_id, str)
        url = "{}/{}s/{}".format(self.endpoint, self.get_item_type(item_id), item_id)
        return self.get_url_json(url)

    @abstractmethod
    def get_dataset(self, dataset_id):
        """ Creates a Dataset object from the cloud dataset with the given id """
        plugin_settings = {
            "type": "analitico/plugin",
            "name": "analitico.plugin.CsvDataframeSourcePlugin",
            "source": {"type": "text/csv", "url": "analitico://datasets/{}/data/csv".format(dataset_id)},
        }
        # Instead of creating a plugin that reads the end product of the dataset
        # pipeline we should consider reading the dataset information from its endpoint,
        # getting the entire plugin chain and recreating it here exactly the same so it
        # can be run in Jupyter with all its plugins, etc.
        plugin = self.get_plugin(**plugin_settings)
        return Dataset(self, plugin=plugin)

    ##
    ## Logging
    ##

    @property
    def logger(self):
        """ A logger that should be used for tracing """
        return analitico.utilities.logger

    ##
    ## with Factory as: lifecycle methods
    ##

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """ Delete any temporary files upon exiting """
        if self._temporary_directory:
            shutil.rmtree(self._temporary_directory, ignore_errors=True)
