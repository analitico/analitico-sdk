import tempfile
import os
import os.path
import requests
import json
import re
import logging
import hashlib
import inspect
import urllib.parse
import io
import pandas as pd
import tempfile

from .mixin import AttributeMixin
from .exceptions import AnaliticoException
from .status import STATUS_FAILED

import analitico.utilities
import analitico.models

from analitico.utilities import id_generator
from analitico.models import Workspace, Item, Dataset, Recipe, Notebook

import logging

logger = logging.getLogger("analitico")

# read http streams in chunks
HTTP_BUFFER_SIZE = 32 * 1024 * 1024  # 32 MiBs

##
## Models used in the SDK
##


class AnaliticoSDK(AttributeMixin):
    """ An SDK for analitico.ai/api. """

    def __init__(self, token=None, endpoint=None, workspace_id: str = None, **kwargs):
        super().__init__(**kwargs)
        if token:
            assert token.startswith("tok_")
            self.set_attribute("token", token)
        if endpoint:
            assert endpoint.startswith("http")
            self.set_attribute("endpoint", endpoint)

        # set default workspace
        if workspace_id:
            self.workspace = self.get_workspace(workspace_id)

        # use current working directory at the time when the factory
        # is created so that the caller can setup a temp directory we
        # should work in
        self._artifacts_directory = os.getcwd()

    ##
    ## Properties and factory context
    ##

    # default workspace
    _workspace: Workspace = None

    @property
    def workspaces(self) -> [Workspace]:
        """ Returns your workspaces. """
        return self.get_items(analitico.WORKSPACE_TYPE)

    @property
    def workspace(self):
        """ Get the default workspace used by the SDK (for example when creating items). """
        return self._workspace

    @workspace.setter
    def workspace(self, workspace: Workspace):
        self._workspace = workspace

    @property
    def token(self):
        """ API token used to call endpoint (optional) """
        return self.get_attribute("token")

    @property
    def endpoint(self):
        """ Endpoint used to call analitico APIs """
        return self.get_attribute("endpoint")

    ##
    ## Temp and cache directories
    ##

    # Temporary directory which is deleted when factory is disposed
    _temp_directory = None

    # Artifacts end up in the current working directory
    _artifacts_directory = None

    def get_temporary_directory(self):
        """ Temporary directory that can be used while a factory is used and deleted afterwards """
        temp_dir = os.path.join(tempfile.gettempdir(), "analitico_temp")
        if not os.path.isdir(temp_dir):
            os.mkdir(temp_dir)
        return temp_dir

    def get_artifacts_directory(self):
        """ 
        A plugin or notebook can produce various file artifacts during execution and place
        them in this directory (datasets, statistics, models, etc). A subclass, for example
        a factory used to run pipelines on the server, may persist files created here to cloud, etc.
        """
        return self._artifacts_directory

    def get_cache_directory(self):
        """ Returns directory to be used for caches """
        cache_dir = os.path.join(tempfile.gettempdir(), "analitico_cache")
        if not os.path.isdir(cache_dir):
            os.mkdir(cache_dir)
        return cache_dir

    def get_cache_filename(self, unique_id):
        """ Returns the fullpath in cache for an item with the given unique_id (eg: a unique url, an md5 or etag, etc) """
        # Tip: if cache contents need to be invalidated for whatever reason, you can change the prefix below...
        return os.path.join(self.get_cache_directory(), "cache_v2_" + hashlib.sha256(unique_id.encode()).hexdigest())

    ##
    ## Internal Utilities
    ##

    # regular expression used to detect assets using analitico:// scheme
    ANALITICO_ASSET_RE = r"(analitico://workspaces/(?P<workspace_id>[-\w.]{4,256})/)"

    EMAIL_RE = r"[^@]+@[^@]+\.[^@]+"  # very rough check

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
        if item_id.startswith(analitico.NOTEBOOK_PREFIX):
            return analitico.NOTEBOOK_TYPE
        if item_id.startswith(analitico.PLUGIN_PREFIX):
            return analitico.PLUGIN_TYPE
        if item_id.startswith(analitico.RECIPE_PREFIX):
            return analitico.RECIPE_TYPE
        if item_id.startswith(analitico.WORKER_PREFIX):
            return analitico.WORKER_TYPE
        if item_id.startswith(analitico.WORKSPACE_PREFIX):
            return analitico.WORKSPACE_TYPE
        if re.match(self.EMAIL_RE, item_id):
            return analitico.USER_TYPE
        self.warning("Factory.get_item_type - couldn't find type for: " + item_id)
        return None

    # TODO could check if it would be possible to switch to an external caching library, eg:
    # https://github.com/ionrock/cachecontrol

    def get_cached_stream(self, stream, unique_id):
        """ Will cache a stream on disk based on a unique_id (like md5 or etag) and return file stream and filename """
        cache_file = self.get_cache_filename(unique_id)
        if not os.path.isfile(cache_file):
            # if not cached already, download and cache
            cache_temp_file = cache_file + ".tmp_" + id_generator()

            with open(cache_temp_file, "wb") as f:
                if hasattr(stream, "read"):
                    for chunk in iter(lambda: stream.read(HTTP_BUFFER_SIZE), b""):
                        f.write(chunk)
                else:
                    for b in stream:
                        f.write(b)

            # TODO add progress bar for slow downloads https://github.com/tqdm/tqdm#iterable-based

            os.rename(cache_temp_file, cache_file)
        # return stream from cached file
        return open(cache_file, "rb"), cache_file

    def get_url_headers(self, url: str) -> (str, dict):
        # If the url uses the analitico:// scheme for assets stored on the cloud
        # service, it will convert the url to a regular https:// scheme.
        # If the url points to an analitico API call, the request will have the
        # ?token= authorization token header added to it.
        # temporarily while all internal urls are updated to analitico://
        assert url and isinstance(url, str)
        if url.startswith("workspaces/ws_"):
            url = analitico.ANALITICO_URL_PREFIX + url
        # see if assets uses analitico://workspaces/... scheme
        if url.startswith("analitico://"):
            if not self.endpoint:
                raise AnaliticoException(
                    f"Analitico SDK is not configured with a valid API endpoint and cannot get {url}"
                )
            url = self.endpoint + url[len("analitico://") :]

        try:
            url_parse = urllib.parse.urlparse(url)
        except Exception:
            pass
        headers = {}
        if url_parse and url_parse.scheme in ("http", "https"):
            if url_parse.hostname and url_parse.hostname.endswith("analitico.ai") and self.token:
                # if url is connecting to analitico.ai add token
                headers = {"Authorization": "Bearer " + self.token}
        return url, headers

    def get_url_stream(
        self,
        url: str,
        data: dict = None,
        json: dict = None,
        files: dict = None,
        binary: bool = False,
        cache: bool = True,
        method: str = "GET",
        status_code: int = 200,
    ):
        """
        Returns a stream to the given url. This works for regular http:// or https://
        and also works for analitico:// assets which are converted to calls to the given
        endpoint with proper authorization tokens. The stream is returned as an iterator.
        """
        url, headers = self.get_url_headers(url)
        # we should not take the raw response stream here as it could be gzipped or encoded.
        # we take the decoded content as a text string and turn it into a stream or we take the
        # decompressed binary content and also turn it into a stream.
        response = requests.request(method, url, data=data, files=files, stream=True, headers=headers)
        if status_code and response.status_code != status_code:
            msg = f"The response from {url} should have been {status_code} but instead it is {response.status_code}."
            raise AnaliticoException(msg)
        # always treat content as binary, utf-8 encoding is done by readers
        response_stream = io.BytesIO(response.content)
        return response_stream

    def get_url_json(self, url: str, json: dict = None, method: str = "GET", status_code: int = 200) -> dict:
        """
        Get a json response from given url. If the url starts with analitico:// it will be
        substituted with the url of the actual endpoint for analitico.ai and a bearer token
        will be attached for authorization.
        
        Arguments:
            url {str} -- An absolute url to be read or an analitico:// url for API calls.
            data {dict} -- An optional dictionary that should be sent (eg: for POST calls).
        
        Keyword Arguments:
            method {str} -- HTTP method to be used (default: {"get"})
        
        Returns:
            dict -- The json response.
        """
        url, headers = self.get_url_headers(url)

        response = requests.request(method, url, headers=headers, json=json)
        if status_code and response.status_code != status_code:
            msg = f"The response from {url} should have been {status_code} but instead it is {response.status_code}."
            raise AnaliticoException(msg)
        try:
            return response.json()
        except Exception:
            return None

    ##
    ## with Factory as: lifecycle methods
    ##

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """ Leave any temporary files upon exiting """
        pass

    ##
    ## SDK v1 methods
    ##

    def create_item(self, item_type: str, workspace: Workspace = None, **kwargs) -> Item:
        if not workspace:
            workspace = self.get_workspace()

        data = {"data": kwargs}
        data["data"]["workspace_id"] = workspace.id
        if "type" in kwargs:
            kwargs.pop("type")
        if "id" in kwargs:
            data["id"] = kwargs.pop("id")

        json = self.get_url_json(f"analitico://{item_type}s", json=data, method="POST", status_code=201)
        assert "data" in json
        return analitico.models.models_factory(self, json["data"])

    def create_dataset(self, workspace: Workspace = None, **kwargs) -> Dataset:
        return self.create_item(workspace=workspace, item_type=analitico.DATASET_TYPE, **kwargs)

    def create_recipe(self, workspace: Workspace = None, **kwargs) -> Recipe:
        return self.create_item(workspace=workspace, item_type=analitico.RECIPE_TYPE, **kwargs)

    def create_notebook(self, workspace: Workspace = None, **kwargs) -> Notebook:
        return self.create_item(workspace=workspace, item_type=analitico.NOTEBOOK_TYPE, **kwargs)

    ##
    ## Retrieve specific items by id
    ##

    def get_items(self, item_type: str) -> Item:
        """ Retrieves item from the server by item_id """
        json = self.get_url_json(f"analitico://{item_type}s")
        items = []
        for item_data in json["data"]:
            items.append(analitico.models.models_factory(self, item_data))
        return items

    def get_item(self, item_id: str) -> Item:
        """ Retrieves item from the server by item_id """
        json = self.get_url_json(f"analitico://{self.get_item_type(item_id)}s/{item_id}")
        assert "data" in json
        return analitico.models.models_factory(self, json["data"])

    def get_workspace(self, workspace_id: str = None) -> Workspace:
        if not workspace_id:
            if not self.workspace:
                workspaces = self.get_items(analitico.WORKSPACE_TYPE)
                if len(workspaces) != 1:
                    raise AnaliticoException("You do not have a default workspace, please assign sdk.workspace")
                self.workspace = workspaces[0]
            return self.workspace

        workspace = self.get_item(workspace_id)
        assert isinstance(workspace, Workspace)
        return workspace

    def get_dataset(self, dataset_id) -> Dataset:
        dataset = self.get_item(dataset_id)
        assert isinstance(dataset, Dataset)
        return dataset

    def get_recipe(self, recipe_id) -> Recipe:
        recipe = self.get_item(recipe_id)
        assert isinstance(recipe, Recipe)
        return recipe

    def get_notebook(self, notebook_id) -> Notebook:
        notebook = self.get_item(notebook_id)
        assert isinstance(notebook, Notebook)
        return notebook
