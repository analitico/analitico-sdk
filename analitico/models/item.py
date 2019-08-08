import os
import pandas as pd
import tempfile
import urllib
import requests
import base64
import io

from analitico import AnaliticoException, logger
from analitico.mixin import AttributeMixin
from analitico.utilities import save_text, subprocess_run
from analitico.constants import CSV_SUFFIXES, PARQUET_SUFFIXES
from analitico.pandas import pd_read_csv

from collections import OrderedDict
from pathlib import Path

DF_SUPPORTED_FORMATS = (".parquet", ".csv")


class Item(AttributeMixin):
    """ Base class for items like datasets, recipes and notebooks on Analitico. """

    # SDK used to communicate with the service
    sdk = None

    def __init__(self, sdk, item_data: dict):
        # item's basics
        self.sdk = sdk
        self.id = item_data["id"]
        self.type = item_data["type"]
        # big bag of open ended properties to be retrieved via get_attribute
        super().__init__(**item_data["attributes"])

    ##
    ## Properties
    ##

    @property
    def title(self) -> str:
        """ The title of this item. """
        return self.get_attribute("title")

    @title.setter
    def title(self, title: str):
        """ Set the title of this item. """
        self.set_attribute("title", title)

    @property
    def url(self):
        """ Url of this item on the service. """
        item_url = f"analitico://{self.sdk.get_item_type(self.id)}s/{self.id}"
        item_url, _ = self.sdk.get_url_headers(item_url)
        return item_url

    ##
    ## Methods
    ##

    def upload(self, filepath: str = None, df: pd.DataFrame = None, remotepath: str = None) -> bool:

        if isinstance(df, pd.DataFrame):
            if not remotepath:
                remotepath = filepath if filepath else "data.parquet"

            # encode dataframe to disk temporarily
            suffix = Path(remotepath).suffix
            with tempfile.NamedTemporaryFile(mode="w+", prefix="df_", suffix=suffix) as f:
                if suffix in PARQUET_SUFFIXES:
                    df.to_parquet(f.name)
                elif suffix in CSV_SUFFIXES:
                    df.to_csv(f.name)
                else:
                    msg = f"{remotepath} is not in a supported format."
                    raise AnaliticoException(msg, status_code=400)
                return self.upload(filepath=f.name, remotepath=remotepath)

        # need to specify a file, directory or Path that should be uploaded to this item's storage
        if not isinstance(filepath, str) and not isinstance(filepath, Path):
            raise AnaliticoException("Please provide a path to the file or files to be uploaded.")

        if not remotepath:
            remotepath = filepath

        # uploading a single file?
        if os.path.isfile(filepath):
            # retrieve workspace associated with this item
            workspace_id = self.get_attribute("workspace_id")
            workspace = self.sdk.get_workspace(workspace_id)

            # storage configuration for this workspace
            storage = workspace.get_attribute("storage")
            assert storage["driver"] == "hetzner-webdav"
            username = storage["credentials"]["username"]
            password = storage["credentials"]["password"]

            if True:
                # TODO move driver to public repo and use mkdirs
                # TODO supports only files in base directory, should support recursive
                remote_url = f"{storage['url']}/{self.sdk.get_item_type(self.id)}s/"
                response = requests.request("MKCOL", remote_url, auth=(username, password))
                remote_url += f"{self.id}/"
                response = requests.request("MKCOL", remote_url, auth=(username, password))

                with open(filepath, "rb") as f:
                    remote_url += str(remotepath)
                    try:
                        response = requests.put(remote_url, data=f, auth=(username, password))
                    except Exception as exc:
                        logger.error(f"upload - cannot upload to {remote_url} because: {exc}")
                        raise exc
                    if response.status_code not in (200, 201, 204):
                        msg = f"Uploaded to {remote_url} and expected a status 200, 201 or 204 but received: {response.status_code}"
                        logger.error(msg)
                        raise AnaliticoException(msg, status_code=response.status_code)

                    return True

            if False:
                rsync_path = f"{username}@{username}.your-storagebox.de"
                #                rsync_path = f"{rsync_path}:./{self.sdk.get_item_type(self.id)}s/{self.id}/{remotepath}"
                rsync_path = f"{rsync_path}:./pippo_{remotepath}"

                with tempfile.NamedTemporaryFile(suffix=".id_rsa") as key:
                    signature = storage["credentials"]["ssh_private_key"]
                    signature = str(base64.b64decode(signature), "ascii")
                    save_text(signature, key.name)
                    os.chmod(key.name, 0o600)

                    with tempfile.NamedTemporaryDirectory() as tempdir:
                        # create relative paths
                        pass

                    sync_cmd = [
                        "rsync",
                        "--recursive",
                        "--progress",
                        "-L",  # follow symlinks
                        "-e",
                        f"'ssh -p23 -o StrictHostKeyChecking=no -i {key.name}'",
                        filepath,
                        rsync_path,
                    ]
                    cmd = " ".join(sync_cmd)
                    os.system(cmd)

                    # a,b = subprocess_run(sync_cmd)

        raise NotImplementedError("Uploading multiple files at once is not yet implemented.")

    def download(
        self, remotepath: str, filepath: str = None, stream: bool = False, binary: bool = True, df: str = None
    ):
        """
        Downloads the file asset associated with this item to a file, stream or dataframe.
        
        Arguments:
            remotepath {str} -- The path of the file asset, eg. data.csv

        Keyword Arguments:
            filepath {str} -- The file path where this asset should be saved, or None.
            stream {bool} -- True if file should be returned as a stream.
            df {bool} -- True if file should be returned as pandas dataframe.
            binary {bool} -- True for binary downloads, false for text. (default: {True})

        Returns:
            The download stream or dataframe or nothing if saved to file.
        """
        url = self.url + "/files/" + remotepath
        url_stream = self.sdk.get_url_stream(url, binary=binary)
        # TODO if we're running serverless or in jupyter the assets may already be on a locally mounted drive (optimize)

        if stream:
            return url_stream

        if filepath:
            with open(filepath, "w+b") as f:
                for chunk in iter(url_stream):
                    f.write(chunk)

        if df:
            suffix = Path(remotepath).suffix
            with tempfile.NamedTemporaryFile(prefix="df_", suffix=suffix) as f:
                for chunk in iter(url_stream):
                    f.write(chunk)
                if suffix in CSV_SUFFIXES:
                    return pd_read_csv(f.name)
                elif suffix in PARQUET_SUFFIXES:
                    return pd.read_parquet(f.name)
                else:
                    msg = f"Can't read {df} to a pandas dataframe, please load .csv or .parquet files."
                    raise AnaliticoException(msg, status_code=400)

    def save(self) -> bool:
        """ Save any changes to the service. """
        json = self.sdk.get_url_json(self.url, method="PUT", json=self.to_dict(), status_code=200)
        self.attributes = json["data"]["attributes"]
        return True

    def delete(self) -> bool:
        """
        Delete this item from the service.
        
        Returns:
            bool -- True if item was deleted.
        """
        self.sdk.get_url_json(self.url, method="DELETE", status_code=204)
        return True

    def to_dict(self) -> dict:
        """ Return item as a dictionary. """
        return {"id": self.id, "type": self.type, "attributes": self.attributes}
