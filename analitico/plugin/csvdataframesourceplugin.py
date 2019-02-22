"""
Plugins that import dataframes from different sources
"""

import pandas
from analitico.utilities import get_dict_dot
from analitico.schema import analitico_to_pandas_type, apply_schema
from .interfaces import IDataframeSourcePlugin, PluginError

##
## CsvDataframeSourcePlugin
##


class CsvDataframeSourcePlugin(IDataframeSourcePlugin):
    """ A plugin that returns a pandas dataframe from a csv file """

    class Meta(IDataframeSourcePlugin.Meta):
        name = "analitico.plugin.CsvDataframeSourcePlugin"

    def run(self, *args, action=None, **kwargs):
        """ Creates a pandas dataframe from the csv source """
        try:
            url = self.get_attribute("source.url")
            if not url:
                raise PluginError("URL of csv file cannot be empty.", plugin=self)

            # source schema is part of the source definition?
            schema = self.get_attribute("source.schema")

            # no schema was provided but the url is that of an analitico dataset in the cloud
            if not schema and url.startswith("analitico://") and url.endswith("/data/csv"):
                info_url = url.replace("/data/csv", "/data/info")
                info = self.factory.get_url_json(info_url)
                schema = get_dict_dot(info, "data.schema")

            # array of types for each column in the source
            columns = schema.get("columns") if schema else None

            dtype = None
            parse_dates = None

            if columns:
                dtype = {}
                parse_dates = []
                for idx, column in enumerate(columns):
                    if "type" in column:  # type is optionally defined
                        if column["type"] == "datetime":
                            # ISO8601 dates only for now
                            # TODO use converters to apply date patterns #16
                            parse_dates.append(idx)
                        elif column["type"] == "timespan":
                            # timedelta needs to be applied later on or else we will get:
                            # 'the dtype timedelta64 is not supported for parsing'
                            dtype[column["name"]] = "object"
                        else:
                            dtype[column["name"]] = analitico_to_pandas_type(column["type"])

            stream = self.factory.get_url_stream(url, binary=False)
            df = pandas.read_csv(stream, dtype=dtype, parse_dates=parse_dates, encoding="utf-8")

            if schema:
                # reorder, filter, apply types, rename columns as requested in schema
                df = apply_schema(df, schema)

            return df
        except Exception as exc:
            self.exception("Error while processing: %s", url, exc_info=exc)
