import os
import time
import pandas as pd
import json
import logging
import socket
import platform
import multiprocessing
import psutil
import collections
import subprocess
import sys
import random
import string
import dateutil
from io import StringIO

from django.conf import settings
from datetime import datetime

try:
    import distro
    import GPUtil
except Exception:
    pass

import analitico.schema
from analitico.schema import analitico_to_pandas_type, NA_VALUES

##
## Pandas utilities
##


def pd_date_parser(x):
    if not x:
        return None
    lower_x = x.lower()
    if lower_x in ("none", "null", "nan", "empty"):
        return None
    date = dateutil.parser.parser(x)
    return date


def pd_cast_datetime(df, column):
    """ Casts a string column to a date column, assumes format is recognizable """
    df[column] = pd.to_datetime(df[column], infer_datetime_format=True, errors="coerce")


def pd_augment_date(df, column):
    """ Splits a datetime column into year, month, day, dayofweek, hour, minute then removes the original column """
    loc = df.columns.get_loc(column) + 1
    # create separate columns for each parameter
    dates = pd.DatetimeIndex(df[column])
    df.insert(loc + 1, column + ".year", dates.year.astype("category", copy=False))
    df.insert(loc + 2, column + ".month", dates.month.astype("category", copy=False))
    df.insert(loc + 3, column + ".day", dates.day.astype("category", copy=False))
    df.insert(loc + 4, column + ".hour", dates.hour.astype("category", copy=False))
    df.insert(loc + 5, column + ".minute", dates.minute.astype("category", copy=False))
    df.insert(loc + 6, column + ".dayofweek", dates.dayofweek.astype("category", copy=False))
    # df.drop([column], axis=1, inplace=True)


def pd_timediff_min(df, column_start, column_end, column_diff):
    """ Creates a new column with difference in minutes between the two named columns """
    pd_cast_datetime(df, column_start)
    pd_cast_datetime(df, column_end)
    df[column_diff] = df[column_end] - df[column_start]
    df[column_diff] = df[column_diff].dt.total_seconds() / 60.0


def pd_columns_to_string(df):
    """ Returns a single string with a list of columns, eg: 'col1', 'col2', 'col3' """
    columns = "".join("'" + column + "', " for column in df.columns)
    return columns[:-2]


def pd_read_csv(filepath_or_buffer, schema=None):
    """ Read csv file from file or stream and apply optional schema """
    dtype = None

    if schema:
        # array of types for each column in the source
        columns = schema.get("columns")
        if columns:
            dtype = {}
            for column in columns:
                if "type" in column:  # type is optionally defined
                    if column["type"] == "datetime":
                        dtype[column["name"]] = "object"
                    elif column["type"] == "timespan":
                        dtype[column["name"]] = "object"
                    else:
                        dtype[column["name"]] = analitico_to_pandas_type(column["type"])

    # read csv from file or stream
    df = pd.read_csv(filepath_or_buffer, dtype=dtype, encoding="utf-8", na_values=NA_VALUES)

    if schema:
        # reorder, filter, apply types, rename columns as requested in schema
        df = analitico.schema.apply_schema(df, schema)
    return df


def pd_drop_column(df, column, inplace=False):
    """ Drops a column, no exceptions if it's not there """
    try:
        df.drop([column], axis=1, inplace=inplace)
    except:
        pass


def pd_to_dict(df):
    """ Convert a dataframe to json, encodes all dates and timestamps to ISO8601 """
    assert isinstance(df, pd.DataFrame), "pd_to_dict - requires a pd.DataFrame"
    with StringIO() as io:
        df.to_json(io, orient="records", date_format="iso", date_unit="s", double_precision=6)
        io.seek(0)
        return json.load(io)


def pd_sample(df, n=20):
    """ Returns a sample from the given DataFrame, either number of rows or percentage. """
    if n < 1:
        return df.sample(frac=n)
    if n < len(df.index):
        return df.sample(n=n)
    return df
