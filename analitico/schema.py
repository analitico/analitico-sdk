""" Utility methods to convert between pandas and analitico's schemas """

import time
import numpy as np
import pandas as pd
import json
import logging
import holidays

from catboost import Pool
from datetime import datetime

##
## Schema
##


def analitico_to_pandas_type(type: str):
    """ Converts an analitico data type to the equivalent dtype string for pandas dataframes """
    try:
        ANALITICO_TO_PANDAS_TYPES = {
            "string": "str",
            "integer": "int64",
            "float": "float64",
            "boolean": "bool",
            "datetime": "datetime64",
            "timespan": "timedelta64",
            "category": "category",
        }
        return ANALITICO_TO_PANDAS_TYPES[type]
    except KeyError as exc:
        raise KeyError("analitico_to_pandas_type - unknown type: " + type, exc)


def pandas_to_analitico_type(dtype):
    """ Return the analitico schema data type of a pandas dtype """
    if dtype == "int":
        return "integer"
    if dtype == "float":
        return "float"
    if dtype == "bool":
        return "boolean"
    if dtype.name == "category":
        return "category"  # dtype alone doesn't ==
    if dtype == "object":
        return "string"
    if dtype == "datetime64[ns]":
        return "datetime"
    if dtype == "timedelta64[ns]":
        return "timespan"
    raise KeyError("_pandas_to_analitico_type - unknown dtype: " + str(dtype))


def generate_schema(df: pd.DataFrame) -> dict:
    """ Generates an analitico schema from a pandas dataframe """
    columns = []
    for name in df.columns:
        ctype = pandas_to_analitico_type(df[name].dtype)
        column = {"name": name, "type": ctype}
        if df.index.name == name:
            column["index"] = True
        columns.append(column)
    return {"columns": columns}


def apply_type(df: pd.DataFrame, **kwargs):
    """ Apply given type to the column (parameters are type, name, etc from schema column) """
    assert isinstance(df, pd.DataFrame)
    cname = kwargs["name"]
    ctype = kwargs["type"]
    missing = cname not in df.columns
    if ctype == "string":
        if missing:
            df[cname] = None
        df[cname] = df[cname].astype(str)
    elif ctype == "float":
        if missing:
            df[cname] = np.nan
        df[cname] = df[cname].astype(float)
    elif ctype == "boolean":
        if missing:
            df[cname] = False
        df[cname] = df[cname].astype(bool)
    elif ctype == "integer":
        if missing:
            df[cname] = 0
        df[cname] = df[cname].astype(int)
    elif ctype == "datetime":
        if missing:
            df[cname] = None
        df[cname] = df[cname].astype("datetime64[ns]")
    elif ctype == "timespan":
        if missing:
            df[cname] = None
        df[cname] = pd.to_timedelta(df[cname])
    elif ctype == "category":
        if missing:
            df[cname] = None
        df[cname] = df[cname].astype("category")
    else:
        raise Exception("analitico.schema.apply_type - unknown type: " + ctype)


def apply_schema(df: pd.DataFrame, schema):
    """ 
    Applies the given schema to the dataframe. The method will scan columns
    in the schema and apply their type to columns in the dataframe. It will
    then sort and filter columns according to schema.
    """
    assert isinstance(df, pd.DataFrame)
    names = []
    for column in schema["columns"]:
        names.append(column["name"])
        apply_type(df, **column)
    return df[names]
