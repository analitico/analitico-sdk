""" Utility methods to convert between pandas and analitico's schemas """

import numpy as np
import pandas as pd

##
## Schema
##

# pandas types for analitico's types
PD_TYPE_INTEGER = "int64"
PD_TYPE_FLOAT = "float64"
PD_TYPE_STRING = "str"
PD_TYPE_BOOLEAN = "bool"
PD_TYPE_DATETIME = "datetime64"
PD_TYPE_TIMESPAN = "timedelta64"
PD_TYPE_CATEGORY = "category"

ANALITICO_TYPE_INTEGER = "integer"
ANALITICO_TYPE_FLOAT = "float"
ANALITICO_TYPE_STRING = "string"
ANALITICO_TYPE_BOOLEAN = "boolean"
ANALITICO_TYPE_DATETIME = "datetime"
ANALITICO_TYPE_TIMESPAN = "timespan"
ANALITICO_TYPE_CATEGORY = "category"

# these values are replaced with pd.NaN
NA_VALUES = ['None', '', '#N/A', '#N/A', 'N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null']

# these values are replaced with pd.NaT
NA_DATES = NA_VALUES + [0, '0']


def analitico_to_pandas_type(data_type: str):
    """ Converts an analitico data type to the equivalent dtype string for pandas dataframes """
    try:
        ANALITICO_TO_PANDAS_TYPES = {
            ANALITICO_TYPE_STRING: PD_TYPE_STRING,
            ANALITICO_TYPE_INTEGER: PD_TYPE_INTEGER,
            ANALITICO_TYPE_FLOAT: PD_TYPE_FLOAT,
            ANALITICO_TYPE_BOOLEAN: PD_TYPE_BOOLEAN,
            ANALITICO_TYPE_DATETIME: PD_TYPE_DATETIME,
            ANALITICO_TYPE_TIMESPAN: PD_TYPE_TIMESPAN,
            ANALITICO_TYPE_CATEGORY: PD_TYPE_CATEGORY,
        }
        return ANALITICO_TO_PANDAS_TYPES[data_type]
    except KeyError as exc:
        raise KeyError("analitico_to_pandas_type - unknown type: " + data_type, exc)


def pandas_to_analitico_type(data_type):
    """ Return the analitico schema data type of a pandas dtype """
    if data_type == "int":
        return "integer"
    if data_type == "float":
        return "float"
    if data_type == "bool":
        return "boolean"
    if data_type.name == "category":
        return "category"  # dtype alone doesn't ==
    if data_type == "object":
        return "string"
    if data_type == "datetime64[ns]":
        return "datetime"
    if data_type == "timedelta64[ns]":
        return "timespan"
    raise KeyError("_pandas_to_analitico_type - unknown data_type: " + str(data_type))


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

    # no type? no op, some schemas may only have partial information
    if "type" not in kwargs:
        return

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
        else:
            # strings like no
            for not_a_date in NA_DATES:
                df[cname].replace(not_a_date, np.nan, inplace=True)
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
    then sort, filter and rename columns according to schema.
    """
    assert isinstance(df, pd.DataFrame), "apply_schema should be passed a pd.DataFrame, received: " + str(df)
    assert isinstance(schema, dict), "apply_schema should be passed a schema dictionary"

    # select columns and apply types to columns
    names = []
    for column in schema["columns"]:
        name = column["name"]
        names.append(name)
        apply_type(df, **column)

        # make requested column index
        index = column.get("index", False)
        if index:
            df = df.set_index(name, drop=False)

    # reorder and remove extra columns
    df = df[names]

    # see if there are columns that need to be renamed
    rename = {}
    for column in schema["columns"]:
        if "rename" in column:
            rename[column["name"]] = column["rename"]
    if len(rename) > 0:
        df = df.rename(index=str, columns=rename)

    return df
