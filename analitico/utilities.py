# Utility methods to process dataframes and simplify workflow.
# Copyright (C) 2018 by Analitico.ai. All rights reserved.

import time
import pandas as pd
import json
import logging
import socket
import platform
import multiprocessing
import psutil

from datetime import datetime

try:
    import distro
    import GPUtil
except Exception:
    pass

# default logger for analitico's libraries
logger = logging.getLogger("analitico")

##
## Runtime
##

MB = 1024 * 1024


def get_runtime():
    """ Collect information on runtime environment, platform, python, hardware, etc """
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    disk = psutil.disk_usage("/")
    boot_time = datetime.fromtimestamp(psutil.boot_time())
    uptime = (datetime.now() - boot_time).total_seconds() / 3600
    runtime = {
        "hostname": socket.gethostname(),
        "ip": socket.gethostbyname(socket.gethostname()),
        "platform": {"system": platform.system(), "version": platform.version()},
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation()},
        "hardware": {
            "cpu": {
                "type": platform.processor(),
                "count": multiprocessing.cpu_count(),
                "freq": int(psutil.cpu_freq()[2]),
            },
            "gpu": [],
            "memory": {
                "total_mb": int(memory.total / MB),
                "available_mb": int(memory.available / MB),
                "used_mb": int(memory.used / MB),
                "swap_mb": int(swap.total / MB),
                "swap_perc": round(swap.percent, 2),
            },
            "disk": {
                "total_mb": int(disk.total / MB),
                "available_mb": int(disk.free / MB),
                "used_mb": int(disk.used / MB),
                "used_perc": round(disk.percent, 2),
            },
        },
        "uptime": {"since": boot_time.strftime("%Y-%m-%d %H:%M:%S"), "hours": round(uptime, 2)},
    }
    try:
        # optional package
        runtime["platform"]["name"] = distro.name()
        runtime["platform"]["version"] = distro.version()
    except Exception:
        pass
    try:
        # optional package
        GPUs = GPUtil.getGPUs()
        if GPUs:
            for GPU in GPUs:
                runtime["hardware"]["gpu"].append(
                    {
                        "uuid": GPU.uuid,
                        "name": GPU.name,
                        "driver": GPU.driver,
                        "temperature": int(GPU.temperature),
                        "load": round(GPU.load, 2),
                        "memory": {
                            "total_mb": int(GPU.memoryTotal),
                            "available_ms": int(GPU.memoryFree),
                            "used_mb": int(GPU.memoryUsed),
                            "used_perc": round(GPU.memoryUtil, 2),
                        },
                    }
                )
    except Exception:
        pass
    return runtime


##
## Json utilities
##


def save_json(data, filename, indent=4):
    """ Saves given data in a json file (we love pretty, so prettified by default) """
    with open(filename, "w") as f:
        json.dump(data, f, indent=indent)


def read_json(filename, encoding="utf-8"):
    """ Reads, decodes and returns the contents of a json file """
    try:
        with open(filename, encoding=encoding) as f:
            return json.load(f)
    except Exception as exc:
        detail = "analitico.utilities.read_json: error while reading {}, exception: {}".format(filename, exc)
        logger.error(detail)
        raise Exception(detail, exc)


def save_text(text, filename):
    with open(filename, "w") as text_file:
        text_file.write(text)


##
## Time utilities
##


def time_ms(started_on=None):
    """ Returns the time elapsed since given time in ms """
    return datetime.now() if started_on is None else int((datetime.now() - started_on).total_seconds() * 1000)


def time_it(code):
    """ Returns the time elapsed to execute the given call in ms """
    started_on = datetime.now()
    code()
    return int((datetime.now() - started_on).total_seconds() * 1000)


##
## Timestamp utilities
##


def timestamp_to_time(ts: str, ts_format="%Y-%m-%d %H:%M:%S") -> time.struct_time:
    """ Converts a timestamp string in the given format to a time object """
    try:
        return time.strptime(ts, ts_format)
    except TypeError:
        return None


def timestamp_to_secs(ts: str, ts_format="%Y-%m-%d %H:%M:%S") -> float:
    """ Converts a timestamp string to number of seconds since epoch """
    return time.mktime(time.strptime(ts, ts_format))


def timestamp_diff_secs(ts1, ts2):
    t1 = timestamp_to_secs(ts1)
    t2 = timestamp_to_secs(ts2)
    return t1 - t2


##
## Dictionary utilities
##


def get_dict_dot(d: dict, key: str, default=None):
    """ Gets an entry from a dictionary using dot notation key, eg: this.that.something """
    try:
        if isinstance(d, dict) and key:
            split = key.split(".")
            value = d.get(split[0])
            if value:
                if len(split) == 1:
                    return value
                return get_dict_dot(value, key[len(split[0]) + 1 :], default)
    except KeyError:
        pass
    return default


def set_dict_dot(d: dict, key: str, value=None):
    """ Sets an entry from a dictionary using dot notation key, eg: this.that.something """
    if isinstance(d, dict) and key:
        split = key.split(".")
        subkey = split[0]
        if len(split) == 1:
            d[subkey] = value
            return
        if not (subkey in d):
            d[subkey] = None
        set_dict_dot(d[subkey], key[len(subkey) + 1 :], value)


##
## Pandas utilities
##


def pd_cast_datetime(df, column):
    """ Casts a string column to a date column, assumes format is recognizable """
    df[column] = pd.to_datetime(df[column], infer_datetime_format=True, errors="coerce")


def pd_augment_date(df, column):
    """ Splits a datetime column into year, month, day, dayofweek, hour, minute then removes the original column """
    loc = df.columns.get_loc(column) + 1
    # create separate columns for each parameter
    df.insert(loc, column + ".minute", df[column].apply(lambda ts: ts.minute))
    df.insert(loc, column + ".hour", df[column].apply(lambda ts: ts.hour))
    df.insert(loc, column + ".day", df[column].apply(lambda ts: ts.day))
    df.insert(loc, column + ".month", df[column].apply(lambda ts: ts.month))
    df.insert(loc, column + ".year", df[column].apply(lambda ts: ts.year))
    df.insert(loc, column + ".dayofweek", df[column].apply(lambda ts: ts.dayofweek))
    df.drop([column], axis=1, inplace=True)


def pd_timediff_min(df, column_start, column_end, column_diff):
    """ Creates a new column with difference in minutes between the two named columns """
    pd_cast_datetime(df, column_start)
    pd_cast_datetime(df, column_end)
    df[column_diff] = df[column_end] - df[column_start]
    df[column_diff] = df[column_diff].dt.total_seconds() / 60.0


##
## CSV
##


def get_csv_row_count(filename):
    """ Returns the number of rows in the given csv file (one row is deducted for the header) """
    with open(filename, "r") as f:
        return sum(1 for row in f) - 1
