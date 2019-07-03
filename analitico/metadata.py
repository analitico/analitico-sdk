import os
import json
import logging
import collections

from analitico.utilities import read_json, save_json, get_dict_dot, set_dict_dot

# model metadata is saved in training.json
METADATA_FILENAME = "training.json"


def get_metadata(metadata_filename=METADATA_FILENAME):
    """ Returns metadata dictionary for current recipe. """
    return read_json(metadata_filename) if os.path.isfile(metadata_filename) else collections.OrderedDict()


def set_score(
    score: str,
    value,
    title: str = None,
    subtitle: str = None,
    priority: int = None,
    category: str = None,
    category_title: str = None,
    category_subtitle: str = None,
    metadata_filename=METADATA_FILENAME,
):
    """ 
    Collects a score in the metadata file. These scores can be useful
    to track performance of a trained model and can also be shown in 
    analitico's UI next to the trained models' information. A score
    has a machine readable id like number_of_lines and a value, eg. 100.
    Optionally a score can have a human readable title like "Number of lines"
    and more descriptive subtitle. Scores can be grouped into related categories
    that can also have titles and subtitles. Scores with priorities are sorted
    according to priority (priotity 1 is more important than 2 and comes first).
    """
    metadata = get_metadata()

    if title or priority:
        value = {"value": value}
        if title:
            value["title"] = title
        if subtitle:
            value["subtitle"] = subtitle
        if priority:
            value["priority"] = priority

    key = f"scores.{category}.{score}" if category else f"scores.{score}"
    set_dict_dot(metadata, key, value)

    if category:
        if category_title:
            set_dict_dot(metadata, f"scores.{category}.title", category_title)
        if category_subtitle:
            set_dict_dot(metadata, f"scores.{category}.subtitle", category_subtitle)

    save_json(metadata, METADATA_FILENAME)
