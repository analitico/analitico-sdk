import os
import json
import logging
import collections

import sklearn
import sklearn.metrics

from analitico import logger
from analitico.utilities import read_json, save_json, get_dict_dot, set_dict_dot

# model metadata is saved in training.json
METADATA_FILENAME = "training.json"


def get_metadata(metadata_filename=METADATA_FILENAME):
    """ Returns metadata dictionary for current recipe. """
    return read_json(metadata_filename) if os.path.isfile(metadata_filename) else collections.OrderedDict()


def set_metric(
    metric: str,
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
    Collects metrics in the metadata file. These metrics can be useful
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

    key = f"scores.{category}.{metric}" if category else f"scores.{metric}"
    set_dict_dot(metadata, key, value)

    if category:
        if category_title:
            set_dict_dot(metadata, f"scores.{category}.title", category_title)
        if category_subtitle:
            set_dict_dot(metadata, f"scores.{category}.subtitle", category_subtitle)

    save_json(metadata, METADATA_FILENAME)


def set_model_metrics(
    model,
    y_true,
    y_pred,
    category=None,
    category_title=None,
    category_subtitle=None,
    metadata_filename=METADATA_FILENAME,
):
    """
    Takes a model (derived from sklearn base estimator) and two array of values
    and predictions and saves a number of statistical scores regarding the accuracy
    of the predictions.
    """
    category = category if category else "sklearn_metrics"
    category_title = category_title if category_title else "Scikit Learn Metrics"

    # model is a sklearn estimator?
    if not isinstance(model, sklearn.base.BaseEstimator):
        logger.warning("set_model_metrics - we only support sklearn models for now")
        return

    if sklearn.base.is_regressor(model):
        set_metric(
            metric="mean_abs_error",
            value=round(sklearn.metrics.mean_absolute_error(y_true, y_pred), 5),
            title="Mean absolute regression loss",
            subtitle="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html",
            category=category,
            category_title=category_title,
            category_subtitle=category_subtitle,
        )
        set_metric(
            metric="median_abs_error",
            value=round(sklearn.metrics.median_absolute_error(y_true, y_pred), 5),
            title="Median absolute error regression loss",
            subtitle="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html",
            category=category,
            category_title=category_title,
            category_subtitle=category_subtitle,
        )
        set_metric(
            metric="mean_squared_error",
            value=round(sklearn.metrics.mean_squared_error(y_true, y_pred), 5),
            title="Mean squared error regression loss",
            subtitle="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html",
            category=category,
            category_title=category_title,
            category_subtitle=category_subtitle,
        )

    if sklearn.base.is_classifier(model):
        log_loss = round(sklearn.metrics.log_loss(y_true, y_pred), 5)
        set_metric(
            metric="log_loss",
            value=log_loss,
            title="Log loss, aka logistic loss or cross-entropy loss",
            subtitle="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html",
            category=category,
            category_title=category_title,
            category_subtitle=category_subtitle,
        )
