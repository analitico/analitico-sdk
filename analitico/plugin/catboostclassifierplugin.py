""" Regression and classification plugins based on CatBoost """

import analitico
import catboost
import collections
import numpy as np
import pandas as pd
import requests
import sklearn
import os.path

from abc import ABC, abstractmethod
from analitico.utilities import analitico_to_pandas_type, get_dict_dot
from analitico.utilities import time_ms, save_json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.metrics import precision_score, recall_score, accuracy_score

from .plugin import IAlgorithmPlugin, PluginError
from .catboostplugin import CatBoostPlugin

##
## CatBoostClassifierPlugin
##


class CatBoostClassifierPlugin(CatBoostPlugin):
    """ A tabular data classifier based on CatBoost """

    class Meta(CatBoostPlugin.Meta):
        name = "analitico.plugin.CatBoostClassifierPlugin"

    def create_model(self):
        """ Creates a CatBoostClassifier configured as requested """
        iterations = self.get_attribute("parameters.iterations", 50)
        learning_rate = self.get_attribute("parameters.learning_rate", 1)
        self.logger.info("iterations: " + iterations)
        self.logger.info("learning_rate: " + learning_rate)
        # loss function could be LogLoss for binary classification
        return catboost.CatBoostClassifier(
            iterations=iterations, learning_rate=learning_rate, loss_function="MultiClass"
        )

    def score_training(self, model, test_df, test_pool, test_labels, results):
        """ Scores the results of this training for the CatBoostClassifier model """
        # There are many metrics available:
        # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
        scores = results["data"]["scores"] = {}

        train_classes = results["data"]["classes"]  # the classes (actual strings)
        train_classes_codes = list(range(0, len(train_classes)))  # the codes, eg: 0, 1, 2...

        test_true = list(test_labels)  # test true labels
        test_preds = model.predict(test_pool, prediction_type="Class")  # prediction for each test sample
        test_probs = model.predict_proba(test_pool, verbose=True)  # probability for each class for each sample

        # Log loss, aka logistic loss or cross-entropy loss.
        scores["log_loss"] = round(sklearn.metrics.log_loss(test_true, test_probs, labels=train_classes_codes), 5)

        # In multilabel classification, this function computes subset accuracy:
        # the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
        scores["accuracy_score"] = round(accuracy_score(test_true, test_preds), 5)

        # The precision is the ratio tp / (tp + fp) where tp is the number of true positives
        # and fp the number of false positives. The precision is intuitively the ability
        # of the classifier not to label as positive a sample that is negative.
        # The best value is 1 and the worst value is 0.
        scores["precision_score_micro"] = round(precision_score(test_true, test_preds, average="micro"), 5)
        scores["precision_score_macro"] = round(precision_score(test_true, test_preds, average="macro"), 5)
        scores["precision_score_weighted"] = round(precision_score(test_true, test_preds, average="weighted"), 5)

        # The recall is the ratio tp / (tp + fn) where tp is the number of true positives
        # and fn the number of false negatives. The recall is intuitively the ability
        # of the classifier to find all the positive samples.
        scores["recall_score_micro"] = round(recall_score(test_true, test_preds, average="micro"), 5)
        scores["recall_score_macro"] = round(recall_score(test_true, test_preds, average="macro"), 5)
        scores["recall_score_weighted"] = round(recall_score(test_true, test_preds, average="weighted"), 5)

        self.info("log_loss: %f", scores["log_loss"])
        self.info("accuracy_score: %f", scores["accuracy_score"])
        self.info("precision_score_micro: %f", scores["precision_score_micro"])
        self.info("precision_score_macro: %f", scores["precision_score_macro"])

        # Report precision and recall for each of the classes
        scores["classes_scores"] = {}
        count = collections.Counter(test_true)
        precision_scores = precision_score(test_true, test_preds, average=None)
        recall_scores = recall_score(test_true, test_preds, average=None)
        for idx, val in enumerate(train_classes):
            scores["classes_scores"][val] = {
                "count": count[idx],
                "precision": round(precision_scores[idx], 5),
                "recall": round(recall_scores[idx], 5),
            }
        # superclass will save test.csv
        super().score_training(model, test_df, test_pool, test_labels, results)
