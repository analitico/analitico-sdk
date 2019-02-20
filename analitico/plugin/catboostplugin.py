""" Regression and classification plugins based on CatBoost """

import catboost
import pandas as pd
import os.path

from abc import abstractmethod
from sklearn.model_selection import train_test_split

from analitico.utilities import get_dict_dot
from analitico.utilities import time_ms
from analitico.schema import generate_schema

from .interfaces import IAlgorithmPlugin, PluginError

##
## CatBoostPlugin
##


class CatBoostPlugin(IAlgorithmPlugin):
    """ Base class for CatBoost regressor and classifier plugins """

    results = None

    class Meta(IAlgorithmPlugin.Meta):
        name = "analitico.plugin.CatBoostPlugin"

    @abstractmethod
    def create_model(self, results):
        """ Creates actual CatBoostClassifier or CatBoostRegressor model in subclass """
        pass

    def get_categorical_idx(self, df):
        """ Return indexes of the columns that should be considered categorical for the purpose of catboost training """
        categorical_idx = []
        for column in df.columns:
            if df[column].dtype.name == "category":
                categorical_idx.append(df.columns.get_loc(column))
        return categorical_idx

    def validate_schema(self, train_df, test_df):
        """ Checks training and test dataframes to make sure they have matching schemas """
        train_schema = generate_schema(train_df)
        if test_df:
            test_schema = generate_schema(test_df)
            train_columns = train_schema["columns"]
            test_columns = test_schema["columns"]
            if len(train_columns) != len(test_columns):
                msg = "{} - training data has {} columns while test data has {} columns".format(
                    self.name, len(train_columns), len(test_columns)
                )
                raise PluginError(msg)
            for i in range(0, len(train_columns)):
                if train_columns[i]["name"] != test_columns[i]["name"]:
                    msg = "{} - column {} of train '{}' and test '{}' have different names".format(
                        self.name, i, train_columns[i]["name"], test_columns[i]["name"]
                    )
                    raise PluginError(msg)
                if train_columns[i]["type"] != test_columns[i]["type"]:
                    msg = "- column %d of train '%s' and test '%s' have different names".format(
                        self.name, i, train_columns[i]["type"], test_columns[i]["type"]
                    )
                    raise PluginError(msg)
        return train_schema

    def score_training(
        self,
        model: catboost.CatBoost,
        test_df: pd.DataFrame,
        test_pool: catboost.Pool,
        test_labels: pd.DataFrame,
        results: dict,
    ):
        """ Scores the results of this training """
        for key, value in model.get_params().items():
            results["parameters"][key] = value

        results["scores"]["best_iteration"] = model.get_best_iteration()
        results["scores"]["best_score"] = model.get_best_score()

        # catboost can tell which features weigh more heavily on the predictions
        self.info("features importance:")
        features_importance = results["scores"]["features_importance"] = {}
        for label, importance in model.get_feature_importance(prettified=True):
            features_importance[label] = round(importance, 5)
            self.info("%24s: %8.4f", label, importance)

        # make the prediction using the resulting model
        # output test set with predictions
        # after moving label to the end for easier reading
        test_predictions = model.predict(test_pool)
        label = test_labels.name
        test_df = test_df.copy().tail(100)  # just sampling
        test_df[label] = test_labels
        cols = list(test_df.columns.values)
        cols.pop(cols.index(label))
        test_df = test_df[cols + [label]]
        test_df["prediction"] = test_predictions[-100:]  # match sample above
        artifacts_path = self.factory.get_artifacts_directory()
        test_df.to_csv(os.path.join(artifacts_path, "test.csv"))

    def train(self, train, test, results, *args, **kwargs):
        """ Train with algorithm and given data to produce a trained model """
        try:
            assert isinstance(train, pd.DataFrame) and len(train.columns) > 1
            train_df = train
            test_df = test

            # if not specified the prediction target will be the last column of the dataset
            label = self.get_attribute("data.label")
            if not label:
                label = train_df.columns[len(train_df.columns) - 1]
            results["data"]["label"] = label

            # remove rows with missing label from training and test sets
            train_rows = len(train_df)
            train_df = train_df.dropna(subset=[label])
            if len(train_df) < train_rows:
                self.warning("training data has %s rows without '%s' label", train_rows - len(train_df), label)
            if test_df:
                test_rows = len(test_df)
                test_df = test_df.dropna(subset=[label])
                if len(test_df) < test_rows:
                    self.warning("test data has %s rows without '%s' label", test_rows - len(test_df), label)

            # make sure schemas match
            train_schema = self.validate_schema(train_df, test_df)

            # shortened training was requested?
            tail = self.get_attribute("parameters.tail", 0)
            if tail > 0:
                self.info("tail: %d, cutting training data", tail)
                train_df = train_df.tail(tail).copy()

            # create test set from training set if not provided
            if not test_df:
                # decide how to create test set from settings variable
                # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
                chronological = self.get_attribute("data.chronological", False)
                test_size = self.get_attribute("parameters.test_size", 0.20)
                results["data"]["chronological"] = chronological
                results["parameters"]["test_size"] = test_size
                if chronological:
                    # test set if from the last rows (chronological order)
                    self.info("chronological test split")
                    test_rows = int(len(train_df) * test_size)
                    test_df = train_df[-test_rows:]
                    train_df = train_df[:-test_rows]
                else:
                    # test set if from a random assortment of rows
                    self.info("random test set split")
                    train_df, test_df, = train_test_split(train_df, test_size=test_size, random_state=42)

            self.info("training set %d rows", len(train_df))
            self.info("test set %d rows", len(test_df))

            # validate data types
            for column in train_schema["columns"]:
                if column["type"] not in ("integer", "float", "boolean", "category"):
                    self.warning(
                        "column '%s' of type '%s' is incompatible and will be dropped", column["name"], column["type"]
                    )
                    train_df = train_df.drop(column["name"], axis=1)
                    test_df = test_df.drop(column["name"], axis=1)

            # save schema after dropping unused columns
            results["data"]["schema"] = generate_schema(train_df)
            results["data"]["source_records"] = len(train)
            results["data"]["training_records"] = len(train_df)
            results["data"]["test_records"] = len(test_df)
            results["data"]["dropped_records"] = len(train) - len(train_df) - len(test_df)

            # save some training data for debugging
            # train_df.tail(500).to_json("/home/xxx/xxx.json", orient="records")

            # split data and labels
            train_labels = train_df[label]
            train_df = train_df.drop([label], axis=1)
            test_labels = test_df[label]
            test_df = test_df.drop([label], axis=1)

            # indexes of columns that should be considered categorical
            categorical_idx = self.get_categorical_idx(train_df)
            train_pool = catboost.Pool(train_df, train_labels, cat_features=categorical_idx)
            test_pool = catboost.Pool(test_df, test_labels, cat_features=categorical_idx)

            # create regressor or classificator then train
            training_on = time_ms()
            model = self.create_model(results)
            model.fit(train_pool, eval_set=test_pool)
            results["performance"]["training_ms"] = time_ms(training_on)

            # score test set, add related metrics to results
            self.score_training(model, test_df, test_pool, test_labels, results)

            # save model file and training results
            artifacts_path = self.factory.get_artifacts_directory()
            model_path = os.path.join(artifacts_path, "model.cbm")
            model.save_model(model_path)
            results["scores"]["model_size"] = os.path.getsize(model_path)
            self.info("saved %s (%d bytes)", model_path, os.path.getsize(model_path))

            return results

        except Exception as exc:
            self.error("error while training: %s", exc)
            self.logger.exception(exc)
            raise exc
