import catboost
import numpy as np
import os.path

from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error

from analitico.utilities import time_ms
from .catboostplugin import CatBoostPlugin


##
## CatBoostRegressorPlugin
##


class CatBoostRegressorPlugin(CatBoostPlugin):
    """ A tabular data regressor based on CatBoost library """

    class Meta(CatBoostPlugin.Meta):
        name = "analitico.plugin.CatBoostRegressorPlugin"

    def create_model(self, results=None):
        """ Creates a CatBoostRegressor configured as requested """
        iterations = self.get_attribute("parameters.iterations", 50)
        learning_rate = self.get_attribute("parameters.learning_rate", 1)
        depth = self.get_attribute("parameters.depth", 8)
        if results:
            results["parameters"]["iterations"] = iterations
            results["parameters"]["learning_rate"] = learning_rate
            results["parameters"]["depth"] = depth
        return catboost.CatBoostRegressor(iterations=iterations, learning_rate=learning_rate, depth=8)

    def score_training(self, model, test_df, test_pool, test_labels, results):
        """ Runs predictions on test set then stores metrics in results["scores"] """
        test_preds = model.predict(test_pool)
        results["scores"]["median_abs_error"] = round(median_absolute_error(test_preds, test_labels), 5)
        results["scores"]["mean_abs_error"] = round(mean_absolute_error(test_preds, test_labels), 5)
        results["scores"]["sqrt_mean_squared_error"] = round(np.sqrt(mean_squared_error(test_preds, test_labels)), 5)
        return super().score_training(model, test_df, test_pool, test_labels, results)

    def predict(self, data, training, results, *args, **kwargs):
        """ Return predictions from trained model """
        # initialize data pool to be tested
        categorical_idx = self.get_categorical_idx(data)
        data_pool = catboost.Pool(data, cat_features=categorical_idx)

        # create model object from stored file
        loading_on = time_ms()
        model_filename = os.path.join(self.factory.get_artifacts_directory(), "model.cbm")
        model = self.create_model()
        model.load_model(model_filename)
        results["performance"]["loading_ms"] = time_ms(loading_on)

        # create predictions with assigned class and probabilities
        predictions = model.predict(data_pool)
        predictions = np.around(predictions, decimals=3)
        results["predictions"] = list(predictions)
        return results
