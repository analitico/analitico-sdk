import unittest
import os
import os.path

import pandas as pd

from sklearn.datasets import load_boston

from analitico.factory import Factory
from analitico.plugin import *
from .test_mixin import TestMixin

# pylint: disable=no-member

ASSETS_PATH = os.path.dirname(os.path.realpath(__file__)) + "/assets"


class AlgorithmTests(unittest.TestCase, TestMixin):
    """ Unit testing of machine learning algorithms """

    def test_catboost_binary_classifier(self):
        """ Test training catboost as a binary classifier """
        try:
            with Factory() as factory:
                csv_path = self.get_asset_path("titanic_1.csv")
                recipe = RecipePipelinePlugin(
                    factory=factory,
                    plugins=[
                        CsvDataframeSourcePlugin(
                            source={"url": csv_path, "schema": {"apply": [{"name": "Survived", "type": "category"}]}}
                        ),
                        CatBoostPlugin(parameters={"learning_rate": 0.2}),
                    ],
                )

                self.assertEqual(len(recipe.plugins), 2)
                self.assertTrue(isinstance(recipe, RecipePipelinePlugin))
                self.assertTrue(isinstance(recipe.plugins[0], CsvDataframeSourcePlugin))
                self.assertTrue(isinstance(recipe.plugins[1], CatBoostPlugin))

                # run training
                results = recipe.run(action="recipe/train")

                self.assertIsNotNone(results)
                self.assertEqual(results["data"]["label"], "Survived")
                self.assertEqual(results["data"]["source_records"], 891)
                self.assertEqual(results["data"]["training_records"], 712)
                self.assertEqual(results["data"]["test_records"], 179)

                self.assertEqual(len(results["data"]["classes"]), 2)
                self.assertEqual(results["data"]["classes"][0], 0)
                self.assertEqual(results["data"]["classes"][1], 1)

                self.assertEqual(results["parameters"]["loss_function"], "Logloss")
                self.assertEqual(results["parameters"]["test_size"], 0.2)
                self.assertEqual(results["parameters"]["learning_rate"], 0.2)

                # model was saved?
                artifacts = factory.get_artifacts_directory()
                model_path = os.path.join(artifacts, "model.cbm")
                self.assertTrue(os.path.isfile(model_path))

        except Exception as exc:
            factory.error("test_catboost_binary_classifier - " + str(exc))
            pass

    def test_catboost_multiclass_classifier(self):
        """ Test training catboost as a multiclass classifier """
        try:
            with Factory() as factory:
                csv_path = self.get_asset_path("iris_1.csv")
                recipe = RecipePipelinePlugin(
                    factory=factory,
                    plugins=[
                        CsvDataframeSourcePlugin(
                            source={
                                "url": csv_path,
                                "schema": {
                                    # Id column is not used to model
                                    "drop": [{"name": "Id"}],
                                    # Species column is marked as categorical to trigger multiclass classifier
                                    "apply": [{"name": "Species", "type": "category"}],
                                },
                            }
                        ),
                        CatBoostPlugin(parameters={"learning_rate": 0.2}),
                    ],
                )

                self.assertEqual(len(recipe.plugins), 2)
                self.assertTrue(isinstance(recipe, RecipePipelinePlugin))
                self.assertTrue(isinstance(recipe.plugins[0], CsvDataframeSourcePlugin))
                self.assertTrue(isinstance(recipe.plugins[1], CatBoostPlugin))

                # run training
                results = recipe.run(action="recipe/train")

                self.assertIsNotNone(results)
                self.assertEqual(results["data"]["label"], "Species")
                self.assertEqual(results["data"]["source_records"], 150)
                self.assertEqual(results["data"]["training_records"], 120)
                self.assertEqual(results["data"]["test_records"], 30)

                self.assertEqual(len(results["data"]["classes"]), 3)
                self.assertEqual(results["data"]["classes"][0], "Iris-setosa")
                self.assertEqual(results["data"]["classes"][1], "Iris-versicolor")
                self.assertEqual(results["data"]["classes"][2], "Iris-virginica")

                self.assertEqual(results["parameters"]["loss_function"], "MultiClass")
                self.assertEqual(results["parameters"]["test_size"], 0.2)
                self.assertEqual(results["parameters"]["learning_rate"], 0.2)
                self.assertEqual(results["parameters"]["iterations"], 50)

                self.assertEqual(results["scores"]["accuracy_score"], 1.0)
                self.assertLess(results["scores"]["log_loss"], 0.10)

                self.assertEqual(len(results["scores"]["features_importance"]), 4)
                self.assertIn("PetalLengthCm", results["scores"]["features_importance"])
                self.assertIn("PetalWidthCm", results["scores"]["features_importance"])
                self.assertIn("SepalLengthCm", results["scores"]["features_importance"])
                self.assertIn("SepalWidthCm", results["scores"]["features_importance"])

                # model was saved?
                artifacts = factory.get_artifacts_directory()
                model_path = os.path.join(artifacts, "model.cbm")
                self.assertTrue(os.path.isfile(model_path))

        except Exception as exc:
            factory.error("test_catboost_multiclass_classifier - " + str(exc))
            pass

    def test_catboost_regressor(self):
        """ Test training catboost as a regressor """
        try:
            with Factory() as factory:
                # bare bones, just run the plugin by itself w/o pipeline
                catboost = CatBoostPlugin(factory=factory, parameters={"learning_rate": 0.2})

                boston_dataset = load_boston()
                boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
                boston["MEDV"] = boston_dataset.target

                results = catboost.run(boston, action="recipe/train")

                self.assertIsNotNone(results)
                self.assertEqual(results["data"]["label"], "MEDV")  # median value
                self.assertEqual(results["data"]["source_records"], 506)
                self.assertEqual(results["data"]["training_records"], 404)
                self.assertEqual(results["data"]["test_records"], 102)

                # not a classifier
                self.assertFalse("classes" in results["data"])
                self.assertFalse("accuracy_score" in results["scores"])
                self.assertFalse("log_loss" in results["scores"])

                self.assertEqual(results["parameters"]["loss_function"], "RMSE")
                self.assertEqual(results["parameters"]["test_size"], 0.2)
                self.assertEqual(results["parameters"]["learning_rate"], 0.2)
                self.assertEqual(results["parameters"]["iterations"], 50)

                self.assertEqual(len(results["scores"]["features_importance"]), 13)
                self.assertGreater(results["scores"]["features_importance"]["AGE"], 8.0)
                self.assertGreater(results["scores"]["features_importance"]["DIS"], 8.0)
                self.assertGreater(results["scores"]["features_importance"]["LSTAT"], 20.0)

                self.assertIn("mean_abs_error", results["scores"])
                self.assertIn("median_abs_error", results["scores"])
                self.assertIn("sqrt_mean_squared_error", results["scores"])
                self.assertIn("mean_abs_error", results["scores"])

                # model was saved?
                artifacts = factory.get_artifacts_directory()
                model_path = os.path.join(artifacts, "model.cbm")
                self.assertTrue(os.path.isfile(model_path))

        except Exception as exc:
            factory.error("test_catboost_regressor - " + str(exc))
            pass
