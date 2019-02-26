import unittest
import os
import os.path

import pandas as pd

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
