import unittest
import tempfile
import numpy as np
import os


from sklearn import datasets
from sklearn import linear_model
import sklearn
import statistics

import pandas

from analitico.metadata import *


class MetadataTests(unittest.TestCase):
    """ Test utilities to write metrics and scores to training.json """

    def setUp(self):
        if os.path.exists(METADATA_FILENAME):
            os.remove(METADATA_FILENAME)

    def tearDown(self):
        if os.path.exists(METADATA_FILENAME):
            os.remove(METADATA_FILENAME)

    def test_metadata_set_score(self):
        set_metric("number_of_lines", 100)

        metadata = get_metadata()
        self.assertEqual(len(metadata["scores"]), 1)
        self.assertEqual(metadata["scores"]["number_of_lines"], 100)

    def test_metadata_set_score_with_title(self):
        set_metric("number_of_lines2", 100, title="Number of lines", subtitle="A longer description")

        metadata = get_metadata()
        self.assertEqual(len(metadata["scores"]), 1)
        self.assertEqual(metadata["scores"]["number_of_lines2"]["value"], 100)
        self.assertEqual(metadata["scores"]["number_of_lines2"]["title"], "Number of lines")
        self.assertEqual(metadata["scores"]["number_of_lines2"]["subtitle"], "A longer description")

    def test_metadata_set_score_with_title_and_priority(self):
        set_metric("number_of_lines3", 100, title="Number of lines", priority=1)

        metadata = get_metadata()
        self.assertEqual(len(metadata["scores"]), 1)
        self.assertEqual(len(metadata["scores"]["number_of_lines3"]), 3)
        self.assertEqual(metadata["scores"]["number_of_lines3"]["value"], 100)
        self.assertEqual(metadata["scores"]["number_of_lines3"]["title"], "Number of lines")
        self.assertEqual(metadata["scores"]["number_of_lines3"]["priority"], 1)

    def test_metadata_set_multiple_scores(self):
        set_metric("metric1", 100)
        set_metric("metric2", "hello")

        metadata = get_metadata()
        self.assertEqual(metadata["scores"]["metric1"], 100)
        self.assertEqual(metadata["scores"]["metric2"], "hello")

    def test_metadata_set_category_scores_with_category(self):
        set_metric("metric1", 100)
        set_metric("metric2", "hello", category="category2")

        metadata = get_metadata()
        self.assertEqual(len(metadata["scores"]), 2)
        self.assertEqual(metadata["scores"]["metric1"], 100)
        self.assertEqual(metadata["scores"]["category2"]["metric2"], "hello")

    def test_metadata_set_category_scores_with_category_title(self):
        set_metric("metric1", 100)
        set_metric("metric2", "hello", category="category2", category_title="Category Two")

        metadata = get_metadata()
        self.assertEqual(len(metadata["scores"]), 2)
        self.assertEqual(metadata["scores"]["metric1"], 100)
        self.assertEqual(metadata["scores"]["category2"]["title"], "Category Two")
        self.assertEqual(metadata["scores"]["category2"]["metric2"], "hello")

    def test_metadata_set_category_scores_with_category_title_and_scores_extras(self):
        set_metric("metric1", 100)
        set_metric(
            "metric2",
            "hello",
            priority=1,
            title="VIP Metric",
            category="category2",
            category_title="Category Two",
            category_subtitle="A longer subtitle",
        )

        metadata = get_metadata()
        self.assertEqual(len(metadata["scores"]), 2)
        self.assertEqual(metadata["scores"]["metric1"], 100)

        self.assertEqual(metadata["scores"]["category2"]["title"], "Category Two")
        self.assertEqual(metadata["scores"]["category2"]["subtitle"], "A longer subtitle")
        self.assertEqual(metadata["scores"]["category2"]["metric2"]["value"], "hello")
        self.assertEqual(metadata["scores"]["category2"]["metric2"]["priority"], 1)
        self.assertEqual(metadata["scores"]["category2"]["metric2"]["title"], "VIP Metric")

    def test_metadata_scores_scikit_boston(self):
        data = sklearn.datasets.load_boston()
        df = pandas.DataFrame(data.data, columns=data.feature_names)
        target = pandas.DataFrame(data.target, columns=["MEDV"])

        X = df
        y_true = target["MEDV"]
        lm = linear_model.LinearRegression()
        model = lm.fit(X, y_true)
        y_pred = lm.predict(X)

        # save a variety of scores derived from model, data and predictions
        set_model_metrics(model, y_true, y_pred)

        # check regressor scores were saved correctly 
        scores = get_metadata()["scores"]["sklearn_metrics"]
        self.assertIn("mean_abs_error", scores)
        self.assertAlmostEqual(scores["mean_abs_error"]["value"], 3.27086, 2)
        self.assertIn("mean_squared_error", scores)
        self.assertAlmostEqual(scores["mean_squared_error"]["value"], 21.89483, 2)
        self.assertIn("median_abs_error", scores)
        self.assertAlmostEqual(scores["median_abs_error"]["value"], 2.45231, 2)
