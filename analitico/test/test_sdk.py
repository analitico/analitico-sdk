import unittest
import os.path
import pandas as pd
import random
import string
import io
import json
import datetime

import analitico
from analitico import logger, authorize_sdk
from analitico.factory import Factory
from analitico.schema import generate_schema

from .test_mixin import TestMixin

# pylint: disable=no-member

ANALITICO_TEST_TOKEN = os.environ["ANALITICO_TEST_TOKEN"]
ANALITICO_TEST_WORKSPACE_ID = os.environ["ANALITICO_TEST_WORKSPACE_ID"]
assert ANALITICO_TEST_TOKEN, "The enviroment variable ANALITICO_TEST_TOKEN should be set with a test token."
assert (
    ANALITICO_TEST_WORKSPACE_ID
), "The enviroment variable ANALITICO_TEST_WORKSPACE_ID should be set with a test token."


TITANIC_PUBLIC_URL = "https://storage.googleapis.com/public.analitico.ai/data/titanic/train.csv"


class SDKTests(unittest.TestCase, TestMixin):
    """ Testing of Factory/sdk functionality: caching, creating items, plugins, etc """

    # Create a factory/sdk with out test token that can
    # access our test workspace in read and write model
    sdk = analitico.authorize_sdk(
        token=ANALITICO_TEST_TOKEN,
        workspace_id=ANALITICO_TEST_WORKSPACE_ID,
        endpoint="https://staging.analitico.ai/api/",
    )

    def setUp(self):
        logger.info("SDKTests.setUp")

    ##
    ## get
    ##

    def test_sdk_get_item(self):
        item = self.sdk.get_item("rx_ho374b88")
        self.assertIsNotNone(item)
        self.assertIsInstance(item, analitico.Recipe)
        self.assertTrue(item.id.startswith(analitico.RECIPE_PREFIX))
        self.assertEqual(item.type, "analitico/recipe")

    def test_sdk_get_recipe(self):
        recipe = self.sdk.get_recipe("rx_ho374b88")
        self.assertIsNotNone(recipe)
        self.assertIsInstance(recipe, analitico.Recipe)
        self.assertTrue(recipe.id.startswith(analitico.RECIPE_PREFIX))
        self.assertEqual(recipe.type, "analitico/recipe")

    ##
    ## create
    ##

    def test_sdk_create_item(self):
        item = None
        try:
            item = self.sdk.create_item(analitico.DATASET_TYPE)
            self.assertIsNotNone(item)
            self.assertIsInstance(item, analitico.Dataset)
            self.assertTrue(item.id.startswith(analitico.DATASET_PREFIX))
            self.assertEqual(item.type, "analitico/dataset")
        finally:
            if item:
                item.delete()

    def test_sdk_create_item_with_title(self):
        item = None
        try:
            title = "Testing at " + datetime.datetime.utcnow().isoformat()
            item = self.sdk.create_item(analitico.DATASET_TYPE, title=title)
            self.assertEqual(item.get_attribute("title"), title)
            self.assertEqual(item.title, title)
        finally:
            if item:
                item.delete()

    ##
    ## save
    ##

    def test_sdk_save_item_with_updates(self):
        item = None
        try:
            title = "Testing at " + datetime.datetime.utcnow().isoformat()
            item = self.sdk.create_item(analitico.DATASET_TYPE, title=title)
            self.assertEqual(item.get_attribute("title"), title)
            self.assertEqual(item.title, title)

            # update title, save updates on service
            title_v2 = title + " v2"
            item.title = title_v2
            item.save()
            self.assertEqual(item.get_attribute("title"), title_v2)
            self.assertEqual(item.title, title_v2)

            # retrieve new object from service
            item_again = self.sdk.get_item(item.id)
            self.assertEqual(item_again.get_attribute("title"), title_v2)
            self.assertEqual(item_again.title, title_v2)

        finally:
            if item:
                item.delete()

    ##
    ## upload
    ##

    def test_sdk_upload_dataframe(self):
        dataset = None
        try:
            title = "Boston test at " + datetime.datetime.utcnow().isoformat()
            dataset = self.sdk.create_item(analitico.DATASET_TYPE, title=title)

            from sklearn.datasets import load_boston

            boston_dataset = load_boston()
            boston_df = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
            dataset.upload(df=boston_df)

        finally:
            if dataset:
                dataset.delete()

    def test_upload_ilpes(self):
        sdk = analitico.authorize_sdk(
            endpoint="https://staging.analitico.ai/api/", token="tok_pbn6gxbj", workspace_id="ws_2q2fq3wg"
        )

        dataset = sdk.get_dataset("ds_aqp195k4")

        from sklearn.datasets import load_boston

        boston_dataset = load_boston()
        boston_df = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
        dataset.upload("boston.parquet", df=boston_df)
