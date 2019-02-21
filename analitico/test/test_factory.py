import unittest

import pandas as pd

from analitico.factory import Factory
from analitico.schema import generate_schema

from .test_mixin import TestMixin

# pylint: disable=no-member

TITANIC_PUBLIC_URL = "https://storage.googleapis.com/public.analitico.ai/data/titanic/train.csv"


class FactoryTests(unittest.TestCase, TestMixin):
    """ Unit testing of Factory functionality: caching, creating items, plugins, etc """

    def test_factory_get_url_stream(self):
        factory = Factory()
        stream = factory.get_url_stream(TITANIC_PUBLIC_URL)
        df = pd.read_csv(stream)

        self.assertEqual(len(df), 891)
        self.assertEqual(df.columns[1], "Survived")
        self.assertEqual(df.loc[0, "Name"], "Braund, Mr. Owen òèéàù Harris")
