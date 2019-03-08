from .interfaces import IFactory
from .constants import *
from .exceptions import *

import analitico.interfaces
import analitico.constants
import analitico.utilities
import analitico.mixin
import analitico.plugin
import analitico.dataset
import analitico.factory
import analitico.status


def authorize(token=None, endpoint=analitico.constants.ANALITICO_STAGING_API_ENDPOINT) -> IFactory:
    """ Returns an API factory which can create datasets, models, plugins, etc """
    return analitico.factory.Factory(token=token, endpoint=endpoint)
