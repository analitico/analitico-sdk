from .constants import *
from .exceptions import *

import analitico.factory
import analitico.utilities
import analitico.mixin
import analitico.plugin
import analitico.dataset
import analitico.status

def authorize(token=None, endpoint=ANALITICO_STAGING_API_ENDPOINT) -> analitico.factory.Factory:
    """ Returns an API factory which can create datasets, models, run notebooks, plugins, etc """
    try:
        import api.factory
        # if environment implements it, use server side factory
        return api.factory.ServerFactory(token=token, endpoint=endpoint)
    except:
        pass

    # use client side factory
    return analitico.factory.Factory(token=token, endpoint=endpoint)
