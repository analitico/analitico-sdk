import sys


class AnaliticoException(Exception):
    """ Base exception used in the project that can carry extra information with it in the form of a dictionary """

    exception = None

    message = None

    code = None

    extra = {}

    def __init__(self, msg, *args, code=None, exception=None, extra=None, **kwargs):
        self.message = msg % (args)

        # retain exception chain
        self.exception = exception if exception else sys.exc_info()[1]

        if "extra" in kwargs:
            self.extra = kwargs.pop("extra")
        for key, value in kwargs.items():
            self.extra[key] = value

    def __str__(self):
        return self.message
