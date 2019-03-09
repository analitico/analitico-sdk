import sys


class AnaliticoException(Exception):
    """ Base exception used in the project that can carry extra information with it in the form of a dictionary """

    default_message = "An error occurred."
    default_code = "error"

    message = None
    code = None
    extra = None

    def __init__(self, message, *args, code=None, extra=None, **kwargs):
        self.message = message % (args) if message else self.default_message
        self.code = code if code else self.default_code

        self.extra = extra if extra else {}
        for key, value in kwargs.items():
            self.extra[key] = value

    def __str__(self):
        return self.message
