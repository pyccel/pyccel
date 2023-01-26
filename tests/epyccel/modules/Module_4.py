# pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.decorators import types

@types('int')
def basic_optional(a = None):
    if a is  None :
        return 2
    return a

@types()
def call_optional_1():
    return basic_optional()

@types('int')
def call_optional_2(b = None):
    return basic_optional(b)
