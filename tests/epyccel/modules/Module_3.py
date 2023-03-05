# pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.decorators import types

@types('int')
def add_one(x):
    return x + 1

@types('int')
def func(x = None):
    if x is None:
        return 2
    return add_one(x)
