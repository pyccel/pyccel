# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import types


@types('int')
def func(n,m):
    return n + m
