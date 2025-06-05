# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import types

def f(a : 'int | int[:]'):
    return a+3
