# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import types

@types('int')
@types('int[:]')
def f(a):
    return a+3
