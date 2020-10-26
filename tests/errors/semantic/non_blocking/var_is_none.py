# pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.decorators import types

@types('int')
def f(a):
    b = 0
    if a is not None:
        b = b + a
    return b
