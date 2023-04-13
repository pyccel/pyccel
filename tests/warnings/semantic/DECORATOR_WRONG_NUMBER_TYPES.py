# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import types

@types('int', 'int')
def func(n):
    return n
