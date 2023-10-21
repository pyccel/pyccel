# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import types

@types('int', 'int', results=['int'])
@types('float', 'float', results=['float'])
def f(a, b):
    return a+b
