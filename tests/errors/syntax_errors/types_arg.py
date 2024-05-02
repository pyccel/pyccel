# pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.decorators import types

@types(x='int',y='double')
def f(x,y):
    pass
