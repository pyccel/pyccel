# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import types


@types('real [:]')
def f(x):
    x[0] = 2.

@types('real [:]')
def g(x):
    x[1] = 4.

@types('real [:]')
def h(x):
    x[2] = 8.
