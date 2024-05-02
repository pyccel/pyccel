from pyccel.decorators import types
from pyccel.decorators import external

@external
@types('real [:]')
def f(x):
    x[0] = 2.

@external
@types('real [:]')
def g(x):
    x[1] = 4.

@types('real [:]')
def h(x):
    x[2] = 8.

