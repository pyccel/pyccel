# pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.decorators import types


@types('int', 'int', 'real [:,:]')
def f6(m1, m2, x):
    x[:,:] = 0.
    for i in range(0, m1):
        for j in range(0, m2):
            x[i,j] = (2*i+j) * 1.

@types('real [:]')
def h(x):
    x[2] = 8.
