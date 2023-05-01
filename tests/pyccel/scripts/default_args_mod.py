# pylint: disable=missing-function-docstring, missing-module-docstring
# coding: utf-8
from pyccel.decorators import types

#------------------------------------------------------------------------------
@types('int')
def f1(x = 1):
    y = x - 1
    return y

@types('real [:]','int')
def f5(x, m1 = 2):
    x[:] = 0.
    for i in range(0, m1):
        x[i] = i * 1.

#------------------------------------------------------------------------------
@types('real','real')
def f3(x = 1.5, y = 2.5):
    return x+y

@types('bool')
def is_nil_default_arg(a = None):
    c = False
    if a is None:
        c = True
    return c

@types('bool')
def is_not_nil(z = None):
    c = False
    if z is not None:
        c = True
    return c

@types('real','real','bool')
def recursivity(x, y = 0.0, z = None):

    tmp = is_not_nil(z)
    if (tmp):
        y = 2.5
    return x + y

def print_var(n : int = 0):
    print(n)
