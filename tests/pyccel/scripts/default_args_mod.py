# pylint: disable=missing-function-docstring, missing-module-docstring
# coding: utf-8

#------------------------------------------------------------------------------
def f1(x  : 'int' =  1):
    y = x - 1
    return y

def f5(x : 'float [:]', m1  : 'int' =  2):
    x[:] = 0.
    for i in range(0, m1):
        x[i] = i * 1.

#------------------------------------------------------------------------------
def f3(x  : 'float' =  1.5, y  : 'float' =  2.5):
    return x+y

def is_nil_default_arg(a  : 'bool' =  None):
    c = False
    if a is None:
        c = True
    return c

def is_not_nil(z  : 'bool' =  None):
    c = False
    if z is not None:
        c = True
    return c

def recursivity(x : 'float', y  : 'float' =  0.0, z  : 'bool' =  None):

    tmp = is_not_nil(z)
    if (tmp):
        y = 2.5
    return x + y

def print_var(n : int = 0):
    print(n)

def f7(*, x  : 'float' =  1.5, y  : 'float' =  2.5):
    return x+y
