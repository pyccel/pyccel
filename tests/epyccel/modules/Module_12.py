# pylint: disable=missing-function-docstring, missing-module-docstring

def add_2(a : float):
    return a + 2

def times_3(a : 'float|complex'):
    b = 1.0
    b = add_2(b)
    a *= b
    return a
