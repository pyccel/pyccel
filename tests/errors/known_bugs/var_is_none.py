# pylint: disable=missing-function-docstring, missing-module-docstring

def f(a : 'int'):
    b = 0
    if a is not None:
        b = b + a
    return b
