# pylint: disable=missing-function-docstring, missing-module-docstring

def add_one(x : 'int'):
    return x + 1

def func(x  : 'int' =  None):
    if x is None:
        return 2
    return add_one(x)
