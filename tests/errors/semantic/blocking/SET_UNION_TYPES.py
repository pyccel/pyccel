# pylint: disable=missing-function-docstring, missing-module-docstring

def f():
    a = {1,2,4}
    b = {1.3,4.6, 9.2}
    c = a.union(b)
    return c
