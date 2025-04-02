# pylint: disable=missing-function-docstring, missing-module-docstring

def f():
    a = [1, 2, 3]
    b = [[4,5,6]]
    b.insert(0, a)
    return b
