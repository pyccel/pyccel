# pylint: disable=missing-function-docstring, missing-module-docstring

def f():
    a = [1, 2, 3]
    b = [7, 8, 9]
    c = [[4,5,6]]
    c.extend([a, b])
    return c
