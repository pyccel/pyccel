# pylint: disable=missing-function-docstring, missing-module-docstring


def f(a : int, b : int, c: int = 0):
    d = a * b + c
    e = a - b
    g = d + e + a
    g += b
    return g
