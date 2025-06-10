# pylint: disable=missing-function-docstring, missing-module-docstring
from math import gcd

def f(a : int, b : int):
    s = gcd(a, b)
    return s + 1
