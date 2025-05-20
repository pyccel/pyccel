# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import TypeVar

O = TypeVar('O', int, float)

def tmplt_1(x : O, y : O):
    return x + y

def tst_tmplt_1():
    x = tmplt_1(5, 4)
    y = tmplt_1(6.56, 3.3)
    return x * y
