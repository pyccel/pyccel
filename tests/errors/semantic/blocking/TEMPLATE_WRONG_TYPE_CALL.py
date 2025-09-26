# Arguments types provided to multi_tmplt_2 are incompatible
# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import TypeVar

Y = TypeVar('Y', int, float)
Z = TypeVar('Z', int, float)
X = TypeVar('X', int)

def multi_tmplt_1(x : Z, y : Z, z : Y):
    return x + y + z

def multi_tmplt_2(y : X, z : Y):
    return y + z

def tst_multi_tmplt_2():
    x = multi_tmplt_2(5.4, 5.4)
    return x
