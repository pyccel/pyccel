# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import TypeVar
from pyccel.decorators import pure

T = TypeVar('T', int, float)

@pure
def add2(x : T, y : T):
    return x + y

@pure
def sum_to_n(n : 'int'):
    result = 0
    for i in range(1, n+1):
        result = add2(result, i)
    return result
