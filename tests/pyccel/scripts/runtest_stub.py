# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
from typing import Final, TypeVar

def f(a : int, b : 'float[:]'):
    return (a, b[0])

def g():
    return 2.5

def h(arg : Final[list[int]]):
    print(arg[0])

T = TypeVar('T', int, float, complex)

def k(a : T):
    return (a, 2*a, 3*a)

S = TypeVar('S', 'Final[int]', 'Final[float]', 'Final[complex]')

def l(a : 'S'):
    tup = (a, 2*a, 3*a)
    for i in range(3):
        print(tup[i])
    return tup

def m(b : int):
    B = b + 2
    return B

def n(arg : Final[list[int]]) -> None:
    print(arg[0])

def p(a : 'int | float | complex'):
    return a + 3.0

def high_int_1(function : '(int)(int)', a : int):
    x = function(a)
    return x

class A:
    def __init__(self, x : int):
        self._x = x

    @property
    def x(self):
        return self._x
