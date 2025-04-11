# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
from typing import Final
from pyccel.decorators import template

def f(a : int, b : 'float[:]'):
    return (a, b[0])

def g():
    return 2.5

def h(arg : Final[list[int]]):
    print(arg[0])

@template('T', [int, float, complex])
def k(a : 'T'):
    return (a, 2*a, 3*a)

@template('T', [int, float, complex])
def l(a : 'T'):
    tup = (a, 2*a, 3*a)
    for i in range(3):
        print(tup[i])
    return tup

def m(b : int):
    B = b + 2
    return B

class A:
    def __init__(self, x : int):
        self._x = x

    @property
    def x(self):
        return self._x
