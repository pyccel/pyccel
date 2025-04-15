from typing import Final
from pyccel.decorators import template
from numpy import float64


class A:
    _x : 'int'
    is_freed : 'bool'
    
    def __init__(self : 'A', x : 'int') -> None:
        ...
    
    def x(self : 'A') -> 'int':
        ...
    
    def __del__(self : 'A') -> None:
        ...

def f(a : 'int', b : 'float[:]') -> 'tuple[int, float64]':
    ...

def g() -> 'float':
    ...

def h(arg : 'Final[list[int]]') -> None:
    ...

@overload
def k(a : 'T') -> 'tuple[int, int, int]':
    ...

@overload
def k(a : 'T') -> 'tuple[float, float, float]':
    ...

@overload
def k(a : 'T') -> 'tuple[complex, complex, complex]':
    ...

@overload
def l(a : 'T') -> 'tuple[int, ...]':
    ...

@overload
def l(a : 'T') -> 'tuple[float, ...]':
    ...

@overload
def l(a : 'T') -> 'tuple[complex, ...]':
    ...

def m(b : 'int') -> 'int':
    ...

def n(arg : 'Final[list[int]]') -> None:
    ...

def high_int_1(function : '(int)(int)', a : 'int') -> 'int':
    ...
