from typing import Final, TypeVar, overload
from pyccel.decorators import low_level
from numpy import float64

T = TypeVar('T', 'int', 'float', 'complex')
S = TypeVar('S', 'Final[int]', 'Final[float]', 'Final[complex]')

@low_level('A')
class A:
    _x : 'int'
    is_freed : 'bool'
    
    @low_level('__init__')
    def __init__(self : 'A', x : 'int') -> None:
        ...
    
    @property
    @low_level('x')
    def x(self : 'A') -> 'int':
        ...
    
    @low_level('__del__')
    def __del__(self : 'A') -> None:
        ...

@low_level('f')
def f(a : 'int', b : 'float64[:]') -> 'tuple[int, float64]':
    ...

@low_level('g')
def g() -> 'float':
    ...

@low_level('h')
def h(arg : 'Final[list[int]]') -> None:
    ...

@low_level('m')
def m(b : 'int') -> 'int':
    ...

@low_level('n')
def n(arg : 'Final[list[int]]') -> None:
    ...

@low_level('high_int_1')
def high_int_1(function : '(int)(int)', a : 'int') -> 'int':
    ...

@low_level('k_0000')
@overload
def k(a : 'int') -> 'tuple[int, int, int]':
    ...

@low_level('k_0001')
@overload
def k(a : 'float') -> 'tuple[float, float, float]':
    ...

@low_level('k_0002')
@overload
def k(a : 'complex') -> 'tuple[complex, complex, complex]':
    ...

@low_level('l_0000')
@overload
def l(a : 'Final[int]') -> 'tuple[int, ...]':
    ...

@low_level('l_0001')
@overload
def l(a : 'Final[float]') -> 'tuple[float, ...]':
    ...

@low_level('l_0002')
@overload
def l(a : 'Final[complex]') -> 'tuple[complex, ...]':
    ...

@low_level('p_0000')
@overload
def p(a : 'int') -> 'float':
    ...

@low_level('p_0001')
@overload
def p(a : 'float') -> 'float':
    ...

@low_level('p_0002')
@overload
def p(a : 'complex') -> 'complex':
    ...

