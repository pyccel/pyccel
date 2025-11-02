#$ header metavar printer_imports="stdlib, stdint, complex, stc/cspan, stdio, inttypes, stc/vec, stdbool, STC_Extensions/List_extensions"
from typing import Final, TypeVar
from typing import Final, TypeVar, overload
from pyccel.decorators import low_level
from numpy import float64

T = TypeVar('T', 'int', 'float', 'complex')
S = TypeVar('S', 'Final[int]', 'Final[float]', 'Final[complex]')

@low_level('runtest_stub__A')
class A:
    _x : 'int'
    is_freed : 'bool'
    
    @low_level('runtest_stub__A__init')
    def __init__(self : 'A', x : 'int') -> None:
        ...
    
    @low_level('runtest_stub__A__x')
    def x(self : 'A') -> 'int':
        ...
    
    @low_level('runtest_stub__A__drop')
    def __del__(self : 'A') -> None:
        ...

@low_level('runtest_stub__f')
def f(a : 'int', b : 'float64[:]') -> 'tuple[int, float64]':
    ...

@low_level('runtest_stub__g')
def g() -> 'float':
    ...

@low_level('runtest_stub__h')
def h(arg : 'Final[list[int]]') -> None:
    ...

@low_level('runtest_stub__m')
def m(b : 'int') -> 'int':
    ...

@low_level('runtest_stub__n')
def n(arg : 'Final[list[int]]') -> None:
    ...

@low_level('runtest_stub__high_int_1')
def high_int_1(function : '(int)(int)', a : 'int') -> 'int':
    ...

@low_level('runtest_stub__k_0000')
@overload
def k(a : 'int') -> 'tuple[int, int, int]':
    ...

@low_level('runtest_stub__k_0001')
@overload
def k(a : 'float') -> 'tuple[float, float, float]':
    ...

@low_level('runtest_stub__k_0002')
@overload
def k(a : 'complex') -> 'tuple[complex, complex, complex]':
    ...

@low_level('runtest_stub__l_0000')
@overload
def l(a : 'int') -> 'tuple[int, ...]':
    ...

@low_level('runtest_stub__l_0001')
@overload
def l(a : 'float') -> 'tuple[float, ...]':
    ...

@low_level('runtest_stub__l_0002')
@overload
def l(a : 'complex') -> 'tuple[complex, ...]':
    ...

@low_level('runtest_stub__p_0000')
@overload
def p(a : 'int') -> 'float':
    ...

@low_level('runtest_stub__p_0001')
@overload
def p(a : 'float') -> 'float':
    ...

@low_level('runtest_stub__p_0002')
@overload
def p(a : 'complex') -> 'complex':
    ...

