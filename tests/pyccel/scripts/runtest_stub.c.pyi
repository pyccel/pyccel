from typing import Final
from pyccel.decorators import template
from numpy import float64


class A:
    private_x : 'int'
    is_freed : 'bool'
    
    def __init__(self : 'A', x : 'int') -> None:
        ...
    
    def x(self : 'A') -> 'int':
        result : 'int'
        return result
    
    def __del__(self : 'A') -> None:
        ...

def f(a : 'int', b : 'float[:]') -> 'tuple[int, float64]':
    result_0 : 'int'
    result_1 : 'float64'
    return (result_0, result_1)

def g() -> 'float':
    result : 'float'
    return result

def h(arg : 'Final[list[int]]') -> None:
    ...

@overload
def k(a : 'T') -> 'tuple[int, int, int]':
    result_0 : 'int'
    result_1 : 'int'
    result_2 : 'int'
    return (result_0, result_1, result_2)

@overload
def k(a : 'T') -> 'tuple[float, float, float]':
    result_0 : 'float'
    result_1 : 'float'
    result_2 : 'float'
    return (result_0, result_1, result_2)

@overload
def k(a : 'T') -> 'tuple[complex, complex, complex]':
    result_0 : 'complex'
    result_1 : 'complex'
    result_2 : 'complex'
    return (result_0, result_1, result_2)

@overload
def l(a : 'T') -> 'tuple[int, ...]':
    tup : 'tuple[int, ...]'
    return tup

@overload
def l(a : 'T') -> 'tuple[float, ...]':
    tup : 'tuple[float, ...]'
    return tup

@overload
def l(a : 'T') -> 'tuple[complex, ...]':
    tup : 'tuple[complex, ...]'
    return tup

def m(b : 'int') -> 'int':
    B : 'int'
    return B
