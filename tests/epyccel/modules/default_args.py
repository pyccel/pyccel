# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import TypeVar


def f1(x: "int" = 1):
    y = x - 1
    return y


def f2(x: "float [:]", m1: "int" = 2):
    x[:] = 0.0
    for i in range(0, m1):
        x[i] = i * 1.0


def f3(x: "float" = 1.5, y: "float" = 2.5):
    return x + y


def f4(x: "bool" = True):
    if x:
        return 1
    else:
        return 2


def f5(x: "complex" = 1j):
    y = x - 1
    return y


T_max_abs = TypeVar("T_max_abs", float, complex)


def max_abs(a: T_max_abs, b: T_max_abs = 3.0) -> float:
    if b is None:
        return abs(a)
    else:
        return max(abs(a), abs(b))
