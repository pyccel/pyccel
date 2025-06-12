# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import TypeVar
import numpy as np

def single_return():
    a = np.array([1,2,3,4])
    return a

def multi_returns():
    x = np.ones(5)
    y = np.array([1,2,3,4,5])
    return x, y

T = TypeVar('T', bool, int, float, complex)

def f(a : T, b : T):
    c = np.array([a,b])
    return c

a = single_return()
b = f(1, 3)
c = f(1., 3.)
d = f(False, True)
e = f(1+2j, 3+4j)
h,g = multi_returns()
k = single_return() + 1

def Mi(x: float, M0: 'float[:, :]') -> 'float[:, :]':
    M1 = x * M0 + 1.
    return M1

def M_GEN(x: float) -> 'float[:, :, :]':
    M = np.ones((4, 4), dtype = float)
    M2 = np.empty((3, 4, 4), dtype = float)
    for i in range(3):
        M2[i] = Mi(x * i, M)
    return M2

if __name__ == '__main__':
    print(a, b, c, d, e, h, g, k)
    print(np.array([1,2,3,4]), np.array([1, 3]), np.array([1., 3.]), np.array([False, True]), np.array([1+2j, 3+4j]))

    x = M_GEN(2.0)
    print(x)
