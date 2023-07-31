# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import types
import numpy as np

a = np.ones(6)
b = np.array([1,2,3,4,5])
c = np.zeros((2,3), dtype=np.int32)
d = np.array([1+2j, 3+4j])
e = np.empty((2,3,4))
F = [False for _ in range(5)]

def update_a():
    a[:] = a+1

def reset_a():
    a[:] = 1

def reset_c():
    c[:] = 0

def reset_e():
    for i in range(2):
        for j in range(3):
            for k in range(4):
                e[i,j,k] = i*12+j*4+k

def get_elem_a(i : int):
    return a[i]

def get_elem_b(i : int):
    return b[i]

def get_elem_c(i : int, j : int):
    return c[i,j]

def get_elem_d(i : int):
    return d[i]

def get_elem_e(i : int, j : int, k : int):
    return e[i,j,k]



reset_e()
