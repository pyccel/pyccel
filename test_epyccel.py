# coding: utf-8
from pyccel.decorators import types
@types(int)
def f_py(x):
    y = x+1
    return y

from pyccel.epyccel import epyccel
f_f90 = epyccel(f_py)
assert(f_py(3) == f_f90(3))
