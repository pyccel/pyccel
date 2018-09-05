# coding: utf-8
from pyccel.decorators import types
@types(int)
def f(x):
    y = x+1
    return y
f(2)
from pyccel.epyccel import epyccel
g = epyccel(f)
