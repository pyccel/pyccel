# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import types
import numpy as np

def f(a : 'int32' = 1):
    return a

def g(a : 'int32' = np.int32(1)):
    return a

def get_f():
    return f()

def get_g():
    return g()
