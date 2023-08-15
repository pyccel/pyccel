# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np

def f(k : 'int'):
    a = np.ones(k)
    c = a[1:] + a[2:]
    return c[0]
