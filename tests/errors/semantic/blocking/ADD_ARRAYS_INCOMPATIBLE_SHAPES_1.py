# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np
from pyccel.decorators import types

@types('int')
def f(k):
    a = np.ones(k)
    c = a[1:] + a[2:]
    return c[0]
