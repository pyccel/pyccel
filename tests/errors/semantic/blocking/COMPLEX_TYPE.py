# pylint: disable=missing-function-docstring, missing-module-docstring

def f():
    import numpy as np
    a = np.zeros(10)
    b = np.ones(9,dtype=type(a))
    return b[0]
