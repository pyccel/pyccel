# pylint: disable=missing-function-docstring, disable=unused-variable, missing-module-docstring/
import numpy as np
from pyccel.decorators import types

def single_return():
    a = np.array([1,2,3,4])
    return a

@types('bool', 'bool')
@types('int', 'int')
@types('float', 'float')
@types('complex', 'complex')
def f(a, b):
    c = np.array([a,b])
    return c

a = single_return()
b = f(1, 3)
c = f(1., 3.)
d = f(False, True)
e = f(1+2j, 3+4j)

print(a, b, c, d, e)
