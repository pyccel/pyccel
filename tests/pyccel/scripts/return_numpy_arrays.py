# pylint: disable=missing-function-docstring, disable=unused-variable, missing-module-docstring
import numpy as np
from pyccel.decorators import types

def single_return():
    a = np.array([1,2,3,4])
    return a

def multi_returns():
    x = np.ones(5)
    y = np.array([1,2,3,4,5])
    return x, y

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
h,g = multi_returns()
k = single_return() + 1

if __name__ == '__main__':
    print(a, b, c, d, e, h, g, k)
