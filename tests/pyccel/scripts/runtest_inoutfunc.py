# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import pure

@pure
def mult_2(x : 'double[:]'):
    for i, xi in enumerate(x):
        x[i] = xi * 2

@pure
def add_2(a : 'double[:]', b : 'double[:]'):
    mult_2(a)
    b[:] = b[:] + a[:]

if __name__ == '__main__':
    import numpy as np

    x = np.ones(4)
    y = np.full_like(x,6)

    add_2(x,y)

    print(y)
