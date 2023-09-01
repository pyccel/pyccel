# pylint: disable=missing-function-docstring, missing-module-docstring


def f6(m1 : 'int', m2 : 'int', x : 'float [:,:]'):
    x[:,:] = 0.
    for i in range(0, m1):
        for j in range(0, m2):
            x[i,j] = (2*i+j) * 1.

def h(x : 'float [:]'):
    x[2] = 8.
