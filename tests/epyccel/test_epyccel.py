# coding: utf-8

import numpy as np
import os

from pyccel.epyccel import epyccel
from pyccel.decorators import types

def clean_test():
    cmd = 'rm -f *.f90 *.so'
    os.system(cmd)

# ..............................................
def f1(x):
    y = x - 1
    return y

def f2(x):
    y = x[0] - 1
    return y

def f3(x):
    y = x - 1
    return y

def f4(x):
    y = x - 1.0
    return y

def f5(m1, x):
    x[:] = 0.
    for i in range(0, m1):
        x[i] = i * 1.

def f6(m1, m2, x):
    x[:,:] = 0.
    for i in range(0, m1):
        for j in range(0, m2):
            x[i,j] = (i+j) * 1.

@types(int)
def g1(x):
    y = x+1
    return y
# ..............................................


def test_f1():
    f = epyccel(f1, '#$ header procedure f1(int)')

    # ...
    assert(f(3) == f1(3))
    # ...

    clean_test()

def test_f2():
    f = epyccel(f2, '#$ header procedure f2(int [:])')

    # ...
    x = np.array([3, 4, 5, 6], dtype=int)
    assert(f(x) == f2(x))
    # ...

    # ...
    x = [3, 4, 5, 6]
    assert(f(x) == f2(x))
    # ...

    clean_test()

def test_f3():
    f = epyccel(f3, '#$ header procedure f3(int [:])')

    # ...
    x = np.array([3, 4, 5, 6], dtype=int)
    assert(np.allclose(f(x), f3(x)))
    # ...

    clean_test()

def test_f4():
    f = epyccel(f4, '#$ header procedure f4(double [:,:])')

    # ...
    x = np.random.random((2, 3))
    assert(np.allclose(f(x), f4(x)))
    # ...

    clean_test()

def test_f5():
    f = epyccel(f5, '#$ header procedure f5(int, double [:])')

    # ...
    m1 = 3

    x = np.zeros(m1)
    f(m1, x)

    x_expected = np.zeros(m1)
    f5(m1, x_expected)

    assert(np.allclose(x, x_expected))
    # ...

    clean_test()

def test_f6():
    f = epyccel(f6, '#$ header procedure f6(int, int, double [:,:](order = F))')

    # ...
    m1 = 2 ; m2 = 3

    x = np.zeros((m1,m2), order='F')
    f(m1, m2, x)

    x_expected = np.zeros((m1,m2), order='F')
    f6(m1, m2, x_expected)

    assert(np.allclose(x, x_expected))
    # ...

    clean_test()

def test_g1():

    # ...
    f = epyccel(g1)
    assert(f(3) == g1(3))
    # ...

    clean_test()

if __name__ == '__main__':
    # ... using headers
    test_f1()
    test_f2()
    test_f3()
    test_f4()
    test_f5()
    test_f6()
    # ...

    # ... using decorators
    test_g1()
    # ...
