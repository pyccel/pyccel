# coding: utf-8

import pytest
import numpy as np
import os

from pyccel.epyccel import epyccel
from pyccel.decorators import types

#VERBOSE = False
VERBOSE = True

def clean_test():
    cmd = 'rm -rf __pycache__/*'
    os.system(cmd)


#------------------------------------------------------------------------------
def test_decorator_f1():
    @types('int')
    def f1(x):
        y = x - 1
        return y

    f = epyccel(f1)

    # ...
    assert f(3) == f1(3)
    # ...

#------------------------------------------------------------------------------
def test_decorator_f2():
    @types('int [:]')
    def f2(x):
        y = x[0] - 1
        return y

    f = epyccel(f2)

    # ...
    x = np.array([3, 4, 5, 6], dtype=int)
    assert f(x) == f2(x)
    # ...

    # ...
    x = [3, 4, 5, 6]
    assert f(x) == f2(x)
    # ...

#------------------------------------------------------------------------------
# TODO we need to pass the size of the returned array for the moment
def test_decorator_f3():
    @types('int [:]')
    def f3(x):
        y = x - 1
        return y

    f = epyccel(f3)

    # ...
    x = np.array([3, 4, 5, 6], dtype=int)
    assert np.array_equal( f(x, len(x)), f3(x) )
    # ...

#------------------------------------------------------------------------------
# TODO we need to pass the shape of the returned array for the moment
def test_decorator_f4():
    @types('real [:,:]')
    def f4(x):
        y = x - 1.0
        return y

    f = epyccel(f4)

    # ...
    x = np.random.random((2, 3))
    assert np.allclose( f(x, *x.shape), f4(x), rtol=1e-15, atol=1e-15 )
    # ...

#------------------------------------------------------------------------------
def test_decorator_f5():
    @types('int', 'real [:]')
    def f5(m1, x):
        x[:] = 0.
        for i in range(0, m1):
            x[i] = i * 1.

    f = epyccel(f5)

    # ...
    m1 = 3

    x = np.zeros(m1)
    f(m1, x)

    x_expected = np.zeros(m1)
    f5(m1, x_expected)

    assert np.allclose( x, x_expected, rtol=1e-15, atol=1e-15 )
    # ...

#------------------------------------------------------------------------------
# TODO remove transpose
def test_decorator_f6():
    @types('int', 'int', 'real [:,:]')
    def f6(m1, m2, x):
        x[:,:] = 0.
        for i in range(0, m1):
            for j in range(0, m2):
                x[i,j] = (2*i+j) * 1.

    f = epyccel(f6)

    # ...
    m1 = 2 ; m2 = 3

    x = np.zeros((m1,m2))
    f(m1, m2, x.transpose())

    x_expected = np.zeros((m1,m2))
    f6(m1, m2, x_expected)

    assert np.allclose( x, x_expected, rtol=1e-15, atol=1e-15 )
    # ...



##==============================================================================
## CLEAN UP GENERATED FILES AFTER RUNNING TESTS
##==============================================================================
#
#def teardown_module():
#    clean_test()
#

######################################
if __name__ == '__main__':
    test_decorator_f1()
    test_decorator_f2()
    test_decorator_f3()
    test_decorator_f4()
    test_decorator_f5()
    test_decorator_f6()

