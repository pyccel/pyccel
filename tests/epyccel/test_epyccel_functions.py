# coding: utf-8

import pytest
import numpy as np
import shutil

from pyccel.epyccel import epyccel
from pyccel.decorators import types
from conftest       import *


def clean_test():
    shutil.rmtree('__pycache__', ignore_errors=True)
    shutil.rmtree('__epyccel__', ignore_errors=True)


def test_func_no_args_f1():
    def f1():
        from numpy import pi
        value = (2*pi)**(3/2)
        return value

    f = epyccel(f1)
    assert abs(f()-f1()) < 1e-13
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
def test_decorator_f3():
    @types('int [:]')
    def f3(x):
        from numpy import empty_like
        y = empty_like(x)
        y[:] = x - 1
        return y

    with pytest.raises(RuntimeError):
        epyccel(f3)

#------------------------------------------------------------------------------
def test_decorator_f4():
    @types('real [:,:]')
    def f4(x):
        from numpy import empty_like
        y = empty_like(x)
        y[:] = x - 1.0
        return y

    with pytest.raises(RuntimeError):
        epyccel(f4)

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
def test_decorator_f6():
    @types('int', 'int', 'real [:,:]')
    def f6_1(m1, m2, x):
        x[:,:] = 0.
        for i in range(0, m1):
            for j in range(0, m2):
                x[i,j] = (2*i+j) * 1.

    f = epyccel(f6_1)

    # ...
    m1 = 2 ; m2 = 3

    x = np.zeros((m1,m2))
    f(m1, m2, x)

    x_expected = np.zeros((m1,m2))
    f6_1(m1, m2, x_expected)

    assert np.allclose( x, x_expected, rtol=1e-15, atol=1e-15 )
    # ...

#------------------------------------------------------------------------------
# in order to call the pyccelized function here, we have to create x with
# Fortran ordering
def test_decorator_f7():

    @types('int', 'int', 'real [:,:](order=F)')
    def f7(m1, m2, x):
        x[:,:] = 0.
        for i in range(0, m1):
            for j in range(0, m2):
                x[i,j] = (2*i+j) * 1.

    f = epyccel(f7)

    # ...
    m1 = 2 ; m2 = 3
    x_expected = np.zeros((m1,m2))
    f7(m1, m2, x_expected)

    x = np.zeros((m1,m2), order='F')
    f(m1, m2, x)

    assert np.allclose( x, x_expected, rtol=1e-15, atol=1e-15 )
    # ...

#------------------------------------------------------------------------------
def test_decorator_f8():
    @types('int','bool')
    def f8(x,b):
        a = x if b else 2
        return a

    f = epyccel(f8)

    # ...
    assert f(3,True)  == f8(3,True)
    assert f(3,False) == f8(3,False)
    # ...


def test_arguments_f9():
    @types('int64[:]')
    def f9(x):
        x += 1

    f = epyccel(f9)

    x = np.zeros(10, dtype='int64')
    x_expected = x.copy()

    f9(x)
    f(x_expected)
    assert np.array_equal(x, x_expected)

def test_arguments_f10():
    @types('int64[:]')
    def f10(x):
        x[:] += 1

    f = epyccel(f10)

    x = np.zeros(10, dtype='int64')
    x_expected = x.copy()

    f10(x)
    f(x_expected)
    assert np.array_equal(x, x_expected)

##==============================================================================
## CLEAN UP GENERATED FILES AFTER RUNNING TESTS
##==============================================================================
#
#def teardown_module():
#    clean_test()
#
