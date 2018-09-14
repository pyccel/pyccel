# coding: utf-8

import pytest
import numpy as np
import os

from pyccel.epyccel import epyccel
from pyccel.decorators import types

def clean_test():
    cmd = 'rm -f *.f90 *.so'
    os.system(cmd)

# ..............................................
#  functions for which we will provide headers
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

# ..............................................
#        functions with types decorator
# ..............................................
@types(int)
def g1(x):
    y = x+1
    return y

@types('int [:]')
def g2(x):
    y = x + 1
    return y

@types(int,int,int)
def g3(x, n=2, m=3):
    y = x - n*m
    return y

@types(int,int)
def g4(x, m=None):
    if m is None:
        y = x + 1
    else:
        y = x - 1
    return y


#==============================================================================
# TEST FUNCTIONS WITH HEADERS
#==============================================================================

def test_header_f1():
    f = epyccel(f1, '#$ header procedure f1(int)')

    # ...
    assert f(3) == f1(3)
    # ...

    clean_test()

#------------------------------------------------------------------------------
def test_header_f2():
    f = epyccel(f2, '#$ header procedure f2(int [:])')

    # ...
    x = np.array([3, 4, 5, 6], dtype=int)
    assert f(x) == f2(x)
    # ...

    # ...
    x = [3, 4, 5, 6]
    assert f(x) == f2(x)
    # ...

    clean_test()

#------------------------------------------------------------------------------
# TODO: functions returning arrays are not supported yet
@pytest.mark.xfail
def test_header_f3():

    f = epyccel(f3, '#$ header procedure f3(int [:])')

    # ...
    x = np.array([3, 4, 5, 6], dtype=int)
    assert np.allclose(f(x), f3(x))
    # ...

    clean_test()

#------------------------------------------------------------------------------
def test_header_f4():
    f = epyccel(f4, '#$ header procedure f4(double [:,:])')

    # ...
    x = np.random.random((2, 3))
    assert np.allclose(f(x), f4(x))
    # ...

    clean_test()

#------------------------------------------------------------------------------
def test_header_f5():
    f = epyccel(f5, '#$ header procedure f5(int, double [:])')

    # ...
    m1 = 3

    x = np.zeros(m1)
    f(m1, x)

    x_expected = np.zeros(m1)
    f5(m1, x_expected)

    assert np.allclose(x, x_expected)
    # ...

    clean_test()

#------------------------------------------------------------------------------
def test_header_f6():
    f = epyccel(f6, '#$ header procedure f6(int, int, double [:,:](order = F))')

    # ...
    m1 = 2 ; m2 = 3

    x = np.zeros((m1,m2), order='F')
    f(m1, m2, x)

    x_expected = np.zeros((m1,m2), order='F')
    f6(m1, m2, x_expected)

    assert np.allclose(x, x_expected)
    # ...

    clean_test()

#==============================================================================
# TEST FUNCTIONS WITH DECORATORS
#==============================================================================

def test_decorators_g1():

    f = epyccel(g1)
    assert f(3) == g1(3)

    clean_test()

#------------------------------------------------------------------------------
# TODO: functions returning arrays are not supported yet
@pytest.mark.xfail
def test_decorators_g2():

    f = epyccel(g2)

    x = np.array([2, 3, 4], dtype=int)
    x_expected = x.copy()

    f(x)
    g2(x_expected)

    assert np.allclose(x, x_expected)

    clean_test()

#------------------------------------------------------------------------------
def test_decorators_g3():

    f = epyccel(g3)
    assert f(3,2,4) == g3(3,2,4)

    clean_test()

#------------------------------------------------------------------------------
# TODO: not working yet. optional arg is placed before out!
@pytest.mark.xfail
def test_decorators_g4():

    f = epyccel(g4)
    assert f(3) == g4(3)

    print(f(3,2))
    print(g4(3,2))
    assert f(3, 2) == g4(3, 2)
    # ...

    clean_test()

#==============================================================================
# TEST MODULE
#==============================================================================

# TODO: functions returning arrays are not supported yet
@pytest.mark.xfail
def test_module1():

    import mod_test1 as mod

    # ...
    m = epyccel(mod)
    assert m.g1(3) == mod.g1(3)
    # ...

    # ...
    x = np.array([2, 3, 4], dtype=int)
    x_expected = x.copy()

    m.g2(x)
    mod.g2(x_expected)

    assert np.allclose(x, x_expected)
    # ...

    clean_test()

#==============================================================================
# CLEAN UP GENERATED FILES AFTER RUNNING TESTS
#==============================================================================

def teardown_module():
    clean_test()
