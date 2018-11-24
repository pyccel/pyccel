# coding: utf-8

import pytest
import numpy as np
import os

from pyccel.epyccel import epyccel
from pyccel.decorators import types


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
def test_decorator_f3():
    @types('int [:]')
    def f3(x):
        y = x - 1
        return y

    from pyccel.ast import AstFunctionResultError
    with pytest.raises(AstFunctionResultError):
        f = epyccel(f3)

#------------------------------------------------------------------------------
def test_decorator_f4():
    @types('real [:,:]')
    def f4(x):
        y = x - 1.0
        return y

    from pyccel.ast import AstFunctionResultError
    with pytest.raises(AstFunctionResultError):
        f = epyccel(f4)

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
def test_decorator_f6_1():
    @types('int', 'int', 'real [:,:]')
    def f6_1(m1, m2, x):
        x[:,:] = 0.
        for i in range(0, m1):
            for j in range(0, m2):
                x[i,j] = (2*i+j) * 1.

    # default value for assert_contiguous is False
    f = epyccel(f6_1, assert_contiguous=False)

    # ...
    m1 = 2 ; m2 = 3

    x = np.zeros((m1,m2))
    f(m1, m2, x)

    x_expected = np.zeros((m1,m2))
    f6_1(m1, m2, x_expected)

    assert np.allclose( x, x_expected, rtol=1e-15, atol=1e-15 )
    # ...

#------------------------------------------------------------------------------
# in order to call the pyccelized function here, we have either to
#   - give the transpose view of x: x.transpose()
#   - create x with Fortran ordering
def test_decorator_f6_2():
    @types('int', 'int', 'real [:,:]')
    def f6_2(m1, m2, x):
        x[:,:] = 0.
        for i in range(0, m1):
            for j in range(0, m2):
                x[i,j] = (2*i+j) * 1.

    f = epyccel(f6_2, assert_contiguous=True)

    # ...
    m1 = 2 ; m2 = 3

    x_expected = np.zeros((m1,m2))
    f6_2(m1, m2, x_expected)
    # ...

    # ... BAD CALL
    x = np.zeros((m1,m2))
    with pytest.raises(ValueError):
        #  in this case we should get the following error
        #  ValueError: failed to initialize intent(inout) array -- input not fortran contiguous
        f(m1, m2, x)
    # ...

    # ... GOOD CALL
    x = np.zeros((m1,m2))
    f(m1, m2, x.transpose())

    assert np.allclose( x, x_expected, rtol=1e-15, atol=1e-15 )
    # ...

#------------------------------------------------------------------------------
# in order to call the pyccelized function here, we have either to
#   - give the transpose view of x: x.transpose()
#   - create x with Fortran ordering
def test_decorator_f6_3():

    @types('int', 'int', 'real [:,:](order=F)')
    def f6_3(m1, m2, x):
        x[:,:] = 0.
        for i in range(0, m1):
            for j in range(0, m2):
                x[i,j] = (2*i+j) * 1.

    f = epyccel(f6_3, assert_contiguous=True)

    m1 = 2 ; m2 = 3
    x_expected = np.zeros((m1,m2))
    f6_3(m1, m2, x_expected)

    # ... GOOD CALL
    x = np.zeros((m1,m2), order='F')
    f(m1, m2, x)

    x = np.ascontiguousarray(x)
    assert np.allclose( x, x_expected, rtol=1e-15, atol=1e-15 )
    # ...



##==============================================================================
## CLEAN UP GENERATED FILES AFTER RUNNING TESTS
##==============================================================================
#
#def teardown_module():
#    clean_test()
#
