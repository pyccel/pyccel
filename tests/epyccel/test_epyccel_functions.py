# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8

import pytest
import numpy as np

from pyccel.epyccel import epyccel
from pyccel.decorators import types

RTOL = 2e-14
ATOL = 1e-15

def test_func_no_args_1(language):
    '''test function with return value but no args'''
    def free_gift():
        gift = 10
        return gift

    c_gift = epyccel(free_gift, language=language)
    assert c_gift() == free_gift()
    assert isinstance(c_gift(), type(free_gift()))
    unexpected_arg = 0
    with pytest.raises(TypeError):
        c_gift(unexpected_arg)

def test_func_no_args_2(language):
    '''test function with negative return value but no args'''
    def p_lose():
        lose = -10
        return lose

    c_lose = epyccel(p_lose, language=language)
    assert c_lose() == p_lose()
    assert isinstance(c_lose(), type(p_lose()))
    unexpected_arg = 0
    with pytest.raises(TypeError):
        c_lose(unexpected_arg)

@pytest.mark.parametrize( 'language', [
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python),
    ]
)
def test_func_no_return_1(language):
    '''Test function with args and no return '''
    @types(int)
    def p_func(x):
        x *= 2

    c_func = epyccel(p_func, language=language)
    x = np.random.randint(100)
    assert c_func(x) == p_func(x)
    # Test type return sould be NoneType
    x = np.random.randint(100)
    assert isinstance(c_func(x), type(p_func(x)))

def test_func_no_return_2(language):
    '''Test function with no args and no return '''
    def p_func():
        x = 2
        x *= 2

    c_func = epyccel(p_func, language=language)
    assert c_func() == p_func()
    assert isinstance(c_func(), type(p_func()))
    unexpected_arg = 0
    with pytest.raises(TypeError):
        c_func(unexpected_arg)

def test_func_no_args_f1(language):
    def f1():
        from numpy import pi
        value = (2*pi)**(3/2)
        return value

    f = epyccel(f1)
    assert np.isclose(f(), f1(), rtol=RTOL, atol=ATOL)

def test_func_return_constant(language):
    def f1():
        from numpy import pi
        return pi

    f = epyccel(f1)
    assert np.isclose(f(), f1(), rtol=RTOL, atol=ATOL)

#------------------------------------------------------------------------------
def test_decorator_f1(language):
    @types('int')
    def f1(x):
        y = x - 1
        return y

    f = epyccel(f1, language=language)

    # ...
    assert f(3) == f1(3)
    # ...

#------------------------------------------------------------------------------
def test_decorator_f2(language):
    @types('int [:]')
    def f2(x):
        y = x[0] - 1
        return y

    f = epyccel(f2, language=language)

    # ...
    x = np.array([3, 4, 5, 6], dtype=int)
    assert f(x) == f2(x)
    # ...

    # ...
    x = np.array([3, 4, 5, 6], dtype=int)
    assert f(x) == f2(x)
    # ...

#------------------------------------------------------------------------------
def test_decorator_f3(language):
    @types('int [:]')
    def f3(x):
        from numpy import empty_like
        y = empty_like(x)
        y[:] = x - 1
        return y

    f = epyccel(f3, language=language)
    x = np.array([3, 4, 5, 6], dtype=int)
    assert np.all(f(x) == f3(x))

#------------------------------------------------------------------------------
def test_decorator_f4(language):
    @types('real [:,:]')
    def f4(x):
        from numpy import empty_like
        y = empty_like(x)
        y[:] = x - 1.0
        return y

    f = epyccel(f4, language=language)
    x = np.array([[3, 4, 5, 6],[3, 4, 5, 6]], dtype=float)
    assert np.all(f(x) == f4(x))

#------------------------------------------------------------------------------
def test_decorator_f5(language):
    @types('int', 'real [:]')
    def f5(m1, x):
        x[:] = 0.
        for i in range(0, m1):
            x[i] = i * 1.

    f = epyccel(f5, language=language)

    # ...
    m1 = 3

    x = np.zeros(m1)
    f(m1, x)

    x_expected = np.zeros(m1)
    f5(m1, x_expected)

    assert np.allclose( x, x_expected, rtol=RTOL, atol=ATOL )
    # ...

#------------------------------------------------------------------------------
def test_decorator_f6(language):
    @types('int', 'int', 'real [:,:]')
    def f6_1(m1, m2, x):
        x[:,:] = 0.
        for i in range(0, m1):
            for j in range(0, m2):
                x[i,j] = (2*i+j) * 1.

    f = epyccel(f6_1, language=language)

    # ...
    m1 = 2 ; m2 = 3

    x = np.zeros((m1,m2))
    f(m1, m2, x)

    x_expected = np.zeros((m1,m2))
    f6_1(m1, m2, x_expected)

    assert np.allclose( x, x_expected, rtol=RTOL, atol=ATOL )
    # ...

#------------------------------------------------------------------------------
# in order to call the pyccelized function here, we have to create x with
# Fortran ordering
def test_decorator_f7(language):

    @types('int', 'int', 'real [:,:](order=F)')
    def f7(m1, m2, x):
        x[:,:] = 0.
        for i in range(0, m1):
            for j in range(0, m2):
                x[i,j] = (2*i+j) * 1.

    f = epyccel(f7, language=language)

    # ...
    m1 = 2 ; m2 = 3
    x_expected = np.zeros((m1,m2))
    f7(m1, m2, x_expected)

    x = np.zeros((m1,m2), order='F')
    f(m1, m2, x)

    assert np.allclose( x, x_expected, rtol=RTOL, atol=ATOL )
    # ...

#------------------------------------------------------------------------------
def test_decorator_f8(language):
    @types('int','bool')
    def f8(x,b):
        a = x if b else 2
        return a

    f = epyccel(f8, language=language)

    # ...
    assert f(3,True)  == f8(3,True)
    assert f(3,False) == f8(3,False)
    # ...


@pytest.mark.parametrize( 'language', [
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python),
    ]
)
def test_arguments_f9(language):
    @types('int64[:]')
    def f9(x):
        x += 1

    f = epyccel(f9, language = language)

    x = np.zeros(10, dtype='int64')
    x_expected = x.copy()

    f9(x)
    f(x_expected)
    assert np.array_equal(x, x_expected)

@pytest.mark.parametrize( 'language', [
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python),
    ]
)
def test_arguments_f10(language):
    @types('int64[:]')
    def f10(x):
        x[:] += 1

    f = epyccel(f10, language = language)

    x = np.zeros(10, dtype='int64')
    x_expected = x.copy()

    f10(x)
    f(x_expected)
    assert np.array_equal(x, x_expected)

def test_multiple_returns_f11(language):
    @types('int', 'int', results='int')
    def ackermann(m, n):
        if m == 0:
            return n + 1
        elif n == 0:
            return ackermann(m - 1, 1)
        else:
            return ackermann(m - 1, ackermann(m, n - 1))

    f = epyccel(ackermann, language=language)
    assert f(2,3) == ackermann(2,3)

def test_multiple_returns_f12(language):
    @types('int')
    def non_negative(i):
        if i < 0:
            return False
        else:
            return True

    f = epyccel(non_negative, language=language)
    assert f(2) == non_negative(2)
    assert f(-1) == non_negative(-1)

def test_multiple_returns_f13(language):
    @types('int', 'int')
    def get_min(a, b):
        if a<b:
            return a
        else:
            return b

    f = epyccel(get_min, language=language)
    assert f(2,3) == get_min(2,3)

def test_multiple_returns_f14(language):
    @types('int', 'int')
    def g(x, y):
        return x,y,y,y,x

    f = epyccel(g, language=language)
    assert f(2,1) == g(2,1)


def test_decorator_f15(language):
    @types('bool', 'int8', 'int16', 'int32', 'int64')
    def f15(a,b,c,d,e):
        from numpy import int64
        if a:
            return int64(b + c)
        else:
            return d + e

    f = epyccel(f15, language=language)
    assert f(True, np.int8(1), np.int16(2), np.int32(3), np.int64(4)) == \
           f15(True, np.int8(1), np.int16(2), np.int32(3), np.int64(4))
    assert f(False, np.int8(1), np.int16(2), np.int32(3), np.int64(4)) == \
           f15(False, np.int8(1), np.int16(2), np.int32(3), np.int64(4))


def test_decorator_f16(language):
    @types('int16')
    def f16(a):
        b = a
        return b
    f = epyccel(f16, language=language)
    assert f(np.int16(17)) == f16(np.int16(17))

def test_decorator_f17(language):
    @types('int8')
    def f17(a):
        b = a
        return b
    f = epyccel(f17, language=language)
    assert f(np.int8(2)) == f17(np.int8(2))

def test_decorator_f18(language):
    @types('int32')
    def f18(a):
        b = a
        return b
    f = epyccel(f18, language=language)
    assert f(np.int32(5)) == f18(np.int32(5))

def test_decorator_f19(language):
    @types('int64')
    def f19(a):
        b = a
        return b
    f = epyccel(f19, language=language)
    assert f(np.int64(1)) == f19(np.int64(1))

def test_decorator_f20(language):
    @types('complex')
    def f20(a):
        b = a
        return b
    f = epyccel(f20, language=language)
    assert f(complex(1, 2.2)) == f20(complex(1, 2.2))

def test_decorator_f21(language):
    @types('complex64')
    def f21(a):
        b = a
        return b
    f = epyccel(f21, language=language)
    assert f(np.complex64(1+ 2.2j)) == f21(np.complex64(1+ 2.2j))

def test_decorator_f22(language):
    @types('complex128')
    def f22(a):
        b = a
        return b
    f = epyccel(f22, language=language)
    assert f(np.complex128(1+ 2.2j)) == f22(np.complex128(1+ 2.2j))

##==============================================================================
## CLEAN UP GENERATED FILES AFTER RUNNING TESTS
##==============================================================================
#
#def teardown_module():
#    clean_test()
