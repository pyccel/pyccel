# pylint: disable=missing-function-docstring, missing-module-docstring
# coding: utf-8
import sys
from typing import TypeVar, Final

import pytest
import numpy as np
from numpy.random import randint

from pyccel import epyccel

RTOL = 2e-14
ATOL = 1e-15

def test_func_no_args_1(language):
    '''test function with return value but no args'''
    def free_gift():
        gift = 10
        return gift

    c_gift = epyccel(free_gift, language=language, folder='__pyccel__test_folder__')
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

def test_func_no_return_1(language):
    '''Test function with args and no return '''
    def p_func(x : int):
        x *= 2

    c_func = epyccel(p_func, language=language)
    x = np.random.randint(100)
    assert c_func(x) == p_func(x)
    # Test type return should be NoneType
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

    f = epyccel(f1, language=language)
    assert np.isclose(f(), f1(), rtol=RTOL, atol=ATOL)

def test_func_return_constant(language):
    def f1():
        from numpy import pi
        return pi

    f = epyccel(f1, language=language)
    assert np.isclose(f(), f1(), rtol=RTOL, atol=ATOL)

#------------------------------------------------------------------------------
def test_decorator_f1(language):
    def f1(x : 'int'):
        y = x - 1
        return y

    f = epyccel(f1, language=language)

    # ...
    assert f(3) == f1(3)
    # ...

#------------------------------------------------------------------------------
def test_decorator_f2(language):
    def f2(x : 'int [:]'):
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
    def f3(x : 'int [:]'):
        from numpy import empty_like
        y = empty_like(x)
        y[:] = x - 1
        return y

    f = epyccel(f3, language=language)
    x = np.array([3, 4, 5, 6], dtype=int)
    assert np.all(f(x) == f3(x))

#------------------------------------------------------------------------------
def test_decorator_f4(language):
    def f4(x : 'float [:,:]'):
        from numpy import empty_like
        y = empty_like(x)
        y[:] = x - 1.0
        return y

    f = epyccel(f4, language=language)
    x = np.array([[3, 4, 5, 6],[3, 4, 5, 6]], dtype=float)
    assert np.all(f(x) == f4(x))

#------------------------------------------------------------------------------
def test_decorator_f5(language):
    def f5(m1 : 'int', x : 'float [:]'):
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
    def f6_1(m1 : 'int', m2 : 'int', x : 'float [:,:]'):
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

    def f7(m1 : 'int', m2 : 'int', x : 'float [:,:](order=F)'):
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
    def f8(x : 'int', b : 'bool'):
        a = x if b else 2
        return a

    f = epyccel(f8, language=language)

    # ...
    assert f(3,True)  == f8(3,True)
    assert f(3,False) == f8(3,False)
    # ...


def test_arguments_f9(language):
    def f9(x : 'int64[:]'):
        x += 1

    f = epyccel(f9, language = language)

    x = np.zeros(10, dtype='int64')
    x_expected = x.copy()

    f9(x)
    f(x_expected)
    assert np.array_equal(x, x_expected)

def test_arguments_f10(language):
    def f10(x : 'int64[:]'):
        x[:] += 1

    f = epyccel(f10, language = language)

    x = np.zeros(10, dtype='int64')
    x_expected = x.copy()

    f10(x)
    f(x_expected)
    assert np.array_equal(x, x_expected)

def test_multiple_returns_f11(language):
    def ackermann(m : 'int', n : 'int') -> int:
        if m == 0:
            return n + 1
        elif n == 0:
            return ackermann(m - 1, 1)
        else:
            return ackermann(m - 1, ackermann(m, n - 1))

    f = epyccel(ackermann, language=language)
    assert f(2,3) == ackermann(2,3)

def test_multiple_returns_f12(language):
    def non_negative(i : 'int'):
        if i < 0:
            return False
        else:
            return True

    f = epyccel(non_negative, language=language)
    assert f(2) == non_negative(2)
    assert f(-1) == non_negative(-1)

def test_multiple_returns_f13(language):
    def get_min(a : 'int', b : 'int'):
        if a<b:
            return a
        else:
            return b

    f = epyccel(get_min, language=language)
    assert f(2,3) == get_min(2,3)

def test_multiple_returns_f14(language):
    def g(x : 'int', y : 'int'):
        return x,y,y,y,x

    f = epyccel(g, language=language)
    assert f(2,1) == g(2,1)


def test_decorator_f15(language):
    def f15(a : 'bool', b : 'int8', c : 'int16', d : 'int32', e : 'int64'):
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
    def f16(a : 'int16'):
        b = a
        return b
    f = epyccel(f16, language=language)
    assert f(np.int16(17)) == f16(np.int16(17))

def test_decorator_f17(language):
    def f17(a : 'int8'):
        b = a
        return b
    f = epyccel(f17, language=language)
    assert f(np.int8(2)) == f17(np.int8(2))

def test_decorator_f18(language):
    def f18(a : 'int32'):
        b = a
        return b
    f = epyccel(f18, language=language)
    assert f(np.int32(5)) == f18(np.int32(5))

def test_decorator_f19(language):
    def f19(a : 'int64'):
        b = a
        return b
    f = epyccel(f19, language=language)
    assert f(np.int64(1)) == f19(np.int64(1))

def test_decorator_f20(language):
    def f20(a : 'complex'):
        b = a
        return b
    f = epyccel(f20, language=language)
    assert f(complex(1, 2.2)) == f20(complex(1, 2.2))

def test_decorator_f21(language):
    def f21(a : 'complex64'):
        b = a
        return b
    f = epyccel(f21, language=language)
    assert f(np.complex64(1+ 2.2j)) == f21(np.complex64(1+ 2.2j))

def test_decorator_f22(language):
    def f22(a : 'complex128'):
        b = a
        return b
    f = epyccel(f22, language=language)
    assert f(np.complex128(1+ 2.2j)) == f22(np.complex128(1+ 2.2j))

@pytest.mark.skipif(sys.version_info < (3, 10), reason="PEP604 (writing union types as X | Y) implemented in Python 3.10")
def test_union_type(language):
    def square(a : int | float): #pylint: disable=unsupported-binary-operation
        return a*a

    f = epyccel(square, language=language)
    x = np.random.randint(40)
    y = np.random.uniform()

    assert np.isclose(f(x), square(x), rtol=RTOL, atol=ATOL)
    assert isinstance(f(x), type(square(x)))
    assert np.isclose(f(y), square(y), rtol=RTOL, atol=ATOL)
    assert isinstance(f(y), type(square(y)))

def test_return_annotation(language):
    def get_2() -> int:
        my_var : int = 2
        return my_var

    f = epyccel(get_2, language=language)
    assert f() == get_2()

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
    )
)
def test_wrong_argument_type(language):
    def f(integer_arg : int):
        return integer_arg + 1
    epyc_f = epyccel(f, language=language)
    test_arg = 3.5
    with pytest.raises(TypeError) as err:
        epyc_f(test_arg)
    assert 'integer_arg' in str(err.value)
    assert str(type(test_arg)) in str(err.value)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
    )
)
def test_wrong_known_argument_type_in_interface(language):
    T = TypeVar('T', int, float)

    def f(a : T, integer_arg : int):
        return a + 1
    epyc_f = epyccel(f, language=language)
    test_arg = 4.5
    with pytest.raises(TypeError) as err:
        epyc_f(3.5, test_arg)
    assert 'integer_arg' in str(err.value)
    assert str(type(test_arg)) in str(err.value)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
    )
)
def test_wrong_known_argument_type_in_interface_with_default(language):
    T = TypeVar('T', int, float)

    def f(a : T, integer_arg : int = 5):
        return a + 1
    epyc_f = epyccel(f, language=language)
    test_arg = 4.5
    with pytest.raises(TypeError) as err:
        epyc_f(3.5, test_arg)
    assert 'integer_arg' in str(err.value)
    assert str(type(test_arg)) in str(err.value)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
    )
)
def test_wrong_unknown_argument_type_in_interface(language):
    T = TypeVar('T', int, float)

    def f(templated_arg : T, b : int):
        return templated_arg + 1
    epyc_f = epyccel(f, language=language)
    test_arg = 3.5+1j
    with pytest.raises(TypeError) as err:
        epyc_f(test_arg, 4.5)
    assert 'templated_arg' in str(err.value)
    assert str(type(test_arg)) in str(err.value)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
    )
)
def test_wrong_argument_combination_in_interface(language):
    T = TypeVar('T', int, float)

    def f(a : T, b : T):
        return a + 1
    epyc_f = epyccel(f, language=language)
    with pytest.raises(TypeError):
        epyc_f(3.5, 4)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
    )
)
def test_argument_checks_with_interfaces(language):
    from modules import Module_12 as mod
    modnew = epyccel(mod, language=language)
    with pytest.raises(TypeError):
        modnew.times_3(1)
    with pytest.raises(TypeError):
        modnew.add_2(1)

def test_container_interface(language):
    T = TypeVar('T', 'int[:]', list[int], set[int])

    def f(a : Final[T]):
        return len(a)

    epyc_f = epyccel(f, language=language)
    assert f([1,2]) == epyc_f([1,2])
    assert f({1,2}) == epyc_f({1,2})
    assert f(np.array([1,2])) == epyc_f(np.array([1,2]))

def test_lambda(language):
    def f(a : int):
        f1 = lambda x: x**2 + 1 # pylint: disable=unnecessary-lambda-assignment
        g1 = lambda x: f1(x)**2 + 1 # pylint: disable=unnecessary-lambda-assignment
        return g1(a)

    epyc_f = epyccel(f, language=language)
    val = randint(20)
    assert f(val) == epyc_f(val)
    assert isinstance(epyc_f(val), type(epyc_f(val)))

def test_lambda_2(language):
    def f(a : int):
        f2 = lambda x,y: x**2 + y**2 + 1 # pylint: disable=unnecessary-lambda-assignment
        return f2(a, 3*a)

    epyc_f = epyccel(f, language=language)
    val = randint(20)
    assert f(val) == epyc_f(val)
    assert isinstance(epyc_f(val), type(epyc_f(val)))

def test_argument_types():
    def f(a : int, /, b : int, *args : int, c : int, **kwargs : int):
        my_sum = sum(v for v in kwargs.values())
        return my_sum + 2*a + 3*b + 5*c + 7*sum(args)

    epyc_f = epyccel(f, language = 'python')
    a = 8
    b = 9
    c = 25
    args = (7, 14, 21)
    kwargs = {'d': 11, 'f': 13}
    assert f(a, b, *args, c=c, **kwargs) == epyc_f(a, b, *args, c=c, **kwargs)

def test_positional_only_arguments(language):
    def f(a : int, /, b : int):
        return 2*a + 3*b

    epyc_f = epyccel(f, language = language)
    a = 8
    b = 9
    assert f(a, b) == epyc_f(a, b)
    assert f(a, b=b) == epyc_f(a, b=b)
    with pytest.raises(TypeError):
        epyc_f(a=a, b=b)

def test_keyword_only_arguments(language):
    def f(a : int, *, b : int):
        return 2*a + 3*b

    epyc_f = epyccel(f, language = language)
    a = 8
    b = 9
    assert f(a, b=b) == epyc_f(a, b=b)
    with pytest.raises(TypeError):
        epyc_f(a, b)

def test_lambda_usage(language):
    f = lambda x: x+1 # pylint: disable=unnecessary-lambda-assignment

    def g(a : 'int[:]'):
        for i, ai in enumerate(a):
            a[i] = f(ai)

    epyc_g = epyccel(g, language=language)
    val = randint(20, size=(10,))
    val_epyc = val.copy()
    g(val)
    epyc_g(val_epyc)
    assert np.array_equal(val, val_epyc)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Function in function is not implemented yet in C language"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_func_usage(language):
    def f(x : int):
        return x+1

    def g(a : 'int[:]'):
        for i, ai in enumerate(a):
            a[i] = f(ai)

    epyc_g = epyccel(g, language=language)
    val = randint(20, size=(10,))
    val_epyc = val.copy()
    g(val)
    epyc_g(val_epyc)
    assert np.array_equal(val, val_epyc)
