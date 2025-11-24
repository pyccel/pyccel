# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import TypeVar
import pytest
import numpy as np
from numpy.random import randint

from pyccel import epyccel

def test_sum_range(language):
    def f(a0 : 'int[:]'):
        return sum(a0[i] for i in range(len(a0)))

    n = randint(1,50)
    x = np.array(randint(100, size=n), dtype=int)

    f_epyc = epyccel(f, language = language)

    assert f(x) == f_epyc(x)

def test_sum_var(language):
    def f(a : 'int[:]'):
        return sum(ai for ai in a)

    n = randint(1,50)
    x = np.array(randint(100, size=n), dtype=int)

    f_epyc = epyccel(f, language = language)

    assert f(x) == f_epyc(x)

def test_sum_var2(language):
    def f(a : 'int[:,:]'):
        return sum(aii for ai in a for aii in ai)

    n1 = randint(1,10)
    n2 = randint(1,10)
    x = np.array(randint(10, size=(n1,n2)), dtype=int)

    f_epyc = epyccel(f, language = language)

    assert f(x) == f_epyc(x)

def test_sum_var3(language):
    def f(a : 'int[:,:,:]'):
        m,n,p = a.shape
        return sum(a[i,j,k] for i in range(m) for j in range(n) for k in range(p))

    n1 = randint(1,10)
    n2 = randint(1,10)
    n3 = randint(1,10)
    x = np.array(randint(10, size=(n1,n2,n3)), dtype=int)

    f_epyc = epyccel(f, language = language)

    assert f(x) == f_epyc(x)

def test_sum_var4(language):
    def f(a : 'int[:]'):
        s = 3
        return sum(ai for ai in a),s

    n = randint(1,50)
    x = np.array(randint(100, size=n), dtype=int)

    f_epyc = epyccel(f, language = language)

    assert f(x) == f_epyc(x)

def test_sum_var5(language):
    def f(a : 'bool[:]'):
        return sum(ai for ai in a)

    n = randint(1,50)
    x = np.ones(n, dtype=bool)

    f_epyc = epyccel(f, language = language)

    assert f(x) == f_epyc(x)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="Max not implemented in C for integers"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_max(language):
    def f():
        return max(i if i>k else k for i in range(5) for k in range(10))

    f_epyc = epyccel(f, language = language)

    assert f() == f_epyc()
@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="Min not implemented in C for integers"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_min(language):
    def f():
        return min(k if i>k else 0 if i==k else i for i in range(5) for k in range(10))

    f_epyc = epyccel(f, language = language)

    assert f() == f_epyc()

def test_expression1(language):
    def f(b : 'float[:]'):
        n = b.shape[0]
        return (2*sum(b[i] for i in range(n))**5+5)*min(j+1. for j in b)**4+9*max(j+1. for j in b)**4

    n = randint(1,10)
    x = np.array(randint(100, size=n), dtype=float)

    f_epyc = epyccel(f, language = language)

    assert np.isclose(f(x), f_epyc(x), rtol=1e-14, atol=1e-14)

def test_expression2(language):
    def f(b : 'int64[:]'):
        def incr(x : 'int64'):
            y = x + 1
            return y
        n = b.shape[0]
        return 5+incr(2+incr(6+sum(b[i] for i in range(n))))

    n = randint(1,10)
    x = randint(100, size=n).astype(np.int64)

    f_epyc = epyccel(f, language = language)

    assert f(x) == f_epyc(x)

def test_nested_generators1(language):
    def f(a : 'float[:,:,:,:]'):
        return sum(sum(sum(a[i,k,o,2] for i in range(5)) for k in range(5)) for o in range(5))

    x = randint(0, 50, size=(5,5,5,5)).astype(float)

    f_epyc = epyccel(f, language = language)

    assert f(x) == f_epyc(x)

def test_nested_generators2(language):
    def f(a : 'float[:,:,:,:]'):
        return min(min(sum(min(max(a[i,k,o,l]*l for i in range(5)) for k in range(5)) for o in range(5)) for l in range(5)), 0.)

    x = randint(0, 50, size=(5,5,5,5)).astype(float)

    f_epyc = epyccel(f, language = language)

    assert f(x) == f_epyc(x)

def test_nested_generators3(language):
    def f(a : 'float[:,:,:,:]'):
        return sum(sum(a[i,k,4,2] for i in range(5)) for k in range(5))**2

    x = randint(0, 10, size=(5,5,5,5)).astype(float)

    f_epyc = epyccel(f, language = language)

    assert f(x) == f_epyc(x)

def test_nested_generators4(language):
    def f(a : 'float[:,:,:,:]'):
        return min(max(a[i,k,4,2] for i in range(5)) for k in range(5))**2

    x = randint(0, 10, size=(5,5,5,5)).astype(float)

    f_epyc = epyccel(f, language = language)

    assert f(x) == f_epyc(x)

def test_sum_range_overwrite(language):
    def f(a0 : 'int[:]'):
        v = sum(a0[i] for i in range(len(a0)))
        v = sum(a0[i] for i in range(len(a0)))
        return v

    n = randint(1,50)
    x = np.array(randint(100, size=n), dtype=int)

    f_epyc = epyccel(f, language = language)

    assert f(x) == f_epyc(x)

def test_sum_with_condition(language):
    def f():
        v = sum(i for i in range(20) if i % 2 == 1)
        return v

    f_epyc = epyccel(f, language = language)
    assert f() == f_epyc()

def test_sum_with_multiple_conditions(language):
    def f():
        v = sum(i - j for i in range(20) if i % 2 == 1 for j in range(30) if j % 3 == 1)
        return v

    f_epyc = epyccel(f, language = language)
    assert f() == f_epyc()

def test_max_with_condition(language):
    def f():
        v = max(i for i in range(20) if i % 2 == 1)
        return v

    f_epyc = epyccel(f, language = language)
    assert f() == f_epyc()

def test_max_with_condition_float(language):
    def f():
        v = max(i/2 for i in range(20) if i % 2 == 1)
        return v

    f_epyc = epyccel(f, language = language)
    assert f() == f_epyc()

def test_max_with_multiple_conditions(language):
    def f():
        v = max(i - j for i in range(20) if i % 2 == 1 for j in range(30) if j % 3 == 1)
        return v

    f_epyc = epyccel(f, language = language)
    assert f() == f_epyc()

def test_min_with_condition(language):
    def f():
        v = min(i for i in range(20) if i % 2 == 1)
        return v

    f_epyc = epyccel(f, language = language)
    assert f() == f_epyc()

def test_min_with_condition_float(language):
    def f():
        v = min(i/2 for i in range(20) if i % 2 == 1)
        return v

    f_epyc = epyccel(f, language = language)
    assert f() == f_epyc()

def test_min_with_multiple_conditions(language):
    def f():
        v = min(i - j for i in range(20) if i % 2 == 1 for j in range(30) if j % 3 == 1)
        return v

    f_epyc = epyccel(f, language = language)
    assert f() == f_epyc()

def test_sum_with_two_variables(language):
    def f():
        x = sum(i-j for i in range(10) for j in range(7))
        return x

    f_epyc = epyccel(f, language=language)

    assert f() == f_epyc()

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Var arg causes type promotion. See #2251."),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_min_max_values(language):
    T = TypeVar('T', 'int16[:]', 'int32[:]', 'int64[:]', 'float32[:]', 'float64[:]')

    def f(a : T):
        min_val = min(ai for ai in a)
        max_val = max(ai for ai in a)
        return min_val, max_val

    f_epyc = epyccel(f, language=language)

    for dtype in (np.int16, np.int32, np.int64, np.float32, np.float64):
        x = randint(0, 100, size=(5,)).astype(dtype)
        print(f(x))
        print(f_epyc(x))
        assert f(x) == f_epyc(x)
