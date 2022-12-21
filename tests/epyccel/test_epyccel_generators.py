# pylint: disable=missing-function-docstring, missing-module-docstring/
import pytest
import numpy as np
from numpy.random import randint, rand

from pyccel.epyccel import epyccel

RTOL = 2e-14
ATOL = 1e-15

def test_sum_range(language):
    def f(a0 : 'int[:]'):
        return sum(a0[i] for i in range(len(a0)))

    n = randint(1,50)
    x = randint(100,size=n)

    f_epyc = epyccel(f, language = language)

    assert f(x) == f_epyc(x)

def test_sum_var(language):
    def f(a : 'int[:]'):
        return sum(ai for ai in a)

    n = randint(1,50)
    x = randint(100,size=n)

    f_epyc = epyccel(f, language = language)

    assert f(x) == f_epyc(x)

def test_sum_var2(language):
    def f(a : 'int[:,:]'):
        return sum(aii for ai in a for aii in ai)

    n1 = randint(1,10)
    n2 = randint(1,10)
    x = randint(10,size=(n1,n2))

    f_epyc = epyccel(f, language = language)

    assert f(x) == f_epyc(x)

def test_sum_var3(language):
    def f(a : 'int[:,:,:]'):
        m,n,p = a.shape
        return sum(a[i,j,k] for i in range(m) for j in range(n) for k in range(p))

    n1 = randint(1,10)
    n2 = randint(1,10)
    n3 = randint(1,10)
    x = randint(10,size=(n1,n2,n3))

    f_epyc = epyccel(f, language = language)

    assert f(x) == f_epyc(x)

def test_sum_var4(language):
    def f(a : 'int[:]'):
        s = 3
        return sum(ai for ai in a),s

    n = randint(1,50)
    x = randint(100,size=n)

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
        return (2*sum(b[i] for i in range(n))**5+5)*min(j+1. for j in b)**4+9

    n = randint(1,10)
    x = np.array(randint(100,size=n), dtype=float)

    f_epyc = epyccel(f, language = language)

    assert np.isclose(f(x), f_epyc(x), rtol=RTOL, atol=ATOL)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="Function in function not implemented in C"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_expression2(language):
    def f(b : 'int[:]'):
        def incr(x : int):
            y = x + 1
            return y
        n = b.shape[0]
        return 5+incr(2+incr(6+sum(b[i] for i in range(n))))

    n = randint(1,10)
    x = randint(100,size=n)

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
        return min(min(sum(min(max(a[i,k,o,l]*l for i in range(5)) for k in range(5)) for o in range(5)) for l in range(5)),0.)

    x = randint(0, 50, size=(5,5,5,5)).astype(float)

    f_epyc = epyccel(f, language = language)

    assert np.isclose(f(x), f_epyc(x), rtol=RTOL, atol=ATOL)

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

    assert np.isclose(f(x), f_epyc(x), rtol=RTOL, atol=ATOL)
