# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np
import pytest
from numpy.random import randint
from pyccel.epyccel import epyccel

def test_complex_imag(language):
    def f():
        a = 1+2j
        return a.imag

    epyc_f = epyccel(f, language=language)

    r = f()
    epyc_r = epyc_f()

    assert r == epyc_r
    assert isinstance(r, type(epyc_r))

def test_complex_imag_expr(language):
    def f(a : 'complex', b : 'complex'):
        return (a+b).imag

    epyc_f = epyccel(f, language=language)

    a = randint(20)+1j*randint(20)
    b = randint(20)+1j*randint(20)

    r = f(a,b)
    epyc_r = epyc_f(a,b)

    assert r == epyc_r
    assert isinstance(r, type(epyc_r))

def test_complex_real(language):
    def f():
        a = 1+2j
        return a.real

    epyc_f = epyccel(f, language=language)

    r = f()
    epyc_r = epyc_f()

    assert r == epyc_r
    assert isinstance(r, type(epyc_r))

def test_complex_real_expr(language):
    def f(a : 'complex', b : 'complex'):
        return (a+b).real

    epyc_f = epyccel(f, language=language)

    a = randint(20)+1j*randint(20)
    b = randint(20)+1j*randint(20)

    r = f(a,b)
    epyc_r = epyc_f(a,b)

    assert r == epyc_r
    assert isinstance(r, type(epyc_r))

def test_complex_conjugate(language):
    def f(a : 'complex', b : 'complex'):
        return (a+b).conjugate()

    epyc_f = epyccel(f, language=language)

    a = randint(20)+1j*randint(20)
    b = randint(20)+1j*randint(20)

    r = f(a,b)
    epyc_r = epyc_f(a,b)

    assert r == epyc_r
    assert isinstance(r, type(epyc_r))

def test_complex64_conjugate(language):
    def f(a : 'complex64', b : 'complex64'):
        return (a+b).conj()

    epyc_f = epyccel(f, language=language)

    a = np.complex64(randint(20)+1j*randint(20))
    b = np.complex64(randint(20)+1j*randint(20))

    r = f(a,b)
    epyc_r = epyc_f(a,b)

    assert r == epyc_r
    assert isinstance(r, type(epyc_r))

def test_float_conjugate(language):
    def f(a : 'float', b : 'float'):
        return (a+b).conjugate()

    epyc_f = epyccel(f, language=language)

    a = float(randint(20))
    b = float(randint(20))

    r = f(a,b)
    epyc_r = epyc_f(a,b)

    assert r == epyc_r
    assert isinstance(r, type(epyc_r))

def test_float64_conjugate(language):
    def f(a : 'float64', b : 'float64'):
        return (a+b).conj()

    epyc_f = epyccel(f, language=language)

    a = np.float64(randint(20))
    b = np.float64(randint(20))

    r = f(a,b)
    epyc_r = epyc_f(a,b)

    assert r == epyc_r
    assert isinstance(r, type(epyc_r))

def test_int_conjugate(language):
    def f(a : 'int', b : 'int'):
        return (a+b).conjugate()

    epyc_f = epyccel(f, language=language)

    a = randint(20)
    b = randint(20)

    r = f(a,b)
    epyc_r = epyc_f(a,b)

    assert r == epyc_r
    assert isinstance(r, type(epyc_r))

def test_int32_conjugate(language):
    def f(a : 'int32', b : 'int32'):
        return (a+b).conj()

    epyc_f = epyccel(f, language=language)

    a = randint(20, dtype=np.int32)
    b = randint(20, dtype=np.int32)

    r = f(a,b)
    epyc_r = epyc_f(a,b)

    assert r == epyc_r
    assert isinstance(r, type(epyc_r))

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="Class inheritance not fully implemented"),
            pytest.mark.c]),
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="Class inheritance not fully implemented"),
            pytest.mark.fortran]),
        pytest.param("python", marks = pytest.mark.python)
    ]
)
def test_bool_conjugate(language):
    def f(a : 'bool', b : 'bool'):
        return (a or b).conjugate()

    epyc_f = epyccel(f, language=language)

    a = bool(randint(2))
    b = bool(randint(2))

    r = f(a,b)
    epyc_r = epyc_f(a,b)

    assert r == epyc_r
    assert isinstance(r, type(epyc_r))

def test_ndarray_var_from_expr(language):
    def f(x : 'int[:]', y : 'int[:]'):
        z = x + y
        a = z.sum()
        return a

    epyc_f = epyccel(f, language=language)

    a = np.ones(6, dtype=int)
    b = np.ones(6, dtype=int)

    r = f(a,b)
    epyc_r = epyc_f(a,b)

    assert r == epyc_r

def test_ndarray_var_from_slice(language):
    def f(x : 'int[:]'):
        z = x[1:]
        a = z.sum()
        return a

    a = np.ones(6, dtype=int)

    epyc_f = epyccel(f, language=language)

    r = f(a)
    epyc_r = epyc_f(a)
    assert r == epyc_r
