# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np
import pytest
from numpy.random import randint
from pytest_teardown_tools import run_epyccel, clean_test

def test_complex_imag(language):
    def f():
        a = 1+2j
        return a.imag

    epyc_f = run_epyccel(f, language=language)

    r = f()
    epyc_r = epyc_f()

    assert r == epyc_r
    assert isinstance(r, type(epyc_r))

def test_complex_imag_expr(language):
    def f(a : 'complex', b : 'complex'):
        return (a+b).imag

    epyc_f = run_epyccel(f, language=language)

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

    epyc_f = run_epyccel(f, language=language)

    r = f()
    epyc_r = epyc_f()

    assert r == epyc_r
    assert isinstance(r, type(epyc_r))

def test_complex_real_expr(language):
    def f(a : 'complex', b : 'complex'):
        return (a+b).real

    epyc_f = run_epyccel(f, language=language)

    a = randint(20)+1j*randint(20)
    b = randint(20)+1j*randint(20)

    r = f(a,b)
    epyc_r = epyc_f(a,b)

    assert r == epyc_r
    assert isinstance(r, type(epyc_r))

def test_complex_conjugate(language):
    def f(a : 'complex', b : 'complex'):
        return (a+b).conjugate()

    epyc_f = run_epyccel(f, language=language)

    a = randint(20)+1j*randint(20)
    b = randint(20)+1j*randint(20)

    r = f(a,b)
    epyc_r = epyc_f(a,b)

    assert r == epyc_r
    assert isinstance(r, type(epyc_r))

def test_complex_conjugate64(language):
    def f(a : 'complex64', b : 'complex64'):
        return (a+b).conj()

    epyc_f = run_epyccel(f, language=language)

    a = np.complex64(randint(20)+1j*randint(20))
    b = np.complex64(randint(20)+1j*randint(20))

    r = f(a,b)
    epyc_r = epyc_f(a,b)

    assert r == epyc_r
    assert isinstance(r, type(epyc_r))

def test_ndarray_var_from_expr(language):
    def f(x : 'int[:]', y : 'int[:]'):
        z = x + y
        a = z.sum()
        return a

    epyc_f = run_epyccel(f, language=language)

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

    epyc_f = run_epyccel(f, language=language)

    r = f(a)
    epyc_r = epyc_f(a)
    assert r == epyc_r

##==============================================================================
## CLEAN UP GENERATED FILES AFTER RUNNING TESTS
##==============================================================================

def teardown_module(module):
    clean_test()
