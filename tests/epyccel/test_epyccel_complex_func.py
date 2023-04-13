# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np
import pytest

from numpy.random import rand, randint

import modules.complex_func as mod
from pyccel.epyccel import epyccel

ATOL = 1e-15
RTOL = 2e-14

@pytest.mark.parametrize("f", [ mod.create_complex_literal__int_int,
                           mod.create_complex_literal__int_float,
                           mod.create_complex_literal__int_complex,
                           mod.create_complex_literal__float_int,
                           mod.create_complex_literal__float_float,
                           mod.create_complex_literal__float_complex,
                           mod.create_complex_literal__complex_int,
                           mod.create_complex_literal__complex_float,
                           mod.create_complex_literal__complex_complex,
                           mod.cast_complex_literal] )
def test_create_complex_literal(f, language):
    f_epyc = epyccel(f, language = language)
    assert f_epyc() == f()

def test_create_complex_var__int_int(language):
    f = mod.create_complex_var__int_int
    f_epyc = epyccel(f, language = language)

    a = randint(100)
    b = randint(100)
    assert f_epyc(a,b) == f(a,b)

def test_create_complex_var__int_complex(language):
    f = mod.create_complex_var__int_complex
    f_epyc = epyccel(f, language = language)

    a = randint(100)
    b = complex(randint(100), randint(100))
    assert f_epyc(a,b) == f(a,b)

def test_create_complex_var__complex_float(language):
    f = mod.create_complex_var__complex_float
    f_epyc = epyccel(f, language = language)

    a = complex(randint(100), randint(100))
    b = rand()*100
    assert np.allclose(f_epyc(a,b), f(a,b), rtol=RTOL, atol=ATOL)

def test_create_complex_var__complex_complex(language):
    f = mod.create_complex_var__complex_complex
    f_epyc = epyccel(f, language = language)

    a = complex(randint(100), randint(100))
    b = complex(randint(100), randint(100))
    assert np.allclose(f_epyc(a,b), f(a,b), rtol=RTOL, atol=ATOL)

def test_create_complex__int_int(language):
    f = mod.create_complex__int_int
    f_epyc = epyccel(f, language = language)

    a = randint(100)
    assert f_epyc(a) == f(a)

def test_create_complex_0__int_int(language):
    f = mod.create_complex_0__int_int
    f_epyc = epyccel(f, language = language)

    a = randint(100)
    assert f_epyc(a) == f(a)

def test_create_complex__float_float(language):
    f = mod.create_complex__float_float
    f_epyc = epyccel(f, language = language)

    a = rand()*100
    assert np.allclose(f_epyc(a), f(a), rtol=RTOL, atol=ATOL)

def test_create_complex_0__float_float(language):
    f = mod.create_complex_0__float_float
    f_epyc = epyccel(f, language = language)

    a = rand()*100
    assert np.allclose(f_epyc(a), f(a), rtol=RTOL, atol=ATOL)

def test_create_complex__complex_complex(language):
    f = mod.create_complex__complex_complex
    f_epyc = epyccel(f, language = language)

    a = complex(randint(100), randint(100))
    assert np.allclose(f_epyc(a), f(a), rtol=RTOL, atol=ATOL)

def test_cast_complex_1(language):
    f = mod.cast_complex_1
    f_epyc = epyccel(f, language = language)

    a = np.complex64(complex(randint(100), randint(100)))
    assert np.allclose(f_epyc(a), f(a), rtol = 1e-7, atol = 1e-8)

def test_cast_complex_2(language):
    f = mod.cast_complex_2
    f_epyc = epyccel(f, language = language)

    a = np.complex128(complex(randint(100), randint(100)))
    assert np.allclose(f_epyc(a), f(a), rtol=RTOL, atol=ATOL)

def test_cast_float_complex(language):
    f = mod.cast_float_complex
    f_epyc = epyccel(f, language = language)

    a = rand()*100
    b = complex(randint(100), randint(100))
    assert np.allclose(f_epyc(a,b), f(a,b), rtol=RTOL, atol=ATOL)
