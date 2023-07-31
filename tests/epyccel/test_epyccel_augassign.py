# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np
from pyccel.decorators import types
from numpy.random import random
import pytest

import modules.augassign as mod
from pyccel.epyccel import epyccel

# += tests

def test_augassign_add_1d(language):
    f_int     = mod.augassign_add_1d_int
    f_float   = mod.augassign_add_1d_float
    f_complex = mod.augassign_add_1d_complex
    f_int_epyc     = epyccel(f_int, language = language)
    f_float_epyc   = epyccel(f_float, language = language)
    f_complex_epyc = epyccel(f_complex, language = language)

    x1_int     = np.zeros(5, dtype=int)
    x1_float   = np.zeros(5, dtype=float)
    x1_complex = np.zeros(5, dtype=complex)
    x2_int     = np.zeros(5, dtype=int)
    x2_float   = np.zeros(5, dtype=float)
    x2_complex = np.zeros(5, dtype=complex)

    y1_int = f_int(x1_int)
    y1_float = f_float(x1_float)
    y1_complex = f_complex(x1_complex)
    y2_int = f_int_epyc(x2_int)
    y2_float = f_float_epyc(x2_float)
    y2_complex = f_complex_epyc(x2_complex)

    assert y1_int == y2_int and np.array_equal(x1_int, x2_int)
    assert y1_float == y2_float and np.array_equal(x1_float, x2_float)
    assert y1_complex == y2_complex and np.array_equal(x1_complex, x2_complex)

def test_augassign_add_2d(language):
    f_int     = mod.augassign_add_2d_int
    f_float   = mod.augassign_add_2d_float
    f_complex = mod.augassign_add_2d_complex
    f_int_epyc     = epyccel(f_int, language = language)
    f_float_epyc   = epyccel(f_float, language = language)
    f_complex_epyc = epyccel(f_complex, language = language)

    x1_int     = np.zeros((5, 5), dtype=int)
    x1_float   = np.zeros((5, 5), dtype=float)
    x1_complex = np.zeros((5, 5), dtype=complex)
    x2_int     = np.zeros((5, 5), dtype=int)
    x2_float   = np.zeros((5, 5), dtype=float)
    x2_complex = np.zeros((5, 5), dtype=complex)

    y1_int = f_int(x1_int)
    y1_float = f_float(x1_float)
    y1_complex = f_complex(x1_complex)
    y2_int = f_int_epyc(x2_int)
    y2_float = f_float_epyc(x2_float)
    y2_complex = f_complex_epyc(x2_complex)

    assert y1_int == y2_int and np.array_equal(x1_int, x2_int)
    assert y1_float == y2_float and np.array_equal(x1_float, x2_float)
    assert y1_complex == y2_complex and np.array_equal(x1_complex, x2_complex)


# -= tests

def test_augassign_sub_1d(language):
    f_int     = mod.augassign_sub_1d_int
    f_float   = mod.augassign_sub_1d_float
    f_complex = mod.augassign_sub_1d_complex
    f_int_epyc     = epyccel(f_int, language = language)
    f_float_epyc   = epyccel(f_float, language = language)
    f_complex_epyc = epyccel(f_complex, language = language)

    x1_int     = np.zeros(5, dtype=int)
    x1_float   = np.zeros(5, dtype=float)
    x1_complex = np.zeros(5, dtype=complex)
    x2_int     = np.zeros(5, dtype=int)
    x2_float   = np.zeros(5, dtype=float)
    x2_complex = np.zeros(5, dtype=complex)

    y1_int = f_int(x1_int)
    y1_float = f_float(x1_float)
    y1_complex = f_complex(x1_complex)
    y2_int = f_int_epyc(x2_int)
    y2_float = f_float_epyc(x2_float)
    y2_complex = f_complex_epyc(x2_complex)

    assert y1_int == y2_int and np.array_equal(x1_int, x2_int)
    assert y1_float == y2_float and np.array_equal(x1_float, x2_float)
    assert y1_complex == y2_complex and np.array_equal(x1_complex, x2_complex)

def test_augassign_sub_2d(language):
    f_int     = mod.augassign_sub_2d_int
    f_float   = mod.augassign_sub_2d_float
    f_complex = mod.augassign_sub_2d_complex
    f_int_epyc     = epyccel(f_int, language = language)
    f_float_epyc   = epyccel(f_float, language = language)
    f_complex_epyc = epyccel(f_complex, language = language)

    x1_int     = np.zeros((5, 5), dtype=int)
    x1_float   = np.zeros((5, 5), dtype=float)
    x1_complex = np.zeros((5, 5), dtype=complex)
    x2_int     = np.zeros((5, 5), dtype=int)
    x2_float   = np.zeros((5, 5), dtype=float)
    x2_complex = np.zeros((5, 5), dtype=complex)

    y1_int = f_int(x1_int)
    y1_float = f_float(x1_float)
    y1_complex = f_complex(x1_complex)
    y2_int = f_int_epyc(x2_int)
    y2_float = f_float_epyc(x2_float)
    y2_complex = f_complex_epyc(x2_complex)

    assert y1_int == y2_int and np.array_equal(x1_int, x2_int)
    assert y1_float == y2_float and np.array_equal(x1_float, x2_float)
    assert y1_complex == y2_complex and np.array_equal(x1_complex, x2_complex)


# *= tests

def test_augassign_mul_1d(language):
    f_int     = mod.augassign_mul_1d_int
    f_float   = mod.augassign_mul_1d_float
    f_complex = mod.augassign_mul_1d_complex
    f_int_epyc     = epyccel(f_int, language = language)
    f_float_epyc   = epyccel(f_float, language = language)
    f_complex_epyc = epyccel(f_complex, language = language)

    x1_int     = np.zeros(5, dtype=int)
    x1_float   = np.zeros(5, dtype=float)
    x1_complex = np.zeros(5, dtype=complex)
    x2_int     = np.zeros(5, dtype=int)
    x2_float   = np.zeros(5, dtype=float)
    x2_complex = np.zeros(5, dtype=complex)

    y1_int = f_int(x1_int)
    y1_float = f_float(x1_float)
    y1_complex = f_complex(x1_complex)
    y2_int = f_int_epyc(x2_int)
    y2_float = f_float_epyc(x2_float)
    y2_complex = f_complex_epyc(x2_complex)

    assert y1_int == y2_int and np.array_equal(x1_int, x2_int)
    assert y1_float == y2_float and np.array_equal(x1_float, x2_float)
    assert y1_complex == y2_complex and np.array_equal(x1_complex, x2_complex)

def test_augassign_mul_2d(language):
    f_int     = mod.augassign_mul_2d_int
    f_float   = mod.augassign_mul_2d_float
    f_complex = mod.augassign_mul_2d_complex
    f_int_epyc     = epyccel(f_int, language = language)
    f_float_epyc   = epyccel(f_float, language = language)
    f_complex_epyc = epyccel(f_complex, language = language)

    x1_int     = np.zeros((5, 5), dtype=int)
    x1_float   = np.zeros((5, 5), dtype=float)
    x1_complex = np.zeros((5, 5), dtype=complex)
    x2_int     = np.zeros((5, 5), dtype=int)
    x2_float   = np.zeros((5, 5), dtype=float)
    x2_complex = np.zeros((5, 5), dtype=complex)

    y1_int = f_int(x1_int)
    y1_float = f_float(x1_float)
    y1_complex = f_complex(x1_complex)
    y2_int = f_int_epyc(x2_int)
    y2_float = f_float_epyc(x2_float)
    y2_complex = f_complex_epyc(x2_complex)

    assert y1_int == y2_int and np.array_equal(x1_int, x2_int)
    assert y1_float == y2_float and np.array_equal(x1_float, x2_float)
    assert y1_complex == y2_complex and np.array_equal(x1_complex, x2_complex)


# /= tests

def test_augassign_div_1d(language):
    f_float   = mod.augassign_div_1d_float
    f_complex = mod.augassign_div_1d_complex
    f_float_epyc   = epyccel(f_float, language = language)
    f_complex_epyc = epyccel(f_complex, language = language)

    x1_float   = np.zeros(5, dtype=float)
    x1_complex = np.zeros(5, dtype=complex)
    x2_float   = np.zeros(5, dtype=float)
    x2_complex = np.zeros(5, dtype=complex)

    y1_float = f_float(x1_float)
    y1_complex = f_complex(x1_complex)
    y2_float = f_float_epyc(x2_float)
    y2_complex = f_complex_epyc(x2_complex)

    assert y1_float == y2_float and np.array_equal(x1_float, x2_float)
    assert y1_complex == y2_complex and np.array_equal(x1_complex, x2_complex)

def test_augassign_div_2d(language):
    f_float   = mod.augassign_div_2d_float
    f_complex = mod.augassign_div_2d_complex
    f_float_epyc   = epyccel(f_float, language = language)
    f_complex_epyc = epyccel(f_complex, language = language)

    x1_float   = np.zeros((5, 5), dtype=float)
    x1_complex = np.zeros((5, 5), dtype=complex)
    x2_float   = np.zeros((5, 5), dtype=float)
    x2_complex = np.zeros((5, 5), dtype=complex)

    y1_float = f_float(x1_float)
    y1_complex = f_complex(x1_complex)
    y2_float = f_float_epyc(x2_float)
    y2_complex = f_complex_epyc(x2_complex)

    assert y1_float == y2_float and np.array_equal(x1_float, x2_float)
    assert y1_complex == y2_complex and np.array_equal(x1_complex, x2_complex)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="Function in function not implemented in C"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_augassign_func(language):
    func = mod.augassign_func
    func_epyc = epyccel(func, language = language)

    x = random()*100+20
    y = random()*100

    z = func(x,y)
    z_epyc = func_epyc(x,y)

    assert z == z_epyc
    assert isinstance(z, type(z_epyc))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="Function in function not implemented in C"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_augassign_array_func(language):
    func = mod.augassign_array_func
    func_epyc = epyccel(func, language = language)

    x = random(10)*100+20
    y = random(10)*100
    x_epyc = x.copy()

    func(x,y)
    func_epyc(x_epyc,y)

    assert np.array_equal(x, x_epyc)
