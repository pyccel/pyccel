# pylint: disable=missing-function-docstring, missing-module-docstring/

import numpy as np
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

    x1_int     = np.ones(5, dtype=int)
    x1_float   = np.ones(5, dtype=float)
    x1_complex = np.ones(5, dtype=complex)
    x2_int     = np.ones(5, dtype=int)
    x2_float   = np.ones(5, dtype=float)
    x2_complex = np.ones(5, dtype=complex)

    f_int(x1_int)
    f_float(x1_float)
    f_complex(x1_complex)
    f_int_epyc(x2_int)
    f_float_epyc(x2_float)
    f_complex_epyc(x2_complex)

    assert x1_int == x2_int and x1_int.dtype == x2_int.dtype
    assert x1_float == x2_float and x1_float.dtype == x2_float.dtype
    assert x1_complex == x2_complex and x1_complex.dtype == x2_complex.dtype

def test_augassign_add_2d(language):
    f_int     = mod.augassign_add_2d_int
    f_float   = mod.augassign_add_2d_float
    f_complex = mod.augassign_add_2d_complex
    f_int_epyc     = epyccel(f_int, language = language)
    f_float_epyc   = epyccel(f_float, language = language)
    f_complex_epyc = epyccel(f_complex, language = language)

    x1_int     = np.ones((5, 5), dtype=int)
    x1_float   = np.ones((5, 5), dtype=float)
    x1_complex = np.ones((5, 5), dtype=complex)
    x2_int     = np.ones((5, 5), dtype=int)
    x2_float   = np.ones((5, 5), dtype=float)
    x2_complex = np.ones((5, 5), dtype=complex)

    f_int(x1_int)
    f_float(x1_float)
    f_complex(x1_complex)
    f_int_epyc(x2_int)
    f_float_epyc(x2_float)
    f_complex_epyc(x2_complex)

    assert x1_int == x2_int and x1_int.dtype == x2_int.dtype
    assert x1_float == x2_float and x1_float.dtype == x2_float.dtype
    assert x1_complex == x2_complex and x1_complex.dtype == x2_complex.dtype


# -= tests

def test_augassign_sub_1d(language):
    f_int     = mod.augassign_sub_1d_int
    f_float   = mod.augassign_sub_1d_float
    f_complex = mod.augassign_sub_1d_complex
    f_int_epyc     = epyccel(f_int, language = language)
    f_float_epyc   = epyccel(f_float, language = language)
    f_complex_epyc = epyccel(f_complex, language = language)

    x1_int     = np.ones(5, dtype=int)
    x1_float   = np.ones(5, dtype=float)
    x1_complex = np.ones(5, dtype=complex)
    x2_int     = np.ones(5, dtype=int)
    x2_float   = np.ones(5, dtype=float)
    x2_complex = np.ones(5, dtype=complex)

    f_int(x1_int)
    f_float(x1_float)
    f_complex(x1_complex)
    f_int_epyc(x2_int)
    f_float_epyc(x2_float)
    f_complex_epyc(x2_complex)

    assert x1_int == x2_int and x1_int.dtype == x2_int.dtype
    assert x1_float == x2_float and x1_float.dtype == x2_float.dtype
    assert x1_complex == x2_complex and x1_complex.dtype == x2_complex.dtype

def test_augassign_sub_2d(language):
    f_int     = mod.augassign_sub_2d_int
    f_float   = mod.augassign_sub_2d_float
    f_complex = mod.augassign_sub_2d_complex
    f_int_epyc     = epyccel(f_int, language = language)
    f_float_epyc   = epyccel(f_float, language = language)
    f_complex_epyc = epyccel(f_complex, language = language)

    x1_int     = np.ones((5, 5), dtype=int)
    x1_float   = np.ones((5, 5), dtype=float)
    x1_complex = np.ones((5, 5), dtype=complex)
    x2_int     = np.ones((5, 5), dtype=int)
    x2_float   = np.ones((5, 5), dtype=float)
    x2_complex = np.ones((5, 5), dtype=complex)

    f_int(x1_int)
    f_float(x1_float)
    f_complex(x1_complex)
    f_int_epyc(x2_int)
    f_float_epyc(x2_float)
    f_complex_epyc(x2_complex)

    assert x1_int == x2_int and x1_int.dtype == x2_int.dtype
    assert x1_float == x2_float and x1_float.dtype == x2_float.dtype
    assert x1_complex == x2_complex and x1_complex.dtype == x2_complex.dtype


# *= tests

def test_augassign_mul_1d(language):
    f_int     = mod.augassign_mul_1d_int
    f_float   = mod.augassign_mul_1d_float
    f_complex = mod.augassign_mul_1d_complex
    f_int_epyc     = epyccel(f_int, language = language)
    f_float_epyc   = epyccel(f_float, language = language)
    f_complex_epyc = epyccel(f_complex, language = language)

    x1_int     = np.ones(5, dtype=int)
    x1_float   = np.ones(5, dtype=float)
    x1_complex = np.ones(5, dtype=complex)
    x2_int     = np.ones(5, dtype=int)
    x2_float   = np.ones(5, dtype=float)
    x2_complex = np.ones(5, dtype=complex)

    f_int(x1_int)
    f_float(x1_float)
    f_complex(x1_complex)
    f_int_epyc(x2_int)
    f_float_epyc(x2_float)
    f_complex_epyc(x2_complex)

    assert x1_int == x2_int and x1_int.dtype == x2_int.dtype
    assert x1_float == x2_float and x1_float.dtype == x2_float.dtype
    assert x1_complex == x2_complex and x1_complex.dtype == x2_complex.dtype

def test_augassign_mul_2d(language):
    f_int     = mod.augassign_mul_2d_int
    f_float   = mod.augassign_mul_2d_float
    f_complex = mod.augassign_mul_2d_complex
    f_int_epyc     = epyccel(f_int, language = language)
    f_float_epyc   = epyccel(f_float, language = language)
    f_complex_epyc = epyccel(f_complex, language = language)

    x1_int     = np.ones((5, 5), dtype=int)
    x1_float   = np.ones((5, 5), dtype=float)
    x1_complex = np.ones((5, 5), dtype=complex)
    x2_int     = np.ones((5, 5), dtype=int)
    x2_float   = np.ones((5, 5), dtype=float)
    x2_complex = np.ones((5, 5), dtype=complex)

    f_int(x1_int)
    f_float(x1_float)
    f_complex(x1_complex)
    f_int_epyc(x2_int)
    f_float_epyc(x2_float)
    f_complex_epyc(x2_complex)

    assert x1_int == x2_int and x1_int.dtype == x2_int.dtype
    assert x1_float == x2_float and x1_float.dtype == x2_float.dtype
    assert x1_complex == x2_complex and x1_complex.dtype == x2_complex.dtype

# /= tests

def test_augassign_div_1d(language):
    f_int     = mod.augassign_div_1d_int
    f_float   = mod.augassign_div_1d_float
    f_complex = mod.augassign_div_1d_complex
    f_int_epyc     = epyccel(f_int, language = language)
    f_float_epyc   = epyccel(f_float, language = language)
    f_complex_epyc = epyccel(f_complex, language = language)

    x1_int     = np.ones(5, dtype=int)
    x1_float   = np.ones(5, dtype=float)
    x1_complex = np.ones(5, dtype=complex)
    x2_int     = np.ones(5, dtype=int)
    x2_float   = np.ones(5, dtype=float)
    x2_complex = np.ones(5, dtype=complex)

    f_int(x1_int)
    f_float(x1_float)
    f_complex(x1_complex)
    f_int_epyc(x2_int)
    f_float_epyc(x2_float)
    f_complex_epyc(x2_complex)

    assert x1_int == x2_int and x1_int.dtype == x2_int.dtype
    assert x1_float == x2_float and x1_float.dtype == x2_float.dtype
    assert x1_complex == x2_complex and x1_complex.dtype == x2_complex.dtype

def test_augassign_div_2d(language):
    f_int     = mod.augassign_div_2d_int
    f_float   = mod.augassign_div_2d_float
    f_complex = mod.augassign_div_2d_complex
    f_int_epyc     = epyccel(f_int, language = language)
    f_float_epyc   = epyccel(f_float, language = language)
    f_complex_epyc = epyccel(f_complex, language = language)

    x1_int     = np.ones((5, 5), dtype=int)
    x1_float   = np.ones((5, 5), dtype=float)
    x1_complex = np.ones((5, 5), dtype=complex)
    x2_int     = np.ones((5, 5), dtype=int)
    x2_float   = np.ones((5, 5), dtype=float)
    x2_complex = np.ones((5, 5), dtype=complex)

    f_int(x1_int)
    f_float(x1_float)
    f_complex(x1_complex)
    f_int_epyc(x2_int)
    f_float_epyc(x2_float)
    f_complex_epyc(x2_complex)

    assert x1_int == x2_int and x1_int.dtype == x2_int.dtype
    assert x1_float == x2_float and x1_float.dtype == x2_float.dtype
    assert x1_complex == x2_complex and x1_complex.dtype == x2_complex.dtype
