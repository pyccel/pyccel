# pylint: disable=missing-function-docstring, missing-module-docstring/

import numpy as np
import modules.numpy_sign as mod

from pyccel.epyccel import epyccel

def test_sign_complex(language):
    f_pos = mod.complex_pos
    f_neg = mod.complex_neg
    f_pos_epyc = epyccel(f_pos, language = language)
    f_neg_epyc = epyccel(f_neg, language = language)

    assert f_pos_epyc() == f_pos()
    assert f_neg_epyc() == f_neg()

def test_sign_complex64(language):
    f_pos = mod.complex64_pos
    f_neg = mod.complex64_neg
    f_pos_epyc = epyccel(f_pos, language = language)
    f_neg_epyc = epyccel(f_neg, language = language)

    assert f_pos_epyc() == f_pos()
    assert f_neg_epyc() == f_neg()

def test_sign_complex128(language):
    f_pos = mod.complex128_pos
    f_neg = mod.complex128_neg
    f_pos_epyc = epyccel(f_pos, language = language)
    f_neg_epyc = epyccel(f_neg, language = language)

    assert f_pos_epyc() == f_pos()
    assert f_neg_epyc() == f_neg()

def test_sign_int16(language):
    f_pos = mod.int16_pos
    f_neg = mod.int16_neg
    f_pos_epyc = epyccel(f_pos, language = language)
    f_neg_epyc = epyccel(f_neg, language = language)

    assert f_pos_epyc() == f_pos()
    assert f_neg_epyc() == f_neg()

def test_sign_int32(language):
    f_pos = mod.int32_pos
    f_neg = mod.int32_neg
    f_pos_epyc = epyccel(f_pos, language = language)
    f_neg_epyc = epyccel(f_neg, language = language)

    assert f_pos_epyc() == f_pos()
    assert f_neg_epyc() == f_neg()

def test_sign_int64(language):
    f_pos = mod.int64_pos
    f_neg = mod.int64_neg
    f_pos_epyc = epyccel(f_pos, language = language)
    f_neg_epyc = epyccel(f_neg, language = language)

    assert f_pos_epyc() == f_pos()
    assert f_neg_epyc() == f_neg()

def test_sign_float(language):
    f_pos = mod.float_pos
    f_neg = mod.float_neg
    f_nul = mod.float_nul
    f_pos_epyc = epyccel(f_pos, language = language)
    f_neg_epyc = epyccel(f_neg, language = language)
    f_nul_epyc = epyccel(f_nul, language = language)

    assert f_pos_epyc() == f_pos()
    assert f_neg_epyc() == f_neg()
    assert f_nul_epyc() == f_nul()

def test_sign_float64(language):
    f_pos = mod.float64_pos
    f_neg = mod.float64_neg
    f_pos_epyc = epyccel(f_pos, language = language)
    f_neg_epyc = epyccel(f_neg, language = language)

    assert f_pos_epyc() == f_pos()
    assert f_neg_epyc() == f_neg()

def test_sign_literal_complex(language):
    f_pos      = mod.literal_complex_pos
    f_neg      = mod.literal_complex_neg
    f_nul      = mod.literal_complex_nul_nul
    f_nul_imag = mod.literal_complex_nul_imag
    f_real_nul = mod.literal_complex_real_nul
    f_pos_epyc      = epyccel(f_pos, language = language)
    f_neg_epyc      = epyccel(f_neg, language = language)
    f_nul_epyc      = epyccel(f_nul, language = language)
    f_nul_imag_epyc = epyccel(f_nul_imag, language = language)
    f_real_nul_epyc = epyccel(f_real_nul, language = language)

    assert f_pos_epyc() == f_pos()
    assert f_neg_epyc() == f_neg()
    assert f_nul_epyc() == f_nul()
    assert f_nul_imag_epyc() == f_nul_imag()
    assert f_real_nul_epyc() == f_real_nul()

def test_sign_literal_int(language):
    f_pos = mod.literal_int_pos
    f_neg = mod.literal_int_neg
    f_nul = mod.literal_int_nul
    f_pos_epyc = epyccel(f_pos, language = language)
    f_neg_epyc = epyccel(f_neg, language = language)
    f_nul_epyc = epyccel(f_nul, language = language)

    assert f_pos_epyc() == f_pos()
    assert f_neg_epyc() == f_neg()
    assert f_nul_epyc() == f_nul()

def test_sign_literal_float(language):
    f_pos = mod.literal_float_pos
    f_neg = mod.literal_float_neg
    f_nul = mod.literal_float_nul
    f_pos_epyc = epyccel(f_pos, language = language)
    f_neg_epyc = epyccel(f_neg, language = language)
    f_nul_epyc = epyccel(f_nul, language = language)

    assert f_pos_epyc() == f_pos()
    assert f_neg_epyc() == f_neg()
    assert f_nul_epyc() == f_nul()

# Tests on arrays

def test_sign_arr_int(language):
    f_1d = mod.array_int_1d
    f_2d = mod.array_int_2d

    f_1d_epyc = epyccel(f_1d, language = language)
    f_2d_epyc = epyccel(f_2d, language = language)

    x1_1d = np.array([2, 0, 2, -13, 37, 42], dtype=np.int64)
    x1_2d = np.array([[2, 0], [2, -13], [37, 42]], dtype=np.int64)
    x2_1d = np.copy(x1_1d)
    x2_2d = np.copy(x1_2d)

    f_1d(x1_1d)
    f_2d(x1_2d)
    f_1d_epyc(x2_1d)
    f_2d_epyc(x2_2d)

    assert np.array_equal(x1_1d, x2_1d) and (x1_1d.dtype is x2_1d.dtype)
    assert np.array_equal(x1_2d, x2_2d) and (x1_2d.dtype is x2_2d.dtype)

def test_sign_arr_float(language):
    f_1d = mod.array_float_1d
    f_2d = mod.array_float_2d

    f_1d_epyc = epyccel(f_1d, language = language)
    f_2d_epyc = epyccel(f_2d, language = language)

    x1_1d = np.array([0., 1., 2., -1.3, -3.7, -0.], dtype=np.float64)
    x1_2d = np.array([[0., 1.], [2., -1.3], [-3.7, -0.]], dtype=np.float64)
    x2_1d = np.copy(x1_1d)
    x2_2d = np.copy(x1_2d)

    f_1d(x1_1d)
    f_2d(x1_2d)
    f_1d_epyc(x2_1d)
    f_2d_epyc(x2_2d)

    assert np.array_equal(x1_1d, x2_1d) and (x1_1d.dtype is x2_1d.dtype)
    assert np.array_equal(x1_2d, x2_2d) and (x1_2d.dtype is x2_2d.dtype)

def test_sign_arr_complex(language):
    f_1d = mod.array_complex_1d
    f_2d = mod.array_complex_2d

    f_1d_epyc = epyccel(f_1d, language = language)
    f_2d_epyc = epyccel(f_2d, language = language)

    x1_1d = np.array([0.+.1j, -2.-1.3j, -3.7j], dtype=np.complex64)
    x1_2d = np.array([[0.+1.], [-2.-1.3j], [-3.7j]], dtype=np.complex64)
    x2_1d = np.copy(x1_1d)
    x2_2d = np.copy(x1_2d)

    f_1d(x1_1d)
    f_2d(x1_2d)
    f_1d_epyc(x2_1d)
    f_2d_epyc(x2_2d)

    assert np.array_equal(x1_1d, x2_1d) and (x1_1d.dtype is x2_1d.dtype)
    assert np.array_equal(x1_2d, x2_2d) and (x1_2d.dtype is x2_2d.dtype)
