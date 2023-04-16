# pylint: disable=missing-function-docstring, missing-module-docstring

import numpy as np
import modules.numpy_sign as mod

from pytest_teardown_tools import run_epyccel, clean_test

def test_sign_complex(language):
    f_nul = mod.complex_nul
    f_pos = mod.complex_pos
    f_neg = mod.complex_neg
    f_pos_neg = mod.complex_pos_neg
    f_neg_pos = mod.complex_neg_pos
    f_nul_epyc = run_epyccel(f_nul, language = language)
    f_pos_epyc = run_epyccel(f_pos, language = language)
    f_neg_epyc = run_epyccel(f_neg, language = language)
    f_pos_neg_epyc = run_epyccel(f_pos_neg, language = language)
    f_neg_pos_epyc = run_epyccel(f_neg_pos, language = language)

    x1_nul, x2_nul = f_nul(), f_nul_epyc()
    x1_pos, x2_pos = f_pos(), f_pos_epyc()
    x1_neg, x2_neg = f_neg(), f_neg_epyc()
    x1_pos_neg, x2_pos_neg = f_pos_neg(), f_pos_neg_epyc()
    x1_neg_pos, x2_neg_pos = f_neg_pos(), f_neg_pos_epyc()

    assert x1_nul == x2_nul and x1_nul.dtype == x2_nul.dtype
    assert x1_pos == x2_pos and x1_pos.dtype == x2_pos.dtype
    assert x1_neg == x2_neg and x1_neg.dtype == x2_neg.dtype
    assert x1_pos_neg == x2_pos_neg and x1_pos_neg.dtype == x2_pos_neg.dtype
    assert x1_neg_pos == x2_neg_pos and x1_neg_pos.dtype == x2_neg_pos.dtype

def test_sign_complex64(language):
    f_nul = mod.complex64_nul
    f_pos = mod.complex64_pos
    f_neg = mod.complex64_neg
    f_pos_neg = mod.complex64_pos_neg
    f_neg_pos = mod.complex64_neg_pos
    f_nul_epyc = run_epyccel(f_nul, language = language)
    f_pos_epyc = run_epyccel(f_pos, language = language)
    f_neg_epyc = run_epyccel(f_neg, language = language)
    f_pos_neg_epyc = run_epyccel(f_pos_neg, language = language)
    f_neg_pos_epyc = run_epyccel(f_neg_pos, language = language)

    x1_nul, x2_nul = f_nul(), f_nul_epyc()
    x1_pos, x2_pos = f_pos(), f_pos_epyc()
    x1_neg, x2_neg = f_neg(), f_neg_epyc()
    x1_pos_neg, x2_pos_neg = f_pos_neg(), f_pos_neg_epyc()
    x1_neg_pos, x2_neg_pos = f_neg_pos(), f_neg_pos_epyc()

    assert x1_nul == x2_nul and x1_nul.dtype == x2_nul.dtype
    assert x1_pos == x2_pos and x1_pos.dtype == x2_pos.dtype
    assert x1_neg == x2_neg and x1_neg.dtype == x2_neg.dtype
    assert x1_pos_neg == x2_pos_neg and x1_pos_neg.dtype == x2_pos_neg.dtype
    assert x1_neg_pos == x2_neg_pos and x1_neg_pos.dtype == x2_neg_pos.dtype

def test_sign_complex128(language):
    f_nul = mod.complex128_nul
    f_pos = mod.complex128_pos
    f_neg = mod.complex128_neg
    f_pos_neg = mod.complex128_pos_neg
    f_neg_pos = mod.complex128_neg_pos
    f_nul_epyc = run_epyccel(f_nul, language = language)
    f_pos_epyc = run_epyccel(f_pos, language = language)
    f_neg_epyc = run_epyccel(f_neg, language = language)
    f_pos_neg_epyc = run_epyccel(f_pos_neg, language = language)
    f_neg_pos_epyc = run_epyccel(f_neg_pos, language = language)

    x1_nul, x2_nul = f_nul(), f_nul_epyc()
    x1_pos, x2_pos = f_pos(), f_pos_epyc()
    x1_neg, x2_neg = f_neg(), f_neg_epyc()
    x1_pos_neg, x2_pos_neg = f_pos_neg(), f_pos_neg_epyc()
    x1_neg_pos, x2_neg_pos = f_neg_pos(), f_neg_pos_epyc()

    assert x1_nul == x2_nul and x1_nul.dtype == x2_nul.dtype
    assert x1_pos == x2_pos and x1_pos.dtype == x2_pos.dtype
    assert x1_neg == x2_neg and x1_neg.dtype == x2_neg.dtype
    assert x1_pos_neg == x2_pos_neg and x1_pos_neg.dtype == x2_pos_neg.dtype
    assert x1_neg_pos == x2_neg_pos and x1_neg_pos.dtype == x2_neg_pos.dtype

def test_sign_int16(language):
    f_pos = mod.int16_pos
    f_neg = mod.int16_neg
    f_pos_epyc = run_epyccel(f_pos, language = language)
    f_neg_epyc = run_epyccel(f_neg, language = language)

    x1_pos, x2_pos = f_pos(), f_pos_epyc()
    x1_neg, x2_neg = f_neg(), f_neg_epyc()

    assert x1_pos == x2_pos and x1_pos.dtype == x2_pos.dtype
    assert x1_neg == x2_neg and x1_neg.dtype == x2_neg.dtype

def test_sign_int32(language):
    f_pos = mod.int32_pos
    f_neg = mod.int32_neg
    f_pos_epyc = run_epyccel(f_pos, language = language)
    f_neg_epyc = run_epyccel(f_neg, language = language)

    x1_pos, x2_pos = f_pos(), f_pos_epyc()
    x1_neg, x2_neg = f_neg(), f_neg_epyc()

    assert x1_pos == x2_pos and x1_pos.dtype == x2_pos.dtype
    assert x1_neg == x2_neg and x1_neg.dtype == x2_neg.dtype

def test_sign_int64(language):
    f_pos = mod.int64_pos
    f_neg = mod.int64_neg
    f_pos_epyc = run_epyccel(f_pos, language = language)
    f_neg_epyc = run_epyccel(f_neg, language = language)

    x1_pos, x2_pos = f_pos(), f_pos_epyc()
    x1_neg, x2_neg = f_neg(), f_neg_epyc()

    assert x1_pos == x2_pos and x1_pos.dtype == x2_pos.dtype
    assert x1_neg == x2_neg and x1_neg.dtype == x2_neg.dtype

def test_sign_float(language):
    f_pos = mod.float_pos
    f_neg = mod.float_neg
    f_nul = mod.float_nul
    f_pos_epyc = run_epyccel(f_pos, language = language)
    f_neg_epyc = run_epyccel(f_neg, language = language)
    f_nul_epyc = run_epyccel(f_nul, language = language)

    x1_pos, x2_pos = f_pos(), f_pos_epyc()
    x1_neg, x2_neg = f_neg(), f_neg_epyc()
    x1_nul, x2_nul = f_nul(), f_nul_epyc()

    assert x1_pos == x2_pos and x1_pos.dtype == x2_pos.dtype
    assert x1_neg == x2_neg and x1_neg.dtype == x2_neg.dtype
    assert x1_nul == x2_nul and x1_nul.dtype == x2_nul.dtype

def test_sign_float64(language):
    f_pos = mod.float64_pos
    f_neg = mod.float64_neg
    f_pos_epyc = run_epyccel(f_pos, language = language)
    f_neg_epyc = run_epyccel(f_neg, language = language)

    x1_pos, x2_pos = f_pos(), f_pos_epyc()
    x1_neg, x2_neg = f_neg(), f_neg_epyc()

    assert x1_pos == x2_pos and x1_pos.dtype == x2_pos.dtype
    assert x1_neg == x2_neg and x1_neg.dtype == x2_neg.dtype

def test_sign_literal_complex(language):
    f_pos      = mod.literal_complex_pos
    f_neg      = mod.literal_complex_neg
    f_nul      = mod.literal_complex_nul_nul
    f_nul_imag = mod.literal_complex_nul_imag
    f_real_nul = mod.literal_complex_real_nul
    f_pos_epyc      = run_epyccel(f_pos, language = language)
    f_neg_epyc      = run_epyccel(f_neg, language = language)
    f_nul_epyc      = run_epyccel(f_nul, language = language)
    f_nul_imag_epyc = run_epyccel(f_nul_imag, language = language)
    f_real_nul_epyc = run_epyccel(f_real_nul, language = language)

    x1_pos, x2_pos = f_pos(), f_pos_epyc()
    x1_neg, x2_neg = f_neg(), f_neg_epyc()
    x1_nul, x2_nul = f_nul(), f_nul_epyc()
    x1_nul_imag, x2_nul_imag = f_nul_imag(), f_nul_imag_epyc()
    x1_real_nul, x2_real_nul = f_real_nul(), f_real_nul_epyc()

    assert x1_pos == x2_pos and x1_pos.dtype == x2_pos.dtype
    assert x1_neg == x2_neg and x1_neg.dtype == x2_neg.dtype
    assert x1_nul == x2_nul and x1_nul.dtype == x2_nul.dtype
    assert x1_nul_imag == x2_nul_imag and x1_nul_imag.dtype == x2_nul_imag.dtype
    assert x1_real_nul == x2_real_nul and x1_real_nul.dtype == x2_real_nul.dtype

def test_sign_literal_int(language):
    f_pos = mod.literal_int_pos
    f_neg = mod.literal_int_neg
    f_nul = mod.literal_int_nul
    f_pos_epyc = run_epyccel(f_pos, language = language)
    f_neg_epyc = run_epyccel(f_neg, language = language)
    f_nul_epyc = run_epyccel(f_nul, language = language)

    x1_pos, x2_pos = f_pos(), f_pos_epyc()
    x1_neg, x2_neg = f_neg(), f_neg_epyc()
    x1_nul, x2_nul = f_nul(), f_nul_epyc()

    assert x1_pos == x2_pos and x1_pos.dtype == x2_pos.dtype
    assert x1_neg == x2_neg and x1_neg.dtype == x2_neg.dtype
    assert x1_nul == x2_nul and x1_nul.dtype == x2_nul.dtype

def test_sign_literal_float(language):
    f_pos = mod.literal_float_pos
    f_neg = mod.literal_float_neg
    f_nul = mod.literal_float_nul
    f_pos_epyc = run_epyccel(f_pos, language = language)
    f_neg_epyc = run_epyccel(f_neg, language = language)
    f_nul_epyc = run_epyccel(f_nul, language = language)

    x1_pos, x2_pos = f_pos(), f_pos_epyc()
    x1_neg, x2_neg = f_neg(), f_neg_epyc()
    x1_nul, x2_nul = f_nul(), f_nul_epyc()

    assert x1_pos == x2_pos and x1_pos.dtype == x2_pos.dtype
    assert x1_neg == x2_neg and x1_neg.dtype == x2_neg.dtype
    assert x1_nul == x2_nul and x1_nul.dtype == x2_nul.dtype

# Tests on arrays

def test_sign_array_1d_int(language):
    f_int8  = mod.array_1d_int8
    f_int16 = mod.array_1d_int16
    f_int32 = mod.array_1d_int32
    f_int64 = mod.array_1d_int64
    f_int8_epyc  = run_epyccel(f_int8, language = language)
    f_int16_epyc = run_epyccel(f_int16, language = language)
    f_int32_epyc = run_epyccel(f_int32, language = language)
    f_int64_epyc = run_epyccel(f_int64, language = language)

    arr8  = np.array([2, 0, -0, -13, 37, 42], dtype=np.int8)
    arr16 = np.array([2, 0, -0, -13, 37, 42], dtype=np.int16)
    arr32 = np.array([2, 0, -0, -13, 37, 42], dtype=np.int32)
    arr64 = np.array([2, 0, -0, -13, 37, 42], dtype=np.int64)

    x_int8, y_int8 = f_int8(arr8), f_int8_epyc(arr8)
    x_int16, y_int16 = f_int16(arr16), f_int16_epyc(arr16)
    x_int32, y_int32 = f_int32(arr32), f_int32_epyc(arr32)
    x_int64, y_int64 = f_int64(arr64), f_int64_epyc(arr64)

    assert np.array_equal(x_int8, y_int8) and x_int8.dtype == y_int8.dtype
    assert np.array_equal(x_int16, y_int16) and x_int16.dtype == y_int16.dtype
    assert np.array_equal(x_int32, y_int32) and x_int32.dtype == y_int32.dtype
    assert np.array_equal(x_int64, y_int64) and x_int64.dtype == y_int64.dtype

def test_sign_array_2d_int(language):
    f_int8  = mod.array_2d_int8
    f_int16 = mod.array_2d_int16
    f_int32 = mod.array_2d_int32
    f_int64 = mod.array_2d_int64
    f_int8_epyc  = run_epyccel(f_int8, language = language)
    f_int16_epyc = run_epyccel(f_int16, language = language)
    f_int32_epyc = run_epyccel(f_int32, language = language)
    f_int64_epyc = run_epyccel(f_int64, language = language)

    arr8  = np.array([[2, 0], [-0, -13], [37, 42]], dtype=np.int8)
    arr16 = np.array([[2, 0], [-0, -13], [37, 42]], dtype=np.int16)
    arr32 = np.array([[2, 0], [-0, -13], [37, 42]], dtype=np.int32)
    arr64 = np.array([[2, 0], [-0, -13], [37, 42]], dtype=np.int64)

    x_int8, y_int8 = f_int8(arr8), f_int8_epyc(arr8)
    x_int16, y_int16 = f_int16(arr16), f_int16_epyc(arr16)
    x_int32, y_int32 = f_int32(arr32), f_int32_epyc(arr32)
    x_int64, y_int64 = f_int64(arr64), f_int64_epyc(arr64)

    assert np.array_equal(x_int8, y_int8) and x_int8.dtype == y_int8.dtype
    assert np.array_equal(x_int16, y_int16) and x_int16.dtype == y_int16.dtype
    assert np.array_equal(x_int32, y_int32) and x_int32.dtype == y_int32.dtype
    assert np.array_equal(x_int64, y_int64) and x_int64.dtype == y_int64.dtype

def test_sign_array_1d_float(language):
    f_float32 = mod.array_1d_float32
    f_float64 = mod.array_1d_float64
    f_float32_epyc = run_epyccel(f_float32, language = language)
    f_float64_epyc = run_epyccel(f_float64, language = language)

    arr32 = np.array([2., 0., -0., -1.3, 3.7, .42], dtype=np.float32)
    arr64 = np.array([2., 0., -0., -1.3, 3.7, .42], dtype=np.float64)

    x_float32, y_float32 = f_float32(arr32), f_float32_epyc(arr32)
    x_float64, y_float64 = f_float64(arr64), f_float64_epyc(arr64)

    assert np.array_equal(x_float32, y_float32) and x_float32.dtype == y_float32.dtype
    assert np.array_equal(x_float64, y_float64) and x_float64.dtype == y_float64.dtype

def test_sign_array_2d_float(language):
    f_float32 = mod.array_2d_float32
    f_float64 = mod.array_2d_float64
    f_float32_epyc = run_epyccel(f_float32, language = language)
    f_float64_epyc = run_epyccel(f_float64, language = language)

    arr32 = np.array([[2., 0.], [-0., -1.3], [3.7, .42]], dtype=np.float32)
    arr64 = np.array([[2., 0.], [-0., -1.3], [3.7, .42]], dtype=np.float64)

    x_float32, y_float32 = f_float32(arr32), f_float32_epyc(arr32)
    x_float64, y_float64 = f_float64(arr64), f_float64_epyc(arr64)

    assert np.array_equal(x_float32, y_float32) and x_float32.dtype == y_float32.dtype
    assert np.array_equal(x_float64, y_float64) and x_float64.dtype == y_float64.dtype

def test_sign_array_1d_complex(language):
    f_complex64 = mod.array_1d_complex64
    f_complex128 = mod.array_1d_complex128
    f_complex64_epyc = run_epyccel(f_complex64, language = language)
    f_complex128_epyc = run_epyccel(f_complex128, language = language)

    arr64 = np.array([0.+0j, 0.j, 1.+2.j, -1.+2.j, 1.-2.j, -1.-2.j, 2.j, -2.j], dtype=np.complex64)
    arr128 = np.array([0.+0j, 0.j, 1.+2.j, -1.+2.j, 1.-2.j, -1.-2.j, 2.j, -2.j], dtype=np.complex128)

    x_complex64, y_complex64 = f_complex64(arr64), f_complex64_epyc(arr64)
    x_complex128, y_complex128 = f_complex128(arr128), f_complex128_epyc(arr128)

    assert np.array_equal(x_complex64, y_complex64) and x_complex64.dtype == y_complex64.dtype
    assert np.array_equal(x_complex128, y_complex128) and x_complex128.dtype == y_complex128.dtype

def test_sign_array_2d_complex(language):
    f_complex64 = mod.array_2d_complex64
    f_complex128 = mod.array_2d_complex128
    f_complex64_epyc = run_epyccel(f_complex64, language = language)
    f_complex128_epyc = run_epyccel(f_complex128, language = language)

    arr64 = np.array([[0.+0j, 0.j], [1.+2.j, -1.+2.j], [1.-2.j, -1.-2.j], [2.j, -2.j]], dtype=np.complex64)
    arr128 = np.array([[0.+0j, 0.j], [1.+2.j, -1.+2.j], [1.-2.j, -1.-2.j], [2.j, -2.j]], dtype=np.complex128)

    x_complex64, y_complex64 = f_complex64(arr64), f_complex64_epyc(arr64)
    x_complex128, y_complex128 = f_complex128(arr128), f_complex128_epyc(arr128)

    assert np.array_equal(x_complex64, y_complex64) and x_complex64.dtype == y_complex64.dtype
    assert np.array_equal(x_complex128, y_complex128) and x_complex128.dtype == y_complex128.dtype

##==============================================================================
## CLEAN UP GENERATED FILES AFTER RUNNING TESTS
##==============================================================================

def teardown_module(module):
    clean_test()
