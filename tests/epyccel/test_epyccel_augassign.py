# pylint: disable=missing-function-docstring, missing-module-docstring
import modules.augassign as mod
import numpy as np
from numpy.random import random

from pyccel import epyccel

# += tests
RTOL = 1e-12
ATOL = 1e-16


def test_augassign_add_1d(language):
    f_int = mod.augassign_add_1d_int
    f_float = mod.augassign_add_1d_float
    f_complex = mod.augassign_add_1d_complex
    f_int_epyc = epyccel(f_int, language=language)
    f_float_epyc = epyccel(f_float, language=language)
    f_complex_epyc = epyccel(f_complex, language=language)

    x1_int = np.zeros(5, dtype=int)
    x1_float = np.zeros(5, dtype=float)
    x1_complex = np.zeros(5, dtype=complex)
    x2_int = np.zeros(5, dtype=int)
    x2_float = np.zeros(5, dtype=float)
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
    f_int = mod.augassign_add_2d_int
    f_float = mod.augassign_add_2d_float
    f_complex = mod.augassign_add_2d_complex
    f_int_epyc = epyccel(f_int, language=language)
    f_float_epyc = epyccel(f_float, language=language)
    f_complex_epyc = epyccel(f_complex, language=language)

    x1_int = np.zeros((5, 5), dtype=int)
    x1_float = np.zeros((5, 5), dtype=float)
    x1_complex = np.zeros((5, 5), dtype=complex)
    x2_int = np.zeros((5, 5), dtype=int)
    x2_float = np.zeros((5, 5), dtype=float)
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


def test_augassign_add_sum_scalar(language):
    func = mod.augassign_add_sum_scalar
    func_epyc = epyccel(func, language=language)

    a = 10
    b = np.array([1, 2, 3, 4, 5], dtype=np.int64)

    result = func(a, b)
    result_epyc = func_epyc(a, b)

    assert result == result_epyc


def test_augassign_add_sum_array(language):
    func = mod.augassign_add_sum_array
    func_epyc = epyccel(func, language=language)

    a1 = np.array([[10, 20], [30, 40]], dtype=np.int64)
    a2 = a1.copy()
    b = np.array([1, 2, 3], dtype=np.int64)

    func(a1, b)
    func_epyc(a2, b)

    assert np.array_equal(a1, a2)


def test_augassign_add_min_scalar(language):
    func = mod.augassign_add_min_scalar
    func_epyc = epyccel(func, language=language)

    a = 10
    b = np.array([5, 2, 8, 1, 9], dtype=np.int64)

    result = func(a, b)
    result_epyc = func_epyc(a, b)

    assert result == result_epyc


def test_augassign_add_min_array(language):
    func = mod.augassign_add_min_array
    func_epyc = epyccel(func, language=language)

    a1 = np.array([[10, 20], [30, 40]], dtype=np.int64)
    a2 = a1.copy()
    b = np.array([5, 2, 8], dtype=np.int64)

    func(a1, b)
    func_epyc(a2, b)

    assert np.array_equal(a1, a2)


def test_augassign_add_norm_scalar(language):
    func = mod.augassign_add_norm_scalar
    func_epyc = epyccel(func, language=language)

    a = 10.0
    b = np.array([3.0, 4.0], dtype=np.float64)

    result = func(a, b)
    result_epyc = func_epyc(a, b)

    assert np.isclose(result, result_epyc, rtol=RTOL, atol=ATOL)


def test_augassign_add_norm_ord1_scalar(language):
    func = mod.augassign_add_norm_ord1_scalar
    func_epyc = epyccel(func, language=language)

    a = 10.0
    b = np.array([3.0, -4.0], dtype=np.float64)

    result = func(a, b)
    result_epyc = func_epyc(a, b)

    assert np.isclose(result, result_epyc, rtol=RTOL, atol=ATOL)


# -= tests


def test_augassign_sub_1d(language):
    f_int = mod.augassign_sub_1d_int
    f_float = mod.augassign_sub_1d_float
    f_complex = mod.augassign_sub_1d_complex
    f_int_epyc = epyccel(f_int, language=language)
    f_float_epyc = epyccel(f_float, language=language)
    f_complex_epyc = epyccel(f_complex, language=language)

    x1_int = np.zeros(5, dtype=int)
    x1_float = np.zeros(5, dtype=float)
    x1_complex = np.zeros(5, dtype=complex)
    x2_int = np.zeros(5, dtype=int)
    x2_float = np.zeros(5, dtype=float)
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
    f_int = mod.augassign_sub_2d_int
    f_float = mod.augassign_sub_2d_float
    f_complex = mod.augassign_sub_2d_complex
    f_int_epyc = epyccel(f_int, language=language)
    f_float_epyc = epyccel(f_float, language=language)
    f_complex_epyc = epyccel(f_complex, language=language)

    x1_int = np.zeros((5, 5), dtype=int)
    x1_float = np.zeros((5, 5), dtype=float)
    x1_complex = np.zeros((5, 5), dtype=complex)
    x2_int = np.zeros((5, 5), dtype=int)
    x2_float = np.zeros((5, 5), dtype=float)
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


def test_augassign_sub_sum_scalar(language):
    func = mod.augassign_sub_sum_scalar
    func_epyc = epyccel(func, language=language)

    a = 100
    b = np.array([1, 2, 3], dtype=np.int64)

    result = func(a, b)
    result_epyc = func_epyc(a, b)

    assert result == result_epyc


def test_augassign_sub_max_scalar(language):
    func = mod.augassign_sub_max_scalar
    func_epyc = epyccel(func, language=language)

    a = 100
    b = np.array([5, 2, 8, 1, 9], dtype=np.int64)

    result = func(a, b)
    result_epyc = func_epyc(a, b)

    assert result == result_epyc


def test_augassign_sub_max_array(language):
    func = mod.augassign_sub_max_array
    func_epyc = epyccel(func, language=language)

    a1 = np.array([[10, 20], [30, 40]], dtype=np.int64)
    a2 = a1.copy()
    b = np.array([5, 2, 8], dtype=np.int64)

    func(a1, b)
    func_epyc(a2, b)

    assert np.array_equal(a1, a2)


# *= tests


def test_augassign_mul_1d(language):
    f_int = mod.augassign_mul_1d_int
    f_float = mod.augassign_mul_1d_float
    f_complex = mod.augassign_mul_1d_complex
    f_int_epyc = epyccel(f_int, language=language)
    f_float_epyc = epyccel(f_float, language=language)
    f_complex_epyc = epyccel(f_complex, language=language)

    x1_int = np.zeros(5, dtype=int)
    x1_float = np.zeros(5, dtype=float)
    x1_complex = np.zeros(5, dtype=complex)
    x2_int = np.zeros(5, dtype=int)
    x2_float = np.zeros(5, dtype=float)
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
    f_int = mod.augassign_mul_2d_int
    f_float = mod.augassign_mul_2d_float
    f_complex = mod.augassign_mul_2d_complex
    f_int_epyc = epyccel(f_int, language=language)
    f_float_epyc = epyccel(f_float, language=language)
    f_complex_epyc = epyccel(f_complex, language=language)

    x1_int = np.zeros((5, 5), dtype=int)
    x1_float = np.zeros((5, 5), dtype=float)
    x1_complex = np.zeros((5, 5), dtype=complex)
    x2_int = np.zeros((5, 5), dtype=int)
    x2_float = np.zeros((5, 5), dtype=float)
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


def test_augassign_mul_sum_scalar(language):
    func = mod.augassign_mul_sum_scalar
    func_epyc = epyccel(func, language=language)

    a = 10
    b = np.array([1, 2, 3], dtype=np.int64)

    result = func(a, b)
    result_epyc = func_epyc(a, b)

    assert result == result_epyc


# /= tests


def test_augassign_div_1d(language):
    f_float = mod.augassign_div_1d_float
    f_complex = mod.augassign_div_1d_complex
    f_float_epyc = epyccel(f_float, language=language)
    f_complex_epyc = epyccel(f_complex, language=language)

    x1_float = np.zeros(5, dtype=float)
    x1_complex = np.zeros(5, dtype=complex)
    x2_float = np.zeros(5, dtype=float)
    x2_complex = np.zeros(5, dtype=complex)

    y1_float = f_float(x1_float)
    y1_complex = f_complex(x1_complex)
    y2_float = f_float_epyc(x2_float)
    y2_complex = f_complex_epyc(x2_complex)

    assert y1_float == y2_float and np.array_equal(x1_float, x2_float)
    assert y1_complex == y2_complex and np.array_equal(x1_complex, x2_complex)


def test_augassign_div_2d(language):
    f_float = mod.augassign_div_2d_float
    f_complex = mod.augassign_div_2d_complex
    f_float_epyc = epyccel(f_float, language=language)
    f_complex_epyc = epyccel(f_complex, language=language)

    x1_float = np.zeros((5, 5), dtype=float)
    x1_complex = np.zeros((5, 5), dtype=complex)
    x2_float = np.zeros((5, 5), dtype=float)
    x2_complex = np.zeros((5, 5), dtype=complex)

    y1_float = f_float(x1_float)
    y1_complex = f_complex(x1_complex)
    y2_float = f_float_epyc(x2_float)
    y2_complex = f_complex_epyc(x2_complex)

    assert y1_float == y2_float and np.array_equal(x1_float, x2_float)
    assert y1_complex == y2_complex and np.array_equal(x1_complex, x2_complex)


def test_augassign_func(language):
    func = mod.augassign_func
    func_epyc = epyccel(func, language=language)

    x = random() * 100 + 20
    y = random() * 100

    z = func(x, y)
    z_epyc = func_epyc(x, y)

    assert np.isclose(z, z_epyc, rtol=RTOL, atol=ATOL)
    assert isinstance(z, type(z_epyc))


def test_augassign_array_func(language):
    func = mod.augassign_array_func
    func_epyc = epyccel(func, language=language)

    x = random(10) * 100 + 20
    y = random(10) * 100
    x_epyc = x.copy()

    func(x, y)
    func_epyc(x_epyc, y)

    assert np.allclose(x, x_epyc, rtol=RTOL, atol=ATOL)


def test_augassign_floor_div(language):
    func = mod.augassign_floor_div
    func_epyc = epyccel(func, language=language)

    x1_float = random((5,)) * 10
    x2_float = x1_float.copy()

    func(x1_float)
    func_epyc(x2_float)

    assert np.allclose(x1_float, x2_float, rtol=RTOL, atol=ATOL)
