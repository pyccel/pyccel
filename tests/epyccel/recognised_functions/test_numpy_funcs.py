# pylint: disable=missing-function-docstring, missing-module-docstring
import os
import sys
from typing import TypeVar

import numpy as np
import pytest
from modules import numpy_funcs
from numpy import isclose
from numpy.random import rand, randint, randn, uniform

from pyccel import epyccel

from epyccel_utilities import epyccel_module_with_fallback, matching_types
from tolerances import (
    ATOL,
    ATOL32,
    RTOL,
    RTOL32,
    max_float,
    max_float32,
    max_float64,
    max_int,
    max_int8,
    max_int16,
    max_int32,
    max_int64,
    min_float,
    min_float32,
    min_float64,
    min_int,
    min_int8,
    min_int16,
    min_int32,
    min_int64,
)


@pytest.fixture(scope="module")
def epyc_numpy_funcs_mod(language):
    return epyccel_module_with_fallback(numpy_funcs, language)


F = TypeVar(
    "F", "bool", "int", "int8", "int16", "int32", "int64", "float", "float32", "float64"
)
C = TypeVar(
    "C",
    "bool",
    "int",
    "int8",
    "int16",
    "int32",
    "int64",
    "float",
    "float32",
    "float64",
    "complex64",
    "complex128",
)
CT = TypeVar("CT", "complex", "complex64", "complex128")
CNT = TypeVar("CNT", "complex64", "complex128")  # complex numpy types
T = TypeVar(
    "T",
    "int",
    "float",
    "complex",
    "int32",
    "float32",
    "float64",
    "complex64",
    "complex128",
)
S = TypeVar("S", int, "int8", "int16", "int32", "int64", "float", "float32", "float64")


# Functions still to be tested:
#    diag
#    cross
#    # ---


# -------------------------------- Fabs function ------------------------------#
def test_fabs_call_r(epyc_numpy_funcs_mod):
    fabs_call_r = numpy_funcs.fabs_call_r

    f1 = epyc_numpy_funcs_mod.fabs_call_r
    x = uniform(high=1e6)
    assert isclose(f1(x), fabs_call_r(x), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x), fabs_call_r(-x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), fabs_call_r(x))


def test_fabs_call_i(epyc_numpy_funcs_mod):
    fabs_call_i = numpy_funcs.fabs_call_i

    f1 = epyc_numpy_funcs_mod.fabs_call_i
    x = randint(1e6)
    assert isclose(f1(x), fabs_call_i(x), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x), fabs_call_i(-x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), fabs_call_i(x))


def test_fabs_phrase_r_r(epyc_numpy_funcs_mod):
    fabs_phrase_r_r = numpy_funcs.fabs_phrase_r_r

    f2 = epyc_numpy_funcs_mod.fabs_phrase_r_r
    x = uniform(high=1e6)
    y = uniform(high=1e6)
    assert isclose(f2(x, y), fabs_phrase_r_r(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), fabs_phrase_r_r(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), fabs_phrase_r_r(x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y), fabs_phrase_r_r(-x, y), rtol=RTOL, atol=ATOL)


def test_fabs_phrase_i_i(epyc_numpy_funcs_mod):
    fabs_phrase_i_i = numpy_funcs.fabs_phrase_i_i

    f2 = epyc_numpy_funcs_mod.fabs_phrase_i_i
    x = randint(1e6)
    y = randint(1e6)
    assert isclose(f2(x, y), fabs_phrase_i_i(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), fabs_phrase_i_i(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), fabs_phrase_i_i(x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y), fabs_phrase_i_i(-x, y), rtol=RTOL, atol=ATOL)


def test_fabs_phrase_r_i(epyc_numpy_funcs_mod):
    fabs_phrase_r_i = numpy_funcs.fabs_phrase_r_i

    f2 = epyc_numpy_funcs_mod.fabs_phrase_r_i
    x = uniform(high=1e6)
    y = randint(1e6)
    assert isclose(f2(x, y), fabs_phrase_r_i(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), fabs_phrase_r_i(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), fabs_phrase_r_i(x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y), fabs_phrase_r_i(-x, y), rtol=RTOL, atol=ATOL)


def test_fabs_phrase_i_r(epyc_numpy_funcs_mod):
    fabs_phrase_r_i = numpy_funcs.fabs_phrase_i_r

    f2 = epyc_numpy_funcs_mod.fabs_phrase_i_r
    x = randint(1e6)
    y = uniform(high=1e6)
    assert isclose(f2(x, y), fabs_phrase_r_i(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), fabs_phrase_r_i(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), fabs_phrase_r_i(x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y), fabs_phrase_r_i(-x, y), rtol=RTOL, atol=ATOL)


# ------------------------------ isnan function ----------------------------#
def test_numpy_isnan(epyc_numpy_funcs_mod):
    numpy_isnan_test = numpy_funcs.numpy_isnan__numpy_isnan_test
    numpy_isnan_array_test = numpy_funcs.numpy_isnan__numpy_isnan_array_test
    numpy_isnan_expr_test = numpy_funcs.numpy_isnan__numpy_isnan_expr_test

    f = epyc_numpy_funcs_mod.numpy_isnan__numpy_isnan_test
    f_arr = epyc_numpy_funcs_mod.numpy_isnan__numpy_isnan_array_test
    f_expr = epyc_numpy_funcs_mod.numpy_isnan__numpy_isnan_expr_test

    input_data = np.nan
    expected_output = numpy_isnan_test(input_data)
    obtained = f(input_data)

    assert obtained == expected_output
    assert matching_types(obtained, expected_output)

    expected_output = numpy_isnan_expr_test(input_data, 3.0)
    obtained = f_expr(input_data, 3.0)

    assert obtained == expected_output
    assert matching_types(obtained, expected_output)

    expected_output = numpy_isnan_expr_test(3.0, input_data)
    obtained = f_expr(3.0, input_data)

    assert obtained == expected_output
    assert matching_types(obtained, expected_output)

    input_data = np.random.uniform(-1e6, 1e6)
    expected_output = numpy_isnan_test(input_data)
    obtained = f(input_data)

    assert obtained == expected_output
    assert matching_types(obtained, expected_output)

    expected_output = numpy_isnan_expr_test(input_data, 3.0)
    obtained = f_expr(input_data, 3.0)

    assert obtained == expected_output
    assert matching_types(obtained, expected_output)

    expected_output = numpy_isnan_expr_test(3.0, input_data)
    obtained = f_expr(3.0, input_data)

    assert obtained == expected_output
    assert matching_types(obtained, expected_output)

    input_data = np.random.uniform(-1e6, 1e6, size=5)
    expected_output = numpy_isnan_array_test(input_data)
    obtained = f_arr(input_data)

    assert np.array_equal(obtained, expected_output)
    assert matching_types(obtained, expected_output)

    input_data[1] = np.nan

    expected_output = numpy_isnan_array_test(input_data)
    obtained = f_arr(input_data)

    assert np.array_equal(obtained, expected_output)
    assert matching_types(obtained, expected_output)


# ------------------------------ isinf function ----------------------------#
def test_numpy_isinf(epyc_numpy_funcs_mod):
    numpy_isinf_test = numpy_funcs.numpy_isinf__numpy_isinf_test
    numpy_isinf_array_test = numpy_funcs.numpy_isinf__numpy_isinf_array_test
    numpy_isinf_expr_test = numpy_funcs.numpy_isinf__numpy_isinf_expr_test

    f = epyc_numpy_funcs_mod.numpy_isinf__numpy_isinf_test
    f_arr = epyc_numpy_funcs_mod.numpy_isinf__numpy_isinf_array_test
    f_expr = epyc_numpy_funcs_mod.numpy_isinf__numpy_isinf_expr_test

    input_data = np.inf
    expected_output = numpy_isinf_test(input_data)
    obtained = f(input_data)

    assert obtained == expected_output
    assert matching_types(obtained, expected_output)

    expected_output = numpy_isinf_expr_test(input_data, 3.0)
    obtained = f_expr(input_data, 3.0)

    assert obtained == expected_output
    assert matching_types(obtained, expected_output)

    expected_output = numpy_isinf_expr_test(3.0, input_data)
    obtained = f_expr(3.0, input_data)

    assert obtained == expected_output
    assert matching_types(obtained, expected_output)

    input_data = np.random.uniform(-1e6, 1e6)
    expected_output = numpy_isinf_test(input_data)
    obtained = f(input_data)

    assert obtained == expected_output
    assert matching_types(obtained, expected_output)

    expected_output = numpy_isinf_expr_test(input_data, 3.0)
    obtained = f_expr(input_data, 3.0)

    assert obtained == expected_output
    assert matching_types(obtained, expected_output)

    expected_output = numpy_isinf_expr_test(3.0, input_data)
    obtained = f_expr(3.0, input_data)

    assert obtained == expected_output
    assert matching_types(obtained, expected_output)

    input_data = np.random.uniform(-1e6, 1e6, size=5)
    expected_output = numpy_isinf_array_test(input_data)
    obtained = f_arr(input_data)

    assert np.array_equal(obtained, expected_output)
    assert matching_types(obtained, expected_output)

    input_data[1] = np.inf

    expected_output = numpy_isinf_array_test(input_data)
    obtained = f_arr(input_data)

    assert np.array_equal(obtained, expected_output)
    assert matching_types(obtained, expected_output)


# ------------------------------ isfinite function ----------------------------#
@pytest.mark.xfail(
    os.environ.get("PYCCEL_DEFAULT_COMPILER", None) == "intel",
    reason="Different inf representation.",
)
def test_numpy_isfinite(epyc_numpy_funcs_mod):
    numpy_isfinite_test = numpy_funcs.numpy_isfinite__numpy_isfinite_test
    numpy_isfinite_array_test = numpy_funcs.numpy_isfinite__numpy_isfinite_array_test
    numpy_isfinite_expr_test = numpy_funcs.numpy_isfinite__numpy_isfinite_expr_test

    f = epyc_numpy_funcs_mod.numpy_isfinite__numpy_isfinite_test
    f_arr = epyc_numpy_funcs_mod.numpy_isfinite__numpy_isfinite_array_test
    f_expr = epyc_numpy_funcs_mod.numpy_isfinite__numpy_isfinite_expr_test

    input_data = np.inf
    expected_output = numpy_isfinite_test(input_data)
    obtained = f(input_data)

    assert obtained == expected_output
    assert matching_types(obtained, expected_output)

    expected_output = numpy_isfinite_expr_test(input_data, 3.0)
    obtained = f_expr(input_data, 3.0)

    assert obtained == expected_output
    assert matching_types(obtained, expected_output)

    expected_output = numpy_isfinite_expr_test(3.0, input_data)
    obtained = f_expr(3.0, input_data)

    assert obtained == expected_output
    assert matching_types(obtained, expected_output)

    input_data = np.random.uniform(-1e6, 1e6)
    expected_output = numpy_isfinite_test(input_data)
    obtained = f(input_data)

    assert obtained == expected_output
    assert matching_types(obtained, expected_output)

    expected_output = numpy_isfinite_expr_test(input_data, 3.0)
    obtained = f_expr(input_data, 3.0)

    assert obtained == expected_output
    assert matching_types(obtained, expected_output)

    expected_output = numpy_isfinite_expr_test(3.0, input_data)
    obtained = f_expr(3.0, input_data)

    assert obtained == expected_output
    assert matching_types(obtained, expected_output)

    input_data = np.random.uniform(-1e6, 1e6, size=5)
    expected_output = numpy_isfinite_array_test(input_data)
    obtained = f_arr(input_data)

    assert np.array_equal(obtained, expected_output)
    assert matching_types(obtained, expected_output)

    input_data[1] = np.inf

    expected_output = numpy_isfinite_array_test(input_data)
    obtained = f_arr(input_data)

    assert np.array_equal(obtained, expected_output)
    assert matching_types(obtained, expected_output)


# ------------------------------ absolute function ----------------------------#
def test_absolute_call_r(epyc_numpy_funcs_mod):
    absolute_call_r = numpy_funcs.absolute_call_r

    f1 = epyc_numpy_funcs_mod.absolute_call_r
    x = uniform(high=1e6)
    assert f1(x) == absolute_call_r(x)
    assert f1(-x) == absolute_call_r(-x)
    assert matching_types(f1(x), absolute_call_r(x))


def test_absolute_call_i(epyc_numpy_funcs_mod):
    absolute_call_i = numpy_funcs.absolute_call_i

    f1 = epyc_numpy_funcs_mod.absolute_call_i
    x = randint(1e6)
    assert f1(x) == absolute_call_i(x)
    assert f1(-x) == absolute_call_i(-x)
    assert matching_types(f1(x), absolute_call_i(x))


def test_absolute_call_c(epyc_numpy_funcs_mod):
    absolute_call_c = numpy_funcs.absolute_call_c

    f1 = epyc_numpy_funcs_mod.absolute_call_c
    x = uniform(high=1e6) + 1j * uniform(high=1e6)
    assert isclose(f1(x), absolute_call_c(x), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x), absolute_call_c(-x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), absolute_call_c(x))

    x = np.complex64(uniform(high=1e6) - 1j * uniform(high=1e6))
    assert isclose(f1(x), absolute_call_c(x), rtol=RTOL32, atol=ATOL32)
    assert matching_types(f1(x), absolute_call_c(x))

    x = np.complex128(uniform(high=1e6) - 1j * uniform(high=1e6))
    assert isclose(f1(x), absolute_call_c(x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), absolute_call_c(x))


def test_absolute_phrase_r_r(epyc_numpy_funcs_mod):
    absolute_phrase_r_r = numpy_funcs.absolute_phrase_r_r

    f2 = epyc_numpy_funcs_mod.absolute_phrase_r_r
    x = uniform(high=1e6)
    y = uniform(high=1e6)
    assert isclose(f2(x, y), absolute_phrase_r_r(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), absolute_phrase_r_r(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y), absolute_phrase_r_r(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), absolute_phrase_r_r(x, -y), rtol=RTOL, atol=ATOL)


def test_absolute_phrase_i_r(epyc_numpy_funcs_mod):
    absolute_phrase_i_r = numpy_funcs.absolute_phrase_i_r

    f2 = epyc_numpy_funcs_mod.absolute_phrase_i_r
    x = randint(1e6)
    y = uniform(high=1e6)
    assert isclose(f2(x, y), absolute_phrase_i_r(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), absolute_phrase_i_r(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y), absolute_phrase_i_r(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), absolute_phrase_i_r(x, -y), rtol=RTOL, atol=ATOL)


def test_absolute_phrase_r_i(epyc_numpy_funcs_mod):
    absolute_phrase_r_i = numpy_funcs.absolute_phrase_r_i

    f2 = epyc_numpy_funcs_mod.absolute_phrase_r_i
    x = uniform(high=1e6)
    y = randint(1e6)
    assert isclose(f2(x, y), absolute_phrase_r_i(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), absolute_phrase_r_i(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y), absolute_phrase_r_i(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), absolute_phrase_r_i(x, -y), rtol=RTOL, atol=ATOL)


# --------------------------------- sin function ------------------------------#
def test_sin_call_r(epyc_numpy_funcs_mod):
    sin_call_r = numpy_funcs.sin_call_r

    f1 = epyc_numpy_funcs_mod.sin_call_r
    x = uniform(high=1e6)
    assert isclose(f1(x), sin_call_r(x), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x), sin_call_r(-x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), sin_call_r(x))


def test_sin_call_i(epyc_numpy_funcs_mod):
    sin_call_i = numpy_funcs.sin_call_i

    f1 = epyc_numpy_funcs_mod.sin_call_i
    x = randint(1e6)
    assert isclose(f1(x), sin_call_i(x), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x), sin_call_i(-x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), sin_call_i(x))


def test_sin_phrase_r_r(epyc_numpy_funcs_mod):
    sin_phrase_r_r = numpy_funcs.sin_phrase_r_r

    f2 = epyc_numpy_funcs_mod.sin_phrase_r_r
    x = uniform(high=1e6)
    y = uniform(high=1e6)
    assert isclose(f2(x, y), sin_phrase_r_r(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), sin_phrase_r_r(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y), sin_phrase_r_r(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), sin_phrase_r_r(x, -y), rtol=RTOL, atol=ATOL)


def test_sin_phrase_i_i(epyc_numpy_funcs_mod):
    sin_phrase_i_i = numpy_funcs.sin_phrase_i_i

    f2 = epyc_numpy_funcs_mod.sin_phrase_i_i
    x = randint(1e6)
    y = randint(1e6)
    assert isclose(f2(x, y), sin_phrase_i_i(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), sin_phrase_i_i(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y), sin_phrase_i_i(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), sin_phrase_i_i(x, -y), rtol=RTOL, atol=ATOL)


def test_sin_phrase_i_r(epyc_numpy_funcs_mod):
    sin_phrase_i_r = numpy_funcs.sin_phrase_i_r

    f2 = epyc_numpy_funcs_mod.sin_phrase_i_r
    x = randint(1e6)
    y = uniform(high=1e6)
    assert isclose(f2(x, y), sin_phrase_i_r(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), sin_phrase_i_r(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y), sin_phrase_i_r(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), sin_phrase_i_r(x, -y), rtol=RTOL, atol=ATOL)


def test_sin_phrase_r_i(epyc_numpy_funcs_mod):
    sin_phrase_r_i = numpy_funcs.sin_phrase_r_i

    f2 = epyc_numpy_funcs_mod.sin_phrase_r_i
    x = uniform(high=1e6)
    y = randint(1e6)
    assert isclose(f2(x, y), sin_phrase_r_i(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), sin_phrase_r_i(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y), sin_phrase_r_i(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), sin_phrase_r_i(x, -y), rtol=RTOL, atol=ATOL)


# --------------------------------- cos function ------------------------------#
def test_cos_call_i(epyc_numpy_funcs_mod):
    cos_call_i = numpy_funcs.cos_call_i

    f1 = epyc_numpy_funcs_mod.cos_call_i
    x = randint(1e6)
    assert isclose(f1(x), cos_call_i(x), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x), cos_call_i(-x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), cos_call_i(x))


def test_cos_call_r(epyc_numpy_funcs_mod):
    cos_call_r = numpy_funcs.cos_call_r

    f1 = epyc_numpy_funcs_mod.cos_call_r
    x = uniform(high=1e6)
    assert isclose(f1(x), cos_call_r(x), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x), cos_call_r(-x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), cos_call_r(x))


def test_cos_call_out(epyc_numpy_funcs_mod):
    cos_call = numpy_funcs.cos_call_out

    f1 = epyc_numpy_funcs_mod.cos_call_out
    x = uniform(high=1e6, size=5)
    y_epyc = np.empty_like(x)
    y_pyth = np.empty_like(x)
    f1(x, y_epyc)
    cos_call(x, y_pyth)
    assert np.allclose(y_epyc, y_pyth, rtol=RTOL, atol=ATOL)


def test_cos_phrase_i_i(epyc_numpy_funcs_mod):
    cos_phrase_i_i = numpy_funcs.cos_phrase_i_i

    f2 = epyc_numpy_funcs_mod.cos_phrase_i_i
    x = randint(1e6)
    y = randint(1e6)
    assert isclose(f2(x, y), cos_phrase_i_i(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), cos_phrase_i_i(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y), cos_phrase_i_i(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), cos_phrase_i_i(x, -y), rtol=RTOL, atol=ATOL)


def test_cos_phrase_r_r(epyc_numpy_funcs_mod):
    cos_phrase_r_r = numpy_funcs.cos_phrase_r_r

    f2 = epyc_numpy_funcs_mod.cos_phrase_r_r
    x = uniform(high=1e6)
    y = uniform(high=1e6)
    assert isclose(f2(x, y), cos_phrase_r_r(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), cos_phrase_r_r(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y), cos_phrase_r_r(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), cos_phrase_r_r(x, -y), rtol=RTOL, atol=ATOL)


def test_cos_phrase_i_r(epyc_numpy_funcs_mod):
    cos_phrase_i_r = numpy_funcs.cos_phrase_i_r

    f2 = epyc_numpy_funcs_mod.cos_phrase_i_r
    x = randint(1e6)
    y = uniform(high=1e6)
    assert isclose(f2(x, y), cos_phrase_i_r(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), cos_phrase_i_r(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y), cos_phrase_i_r(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), cos_phrase_i_r(x, -y), rtol=RTOL, atol=ATOL)


def test_cos_phrase_r_i(epyc_numpy_funcs_mod):
    cos_phrase_r_i = numpy_funcs.cos_phrase_r_i

    f2 = epyc_numpy_funcs_mod.cos_phrase_r_i
    x = uniform(high=1e6)
    y = randint(1e6)
    assert isclose(f2(x, y), cos_phrase_r_i(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), cos_phrase_r_i(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y), cos_phrase_r_i(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), cos_phrase_r_i(x, -y), rtol=RTOL, atol=ATOL)


# --------------------------------- tan function ------------------------------#
def test_tan_call_i(epyc_numpy_funcs_mod):
    tan_call_i = numpy_funcs.tan_call_i

    f1 = epyc_numpy_funcs_mod.tan_call_i
    x = randint(1e6)
    assert isclose(f1(x), tan_call_i(x), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x), tan_call_i(-x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), tan_call_i(x))


def test_tan_call_r(epyc_numpy_funcs_mod):
    tan_call_r = numpy_funcs.tan_call_r

    f1 = epyc_numpy_funcs_mod.tan_call_r
    x = uniform(high=1e6)
    assert isclose(f1(x), tan_call_r(x), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x), tan_call_r(-x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), tan_call_r(x))


def test_tan_phrase_i_i(epyc_numpy_funcs_mod):
    tan_phrase_i_i = numpy_funcs.tan_phrase_i_i

    f2 = epyc_numpy_funcs_mod.tan_phrase_i_i
    x = randint(1e6)
    y = randint(1e6)
    assert isclose(f2(x, y), tan_phrase_i_i(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), tan_phrase_i_i(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y), tan_phrase_i_i(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), tan_phrase_i_i(x, -y), rtol=RTOL, atol=ATOL)


def test_tan_phrase_r_r(epyc_numpy_funcs_mod):
    tan_phrase_r_r = numpy_funcs.tan_phrase_r_r

    f2 = epyc_numpy_funcs_mod.tan_phrase_r_r
    x = uniform(high=1e6)
    y = uniform(high=1e6)
    assert isclose(f2(x, y), tan_phrase_r_r(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), tan_phrase_r_r(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y), tan_phrase_r_r(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), tan_phrase_r_r(x, -y), rtol=RTOL, atol=ATOL)


def test_tan_phrase_i_r(epyc_numpy_funcs_mod):
    tan_phrase_i_r = numpy_funcs.tan_phrase_i_r

    f2 = epyc_numpy_funcs_mod.tan_phrase_i_r
    x = randint(1e6)
    y = uniform(high=1e6)
    assert isclose(f2(x, y), tan_phrase_i_r(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), tan_phrase_i_r(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y), tan_phrase_i_r(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), tan_phrase_i_r(x, -y), rtol=RTOL, atol=ATOL)


def test_tan_phrase_r_i(epyc_numpy_funcs_mod):
    tan_phrase_r_i = numpy_funcs.tan_phrase_r_i

    f2 = epyc_numpy_funcs_mod.tan_phrase_r_i
    x = uniform(high=1e6)
    y = randint(1e6)
    assert isclose(f2(x, y), tan_phrase_r_i(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), tan_phrase_r_i(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y), tan_phrase_r_i(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), tan_phrase_r_i(x, -y), rtol=RTOL, atol=ATOL)


# --------------------------------- exp function ------------------------------#
def test_exp_call_i(epyc_numpy_funcs_mod):
    exp_call_i = numpy_funcs.exp_call_i

    f1 = epyc_numpy_funcs_mod.exp_call_i
    x = randint(1e2)
    assert isclose(f1(x), exp_call_i(x), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x), exp_call_i(-x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), exp_call_i(x))


def test_exp_call_r(epyc_numpy_funcs_mod):
    exp_call_r = numpy_funcs.exp_call_r

    f1 = epyc_numpy_funcs_mod.exp_call_r
    x = uniform(high=1e2)
    assert isclose(f1(x), exp_call_r(x), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x), exp_call_r(-x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), exp_call_r(x))


def test_exp_phrase_i_i(epyc_numpy_funcs_mod):
    exp_phrase_i_i = numpy_funcs.exp_phrase_i_i

    f2 = epyc_numpy_funcs_mod.exp_phrase_i_i
    x = randint(1e2)
    y = randint(1e2)
    assert isclose(f2(x, y), exp_phrase_i_i(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), exp_phrase_i_i(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y), exp_phrase_i_i(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), exp_phrase_i_i(x, -y), rtol=RTOL, atol=ATOL)


def test_exp_phrase_r_r(epyc_numpy_funcs_mod):
    exp_phrase_r_r = numpy_funcs.exp_phrase_r_r

    f2 = epyc_numpy_funcs_mod.exp_phrase_r_r
    x = uniform(high=1e2)
    y = uniform(high=1e2)
    assert isclose(f2(x, y), exp_phrase_r_r(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), exp_phrase_r_r(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y), exp_phrase_r_r(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), exp_phrase_r_r(x, -y), rtol=RTOL, atol=ATOL)


def test_exp_phrase_i_r(epyc_numpy_funcs_mod):
    exp_phrase_i_r = numpy_funcs.exp_phrase_i_r

    f2 = epyc_numpy_funcs_mod.exp_phrase_i_r
    x = randint(1e2)
    y = uniform(high=1e2)
    assert isclose(f2(x, y), exp_phrase_i_r(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), exp_phrase_i_r(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y), exp_phrase_i_r(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), exp_phrase_i_r(x, -y), rtol=RTOL, atol=ATOL)


def test_exp_phrase_r_i(epyc_numpy_funcs_mod):
    exp_phrase_r_i = numpy_funcs.exp_phrase_r_i

    f2 = epyc_numpy_funcs_mod.exp_phrase_r_i
    x = uniform(high=1e2)
    y = randint(1e2)
    assert isclose(f2(x, y), exp_phrase_r_i(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, y), exp_phrase_r_i(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, y), exp_phrase_r_i(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), exp_phrase_r_i(x, -y), rtol=RTOL, atol=ATOL)


# --------------------------------- expm1 function ------------------------------#
def test_expm1_call_i(epyc_numpy_funcs_mod):
    expm1_call_i = numpy_funcs.expm1_call_i

    f1 = epyc_numpy_funcs_mod.expm1_call_i
    x = randint(100)
    assert isclose(f1(x), expm1_call_i(x), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x), expm1_call_i(-x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), expm1_call_i(x))


def test_expm1_call_f(epyc_numpy_funcs_mod):
    expm1_call_f = numpy_funcs.expm1_call_f

    f1 = epyc_numpy_funcs_mod.expm1_call_f
    x = uniform(high=100)
    assert isclose(f1(x), expm1_call_f(x), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x), expm1_call_f(-x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), expm1_call_f(x))


def test_expm1_call_c(epyc_numpy_funcs_mod):
    expm1_call_c = numpy_funcs.expm1_call_c

    f1 = epyc_numpy_funcs_mod.expm1_call_c
    x = uniform(high=100) + uniform(high=100) * 1j
    assert isclose(f1(x), expm1_call_c(x), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x), expm1_call_c(-x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), expm1_call_c(x))


def test_expm1_call_f_array(epyc_numpy_funcs_mod):
    expm1_call_f = numpy_funcs.expm1_call_f_array

    f1 = epyc_numpy_funcs_mod.expm1_call_f_array
    x = uniform(high=100, size=5)
    assert np.allclose(f1(x), expm1_call_f(x), rtol=RTOL, atol=ATOL)
    assert np.allclose(f1(-x), expm1_call_f(-x), rtol=RTOL, atol=ATOL)


def test_expm1_call_c_array(epyc_numpy_funcs_mod):
    expm1_call_c = numpy_funcs.expm1_call_c_array

    f1 = epyc_numpy_funcs_mod.expm1_call_c_array
    x = uniform(high=100, size=5) + uniform(high=100, size=5) * 1j
    assert np.allclose(f1(x), expm1_call_c(x), rtol=RTOL, atol=ATOL)
    assert np.allclose(f1(-x), expm1_call_c(-x), rtol=RTOL, atol=ATOL)


def test_expm1_call_cast_f(epyc_numpy_funcs_mod):
    expm1_call_f = numpy_funcs.expm1_call_cast_f

    f1 = epyc_numpy_funcs_mod.expm1_call_cast_f
    x = np.float32(uniform(high=30))
    assert isclose(f1(x), expm1_call_f(x), rtol=RTOL32, atol=ATOL32)
    assert matching_types(f1(x), expm1_call_f(x))


def test_expm1_call_cast_c(epyc_numpy_funcs_mod):
    expm1_call_c = numpy_funcs.expm1_call_cast_c

    f1 = epyc_numpy_funcs_mod.expm1_call_cast_c
    x = np.complex64(uniform(high=15) + uniform(high=15) * 1j)
    assert isclose(f1(x), expm1_call_c(x), rtol=RTOL32, atol=ATOL32)
    assert matching_types(f1(x), expm1_call_c(x))


def test_expm1_phrase_i_i(epyc_numpy_funcs_mod):
    expm1_phrase_i_i = numpy_funcs.expm1_phrase_i_i

    f2 = epyc_numpy_funcs_mod.expm1_phrase_i_i
    x = randint(100)
    y = randint(100)
    assert isclose(f2(x, y), expm1_phrase_i_i(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), expm1_phrase_i_i(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y), expm1_phrase_i_i(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), expm1_phrase_i_i(x, -y), rtol=RTOL, atol=ATOL)


def test_expm1_phrase_f_f(epyc_numpy_funcs_mod):
    expm1_phrase_f_f = numpy_funcs.expm1_phrase_f_f

    f2 = epyc_numpy_funcs_mod.expm1_phrase_f_f
    x = uniform(high=100)
    y = uniform(high=100)
    assert isclose(f2(x, y), expm1_phrase_f_f(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), expm1_phrase_f_f(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y), expm1_phrase_f_f(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), expm1_phrase_f_f(x, -y), rtol=RTOL, atol=ATOL)


def test_expm1_phrase_i_f(epyc_numpy_funcs_mod):
    expm1_phrase_i_f = numpy_funcs.expm1_phrase_i_f

    f2 = epyc_numpy_funcs_mod.expm1_phrase_i_f
    x = randint(100)
    y = uniform(high=100)
    assert isclose(f2(x, y), expm1_phrase_i_f(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), expm1_phrase_i_f(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y), expm1_phrase_i_f(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), expm1_phrase_i_f(x, -y), rtol=RTOL, atol=ATOL)


def test_expm1_phrase_f_i(epyc_numpy_funcs_mod):
    expm1_phrase_f_i = numpy_funcs.expm1_phrase_f_i

    f2 = epyc_numpy_funcs_mod.expm1_phrase_f_i
    x = uniform(high=100)
    y = randint(100)
    assert isclose(f2(x, y), expm1_phrase_f_i(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, y), expm1_phrase_f_i(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, y), expm1_phrase_f_i(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), expm1_phrase_f_i(x, -y), rtol=RTOL, atol=ATOL)


def test_expm1_phrase_i_c(epyc_numpy_funcs_mod):
    expm1_phrase_i_c = numpy_funcs.expm1_phrase_i_c

    f2 = epyc_numpy_funcs_mod.expm1_phrase_i_c
    x = randint(100)
    y = uniform(high=100) + uniform(high=100) * 1j
    assert isclose(f2(x, y), expm1_phrase_i_c(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, y), expm1_phrase_i_c(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, y), expm1_phrase_i_c(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), expm1_phrase_i_c(x, -y), rtol=RTOL, atol=ATOL)


# --------------------------------- log function ------------------------------#
def test_log_call_i(epyc_numpy_funcs_mod):
    log_call_i = numpy_funcs.log_call_i

    f1 = epyc_numpy_funcs_mod.log_call_i
    x = randint(low=sys.float_info.min, high=1e6)
    assert isclose(f1(x), log_call_i(x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), log_call_i(x))


def test_log_call_r(epyc_numpy_funcs_mod):
    log_call_r = numpy_funcs.log_call_r

    f1 = epyc_numpy_funcs_mod.log_call_r
    x = uniform(low=sys.float_info.min, high=max_float)
    assert isclose(f1(x), log_call_r(x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), log_call_r(x))


def test_log_phrase(epyc_numpy_funcs_mod):
    log_phrase = numpy_funcs.log_phrase

    f2 = epyc_numpy_funcs_mod.log_phrase
    x = uniform(low=sys.float_info.min, high=1e6)
    y = uniform(low=sys.float_info.min, high=1e6)
    assert isclose(f2(x, y), log_phrase(x, y), rtol=RTOL, atol=ATOL)


# ----------------------------- arcsin function -------------------------------#
def test_arcsin_call_i(epyc_numpy_funcs_mod):
    arcsin_call_i = numpy_funcs.arcsin_call_i

    f1 = epyc_numpy_funcs_mod.arcsin_call_i
    x = randint(2)
    assert isclose(f1(x), arcsin_call_i(x), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x), arcsin_call_i(-x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), arcsin_call_i(x))


def test_arcsin_call_r(epyc_numpy_funcs_mod):
    arcsin_call_r = numpy_funcs.arcsin_call_r

    f1 = epyc_numpy_funcs_mod.arcsin_call_r
    x = rand()
    assert isclose(f1(x), arcsin_call_r(x), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x), arcsin_call_r(-x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), arcsin_call_r(x))


def test_arcsin_phrase(epyc_numpy_funcs_mod):
    arcsin_phrase = numpy_funcs.arcsin_phrase

    f2 = epyc_numpy_funcs_mod.arcsin_phrase
    x = rand()
    y = rand()
    assert isclose(f2(x, y), arcsin_phrase(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), arcsin_phrase(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y), arcsin_phrase(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), arcsin_phrase(x, -y), rtol=RTOL, atol=ATOL)


# ----------------------------- arccos function -------------------------------#


def test_arccos_call_i(epyc_numpy_funcs_mod):
    arccos_call_i = numpy_funcs.arccos_call_i

    f1 = epyc_numpy_funcs_mod.arccos_call_i
    x = randint(2)
    assert isclose(f1(x), arccos_call_i(x), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x), arccos_call_i(-x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), arccos_call_i(x))


def test_arccos_call_r(epyc_numpy_funcs_mod):
    arccos_call_r = numpy_funcs.arccos_call_r

    f1 = epyc_numpy_funcs_mod.arccos_call_r
    x = rand()
    assert isclose(f1(x), arccos_call_r(x), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x), arccos_call_r(-x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), arccos_call_r(x))


def test_arccos_phrase(epyc_numpy_funcs_mod):
    arccos_phrase = numpy_funcs.arccos_phrase

    f2 = epyc_numpy_funcs_mod.arccos_phrase
    x = rand()
    y = rand()
    assert isclose(f2(x, y), arccos_phrase(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), arccos_phrase(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y), arccos_phrase(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), arccos_phrase(x, -y), rtol=RTOL, atol=ATOL)


# ----------------------------- arctan function -------------------------------#
def test_arctan_call_i(epyc_numpy_funcs_mod):
    arctan_call_i = numpy_funcs.arctan_call_i

    f1 = epyc_numpy_funcs_mod.arctan_call_i
    x = randint(1e6)
    assert isclose(f1(x), arctan_call_i(x), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x), arctan_call_i(-x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), arctan_call_i(x))


def test_arctan_call_r(epyc_numpy_funcs_mod):
    arctan_call_r = numpy_funcs.arctan_call_r

    f1 = epyc_numpy_funcs_mod.arctan_call_r
    x = uniform(high=1e6)
    assert isclose(f1(x), arctan_call_r(x), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x), arctan_call_r(-x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), arctan_call_r(x))


def test_arctan_phrase(epyc_numpy_funcs_mod):
    arctan_phrase = numpy_funcs.arctan_phrase

    f2 = epyc_numpy_funcs_mod.arctan_phrase
    x = uniform(high=1e6)
    y = uniform(high=1e6)
    assert isclose(f2(x, y), arctan_phrase(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), arctan_phrase(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y), arctan_phrase(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), arctan_phrase(x, -y), rtol=RTOL, atol=ATOL)


# ------------------------------- sinh function -------------------------------#
def test_sinh_call_i(epyc_numpy_funcs_mod):
    sinh_call_i = numpy_funcs.sinh_call_i

    f1 = epyc_numpy_funcs_mod.sinh_call_i
    x = randint(100)
    assert isclose(f1(x), sinh_call_i(x), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x), sinh_call_i(-x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), sinh_call_i(x))


def test_sinh_call_r(epyc_numpy_funcs_mod):
    sinh_call_r = numpy_funcs.sinh_call_r

    f1 = epyc_numpy_funcs_mod.sinh_call_r
    x = uniform(high=1e2)
    assert isclose(f1(x), sinh_call_r(x), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x), sinh_call_r(-x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), sinh_call_r(x))


def test_sinh_phrase(epyc_numpy_funcs_mod):
    sinh_phrase = numpy_funcs.sinh_phrase

    f2 = epyc_numpy_funcs_mod.sinh_phrase
    x = uniform(high=1e2)
    y = uniform(high=1e2)
    assert isclose(f2(x, y), sinh_phrase(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), sinh_phrase(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y), sinh_phrase(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), sinh_phrase(x, -y), rtol=RTOL, atol=ATOL)


# ------------------------------- cosh function -------------------------------#
def test_cosh_call_i(epyc_numpy_funcs_mod):
    cosh_call_i = numpy_funcs.cosh_call_i

    f1 = epyc_numpy_funcs_mod.cosh_call_i
    x = randint(100)
    assert isclose(f1(x), cosh_call_i(x), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x), cosh_call_i(-x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), cosh_call_i(x))


def test_cosh_call_r(epyc_numpy_funcs_mod):
    cosh_call_r = numpy_funcs.cosh_call_r

    f1 = epyc_numpy_funcs_mod.cosh_call_r
    x = uniform(high=1e2)
    assert isclose(f1(x), cosh_call_r(x), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x), cosh_call_r(-x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), cosh_call_r(x))


def test_cosh_phrase(epyc_numpy_funcs_mod):
    cosh_phrase = numpy_funcs.cosh_phrase

    f2 = epyc_numpy_funcs_mod.cosh_phrase
    x = uniform(high=1e2)
    y = uniform(high=1e2)
    assert isclose(f2(x, y), cosh_phrase(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), cosh_phrase(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y), cosh_phrase(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), cosh_phrase(x, -y), rtol=RTOL, atol=ATOL)


# ------------------------------- tanh function -------------------------------#
def test_tanh_call_i(epyc_numpy_funcs_mod):
    tanh_call_i = numpy_funcs.tanh_call_i

    f1 = epyc_numpy_funcs_mod.tanh_call_i
    x = randint(100)
    assert isclose(f1(x), tanh_call_i(x), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x), tanh_call_i(-x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), tanh_call_i(x))


def test_tanh_call_r(epyc_numpy_funcs_mod):
    tanh_call_r = numpy_funcs.tanh_call_r

    f1 = epyc_numpy_funcs_mod.tanh_call_r
    x = uniform(high=1e2)
    assert isclose(f1(x), tanh_call_r(x), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x), tanh_call_r(-x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), tanh_call_r(x))


def test_tanh_phrase(epyc_numpy_funcs_mod):
    tanh_phrase = numpy_funcs.tanh_phrase

    f2 = epyc_numpy_funcs_mod.tanh_phrase
    x = uniform(high=1e2)
    y = uniform(high=1e2)
    assert isclose(f2(x, y), tanh_phrase(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), tanh_phrase(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y), tanh_phrase(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), tanh_phrase(x, -y), rtol=RTOL, atol=ATOL)


# ------------------------------ arctan2 function -----------------------------#
def test_arctan2_call_i_i(epyc_numpy_funcs_mod):
    arctan2_call = numpy_funcs.arctan2_call_i_i

    f1 = epyc_numpy_funcs_mod.arctan2_call_i_i
    x = randint(100)
    y = randint(100)
    assert isclose(f1(x, y), arctan2_call(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x, -y), arctan2_call(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x, y), arctan2_call(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f1(x, -y), arctan2_call(x, -y), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x, y), arctan2_call(x, y))


def test_arctan2_call_i_r(epyc_numpy_funcs_mod):
    arctan2_call = numpy_funcs.arctan2_call_i_r

    f1 = epyc_numpy_funcs_mod.arctan2_call_i_r
    x = randint(100)
    y = uniform(high=1e2)
    assert isclose(f1(x, y), arctan2_call(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x, -y), arctan2_call(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x, y), arctan2_call(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f1(x, -y), arctan2_call(x, -y), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x, y), arctan2_call(x, y))


def test_arctan2_call_r_i(epyc_numpy_funcs_mod):
    arctan2_call = numpy_funcs.arctan2_call_r_i

    f1 = epyc_numpy_funcs_mod.arctan2_call_r_i
    x = uniform(high=1e2)
    y = randint(100)
    assert isclose(f1(x, y), arctan2_call(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x, -y), arctan2_call(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x, y), arctan2_call(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f1(x, -y), arctan2_call(x, -y), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x, y), arctan2_call(x, y))


def test_arctan2_call_r_r(epyc_numpy_funcs_mod):
    arctan2_call = numpy_funcs.arctan2_call_r_r

    f1 = epyc_numpy_funcs_mod.arctan2_call_r_r
    x = uniform(high=1e2)
    y = uniform(high=1e2)
    assert isclose(f1(x, y), arctan2_call(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x, -y), arctan2_call(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x, y), arctan2_call(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f1(x, -y), arctan2_call(x, -y), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x, y), arctan2_call(x, y))


def test_arctan2_phrase(epyc_numpy_funcs_mod):
    arctan2_phrase = numpy_funcs.arctan2_phrase

    f2 = epyc_numpy_funcs_mod.arctan2_phrase
    x = uniform(high=1e2)
    y = uniform(high=1e2)
    z = uniform(high=1e2)
    assert isclose(f2(x, y, z), arctan2_phrase(x, y, z), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y, z), arctan2_phrase(-x, y, z), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y, z), arctan2_phrase(-x, -y, z), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y, -z), arctan2_phrase(-x, y, -z), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y, z), arctan2_phrase(x, -y, z), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y, -z), arctan2_phrase(x, -y, -z), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, y, -z), arctan2_phrase(x, y, -z), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y, -z), arctan2_phrase(-x, -y, -z), rtol=RTOL, atol=ATOL)


# -------------------------------- sqrt function ------------------------------#
def test_sqrt_call(epyc_numpy_funcs_mod):
    sqrt_call = numpy_funcs.sqrt_call

    f1 = epyc_numpy_funcs_mod.sqrt_call
    x = rand()
    assert isclose(f1(x), sqrt_call(x), rtol=RTOL, atol=ATOL)


def test_sqrt_phrase(epyc_numpy_funcs_mod):
    sqrt_phrase = numpy_funcs.sqrt_phrase

    f2 = epyc_numpy_funcs_mod.sqrt_phrase
    x = rand()
    y = rand()
    assert isclose(f2(x, y), sqrt_phrase(x, y), rtol=RTOL, atol=ATOL)


def test_sqrt_return_type_r(epyc_numpy_funcs_mod):
    sqrt_return_type_real = numpy_funcs.sqrt_return_type_r

    f1 = epyc_numpy_funcs_mod.sqrt_return_type_r
    x = rand()
    assert isclose(f1(x), sqrt_return_type_real(x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), sqrt_return_type_real(x))


def test_sqrt_return_type_c(epyc_numpy_funcs_mod):
    sqrt_return_type_comp = numpy_funcs.sqrt_return_type_c

    f1 = epyc_numpy_funcs_mod.sqrt_return_type_c
    x = rand() + 1j * rand()
    assert isclose(f1(x), sqrt_return_type_comp(x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), sqrt_return_type_comp(x))


# -------------------------------- floor function -----------------------------#
def test_floor_call_i(epyc_numpy_funcs_mod):
    floor_call = numpy_funcs.floor_call_i

    f1 = epyc_numpy_funcs_mod.floor_call_i
    x = randint(1e6)
    assert isclose(f1(x), floor_call(x), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x), floor_call(-x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), floor_call(x))


def test_floor_call_r(epyc_numpy_funcs_mod):
    floor_call = numpy_funcs.floor_call_r

    f1 = epyc_numpy_funcs_mod.floor_call_r
    x = uniform(high=1e6)
    assert isclose(f1(x), floor_call(x), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x), floor_call(-x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), floor_call(x))


def test_floor_phrase(epyc_numpy_funcs_mod):
    floor_phrase = numpy_funcs.floor_phrase

    f2 = epyc_numpy_funcs_mod.floor_phrase
    x = uniform(high=1e6)
    y = uniform(high=1e6)
    assert isclose(f2(x, y), floor_phrase(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), floor_phrase(-x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y), floor_phrase(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), floor_phrase(x, -y), rtol=RTOL, atol=ATOL)


def test_shape_indexed(epyc_numpy_funcs_mod):
    test_shape_1d = numpy_funcs.shape_indexed__test_shape_1d
    test_shape_2d = numpy_funcs.shape_indexed__test_shape_2d
    test_shape_2d_f = numpy_funcs.shape_indexed__test_shape_2d_f

    from numpy import empty

    f1 = epyc_numpy_funcs_mod.shape_indexed__test_shape_1d
    f2 = epyc_numpy_funcs_mod.shape_indexed__test_shape_2d
    f3 = epyc_numpy_funcs_mod.shape_indexed__test_shape_2d_f
    n1 = randint(1, 20)
    n2 = randint(1, 20)
    n3 = randint(1, 20)
    x1 = empty(n1, dtype=int)
    x2 = empty((n2, n3), dtype=int)
    x3 = empty((n1, n2, 1), dtype=int)
    assert f1(x1) == test_shape_1d(x1)
    assert f2(x2) == test_shape_2d(x2)
    assert f3(x2.T) == test_shape_2d_f(x2.T)
    assert f3(x3[0, :, :].T) == test_shape_2d_f(x3[0, :, :].T)


def test_shape_property(epyc_numpy_funcs_mod):
    test_shape_1d = numpy_funcs.shape_property__test_shape_1d
    test_shape_2d = numpy_funcs.shape_property__test_shape_2d

    from numpy import empty

    f1 = epyc_numpy_funcs_mod.shape_property__test_shape_1d
    f2 = epyc_numpy_funcs_mod.shape_property__test_shape_2d
    n1 = randint(1, 20)
    n2 = randint(1, 20)
    n3 = randint(1, 20)
    x1 = empty(n1, dtype=int)
    x2 = empty((n2, n3), dtype=int)
    assert f1(x1) == test_shape_1d(x1)
    assert f2(x2) == test_shape_2d(x2)


def test_shape_tuple_output(epyc_numpy_funcs_mod):
    test_shape_1d = numpy_funcs.shape_tuple_output__test_shape_1d
    test_shape_1d_tuple = numpy_funcs.shape_tuple_output__test_shape_1d_tuple
    test_shape_2d = numpy_funcs.shape_tuple_output__test_shape_2d

    from numpy import empty

    n1 = randint(1, 20)
    n2 = randint(1, 20)
    n3 = randint(1, 20)
    x1 = empty(n1, dtype=int)
    x2 = empty((n2, n3), dtype=int)
    f1 = epyc_numpy_funcs_mod.shape_tuple_output__test_shape_1d
    assert f1(x1) == test_shape_1d(x1)
    f1_t = epyc_numpy_funcs_mod.shape_tuple_output__test_shape_1d_tuple
    assert f1_t(x1) == test_shape_1d_tuple(x1)
    f2 = epyc_numpy_funcs_mod.shape_tuple_output__test_shape_2d
    assert f2(x2) == test_shape_2d(x2)


def test_shape_real(epyc_numpy_funcs_mod):
    test_shape_1d = numpy_funcs.shape_real__test_shape_1d
    test_shape_2d = numpy_funcs.shape_real__test_shape_2d

    from numpy import empty

    f1 = epyc_numpy_funcs_mod.shape_real__test_shape_1d
    f2 = epyc_numpy_funcs_mod.shape_real__test_shape_2d
    n1 = randint(1, 20)
    n2 = randint(1, 20)
    n3 = randint(1, 20)
    x1 = empty(n1, dtype=float)
    x2 = empty((n2, n3), dtype=float)
    assert f1(x1) == test_shape_1d(x1)
    assert f2(x2) == test_shape_2d(x2)


def test_shape_int(epyc_numpy_funcs_mod):
    test_shape_1d = numpy_funcs.shape_int__test_shape_1d
    test_shape_2d = numpy_funcs.shape_int__test_shape_2d

    f1 = epyc_numpy_funcs_mod.shape_int__test_shape_1d
    f2 = epyc_numpy_funcs_mod.shape_int__test_shape_2d

    from numpy import empty

    n1 = randint(1, 20)
    n2 = randint(1, 20)
    n3 = randint(1, 20)
    x1 = empty(n1, dtype=int)
    x2 = empty((n2, n3), dtype=int)
    assert f1(x1) == test_shape_1d(x1)
    assert f2(x2) == test_shape_2d(x2)


def test_shape_bool(epyc_numpy_funcs_mod):
    test_shape_1d = numpy_funcs.shape_bool__test_shape_1d
    test_shape_2d = numpy_funcs.shape_bool__test_shape_2d

    from numpy import empty

    f1 = epyc_numpy_funcs_mod.shape_bool__test_shape_1d
    f2 = epyc_numpy_funcs_mod.shape_bool__test_shape_2d
    n1 = randint(1, 20)
    n2 = randint(1, 20)
    n3 = randint(1, 20)
    x1 = empty(n1, dtype=bool)
    x2 = empty((n2, n3), dtype=bool)
    assert f1(x1) == test_shape_1d(x1)
    assert f2(x2) == test_shape_2d(x2)


def test_full_basic_int(epyc_numpy_funcs_mod):
    create_full_shape_1d = numpy_funcs.full_basic_int__create_full_shape_1d
    create_full_shape_2d = numpy_funcs.full_basic_int__create_full_shape_2d
    create_full_val = numpy_funcs.full_basic_int__create_full_val
    create_full_arg_names = numpy_funcs.full_basic_int__create_full_arg_names

    size = randint(1, 10)

    f_shape_1d = epyc_numpy_funcs_mod.full_basic_int__create_full_shape_1d
    assert f_shape_1d(size) == create_full_shape_1d(size)

    f_shape_2d = epyc_numpy_funcs_mod.full_basic_int__create_full_shape_2d
    assert f_shape_2d(size) == create_full_shape_2d(size)

    f_val = epyc_numpy_funcs_mod.full_basic_int__create_full_val
    assert f_val(size) == create_full_val(size)
    assert matching_types(f_val(size)[0], create_full_val(size)[0])

    f_arg_names = epyc_numpy_funcs_mod.full_basic_int__create_full_arg_names
    assert f_arg_names(size) == create_full_arg_names(size)
    assert matching_types(f_arg_names(size)[0], create_full_arg_names(size)[0])


def test_size(epyc_numpy_funcs_mod):
    test_size_1d = numpy_funcs.size__test_size_1d
    test_size_2d = numpy_funcs.size__test_size_2d
    test_size_axis_variable_2d = numpy_funcs.size__test_size_axis_variable_2d
    test_size_axis_literal_3d = numpy_funcs.size__test_size_axis_literal_3d

    from numpy import empty

    f1 = epyc_numpy_funcs_mod.size__test_size_1d
    f2 = epyc_numpy_funcs_mod.size__test_size_2d
    f3 = epyc_numpy_funcs_mod.size__test_size_axis_variable_2d
    f4 = epyc_numpy_funcs_mod.size__test_size_axis_literal_3d
    n1 = randint(1, 20)
    n2 = randint(1, 20)
    n3 = randint(1, 20)
    axis = randint(2)
    x1 = empty(n1, dtype=int)
    x2 = empty((n1, n2), dtype=int)
    x3 = empty((n1, n3), dtype=int)
    x4 = empty((n1, n2, n3), dtype=int)
    assert f1(x1) == test_size_1d(x1)
    assert f2(x2) == test_size_2d(x2)
    assert f3(x3, axis) == test_size_axis_variable_2d(x3, axis)
    assert f4(x4) == test_size_axis_literal_3d(x4)


def test_size_property(epyc_numpy_funcs_mod):
    test_size_1d = numpy_funcs.size_property__test_size_1d
    test_size_2d = numpy_funcs.size_property__test_size_2d
    test_size_3d = numpy_funcs.size_property__test_size_3d
    test_slice_size_2d = numpy_funcs.size_property__test_slice_size_2d

    from numpy import empty

    f1 = epyc_numpy_funcs_mod.size_property__test_size_1d
    f2 = epyc_numpy_funcs_mod.size_property__test_size_2d
    f3 = epyc_numpy_funcs_mod.size_property__test_size_3d
    f4 = epyc_numpy_funcs_mod.size_property__test_slice_size_2d
    n1 = randint(1, 20)
    n2 = randint(1, 20)
    n3 = randint(1, 20)
    x1 = empty(n1, dtype=int)
    x2 = empty((n1, n2), dtype=int)
    x3 = empty((n1, n2, n3), dtype=int)
    assert f1(x1) == test_size_1d(x1)
    assert f2(x2) == test_size_2d(x2)
    assert f3(x3) == test_size_3d(x3)
    assert f4(x3) == test_slice_size_2d(x3)


def test_full_basic_real(epyc_numpy_funcs_mod):
    create_full_shape_1d = numpy_funcs.full_basic_real__create_full_shape_1d
    create_full_shape_2d = numpy_funcs.full_basic_real__create_full_shape_2d
    create_full_val = numpy_funcs.full_basic_real__create_full_val
    create_full_arg_names = numpy_funcs.full_basic_real__create_full_arg_names

    size = randint(1, 10)
    val = rand() * 5

    f_shape_1d = epyc_numpy_funcs_mod.full_basic_real__create_full_shape_1d
    assert f_shape_1d(size) == create_full_shape_1d(size)

    f_shape_2d = epyc_numpy_funcs_mod.full_basic_real__create_full_shape_2d
    assert f_shape_2d(size) == create_full_shape_2d(size)

    f_val = epyc_numpy_funcs_mod.full_basic_real__create_full_val
    assert f_val(val) == create_full_val(val)
    assert matching_types(f_val(val)[0], create_full_val(val)[0])

    f_arg_names = epyc_numpy_funcs_mod.full_basic_real__create_full_arg_names
    assert f_arg_names(val) == create_full_arg_names(val)
    assert matching_types(f_arg_names(val)[0], create_full_arg_names(val)[0])


def test_full_basic_bool(epyc_numpy_funcs_mod):
    create_full_shape_1d = numpy_funcs.full_basic_bool__create_full_shape_1d
    create_full_shape_2d = numpy_funcs.full_basic_bool__create_full_shape_2d
    create_full_val = numpy_funcs.full_basic_bool__create_full_val
    create_full_arg_names = numpy_funcs.full_basic_bool__create_full_arg_names

    size = randint(1, 10)
    val = bool(randint(2))

    f_shape_1d = epyc_numpy_funcs_mod.full_basic_bool__create_full_shape_1d
    assert f_shape_1d(size) == create_full_shape_1d(size)

    f_shape_2d = epyc_numpy_funcs_mod.full_basic_bool__create_full_shape_2d
    assert f_shape_2d(size) == create_full_shape_2d(size)

    f_val = epyc_numpy_funcs_mod.full_basic_bool__create_full_val
    assert f_val(val) == create_full_val(val)
    assert matching_types(f_val(val)[0], create_full_val(val)[0])

    f_arg_names = epyc_numpy_funcs_mod.full_basic_bool__create_full_arg_names
    assert f_arg_names(val) == create_full_arg_names(val)
    assert matching_types(f_arg_names(val)[0], create_full_arg_names(val)[0])


def test_full_order(epyc_numpy_funcs_mod):
    create_full_shape_C = numpy_funcs.full_order__create_full_shape_C
    create_full_shape_F = numpy_funcs.full_order__create_full_shape_F

    size_1 = randint(1, 10)
    size_2 = randint(1, 10)

    f_shape_C = epyc_numpy_funcs_mod.full_order__create_full_shape_C
    assert f_shape_C(size_1, size_2) == create_full_shape_C(size_1, size_2)

    f_shape_F = epyc_numpy_funcs_mod.full_order__create_full_shape_F
    assert f_shape_F(size_1, size_2) == create_full_shape_F(size_1, size_2)


def test_full_dtype(epyc_numpy_funcs_mod):
    create_full_val_int_int = numpy_funcs.full_dtype__create_full_val_int_int
    create_full_val_int_float = numpy_funcs.full_dtype__create_full_val_int_float
    create_full_val_int_complex = numpy_funcs.full_dtype__create_full_val_int_complex
    create_full_val_real_int32 = numpy_funcs.full_dtype__create_full_val_real_int32
    create_full_val_real_float32 = numpy_funcs.full_dtype__create_full_val_real_float32
    create_full_val_real_float64 = numpy_funcs.full_dtype__create_full_val_real_float64
    create_full_val_real_complex64 = (
        numpy_funcs.full_dtype__create_full_val_real_complex64
    )
    create_full_val_real_complex128 = (
        numpy_funcs.full_dtype__create_full_val_real_complex128
    )

    val_int = randint(100)
    val_float = rand() * 100

    f_int_int = epyc_numpy_funcs_mod.full_dtype__create_full_val_int_int
    assert f_int_int(val_int) == create_full_val_int_int(val_int)
    assert matching_types(f_int_int(val_int), create_full_val_int_int(val_int))

    f_int_float = epyc_numpy_funcs_mod.full_dtype__create_full_val_int_float
    assert isclose(
        f_int_float(val_int), create_full_val_int_float(val_int), rtol=RTOL, atol=ATOL
    )
    assert matching_types(f_int_float(val_int), create_full_val_int_float(val_int))

    f_int_complex = epyc_numpy_funcs_mod.full_dtype__create_full_val_int_complex
    assert isclose(
        f_int_complex(val_int),
        create_full_val_int_complex(val_int),
        rtol=RTOL,
        atol=ATOL,
    )
    assert matching_types(f_int_complex(val_int), create_full_val_int_complex(val_int))

    f_real_int32 = epyc_numpy_funcs_mod.full_dtype__create_full_val_real_int32
    assert f_real_int32(val_float) == create_full_val_real_int32(val_float)
    assert matching_types(
        f_real_int32(val_float), create_full_val_real_int32(val_float)
    )

    f_real_float32 = epyc_numpy_funcs_mod.full_dtype__create_full_val_real_float32
    assert isclose(
        f_real_float32(val_float),
        create_full_val_real_float32(val_float),
        rtol=RTOL,
        atol=ATOL,
    )
    assert matching_types(
        f_real_float32(val_float), create_full_val_real_float32(val_float)
    )

    f_real_float64 = epyc_numpy_funcs_mod.full_dtype__create_full_val_real_float64
    assert isclose(
        f_real_float64(val_float),
        create_full_val_real_float64(val_float),
        rtol=RTOL,
        atol=ATOL,
    )
    assert matching_types(
        f_real_float64(val_float), create_full_val_real_float64(val_float)
    )

    f_real_complex64 = epyc_numpy_funcs_mod.full_dtype__create_full_val_real_complex64
    assert isclose(
        f_real_complex64(val_float),
        create_full_val_real_complex64(val_float),
        rtol=RTOL,
        atol=ATOL,
    )
    assert matching_types(
        f_real_complex64(val_float), create_full_val_real_complex64(val_float)
    )

    f_real_complex128 = epyc_numpy_funcs_mod.full_dtype__create_full_val_real_complex128
    assert isclose(
        f_real_complex128(val_float),
        create_full_val_real_complex128(val_float),
        rtol=RTOL,
        atol=ATOL,
    )
    assert matching_types(
        f_real_complex128(val_float), create_full_val_real_complex128(val_float)
    )


def test_full_dtype_auto(epyc_numpy_funcs_mod):
    create_full_val_auto = numpy_funcs.full_dtype_auto

    integer32 = randint(low=min_int32, high=max_int32, dtype=np.int32)
    integer = int(integer32)

    fl = float(integer)
    fl32 = np.float32(fl)
    fl64 = np.float64(fl)

    cmplx = complex(integer)
    cmplx64 = np.complex64(fl32)
    cmplx128 = np.complex128(fl64)

    f_int = epyc_numpy_funcs_mod.full_dtype_auto
    assert f_int(integer) == create_full_val_auto(integer)
    assert matching_types(f_int(integer), create_full_val_auto(integer))

    f_float = epyc_numpy_funcs_mod.full_dtype_auto
    assert isclose(f_float(fl), create_full_val_auto(fl), rtol=RTOL, atol=ATOL)
    assert matching_types(f_float(fl), create_full_val_auto(fl))

    f_complex = epyc_numpy_funcs_mod.full_dtype_auto
    assert isclose(f_complex(cmplx), create_full_val_auto(cmplx), rtol=RTOL, atol=ATOL)
    assert matching_types(f_complex(cmplx), create_full_val_auto(cmplx))

    f_int32 = epyc_numpy_funcs_mod.full_dtype_auto
    assert f_int32(integer32) == create_full_val_auto(integer32)
    assert matching_types(f_int32(integer32), create_full_val_auto(integer32))

    f_float32 = epyc_numpy_funcs_mod.full_dtype_auto
    assert isclose(f_float32(fl32), create_full_val_auto(fl32), rtol=RTOL, atol=ATOL)
    assert matching_types(f_float32(fl32), create_full_val_auto(fl32))

    f_float64 = epyc_numpy_funcs_mod.full_dtype_auto
    assert isclose(f_float64(fl64), create_full_val_auto(fl64), rtol=RTOL, atol=ATOL)
    assert matching_types(f_float64(fl64), create_full_val_auto(fl64))

    f_complex64 = epyc_numpy_funcs_mod.full_dtype_auto
    assert isclose(
        f_complex64(cmplx64), create_full_val_auto(cmplx64), rtol=RTOL, atol=ATOL
    )
    assert matching_types(f_complex64(cmplx64), create_full_val_auto(cmplx64))

    f_complex128 = epyc_numpy_funcs_mod.full_dtype_auto
    assert isclose(
        f_complex128(cmplx128), create_full_val_auto(cmplx128), rtol=RTOL, atol=ATOL
    )
    assert matching_types(f_complex128(cmplx128), create_full_val_auto(cmplx128))


def test_full_combined_args(epyc_numpy_funcs_mod):
    create_full_1_shape = numpy_funcs.full_combined_args__create_full_1_shape
    create_full_1_val = numpy_funcs.full_combined_args__create_full_1_val
    create_full_2_shape = numpy_funcs.full_combined_args__create_full_2_shape
    create_full_2_val = numpy_funcs.full_combined_args__create_full_2_val
    create_full_3_shape = numpy_funcs.full_combined_args__create_full_3_shape
    create_full_3_val = numpy_funcs.full_combined_args__create_full_3_val

    f1_shape = epyc_numpy_funcs_mod.full_combined_args__create_full_1_shape
    f1_val = epyc_numpy_funcs_mod.full_combined_args__create_full_1_val
    assert f1_shape() == create_full_1_shape()
    assert f1_val() == create_full_1_val()
    assert matching_types(f1_val(), create_full_1_val())

    f2_shape = epyc_numpy_funcs_mod.full_combined_args__create_full_2_shape
    f2_val = epyc_numpy_funcs_mod.full_combined_args__create_full_2_val
    assert f2_shape() == create_full_2_shape()
    assert isclose(f2_val(), create_full_2_val(), rtol=RTOL, atol=ATOL)
    assert matching_types(f2_val(), create_full_2_val())

    f3_shape = epyc_numpy_funcs_mod.full_combined_args__create_full_3_shape
    f3_val = epyc_numpy_funcs_mod.full_combined_args__create_full_3_val
    assert f3_shape() == create_full_3_shape()
    assert isclose(f3_val(), create_full_3_val(), rtol=RTOL, atol=ATOL)
    assert matching_types(f3_val(), create_full_3_val())


def test_empty_basic(epyc_numpy_funcs_mod):
    create_empty_shape_1d = numpy_funcs.empty_basic__create_empty_shape_1d
    create_empty_shape_2d = numpy_funcs.empty_basic__create_empty_shape_2d

    size = randint(1, 10)

    f_shape_1d = epyc_numpy_funcs_mod.empty_basic__create_empty_shape_1d
    assert f_shape_1d(size) == create_empty_shape_1d(size)

    f_shape_2d = epyc_numpy_funcs_mod.empty_basic__create_empty_shape_2d
    assert f_shape_2d(size) == create_empty_shape_2d(size)


def test_empty_order(epyc_numpy_funcs_mod):
    create_empty_shape_C = numpy_funcs.empty_order__create_empty_shape_C
    create_empty_shape_F = numpy_funcs.empty_order__create_empty_shape_F

    size_1 = randint(1, 10)
    size_2 = randint(1, 10)

    f_shape_C = epyc_numpy_funcs_mod.empty_order__create_empty_shape_C
    assert f_shape_C(size_1, size_2) == create_empty_shape_C(size_1, size_2)

    f_shape_F = epyc_numpy_funcs_mod.empty_order__create_empty_shape_F
    assert f_shape_F(size_1, size_2) == create_empty_shape_F(size_1, size_2)


def test_empty_dtype(epyc_numpy_funcs_mod):
    create_empty_val_int = numpy_funcs.empty_dtype__create_empty_val_int
    create_empty_val_float = numpy_funcs.empty_dtype__create_empty_val_float
    create_empty_val_complex = numpy_funcs.empty_dtype__create_empty_val_complex
    create_empty_val_int32 = numpy_funcs.empty_dtype__create_empty_val_int32
    create_empty_val_float32 = numpy_funcs.empty_dtype__create_empty_val_float32
    create_empty_val_float64 = numpy_funcs.empty_dtype__create_empty_val_float64
    create_empty_val_complex64 = numpy_funcs.empty_dtype__create_empty_val_complex64
    create_empty_val_complex128 = numpy_funcs.empty_dtype__create_empty_val_complex128

    f_int_int = epyc_numpy_funcs_mod.empty_dtype__create_empty_val_int
    assert matching_types(f_int_int(), create_empty_val_int())

    f_int_float = epyc_numpy_funcs_mod.empty_dtype__create_empty_val_float
    assert matching_types(f_int_float(), create_empty_val_float())

    f_int_complex = epyc_numpy_funcs_mod.empty_dtype__create_empty_val_complex
    assert matching_types(f_int_complex(), create_empty_val_complex())

    f_real_int32 = epyc_numpy_funcs_mod.empty_dtype__create_empty_val_int32
    assert matching_types(f_real_int32(), create_empty_val_int32())

    f_real_float32 = epyc_numpy_funcs_mod.empty_dtype__create_empty_val_float32
    assert matching_types(f_real_float32(), create_empty_val_float32())

    f_real_float64 = epyc_numpy_funcs_mod.empty_dtype__create_empty_val_float64
    assert matching_types(f_real_float64(), create_empty_val_float64())

    f_real_complex64 = epyc_numpy_funcs_mod.empty_dtype__create_empty_val_complex64
    assert matching_types(f_real_complex64(), create_empty_val_complex64())

    f_real_complex128 = epyc_numpy_funcs_mod.empty_dtype__create_empty_val_complex128
    assert matching_types(f_real_complex128(), create_empty_val_complex128())


def test_empty_combined_args(epyc_numpy_funcs_mod):
    create_empty_1_shape = numpy_funcs.empty_combined_args__create_empty_1_shape
    create_empty_1_val = numpy_funcs.empty_combined_args__create_empty_1_val
    create_empty_2_shape = numpy_funcs.empty_combined_args__create_empty_2_shape
    create_empty_2_val = numpy_funcs.empty_combined_args__create_empty_2_val
    create_empty_3_shape = numpy_funcs.empty_combined_args__create_empty_3_shape
    create_empty_3_val = numpy_funcs.empty_combined_args__create_empty_3_val

    f1_shape = epyc_numpy_funcs_mod.empty_combined_args__create_empty_1_shape
    f1_val = epyc_numpy_funcs_mod.empty_combined_args__create_empty_1_val
    assert f1_shape() == create_empty_1_shape()
    assert matching_types(f1_val(), create_empty_1_val())

    f2_shape = epyc_numpy_funcs_mod.empty_combined_args__create_empty_2_shape
    f2_val = epyc_numpy_funcs_mod.empty_combined_args__create_empty_2_val
    assert f2_shape() == create_empty_2_shape()
    assert matching_types(f2_val(), create_empty_2_val())

    f3_shape = epyc_numpy_funcs_mod.empty_combined_args__create_empty_3_shape
    f3_val = epyc_numpy_funcs_mod.empty_combined_args__create_empty_3_val
    assert f3_shape() == create_empty_3_shape()
    assert matching_types(f3_val(), create_empty_3_val())


def test_ones_basic(epyc_numpy_funcs_mod):
    create_ones_shape_1d = numpy_funcs.ones_basic__create_ones_shape_1d
    create_ones_shape_2d = numpy_funcs.ones_basic__create_ones_shape_2d

    size = randint(1, 10)

    f_shape_1d = epyc_numpy_funcs_mod.ones_basic__create_ones_shape_1d
    assert f_shape_1d(size) == create_ones_shape_1d(size)

    f_shape_2d = epyc_numpy_funcs_mod.ones_basic__create_ones_shape_2d
    assert f_shape_2d(size) == create_ones_shape_2d(size)


def test_ones_order(epyc_numpy_funcs_mod):
    create_ones_shape_C = numpy_funcs.ones_order__create_ones_shape_C
    create_ones_shape_F = numpy_funcs.ones_order__create_ones_shape_F

    size_1 = randint(1, 10)
    size_2 = randint(1, 10)

    f_shape_C = epyc_numpy_funcs_mod.ones_order__create_ones_shape_C
    assert f_shape_C(size_1, size_2) == create_ones_shape_C(size_1, size_2)

    f_shape_F = epyc_numpy_funcs_mod.ones_order__create_ones_shape_F
    assert f_shape_F(size_1, size_2) == create_ones_shape_F(size_1, size_2)


def test_ones_dtype(epyc_numpy_funcs_mod):
    create_ones_val_int = numpy_funcs.ones_dtype__create_ones_val_int
    create_ones_val_float = numpy_funcs.ones_dtype__create_ones_val_float
    create_ones_val_complex = numpy_funcs.ones_dtype__create_ones_val_complex
    create_ones_val_int32 = numpy_funcs.ones_dtype__create_ones_val_int32
    create_ones_val_float32 = numpy_funcs.ones_dtype__create_ones_val_float32
    create_ones_val_float64 = numpy_funcs.ones_dtype__create_ones_val_float64
    create_ones_val_complex64 = numpy_funcs.ones_dtype__create_ones_val_complex64
    create_ones_val_complex128 = numpy_funcs.ones_dtype__create_ones_val_complex128

    f_int_int = epyc_numpy_funcs_mod.ones_dtype__create_ones_val_int
    assert f_int_int() == create_ones_val_int()
    assert matching_types(f_int_int(), create_ones_val_int())

    f_int_float = epyc_numpy_funcs_mod.ones_dtype__create_ones_val_float
    assert isclose(f_int_float(), create_ones_val_float(), rtol=RTOL, atol=ATOL)
    assert matching_types(f_int_float(), create_ones_val_float())

    f_int_complex = epyc_numpy_funcs_mod.ones_dtype__create_ones_val_complex
    assert isclose(f_int_complex(), create_ones_val_complex(), rtol=RTOL, atol=ATOL)
    assert matching_types(f_int_complex(), create_ones_val_complex())

    f_real_int32 = epyc_numpy_funcs_mod.ones_dtype__create_ones_val_int32
    assert f_real_int32() == create_ones_val_int32()
    assert matching_types(f_real_int32(), create_ones_val_int32())

    f_real_float32 = epyc_numpy_funcs_mod.ones_dtype__create_ones_val_float32
    assert isclose(f_real_float32(), create_ones_val_float32(), rtol=RTOL, atol=ATOL)
    assert matching_types(f_real_float32(), create_ones_val_float32())

    f_real_float64 = epyc_numpy_funcs_mod.ones_dtype__create_ones_val_float64
    assert isclose(f_real_float64(), create_ones_val_float64(), rtol=RTOL, atol=ATOL)
    assert matching_types(f_real_float64(), create_ones_val_float64())

    f_real_complex64 = epyc_numpy_funcs_mod.ones_dtype__create_ones_val_complex64
    assert isclose(
        f_real_complex64(), create_ones_val_complex64(), rtol=RTOL, atol=ATOL
    )
    assert matching_types(f_real_complex64(), create_ones_val_complex64())

    f_real_complex128 = epyc_numpy_funcs_mod.ones_dtype__create_ones_val_complex128
    assert isclose(
        f_real_complex128(), create_ones_val_complex128(), rtol=RTOL, atol=ATOL
    )
    assert matching_types(f_real_complex128(), create_ones_val_complex128())


def test_ones_combined_args(epyc_numpy_funcs_mod):
    create_ones_1_shape = numpy_funcs.ones_combined_args__create_ones_1_shape
    create_ones_1_val = numpy_funcs.ones_combined_args__create_ones_1_val
    create_ones_2_shape = numpy_funcs.ones_combined_args__create_ones_2_shape
    create_ones_2_val = numpy_funcs.ones_combined_args__create_ones_2_val
    create_ones_3_shape = numpy_funcs.ones_combined_args__create_ones_3_shape
    create_ones_3_val = numpy_funcs.ones_combined_args__create_ones_3_val

    f1_shape = epyc_numpy_funcs_mod.ones_combined_args__create_ones_1_shape
    f1_val = epyc_numpy_funcs_mod.ones_combined_args__create_ones_1_val
    assert f1_shape() == create_ones_1_shape()
    assert f1_val() == create_ones_1_val()
    assert matching_types(f1_val(), create_ones_1_val())

    f2_shape = epyc_numpy_funcs_mod.ones_combined_args__create_ones_2_shape
    f2_val = epyc_numpy_funcs_mod.ones_combined_args__create_ones_2_val
    assert f2_shape() == create_ones_2_shape()
    assert isclose(f2_val(), create_ones_2_val(), rtol=RTOL, atol=ATOL)
    assert matching_types(f2_val(), create_ones_2_val())

    f3_shape = epyc_numpy_funcs_mod.ones_combined_args__create_ones_3_shape
    f3_val = epyc_numpy_funcs_mod.ones_combined_args__create_ones_3_val
    assert f3_shape() == create_ones_3_shape()
    assert isclose(f3_val(), create_ones_3_val(), rtol=RTOL, atol=ATOL)
    assert matching_types(f3_val(), create_ones_3_val())


def test_ones_in_expression(epyc_numpy_funcs_mod):
    ones_plus_scalar = numpy_funcs.ones_in_expression__ones_plus_scalar
    ones_times_scalar = numpy_funcs.ones_in_expression__ones_times_scalar

    f1 = epyc_numpy_funcs_mod.ones_in_expression__ones_plus_scalar
    assert f1() == ones_plus_scalar()

    f2 = epyc_numpy_funcs_mod.ones_in_expression__ones_times_scalar
    assert isclose(f2(), ones_times_scalar(), rtol=RTOL, atol=ATOL)


def test_zeros_basic(epyc_numpy_funcs_mod):
    create_zeros_shape_1d = numpy_funcs.zeros_basic__create_zeros_shape_1d
    create_zeros_shape_2d = numpy_funcs.zeros_basic__create_zeros_shape_2d

    size = randint(1, 10)

    f_shape_1d = epyc_numpy_funcs_mod.zeros_basic__create_zeros_shape_1d
    assert f_shape_1d(size) == create_zeros_shape_1d(size)

    f_shape_2d = epyc_numpy_funcs_mod.zeros_basic__create_zeros_shape_2d
    assert f_shape_2d(size) == create_zeros_shape_2d(size)


def test_zeros_order(epyc_numpy_funcs_mod):
    create_zeros_shape_C = numpy_funcs.zeros_order__create_zeros_shape_C
    create_zeros_shape_F = numpy_funcs.zeros_order__create_zeros_shape_F

    size_1 = randint(1, 10)
    size_2 = randint(1, 10)

    f_shape_C = epyc_numpy_funcs_mod.zeros_order__create_zeros_shape_C
    assert f_shape_C(size_1, size_2) == create_zeros_shape_C(size_1, size_2)

    f_shape_F = epyc_numpy_funcs_mod.zeros_order__create_zeros_shape_F
    assert f_shape_F(size_1, size_2) == create_zeros_shape_F(size_1, size_2)


def test_zeros_dtype(epyc_numpy_funcs_mod):
    create_zeros_val_int = numpy_funcs.zeros_dtype__create_zeros_val_int
    create_zeros_val_float = numpy_funcs.zeros_dtype__create_zeros_val_float
    create_zeros_val_complex = numpy_funcs.zeros_dtype__create_zeros_val_complex
    create_zeros_val_int32 = numpy_funcs.zeros_dtype__create_zeros_val_int32
    create_zeros_val_float32 = numpy_funcs.zeros_dtype__create_zeros_val_float32
    create_zeros_val_float64 = numpy_funcs.zeros_dtype__create_zeros_val_float64
    create_zeros_val_complex64 = numpy_funcs.zeros_dtype__create_zeros_val_complex64
    create_zeros_val_complex128 = numpy_funcs.zeros_dtype__create_zeros_val_complex128

    f_int_int = epyc_numpy_funcs_mod.zeros_dtype__create_zeros_val_int
    assert f_int_int() == create_zeros_val_int()
    assert matching_types(f_int_int(), create_zeros_val_int())

    f_int_float = epyc_numpy_funcs_mod.zeros_dtype__create_zeros_val_float
    assert isclose(f_int_float(), create_zeros_val_float(), rtol=RTOL, atol=ATOL)
    assert matching_types(f_int_float(), create_zeros_val_float())

    f_int_complex = epyc_numpy_funcs_mod.zeros_dtype__create_zeros_val_complex
    assert isclose(f_int_complex(), create_zeros_val_complex(), rtol=RTOL, atol=ATOL)
    assert matching_types(f_int_complex(), create_zeros_val_complex())

    f_real_int32 = epyc_numpy_funcs_mod.zeros_dtype__create_zeros_val_int32
    assert f_real_int32() == create_zeros_val_int32()
    assert matching_types(f_real_int32(), create_zeros_val_int32())

    f_real_float32 = epyc_numpy_funcs_mod.zeros_dtype__create_zeros_val_float32
    assert isclose(f_real_float32(), create_zeros_val_float32(), rtol=RTOL, atol=ATOL)
    assert matching_types(f_real_float32(), create_zeros_val_float32())

    f_real_float64 = epyc_numpy_funcs_mod.zeros_dtype__create_zeros_val_float64
    assert isclose(f_real_float64(), create_zeros_val_float64(), rtol=RTOL, atol=ATOL)
    assert matching_types(f_real_float64(), create_zeros_val_float64())

    f_real_complex64 = epyc_numpy_funcs_mod.zeros_dtype__create_zeros_val_complex64
    assert isclose(
        f_real_complex64(), create_zeros_val_complex64(), rtol=RTOL, atol=ATOL
    )
    assert matching_types(f_real_complex64(), create_zeros_val_complex64())

    f_real_complex128 = epyc_numpy_funcs_mod.zeros_dtype__create_zeros_val_complex128
    assert isclose(
        f_real_complex128(), create_zeros_val_complex128(), rtol=RTOL, atol=ATOL
    )
    assert matching_types(f_real_complex128(), create_zeros_val_complex128())


def test_zeros_combined_args(epyc_numpy_funcs_mod):
    create_zeros_1_shape = numpy_funcs.zeros_combined_args__create_zeros_1_shape
    create_zeros_1_val = numpy_funcs.zeros_combined_args__create_zeros_1_val
    create_zeros_2_shape = numpy_funcs.zeros_combined_args__create_zeros_2_shape
    create_zeros_2_val = numpy_funcs.zeros_combined_args__create_zeros_2_val
    create_zeros_3_shape = numpy_funcs.zeros_combined_args__create_zeros_3_shape
    create_zeros_3_val = numpy_funcs.zeros_combined_args__create_zeros_3_val

    f1_shape = epyc_numpy_funcs_mod.zeros_combined_args__create_zeros_1_shape
    f1_val = epyc_numpy_funcs_mod.zeros_combined_args__create_zeros_1_val
    assert f1_shape() == create_zeros_1_shape()
    assert f1_val() == create_zeros_1_val()
    assert matching_types(f1_val(), create_zeros_1_val())

    f2_shape = epyc_numpy_funcs_mod.zeros_combined_args__create_zeros_2_shape
    f2_val = epyc_numpy_funcs_mod.zeros_combined_args__create_zeros_2_val
    assert f2_shape() == create_zeros_2_shape()
    assert isclose(f2_val(), create_zeros_2_val(), rtol=RTOL, atol=ATOL)
    assert matching_types(f2_val(), create_zeros_2_val())

    f3_shape = epyc_numpy_funcs_mod.zeros_combined_args__create_zeros_3_shape
    f3_val = epyc_numpy_funcs_mod.zeros_combined_args__create_zeros_3_val
    assert f3_shape() == create_zeros_3_shape()
    assert isclose(f3_val(), create_zeros_3_val(), rtol=RTOL, atol=ATOL)
    assert matching_types(f3_val(), create_zeros_3_val())


def test_zeros_in_expression(epyc_numpy_funcs_mod):
    zeros_plus_scalar = numpy_funcs.zeros_in_expression__zeros_plus_scalar
    zeros_times_scalar = numpy_funcs.zeros_in_expression__zeros_times_scalar
    zeros_2d_plus_scalar = numpy_funcs.zeros_in_expression__zeros_2d_plus_scalar
    zeros_plus_ones = numpy_funcs.zeros_in_expression__zeros_plus_ones

    f1 = epyc_numpy_funcs_mod.zeros_in_expression__zeros_plus_scalar
    assert f1() == zeros_plus_scalar()

    f2 = epyc_numpy_funcs_mod.zeros_in_expression__zeros_times_scalar
    assert isclose(f2(), zeros_times_scalar(), rtol=RTOL, atol=ATOL)

    f3 = epyc_numpy_funcs_mod.zeros_in_expression__zeros_2d_plus_scalar
    assert f3() == zeros_2d_plus_scalar()

    f4 = epyc_numpy_funcs_mod.zeros_in_expression__zeros_plus_ones
    assert isclose(f4(), zeros_plus_ones(), rtol=RTOL, atol=ATOL)


def test_array(epyc_numpy_funcs_mod):
    create_array_list_shape = numpy_funcs.array__create_array_list_shape
    create_array_list_val = numpy_funcs.array__create_array_list_val
    create_array_tuple_shape = numpy_funcs.array__create_array_tuple_shape
    create_array_tuple_val = numpy_funcs.array__create_array_tuple_val
    create_array_tuple_ref = numpy_funcs.array__create_array_tuple_ref

    f1_shape = epyc_numpy_funcs_mod.array__create_array_list_shape
    f1_val = epyc_numpy_funcs_mod.array__create_array_list_val
    assert f1_shape() == create_array_list_shape()
    assert f1_val() == create_array_list_val()
    assert matching_types(f1_val(), create_array_list_val())
    f2_shape = epyc_numpy_funcs_mod.array__create_array_tuple_shape
    f2_val = epyc_numpy_funcs_mod.array__create_array_tuple_val
    assert f2_shape() == create_array_tuple_shape()
    assert f2_val() == create_array_tuple_val()
    assert matching_types(f2_val(), create_array_tuple_val())
    array_tuple_ref = epyc_numpy_funcs_mod.array__create_array_tuple_ref
    tmp_arr = np.ones((3, 4), dtype=int)
    assert np.allclose(array_tuple_ref(tmp_arr), create_array_tuple_ref(tmp_arr))


def test_array_in_expression(epyc_numpy_funcs_mod):
    create_array_list_val = numpy_funcs.array_in_expression

    f1_val = epyc_numpy_funcs_mod.array_in_expression
    assert np.array_equal(f1_val(), create_array_list_val())


def test_array_new_dtype(epyc_numpy_funcs_mod):
    create_float_array_tuple_ref = numpy_funcs.array_new_dtype

    def create_bool_array_tuple_ref(a: "int[:,:]"):
        from numpy import array

        b = (a[0, :], a[1, :])
        c = array(b, dtype=bool)
        return c

    array_float_tuple_ref = epyc_numpy_funcs_mod.array_new_dtype
    tmp_arr = np.ones((3, 4), dtype=int)
    assert np.allclose(
        array_float_tuple_ref(tmp_arr), create_float_array_tuple_ref(tmp_arr)
    )

    array_bool_tuple_ref = epyc_numpy_funcs_mod.array_new_dtype
    tmp_arr = np.ones((3, 4), dtype=int)
    assert np.allclose(
        array_bool_tuple_ref(tmp_arr), create_bool_array_tuple_ref(tmp_arr)
    )


@pytest.mark.parametrize(
    "language",
    (
        pytest.param("fortran", marks=pytest.mark.fortran),
        pytest.param(
            "c", marks=[pytest.mark.skip(reason="rand not implemented"), pytest.mark.c]
        ),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_rand_basic(language):
    def create_val():
        return rand()

    f1 = epyccel(create_val, language=language)
    y = [f1() for i in range(10)]
    assert all(yi < 1 for yi in y)
    assert all(yi >= 0 for yi in y)
    assert all(isinstance(yi, float) for yi in y)
    assert len(set(y)) > 1


@pytest.mark.parametrize(
    "language",
    (
        pytest.param("fortran", marks=pytest.mark.fortran),
        pytest.param(
            "c", marks=[pytest.mark.skip(reason="rand not implemented"), pytest.mark.c]
        ),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_rand_args(language):
    def create_array_size_1d(n: "int"):
        from numpy import shape

        a = rand(n)
        return shape(a)[0]

    def create_array_size_2d(n: "int", m: "int"):
        from numpy import shape

        a = rand(n, m)
        return shape(a)[0], shape(a)[1]

    def create_array_size_3d(n: "int", m: "int", p: "int"):
        from numpy import shape

        a = rand(n, m, p)
        return shape(a)[0], shape(a)[1], shape(a)[2]

    def create_array_vals_1d():
        a = rand(4)
        return a[0], a[1], a[2], a[3]

    def create_array_vals_2d():
        a = rand(2, 2)
        return a[0, 0], a[0, 1], a[1, 0], a[1, 1]

    n = randint(1, 10)
    m = randint(1, 10)
    p = randint(1, 5)
    f_1d = epyccel(create_array_size_1d, language=language)
    assert f_1d(n) == create_array_size_1d(n)

    f_2d = epyccel(create_array_size_2d, language=language)
    assert f_2d(n, m) == create_array_size_2d(n, m)

    f_3d = epyccel(create_array_size_3d, language=language)
    assert f_3d(n, m, p) == create_array_size_3d(n, m, p)

    g_1d = epyccel(create_array_vals_1d, language=language)
    y = g_1d()
    assert all(yi < 1 for yi in y)
    assert all(yi >= 0 for yi in y)
    assert all(isinstance(yi, float) for yi in y)
    assert len(set(y)) > 1

    g_2d = epyccel(create_array_vals_2d, language=language)
    y = g_2d()
    assert all(yi < 1 for yi in y)
    assert all(yi >= 0 for yi in y)
    assert all(isinstance(yi, float) for yi in y)
    assert len(set(y)) > 1


@pytest.mark.parametrize(
    "language",
    (
        pytest.param("fortran", marks=pytest.mark.fortran),
        pytest.param(
            "c", marks=[pytest.mark.skip(reason="rand not implemented"), pytest.mark.c]
        ),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_rand_expr(language):
    def create_val():
        x = 2 * rand()
        return x

    f1 = epyccel(create_val, language=language)
    y = [f1() for i in range(10)]
    assert all(yi < 2 for yi in y)
    assert all(yi >= 0 for yi in y)
    assert all(isinstance(yi, float) for yi in y)
    assert len(set(y)) > 1


msg = "a is not allocated. See #2566"


@pytest.mark.parametrize(
    "language",
    (
        pytest.param("fortran", marks=[pytest.mark.xfail(reason=msg), pytest.mark.c]),
        pytest.param("c", marks=[pytest.mark.xfail(reason=msg), pytest.mark.c]),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_rand_expr_array(language):
    def create_array_vals_2d():
        a = rand(2, 2) * 0.5 + 3
        return a[0, 0], a[0, 1], a[1, 0], a[1, 1]

    f2 = epyccel(create_array_vals_2d, language=language)
    y = f2()
    assert all(yi < 3.5 for yi in y)
    assert all(yi >= 3 for yi in y)
    assert all(isinstance(yi, float) for yi in y)
    assert len(set(y)) > 1


@pytest.mark.parametrize(
    "language",
    (
        pytest.param("fortran", marks=pytest.mark.fortran),
        pytest.param(
            "c",
            marks=[pytest.mark.skip(reason="randint not implemented"), pytest.mark.c],
        ),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_randint_basic(language):
    def create_rand():
        return np.random.randint(-10, 10)

    def create_val(high: "int"):
        return np.random.randint(high)

    def create_val_low(low: "int", high: "int"):
        return np.random.randint(low, high)

    f0 = epyccel(create_rand, language=language)
    y = [f0() for i in range(10)]
    assert all(yi < 10 for yi in y)
    assert all(yi >= -10 for yi in y)
    assert all(isinstance(yi, int) for yi in y)
    assert len(set(y)) > 1

    f1 = epyccel(create_val, language=language)
    y = [f1(100) for i in range(10)]
    assert all(yi < 100 for yi in y)
    assert all(yi >= 0 for yi in y)
    assert all(isinstance(yi, int) for yi in y)
    assert len(set(y)) > 1

    f2 = epyccel(create_val_low, language=language)
    y = [f2(25, 100) for i in range(10)]
    assert all(yi < 100 for yi in y)
    assert all(yi >= 25 for yi in y)
    assert all(isinstance(yi, int) for yi in y)
    assert len(set(y)) > 1


@pytest.mark.parametrize(
    "language",
    (
        pytest.param("fortran", marks=pytest.mark.fortran),
        pytest.param(
            "c",
            marks=[pytest.mark.skip(reason="randint not implemented"), pytest.mark.c],
        ),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_randint_expr(language):
    def create_val(high: "int"):
        x = 2 * np.random.randint(high)
        return x

    def create_val_low(low: "int", high: "int"):
        x = 2 * np.random.randint(low, high)
        return x

    f1 = epyccel(create_val, language=language)
    y = [f1(27) for i in range(10)]
    assert all(yi < 54 for yi in y)
    assert all(yi >= 0 for yi in y)
    assert all(isinstance(yi, int) for yi in y)
    assert len(set(y)) > 1

    f2 = epyccel(create_val_low, language=language)
    y = [f2(21, 46) for i in range(10)]
    assert all(yi < 92 for yi in y)
    assert all(yi >= 42 for yi in y)
    assert all(isinstance(yi, int) for yi in y)
    assert len(set(y)) > 1


@pytest.mark.parametrize(
    "language",
    (
        pytest.param("fortran", marks=pytest.mark.fortran),
        pytest.param(
            "c",
            marks=[pytest.mark.skip(reason="randint not implemented"), pytest.mark.c],
        ),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_randint_size(language):
    def create_arr_high(high: "int"):
        return np.random.randint(high, size=10)

    def create_arr_low_high(low: "int", high: "int"):
        return np.random.randint(low, high, size=10)

    f1 = epyccel(create_arr_high, language=language)
    y = f1(50)
    assert y.shape == (10,)
    assert all(yi < 50 for yi in y)
    assert all(yi >= 0 for yi in y)
    assert np.issubdtype(y.dtype, np.integer)
    assert len(set(y)) > 1

    f2 = epyccel(create_arr_low_high, language=language)
    y = f2(10, 30)
    assert y.shape == (10,)
    assert all(yi < 30 for yi in y)
    assert all(yi >= 10 for yi in y)
    assert np.issubdtype(y.dtype, np.integer)
    assert len(set(y)) > 1


@pytest.mark.parametrize(
    "language",
    (
        pytest.param("fortran", marks=pytest.mark.fortran),
        pytest.param(
            "c",
            marks=[pytest.mark.skip(reason="randint not implemented"), pytest.mark.c],
        ),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_randint_dtype(language):
    def create_arr_i32_high(high: "int"):
        return np.random.randint(high, size=10, dtype=np.int32)

    def create_arr_i32_low_high(low: "int", high: "int"):
        return np.random.randint(low, high, size=10, dtype=np.int32)

    def create_scalar_i32(high: "int"):
        return np.random.randint(high, dtype=np.int32)

    f1 = epyccel(create_arr_i32_high, language=language)
    y = f1(50)
    assert y.shape == (10,)
    assert all(yi < 50 for yi in y)
    assert all(yi >= 0 for yi in y)
    assert y.dtype == np.int32
    assert len(set(y)) > 1

    f2 = epyccel(create_arr_i32_low_high, language=language)
    y = f2(10, 30)
    assert y.shape == (10,)
    assert all(yi < 30 for yi in y)
    assert all(yi >= 10 for yi in y)
    assert y.dtype == np.int32
    assert len(set(y)) > 1

    f3 = epyccel(create_scalar_i32, language=language)
    y = [f3(100) for i in range(10)]
    assert all(yi < 100 for yi in y)
    assert all(yi >= 0 for yi in y)
    assert all(isinstance(yi, np.int32) for yi in y)
    assert len(set(y)) > 1


def test_sum_bool(epyc_numpy_funcs_mod):
    sum_call = numpy_funcs.sum_bool

    f1 = epyc_numpy_funcs_mod.sum_bool
    x = randint(1, size=10, dtype=bool)
    assert f1(x) == sum_call(x)


def test_sum_int(epyc_numpy_funcs_mod):
    sum_call = numpy_funcs.sum_int

    f1 = epyc_numpy_funcs_mod.sum_int
    x = randint(99, size=10, dtype=np.int64)
    assert f1(x) == sum_call(x)


def test_sum_override_builtin(epyc_numpy_funcs_mod):
    sum_call = numpy_funcs.sum_override_builtin

    f1 = epyc_numpy_funcs_mod.sum_override_builtin
    x = randint(99, size=10, dtype=np.int64)
    assert f1(x) == sum_call(x)


def test_sum_real(epyc_numpy_funcs_mod):
    sum_call = numpy_funcs.sum_real

    f1 = epyc_numpy_funcs_mod.sum_real
    x = rand(10)
    assert isclose(f1(x), sum_call(x), rtol=RTOL, atol=ATOL)


def test_sum_type(epyc_numpy_funcs_mod):
    sum_call = numpy_funcs.sum_type

    f1 = epyc_numpy_funcs_mod.sum_type
    x = rand(10).astype(np.float32)
    assert isclose(f1(x), sum_call(x), rtol=RTOL32, atol=ATOL32)
    assert matching_types(f1(x), sum_call(x))


def test_sum_phrase(epyc_numpy_funcs_mod):
    sum_phrase = numpy_funcs.sum_phrase

    f2 = epyc_numpy_funcs_mod.sum_phrase
    x = rand(10)
    y = rand(15)
    assert isclose(f2(x, y), sum_phrase(x, y), rtol=RTOL, atol=ATOL)


def test_sum_property(epyc_numpy_funcs_mod):
    sum_call = numpy_funcs.sum_property

    f1 = epyc_numpy_funcs_mod.sum_property
    x = randint(99, size=10, dtype=np.int64)
    assert f1(x) == sum_call(x)


def test_sum_3d(epyc_numpy_funcs_mod):
    sum_call = numpy_funcs.sum_3d

    f1 = epyc_numpy_funcs_mod.sum_3d
    x = rand(4, 5, 6)
    assert np.allclose(f1(x), sum_call(x), rtol=RTOL, atol=ATOL)


# Skip test if PYCCEL_DEFAULT_COMPILER=LLVM
@pytest.mark.skip_llvm
def test_sum_slice_in_if(language):
    def sum_call(x: "int[:]"):
        from numpy import sum as np_sum

        s = x.shape[0]
        if s < 3:
            return 0
        else:
            n = 1
            m = s - 1
            return np_sum(x[n:m])

    f1 = epyccel(sum_call, language=language, flags="-Werror=uninitialized")
    x = randint(99, size=10, dtype=np.int64)
    assert f1(x) == sum_call(x)


def test_sum_dtype(epyc_numpy_funcs_mod):
    sum_call = numpy_funcs.sum_dtype

    f1 = epyc_numpy_funcs_mod.sum_dtype
    x = randint(99, size=10, dtype=np.int64)
    assert isclose(f1(x), sum_call(x), rtol=RTOL, atol=ATOL)
    assert matching_types(f1(x), sum_call(x))


def test_sum_dtype_2(epyc_numpy_funcs_mod):
    sum_call = numpy_funcs.sum_dtype_2

    f1 = epyc_numpy_funcs_mod.sum_dtype_2
    x = rand(6, 4)
    assert np.array_equal(f1(x), sum_call(x))
    assert matching_types(f1(x), sum_call(x))


def test_sum_axis_2d(epyc_numpy_funcs_mod):
    sum_call = numpy_funcs.sum_axis_2d

    f1 = epyc_numpy_funcs_mod.sum_axis_2d
    x = randint(99, size=(5, 7), dtype=np.int64)

    f_x_pycc = f1(x)
    f_x_pyth = sum_call(x)
    assert np.array_equal(f_x_pycc, f_x_pyth)


def test_sum_keepdims(epyc_numpy_funcs_mod):
    sum_call = numpy_funcs.sum_keepdims

    f1 = epyc_numpy_funcs_mod.sum_keepdims
    x = rand(6, 4)
    f_x_pycc = f1(x)
    f_x_pyth = sum_call(x)
    assert np.allclose(f_x_pycc, f_x_pyth, rtol=RTOL, atol=ATOL)


def test_sum_initial(epyc_numpy_funcs_mod):
    sum_call = numpy_funcs.sum_initial

    f1 = epyc_numpy_funcs_mod.sum_initial
    x = randint(99, size=10, dtype=np.int64)
    f_x_pycc = f1(x)
    f_x_pyth = sum_call(x)
    assert f_x_pycc == f_x_pyth
    assert matching_types(f_x_pycc, f_x_pyth)


def test_sum_axis_keepdims_initial(epyc_numpy_funcs_mod):
    sum_call = numpy_funcs.sum_axis_keepdims_initial

    f1 = epyc_numpy_funcs_mod.sum_axis_keepdims_initial
    x = randint(99, size=(4, 6), dtype=np.int64)
    f_x_pycc = f1(x)
    f_x_pyth = sum_call(x)
    assert np.array_equal(f_x_pycc, f_x_pyth)


def test_sum_dtype_axis(epyc_numpy_funcs_mod):
    sum_call = numpy_funcs.sum_dtype_axis

    f1 = epyc_numpy_funcs_mod.sum_dtype_axis
    x = randint(99, size=(3, 8), dtype=np.int64)
    f_x_pycc = f1(x)
    f_x_pyth = sum_call(x)
    assert np.array_equal(f_x_pycc, f_x_pyth)


def test_sum_3d_multi_axis(epyc_numpy_funcs_mod):
    sum_call = numpy_funcs.sum_3d_multi_axis

    f1 = epyc_numpy_funcs_mod.sum_3d_multi_axis
    x = rand(4, 5, 6)
    assert np.allclose(f1(x), sum_call(x), rtol=RTOL, atol=ATOL)


def test_sum_out_axis_2d(epyc_numpy_funcs_mod):
    sum_call = numpy_funcs.sum_out_axis_2d

    f1 = epyc_numpy_funcs_mod.sum_out_axis_2d
    x = randint(99, size=(5, 7), dtype=np.int64)
    assert np.array_equal(f1(x), sum_call(x))


def test_sum_out_axis_keepdims(epyc_numpy_funcs_mod):
    sum_call = numpy_funcs.sum_out_axis_keepdims

    f1 = epyc_numpy_funcs_mod.sum_out_axis_keepdims
    x = rand(6, 4)
    assert np.allclose(f1(x), sum_call(x), rtol=RTOL, atol=ATOL)


def test_sum_out_reference(epyc_numpy_funcs_mod):
    sum_call = numpy_funcs.sum_out_reference

    f1 = epyc_numpy_funcs_mod.sum_out_reference
    x = rand(6, 4)
    assert np.allclose(f1(x), sum_call(x), rtol=RTOL, atol=ATOL)


def test_min_int(epyc_numpy_funcs_mod):
    min_call = numpy_funcs.min_int

    f1 = epyc_numpy_funcs_mod.min_int
    x = randint(99, size=10, dtype=np.int64)
    assert f1(x) == min_call(x)


def test_min_real(epyc_numpy_funcs_mod):
    min_call = numpy_funcs.min_real

    f1 = epyc_numpy_funcs_mod.min_real
    x = rand(10)
    assert np.array_equal(f1(x), min_call(x))


def test_min_complex(epyc_numpy_funcs_mod):
    min_call = numpy_funcs.min_complex

    f1 = epyc_numpy_funcs_mod.min_complex
    x = randn(10) + 1j * randn(10)
    assert np.array_equal(f1(x), min_call(x))
    x = randn(10) + 1j
    assert np.array_equal(f1(x), min_call(x))
    x = 10 + 1j * randn(10)
    assert np.array_equal(f1(x), min_call(x))


def test_min_bool(epyc_numpy_funcs_mod):
    min_call = numpy_funcs.min_bool

    f1 = epyc_numpy_funcs_mod.min_bool
    x = np.array([True, False, True, False])  # Generating a boolean array
    assert f1(x) == min_call(x)


def test_min_phrase(epyc_numpy_funcs_mod):
    min_phrase = numpy_funcs.min_phrase

    f2 = epyc_numpy_funcs_mod.min_phrase
    x = rand(10)
    y = rand(15)
    assert np.array_equal(f2(x, y), min_phrase(x, y))


def test_min_property(epyc_numpy_funcs_mod):
    min_call = numpy_funcs.min_property

    f1 = epyc_numpy_funcs_mod.min_property
    x = randint(99, size=10, dtype=np.int64)
    assert f1(x) == min_call(x)


def test_amin_1d(epyc_numpy_funcs_mod):
    amin_call = numpy_funcs.amin_1d

    f1 = epyc_numpy_funcs_mod.amin_1d
    x = randint(99, size=10, dtype=np.int64)
    assert f1(x) == amin_call(x)


def test_amin_axis(epyc_numpy_funcs_mod):
    amin_call = numpy_funcs.amin_axis

    f1 = epyc_numpy_funcs_mod.amin_axis
    x = randint(99, size=(6, 8), dtype=np.int64)
    assert np.array_equal(f1(x), amin_call(x))


def test_amin_keepdims(epyc_numpy_funcs_mod):
    amin_call = numpy_funcs.amin_keepdims

    f1 = epyc_numpy_funcs_mod.amin_keepdims
    x = rand(5, 7)
    res_ref = amin_call(x)
    res_cc = f1(x)
    assert np.array_equal(res_cc, res_ref)
    assert res_cc.shape == res_ref.shape


def test_amin_initial(epyc_numpy_funcs_mod):
    amin_call = numpy_funcs.amin_initial

    f1 = epyc_numpy_funcs_mod.amin_initial
    x = randint(99, size=10, dtype=np.int64)
    assert f1(x) == amin_call(x)


def test_amin_out_axis(epyc_numpy_funcs_mod):
    amin_call = numpy_funcs.amin_out_axis

    f1 = epyc_numpy_funcs_mod.amin_out_axis
    x = randint(99, size=(6, 8), dtype=np.int64)
    y_epyc = np.empty(6, dtype=int)
    y_pyth = np.empty(6, dtype=int)
    f1(x, y_epyc)
    amin_call(x, y_pyth)
    assert np.array_equal(y_epyc, y_pyth)


def test_max_int(epyc_numpy_funcs_mod):
    max_call = numpy_funcs.max_int

    f1 = epyc_numpy_funcs_mod.max_int
    x = randint(99, size=10, dtype=np.int64)
    assert f1(x) == max_call(x)


def test_max_real(epyc_numpy_funcs_mod):
    max_call = numpy_funcs.max_real

    f1 = epyc_numpy_funcs_mod.max_real
    x = rand(10)
    assert np.array_equal(f1(x), max_call(x))


def test_max_complex(epyc_numpy_funcs_mod):
    max_call = numpy_funcs.max_complex

    f1 = epyc_numpy_funcs_mod.max_complex
    x = randn(10) + 1j * randn(10)
    assert np.array_equal(f1(x), max_call(x))
    x = randn(10) + 1j
    assert np.array_equal(f1(x), max_call(x))
    x = 10 + 1j * randn(10)
    assert np.array_equal(f1(x), max_call(x))


def test_max_bool(epyc_numpy_funcs_mod):
    max_call = numpy_funcs.max_bool

    f1 = epyc_numpy_funcs_mod.max_bool
    x = np.array([True, False, True, False])  # Generating a boolean array
    assert f1(x) == max_call(x)


def test_max_phrase(epyc_numpy_funcs_mod):
    max_phrase = numpy_funcs.max_phrase

    f2 = epyc_numpy_funcs_mod.max_phrase
    x = rand(10)
    y = rand(15)
    assert np.array_equal(f2(x, y), max_phrase(x, y))


def test_max_property(epyc_numpy_funcs_mod):
    max_call = numpy_funcs.max_property

    f1 = epyc_numpy_funcs_mod.max_property
    x = randint(99, size=10, dtype=np.int64)
    assert f1(x) == max_call(x)


def test_amax_1d(epyc_numpy_funcs_mod):
    amax_call = numpy_funcs.amax_1d

    f1 = epyc_numpy_funcs_mod.amax_1d
    x = randint(99, size=10, dtype=np.int64)
    assert f1(x) == amax_call(x)


def test_amax_axis(epyc_numpy_funcs_mod):
    amax_call = numpy_funcs.amax_axis

    f1 = epyc_numpy_funcs_mod.amax_axis
    x = randint(99, size=(6, 8), dtype=np.int64)
    assert np.array_equal(f1(x), amax_call(x))


def test_amax_keepdims(epyc_numpy_funcs_mod):
    amax_call = numpy_funcs.amax_keepdims

    f1 = epyc_numpy_funcs_mod.amax_keepdims
    x = rand(5, 7)
    res_ref = amax_call(x)
    res_cc = f1(x)
    assert np.array_equal(res_cc, res_ref)
    assert res_cc.shape == res_ref.shape


def test_amax_initial(epyc_numpy_funcs_mod):
    amax_call = numpy_funcs.amax_initial

    f1 = epyc_numpy_funcs_mod.amax_initial
    x = randint(99, size=10, dtype=np.int64)
    assert f1(x) == amax_call(x)


def test_amax_out_axis(epyc_numpy_funcs_mod):
    amax_call = numpy_funcs.amax_out_axis

    f1 = epyc_numpy_funcs_mod.amax_out_axis
    x = randint(99, size=(6, 8), dtype=np.int64)
    y_epyc = np.empty(6, dtype=int)
    y_pyth = np.empty(6, dtype=int)
    f1(x, y_epyc)
    amax_call(x, y_pyth)
    assert np.array_equal(y_epyc, y_pyth)


def test_full_like_basic_int(epyc_numpy_funcs_mod):
    create_full_like_shape_1d = numpy_funcs.full_like_basic_int__create_shape_1d
    create_full_like_shape_2d = numpy_funcs.full_like_basic_int__create_shape_2d
    create_full_like_val = numpy_funcs.full_like_basic_int__create_val
    create_full_like_arg_names = numpy_funcs.full_like_basic_int__create_arg_names

    size = randint(1, 10)

    f_shape_1d = epyc_numpy_funcs_mod.full_like_basic_int__create_shape_1d
    assert f_shape_1d(size) == create_full_like_shape_1d(size)

    f_shape_2d = epyc_numpy_funcs_mod.full_like_basic_int__create_shape_2d
    assert f_shape_2d(size) == create_full_like_shape_2d(size)

    f_val = epyc_numpy_funcs_mod.full_like_basic_int__create_val
    assert f_val(size) == create_full_like_val(size)
    assert matching_types(f_val(size)[0], create_full_like_val(size)[0])

    f_arg_names = epyc_numpy_funcs_mod.full_like_basic_int__create_arg_names
    assert f_arg_names(size) == create_full_like_arg_names(size)
    assert matching_types(f_arg_names(size)[0], create_full_like_arg_names(size)[0])


def test_full_like_basic_real(epyc_numpy_funcs_mod):
    create_full_like_shape_1d = numpy_funcs.full_like_basic_real__create_shape_1d
    create_full_like_shape_2d = numpy_funcs.full_like_basic_real__create_shape_2d
    create_full_like_val = numpy_funcs.full_like_basic_real__create_val
    create_full_like_arg_names = numpy_funcs.full_like_basic_real__create_arg_names

    size = uniform(10)
    val = rand() * 5

    f_shape_1d = epyc_numpy_funcs_mod.full_like_basic_real__create_shape_1d
    assert f_shape_1d(size) == create_full_like_shape_1d(size)

    f_shape_2d = epyc_numpy_funcs_mod.full_like_basic_real__create_shape_2d
    assert f_shape_2d(size) == create_full_like_shape_2d(size)

    f_val = epyc_numpy_funcs_mod.full_like_basic_real__create_val
    assert f_val(val) == create_full_like_val(val)
    assert matching_types(f_val(val)[0], create_full_like_val(val)[0])

    f_arg_names = epyc_numpy_funcs_mod.full_like_basic_real__create_arg_names
    assert f_arg_names(val) == create_full_like_arg_names(val)
    assert matching_types(f_arg_names(val)[0], create_full_like_arg_names(val)[0])


@pytest.mark.parametrize(
    "language",
    (
        pytest.param("fortran", marks=pytest.mark.fortran),
        pytest.param(
            "c",
            marks=[pytest.mark.skip(reason="Tuples not implemented"), pytest.mark.c],
        ),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_full_like_basic_bool(language):
    def create_full_like_shape_1d(n: "int"):
        from numpy import array, full_like, shape

        arr = array([5, 1, 8, 0, 9])
        a = full_like(arr, n, int, "F")
        s = shape(a)
        return len(s), s[0]

    def create_full_like_shape_2d(n: "int"):
        from numpy import array, full_like, shape

        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = full_like(arr, n, int, "F")
        s = shape(a)
        return len(s), s[0], s[1]

    def create_full_like_val(val: "bool"):
        from numpy import array, full_like

        arr = array([5, 1, 8, 0, 9])
        a = full_like(arr, 3, bool, "F")
        return a[0], a[1], a[2]

    def create_full_like_arg_names(val: "bool"):
        from numpy import array, full_like

        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = full_like(arr, fill_value=val, dtype=bool, shape=(2, 3))
        return a[0, 0], a[0, 1], a[0, 2], a[1, 0], a[1, 1], a[1, 2]

    size = randint(1, 10)
    val = bool(randint(2))

    f_shape_1d = epyccel(create_full_like_shape_1d, language=language)
    assert f_shape_1d(size) == create_full_like_shape_1d(size)

    f_shape_2d = epyccel(create_full_like_shape_2d, language=language)
    assert f_shape_2d(size) == create_full_like_shape_2d(size)

    f_val = epyccel(create_full_like_val, language=language)
    assert f_val(val) == create_full_like_val(val)
    assert matching_types(f_val(val)[0], create_full_like_val(val)[0])

    f_arg_names = epyccel(create_full_like_arg_names, language=language)
    assert f_arg_names(val) == create_full_like_arg_names(val)
    assert matching_types(f_arg_names(val)[0], create_full_like_arg_names(val)[0])


def test_full_like_order(epyc_numpy_funcs_mod):
    create_full_like_shape_C = numpy_funcs.full_like_order__create_shape_C
    create_full_like_shape_F = numpy_funcs.full_like_order__create_shape_F

    size = randint(1, 10)

    f_shape_C = epyc_numpy_funcs_mod.full_like_order__create_shape_C
    assert f_shape_C(size) == create_full_like_shape_C(size)

    f_shape_F = epyc_numpy_funcs_mod.full_like_order__create_shape_F
    assert f_shape_F(size) == create_full_like_shape_F(size)


def test_full_like_dtype(epyc_numpy_funcs_mod):

    val_int = randint(100)
    val_float = rand() * 100

    create_full_like_val_int_int = numpy_funcs.full_like_dtype__create_val_int_int
    f_int_int = epyc_numpy_funcs_mod.full_like_dtype__create_val_int_int
    assert f_int_int(val_int) == create_full_like_val_int_int(val_int)
    assert matching_types(f_int_int(val_int), create_full_like_val_int_int(val_int))

    create_full_like_val_int_float = numpy_funcs.full_like_dtype__create_val_int_float
    f_int_float = epyc_numpy_funcs_mod.full_like_dtype__create_val_int_float
    assert isclose(
        f_int_float(val_int),
        create_full_like_val_int_float(val_int),
        rtol=RTOL,
        atol=ATOL,
    )
    assert matching_types(f_int_float(val_int), create_full_like_val_int_float(val_int))

    create_full_like_val_int_complex = (
        numpy_funcs.full_like_dtype__create_val_int_complex
    )
    f_int_complex = epyc_numpy_funcs_mod.full_like_dtype__create_val_int_complex
    assert isclose(
        f_int_complex(val_int),
        create_full_like_val_int_complex(val_int),
        rtol=RTOL,
        atol=ATOL,
    )
    assert matching_types(
        f_int_complex(val_int), create_full_like_val_int_complex(val_int)
    )

    create_full_like_val_real_int32 = numpy_funcs.full_like_dtype__create_val_real_int32
    f_real_int32 = epyc_numpy_funcs_mod.full_like_dtype__create_val_real_int32
    assert f_real_int32(val_float) == create_full_like_val_real_int32(val_float)
    assert matching_types(
        f_real_int32(val_float), create_full_like_val_real_int32(val_float)
    )

    create_full_like_val_real_float32 = (
        numpy_funcs.full_like_dtype__create_val_real_float32
    )
    f_real_float32 = epyc_numpy_funcs_mod.full_like_dtype__create_val_real_float32
    assert isclose(
        f_real_float32(val_float),
        create_full_like_val_real_float32(val_float),
        rtol=RTOL,
        atol=ATOL,
    )
    assert matching_types(
        f_real_float32(val_float), create_full_like_val_real_float32(val_float)
    )

    create_full_like_val_real_float64 = (
        numpy_funcs.full_like_dtype__create_val_real_float64
    )
    f_real_float64 = epyc_numpy_funcs_mod.full_like_dtype__create_val_real_float64
    assert isclose(
        f_real_float64(val_float),
        create_full_like_val_real_float64(val_float),
        rtol=RTOL,
        atol=ATOL,
    )
    assert matching_types(
        f_real_float64(val_float), create_full_like_val_real_float64(val_float)
    )

    create_full_like_val_real_complex64 = (
        numpy_funcs.full_like_dtype__create_val_real_complex64
    )
    f_real_complex64 = epyc_numpy_funcs_mod.full_like_dtype__create_val_real_complex64
    assert isclose(
        f_real_complex64(val_float),
        create_full_like_val_real_complex64(val_float),
        rtol=RTOL,
        atol=ATOL,
    )
    assert matching_types(
        f_real_complex64(val_float), create_full_like_val_real_complex64(val_float)
    )

    create_full_like_val_real_complex128 = (
        numpy_funcs.full_like_dtype__create_val_real_complex128
    )
    f_real_complex128 = epyc_numpy_funcs_mod.full_like_dtype__create_val_real_complex128
    assert isclose(
        f_real_complex128(val_float),
        create_full_like_val_real_complex128(val_float),
        rtol=RTOL,
        atol=ATOL,
    )
    assert matching_types(
        f_real_complex128(val_float), create_full_like_val_real_complex128(val_float)
    )

    create_full_like_val_int_int_auto = (
        numpy_funcs.full_like_dtype__create_val_int_int_auto
    )
    f_int_int_auto = epyc_numpy_funcs_mod.full_like_dtype__create_val_int_int_auto
    assert f_int_int_auto(val_int) == create_full_like_val_int_int_auto(val_int)
    assert matching_types(
        f_int_int(val_int), create_full_like_val_int_int_auto(val_int)
    )

    create_full_like_val_int_float_auto = (
        numpy_funcs.full_like_dtype__create_val_int_float_auto
    )
    f_int_float_auto = epyc_numpy_funcs_mod.full_like_dtype__create_val_int_float_auto
    assert isclose(
        f_int_float_auto(val_int),
        create_full_like_val_int_float_auto(val_int),
        rtol=RTOL,
        atol=ATOL,
    )
    assert matching_types(
        f_int_float_auto(val_int), create_full_like_val_int_float_auto(val_int)
    )

    create_full_like_val_int_complex_auto = (
        numpy_funcs.full_like_dtype__create_val_int_complex_auto
    )
    f_int_complex_auto = (
        epyc_numpy_funcs_mod.full_like_dtype__create_val_int_complex_auto
    )
    assert isclose(
        f_int_complex_auto(val_int),
        create_full_like_val_int_complex_auto(val_int),
        rtol=RTOL,
        atol=ATOL,
    )
    assert matching_types(
        f_int_complex_auto(val_int), create_full_like_val_int_complex_auto(val_int)
    )

    create_full_like_val_real_int32_auto = (
        numpy_funcs.full_like_dtype__create_val_real_int32_auto
    )
    f_real_int32_auto = epyc_numpy_funcs_mod.full_like_dtype__create_val_real_int32_auto
    assert f_real_int32_auto(val_float) == create_full_like_val_real_int32_auto(
        val_float
    )
    assert matching_types(
        f_real_int32_auto(val_float), create_full_like_val_real_int32_auto(val_float)
    )

    create_full_like_val_real_float32_auto = (
        numpy_funcs.full_like_dtype__create_val_real_float32_auto
    )
    f_real_float32_auto = (
        epyc_numpy_funcs_mod.full_like_dtype__create_val_real_float32_auto
    )
    assert isclose(
        f_real_float32_auto(val_float),
        create_full_like_val_real_float32_auto(val_float),
        rtol=RTOL,
        atol=ATOL,
    )
    assert matching_types(
        f_real_float32_auto(val_float),
        create_full_like_val_real_float32_auto(val_float),
    )

    create_full_like_val_real_float64_auto = (
        numpy_funcs.full_like_dtype__create_val_real_float64_auto
    )
    f_real_float64_auto = (
        epyc_numpy_funcs_mod.full_like_dtype__create_val_real_float64_auto
    )
    assert isclose(
        f_real_float64_auto(val_float),
        create_full_like_val_real_float64_auto(val_float),
        rtol=RTOL,
        atol=ATOL,
    )
    assert matching_types(
        f_real_float64_auto(val_float),
        create_full_like_val_real_float64_auto(val_float),
    )

    create_full_like_val_real_complex64_auto = (
        numpy_funcs.full_like_dtype__create_val_real_float64_auto
    )
    f_real_complex64_auto = (
        epyc_numpy_funcs_mod.full_like_dtype__create_val_real_float64_auto
    )
    assert isclose(
        f_real_complex64_auto(val_float),
        create_full_like_val_real_complex64_auto(val_float),
        rtol=RTOL,
        atol=ATOL,
    )
    assert matching_types(
        f_real_complex64_auto(val_float),
        create_full_like_val_real_complex64_auto(val_float),
    )

    create_full_like_val_real_complex128_auto = (
        numpy_funcs.full_like_dtype__create_val_real_complex128_auto
    )
    f_real_complex128_auto = (
        epyc_numpy_funcs_mod.full_like_dtype__create_val_real_complex128_auto
    )
    assert isclose(
        f_real_complex128_auto(val_float),
        create_full_like_val_real_complex128_auto(val_float),
        rtol=RTOL,
        atol=ATOL,
    )
    assert matching_types(
        f_real_complex128_auto(val_float),
        create_full_like_val_real_complex128_auto(val_float),
    )


def test_full_like_combined_args(epyc_numpy_funcs_mod):
    create_full_like_1_shape = numpy_funcs.full_like_combined_args__create_1_shape
    create_full_like_1_val = numpy_funcs.full_like_combined_args__create_1_val
    create_full_like_2_shape = numpy_funcs.full_like_combined_args__create_2_shape
    create_full_like_2_val = numpy_funcs.full_like_combined_args__create_2_val
    create_full_like_3_shape = numpy_funcs.full_like_combined_args__create_3_shape
    create_full_like_3_val = numpy_funcs.full_like_combined_args__create_3_val

    f1_shape = epyc_numpy_funcs_mod.full_like_combined_args__create_1_shape
    f1_val = epyc_numpy_funcs_mod.full_like_combined_args__create_1_val
    assert f1_shape() == create_full_like_1_shape()
    assert f1_val() == create_full_like_1_val()
    assert matching_types(f1_val(), create_full_like_1_val())

    f2_shape = epyc_numpy_funcs_mod.full_like_combined_args__create_2_shape
    f2_val = epyc_numpy_funcs_mod.full_like_combined_args__create_2_val
    assert f2_shape() == create_full_like_2_shape()
    assert isclose(f2_val(), create_full_like_2_val(), rtol=RTOL, atol=ATOL)
    assert matching_types(f2_val(), create_full_like_2_val())

    f3_shape = epyc_numpy_funcs_mod.full_like_combined_args__create_3_shape
    f3_val = epyc_numpy_funcs_mod.full_like_combined_args__create_3_val
    assert f3_shape() == create_full_like_3_shape()
    assert isclose(f3_val(), create_full_like_3_val(), rtol=RTOL, atol=ATOL)
    assert matching_types(f3_val(), create_full_like_3_val())


def test_empty_like_basic(epyc_numpy_funcs_mod):
    create_empty_like_shape_1d = (
        numpy_funcs.empty_like_basic__create_empty_like_shape_1d
    )
    create_empty_like_shape_2d = (
        numpy_funcs.empty_like_basic__create_empty_like_shape_2d
    )

    size = randint(1, 10)

    f_shape_1d = epyc_numpy_funcs_mod.empty_like_basic__create_empty_like_shape_1d
    assert f_shape_1d(size) == create_empty_like_shape_1d(size)

    f_shape_2d = epyc_numpy_funcs_mod.empty_like_basic__create_empty_like_shape_2d
    assert f_shape_2d(size) == create_empty_like_shape_2d(size)


def test_empty_like_order(epyc_numpy_funcs_mod):
    create_empty_like_shape_C = numpy_funcs.empty_like_order__create_empty_like_shape_C
    create_empty_like_shape_F = numpy_funcs.empty_like_order__create_empty_like_shape_F

    size_1 = randint(1, 10)
    size_2 = randint(1, 10)

    f_shape_C = epyc_numpy_funcs_mod.empty_like_order__create_empty_like_shape_C
    assert f_shape_C(size_1, size_2) == create_empty_like_shape_C(size_1, size_2)

    f_shape_F = epyc_numpy_funcs_mod.empty_like_order__create_empty_like_shape_F
    assert f_shape_F(size_1, size_2) == create_empty_like_shape_F(size_1, size_2)


def test_empty_like_dtype(epyc_numpy_funcs_mod):

    create_empty_like_val_int_auto = (
        numpy_funcs.empty_like_dtype__create_empty_like_val_int_auto
    )
    create_empty_like_val_int = numpy_funcs.empty_like_dtype__create_empty_like_val_int
    create_empty_like_val_float_auto = (
        numpy_funcs.empty_like_dtype__create_empty_like_val_float_auto
    )
    create_empty_like_val_float = (
        numpy_funcs.empty_like_dtype__create_empty_like_val_float
    )
    create_empty_like_val_complex_auto = (
        numpy_funcs.empty_like_dtype__create_empty_like_val_complex_auto
    )
    create_empty_like_val_complex = (
        numpy_funcs.empty_like_dtype__create_empty_like_val_complex
    )
    create_empty_like_val_int32_auto = (
        numpy_funcs.empty_like_dtype__create_empty_like_val_int32_auto
    )
    create_empty_like_val_int32 = (
        numpy_funcs.empty_like_dtype__create_empty_like_val_int32
    )
    create_empty_like_val_float32_auto = (
        numpy_funcs.empty_like_dtype__create_empty_like_val_float32_auto
    )
    create_empty_like_val_float32 = (
        numpy_funcs.empty_like_dtype__create_empty_like_val_float32
    )
    create_empty_like_val_float64_auto = (
        numpy_funcs.empty_like_dtype__create_empty_like_val_float64_auto
    )
    create_empty_like_val_float64 = (
        numpy_funcs.empty_like_dtype__create_empty_like_val_float64
    )
    create_empty_like_val_complex64_auto = (
        numpy_funcs.empty_like_dtype__create_empty_like_val_complex64_auto
    )
    create_empty_like_val_complex64 = (
        numpy_funcs.empty_like_dtype__create_empty_like_val_complex64
    )
    create_empty_like_val_complex128_auto = (
        numpy_funcs.empty_like_dtype__create_empty_like_val_complex128_auto
    )
    create_empty_like_val_complex128 = (
        numpy_funcs.empty_like_dtype__create_empty_like_val_complex128
    )

    f_int_auto = epyc_numpy_funcs_mod.empty_like_dtype__create_empty_like_val_int_auto
    assert matching_types(f_int_auto(), create_empty_like_val_int_auto())

    f_int_int = epyc_numpy_funcs_mod.empty_like_dtype__create_empty_like_val_int
    assert matching_types(f_int_int(), create_empty_like_val_int())

    f_float_auto = (
        epyc_numpy_funcs_mod.empty_like_dtype__create_empty_like_val_float_auto
    )
    assert matching_types(f_float_auto(), create_empty_like_val_float_auto())

    f_int_float = epyc_numpy_funcs_mod.empty_like_dtype__create_empty_like_val_float
    assert matching_types(f_int_float(), create_empty_like_val_float())

    f_complex_auto = (
        epyc_numpy_funcs_mod.empty_like_dtype__create_empty_like_val_complex_auto
    )
    assert matching_types(f_complex_auto(), create_empty_like_val_complex_auto())

    f_int_complex = epyc_numpy_funcs_mod.empty_like_dtype__create_empty_like_val_complex
    assert matching_types(f_int_complex(), create_empty_like_val_complex())

    f_int32_auto = (
        epyc_numpy_funcs_mod.empty_like_dtype__create_empty_like_val_int32_auto
    )
    assert matching_types(f_int32_auto(), create_empty_like_val_int32_auto())

    f_real_int32 = epyc_numpy_funcs_mod.empty_like_dtype__create_empty_like_val_int32
    assert matching_types(f_real_int32(), create_empty_like_val_int32())

    f_float32_auto = (
        epyc_numpy_funcs_mod.empty_like_dtype__create_empty_like_val_float32_auto
    )
    assert matching_types(f_float32_auto(), create_empty_like_val_float32_auto())

    f_real_float32 = (
        epyc_numpy_funcs_mod.empty_like_dtype__create_empty_like_val_float32
    )
    assert matching_types(f_real_float32(), create_empty_like_val_float32())

    f_float64_auto = (
        epyc_numpy_funcs_mod.empty_like_dtype__create_empty_like_val_float64_auto
    )
    assert matching_types(f_float64_auto(), create_empty_like_val_float64_auto())

    f_real_float64 = (
        epyc_numpy_funcs_mod.empty_like_dtype__create_empty_like_val_float64
    )
    assert matching_types(f_real_float64(), create_empty_like_val_float64())

    f_complex64_auto = (
        epyc_numpy_funcs_mod.empty_like_dtype__create_empty_like_val_complex64_auto
    )

    assert matching_types(f_complex64_auto(), create_empty_like_val_complex64_auto())

    f_real_complex64 = (
        epyc_numpy_funcs_mod.empty_like_dtype__create_empty_like_val_complex64
    )
    assert matching_types(f_real_complex64(), create_empty_like_val_complex64())

    f_complex128_auto = (
        epyc_numpy_funcs_mod.empty_like_dtype__create_empty_like_val_complex128_auto
    )
    assert matching_types(f_complex128_auto(), create_empty_like_val_complex128_auto())

    f_real_complex128 = (
        epyc_numpy_funcs_mod.empty_like_dtype__create_empty_like_val_complex128
    )
    assert matching_types(f_real_complex128(), create_empty_like_val_complex128())


def test_empty_like_combined_args(epyc_numpy_funcs_mod):

    create_empty_like_1_shape = (
        numpy_funcs.empty_like_combined_args__create_empty_like_1_shape
    )
    create_empty_like_1_val = (
        numpy_funcs.empty_like_combined_args__create_empty_like_1_val
    )
    create_empty_like_2_shape = (
        numpy_funcs.empty_like_combined_args__create_empty_like_2_shape
    )
    create_empty_like_2_val = (
        numpy_funcs.empty_like_combined_args__create_empty_like_2_val
    )
    create_empty_like_3_shape = (
        numpy_funcs.empty_like_combined_args__create_empty_like_3_shape
    )
    create_empty_like_3_val = (
        numpy_funcs.empty_like_combined_args__create_empty_like_3_val
    )

    f1_shape = epyc_numpy_funcs_mod.empty_like_combined_args__create_empty_like_1_shape
    f1_val = epyc_numpy_funcs_mod.empty_like_combined_args__create_empty_like_1_val
    assert f1_shape() == create_empty_like_1_shape()
    assert matching_types(f1_val(), create_empty_like_1_val())

    f2_shape = epyc_numpy_funcs_mod.empty_like_combined_args__create_empty_like_2_shape
    f2_val = epyc_numpy_funcs_mod.empty_like_combined_args__create_empty_like_2_val
    assert f2_shape() == create_empty_like_2_shape()
    assert matching_types(f2_val(), create_empty_like_2_val())

    f3_shape = epyc_numpy_funcs_mod.empty_like_combined_args__create_empty_like_3_shape
    f3_val = epyc_numpy_funcs_mod.empty_like_combined_args__create_empty_like_3_val
    assert f3_shape() == create_empty_like_3_shape()
    assert matching_types(f3_val(), create_empty_like_3_val())


def test_ones_like_basic(epyc_numpy_funcs_mod):
    create_ones_like_shape_1d = numpy_funcs.ones_like_basic__create_ones_like_shape_1d
    create_ones_like_shape_2d = numpy_funcs.ones_like_basic__create_ones_like_shape_2d

    size = randint(1, 10)

    f_shape_1d = epyc_numpy_funcs_mod.ones_like_basic__create_ones_like_shape_1d
    assert f_shape_1d(size) == create_ones_like_shape_1d(size)

    f_shape_2d = epyc_numpy_funcs_mod.ones_like_basic__create_ones_like_shape_2d
    assert f_shape_2d(size) == create_ones_like_shape_2d(size)


def test_ones_like_order(epyc_numpy_funcs_mod):
    create_ones_like_shape_C = numpy_funcs.ones_like_order__create_ones_like_shape_C
    create_ones_like_shape_F = numpy_funcs.ones_like_order__create_ones_like_shape_F

    size_1 = randint(1, 10)
    size_2 = randint(1, 10)

    f_shape_C = epyc_numpy_funcs_mod.ones_like_order__create_ones_like_shape_C
    assert f_shape_C(size_1, size_2) == create_ones_like_shape_C(size_1, size_2)

    f_shape_F = epyc_numpy_funcs_mod.ones_like_order__create_ones_like_shape_F
    assert f_shape_F(size_1, size_2) == create_ones_like_shape_F(size_1, size_2)


def test_ones_like_dtype(epyc_numpy_funcs_mod):

    create_ones_like_val_int = numpy_funcs.ones_like_dtype__create_ones_like_val_int
    create_ones_like_val_float = numpy_funcs.ones_like_dtype__create_ones_like_val_float
    create_ones_like_val_complex = (
        numpy_funcs.ones_like_dtype__create_ones_like_val_complex
    )
    create_ones_like_val_int32 = numpy_funcs.ones_like_dtype__create_ones_like_val_int32
    create_ones_like_val_float32 = (
        numpy_funcs.ones_like_dtype__create_ones_like_val_float32
    )
    create_ones_like_val_float64 = (
        numpy_funcs.ones_like_dtype__create_ones_like_val_float64
    )
    create_ones_like_val_complex64 = (
        numpy_funcs.ones_like_dtype__create_ones_like_val_complex64
    )
    create_ones_like_val_complex128 = (
        numpy_funcs.ones_like_dtype__create_ones_like_val_complex128
    )
    create_ones_like_val_int_auto = (
        numpy_funcs.ones_like_dtype__create_ones_like_val_int_auto
    )
    create_ones_like_val_float_auto = (
        numpy_funcs.ones_like_dtype__create_ones_like_val_float_auto
    )
    create_ones_like_val_complex_auto = (
        numpy_funcs.ones_like_dtype__create_ones_like_val_complex_auto
    )
    create_ones_like_val_int32_auto = (
        numpy_funcs.ones_like_dtype__create_ones_like_val_int32_auto
    )
    create_ones_like_val_float32_auto = (
        numpy_funcs.ones_like_dtype__create_ones_like_val_float32_auto
    )
    create_ones_like_val_float64_auto = (
        numpy_funcs.ones_like_dtype__create_ones_like_val_float64_auto
    )
    create_ones_like_val_complex64_auto = (
        numpy_funcs.ones_like_dtype__create_ones_like_val_complex64_auto
    )
    create_ones_like_val_complex128_auto = (
        numpy_funcs.ones_like_dtype__create_ones_like_val_complex128_auto
    )

    f_int_int = epyc_numpy_funcs_mod.ones_like_dtype__create_ones_like_val_int
    assert f_int_int() == create_ones_like_val_int()
    assert matching_types(f_int_int(), create_ones_like_val_int())

    f_int_float = epyc_numpy_funcs_mod.ones_like_dtype__create_ones_like_val_float
    assert isclose(f_int_float(), create_ones_like_val_float(), rtol=RTOL, atol=ATOL)
    assert matching_types(f_int_float(), create_ones_like_val_float())

    f_int_complex = epyc_numpy_funcs_mod.ones_like_dtype__create_ones_like_val_complex
    assert isclose(
        f_int_complex(), create_ones_like_val_complex(), rtol=RTOL, atol=ATOL
    )
    assert matching_types(f_int_complex(), create_ones_like_val_complex())

    f_real_int32 = epyc_numpy_funcs_mod.ones_like_dtype__create_ones_like_val_int32
    assert f_real_int32() == create_ones_like_val_int32()
    assert matching_types(f_real_int32(), create_ones_like_val_int32())

    f_real_float32 = epyc_numpy_funcs_mod.ones_like_dtype__create_ones_like_val_float32
    assert isclose(
        f_real_float32(), create_ones_like_val_float32(), rtol=RTOL, atol=ATOL
    )
    assert matching_types(f_real_float32(), create_ones_like_val_float32())

    f_real_float64 = epyc_numpy_funcs_mod.ones_like_dtype__create_ones_like_val_float64
    assert isclose(
        f_real_float64(), create_ones_like_val_float64(), rtol=RTOL, atol=ATOL
    )
    assert matching_types(f_real_float64(), create_ones_like_val_float64())

    f_real_complex64 = (
        epyc_numpy_funcs_mod.ones_like_dtype__create_ones_like_val_complex64
    )
    assert isclose(
        f_real_complex64(), create_ones_like_val_complex64(), rtol=RTOL, atol=ATOL
    )
    assert matching_types(f_real_complex64(), create_ones_like_val_complex64())

    f_real_complex128 = (
        epyc_numpy_funcs_mod.ones_like_dtype__create_ones_like_val_complex128
    )
    assert isclose(
        f_real_complex128(), create_ones_like_val_complex128(), rtol=RTOL, atol=ATOL
    )
    assert matching_types(f_real_complex128(), create_ones_like_val_complex128())

    f_int_int_auto = epyc_numpy_funcs_mod.ones_like_dtype__create_ones_like_val_int_auto
    assert f_int_int_auto() == create_ones_like_val_int_auto()
    assert matching_types(f_int_int_auto(), create_ones_like_val_int_auto())

    f_int_float_auto = (
        epyc_numpy_funcs_mod.ones_like_dtype__create_ones_like_val_float_auto
    )
    assert isclose(
        f_int_float_auto(), create_ones_like_val_float_auto(), rtol=RTOL, atol=ATOL
    )
    assert matching_types(f_int_float_auto(), create_ones_like_val_float_auto())

    f_int_complex_auto = (
        epyc_numpy_funcs_mod.ones_like_dtype__create_ones_like_val_complex_auto
    )
    assert isclose(
        f_int_complex_auto(), create_ones_like_val_complex_auto(), rtol=RTOL, atol=ATOL
    )
    assert matching_types(f_int_complex_auto(), create_ones_like_val_complex_auto())

    f_real_int32_auto = (
        epyc_numpy_funcs_mod.ones_like_dtype__create_ones_like_val_int32_auto
    )
    assert f_real_int32_auto() == create_ones_like_val_int32_auto()
    assert matching_types(f_real_int32_auto(), create_ones_like_val_int32_auto())

    f_real_float32_auto = (
        epyc_numpy_funcs_mod.ones_like_dtype__create_ones_like_val_float32_auto
    )
    assert isclose(
        f_real_float32_auto(), create_ones_like_val_float32_auto(), rtol=RTOL, atol=ATOL
    )
    assert matching_types(f_real_float32_auto(), create_ones_like_val_float32_auto())

    f_real_float64_auto = (
        epyc_numpy_funcs_mod.ones_like_dtype__create_ones_like_val_float64_auto
    )
    assert isclose(
        f_real_float64_auto(), create_ones_like_val_float64_auto(), rtol=RTOL, atol=ATOL
    )
    assert matching_types(f_real_float64_auto(), create_ones_like_val_float64_auto())

    f_real_complex64_auto = (
        epyc_numpy_funcs_mod.ones_like_dtype__create_ones_like_val_complex64_auto
    )
    assert isclose(
        f_real_complex64_auto(),
        create_ones_like_val_complex64_auto(),
        rtol=RTOL,
        atol=ATOL,
    )
    assert matching_types(
        f_real_complex64_auto(), create_ones_like_val_complex64_auto()
    )

    f_real_complex128_auto = (
        epyc_numpy_funcs_mod.ones_like_dtype__create_ones_like_val_complex128_auto
    )
    assert isclose(
        f_real_complex128_auto(),
        create_ones_like_val_complex128_auto(),
        rtol=RTOL,
        atol=ATOL,
    )
    assert matching_types(
        f_real_complex128_auto(), create_ones_like_val_complex128_auto()
    )


def test_ones_like_combined_args(epyc_numpy_funcs_mod):

    create_ones_like_1_shape = (
        numpy_funcs.ones_like_combined_args__create_ones_like_1_shape
    )
    create_ones_like_1_val = numpy_funcs.ones_like_combined_args__create_ones_like_1_val
    create_ones_like_2_shape = (
        numpy_funcs.ones_like_combined_args__create_ones_like_2_shape
    )
    create_ones_like_2_val = numpy_funcs.ones_like_combined_args__create_ones_like_2_val
    create_ones_like_3_shape = (
        numpy_funcs.ones_like_combined_args__create_ones_like_3_shape
    )
    create_ones_like_3_val = numpy_funcs.ones_like_combined_args__create_ones_like_3_val

    f1_shape = epyc_numpy_funcs_mod.ones_like_combined_args__create_ones_like_1_shape
    f1_val = epyc_numpy_funcs_mod.ones_like_combined_args__create_ones_like_1_val
    assert f1_shape() == create_ones_like_1_shape()
    assert f1_val() == create_ones_like_1_val()
    assert matching_types(f1_val(), create_ones_like_1_val())

    f2_shape = epyc_numpy_funcs_mod.ones_like_combined_args__create_ones_like_2_shape
    f2_val = epyc_numpy_funcs_mod.ones_like_combined_args__create_ones_like_2_val
    assert f2_shape() == create_ones_like_2_shape()
    assert isclose(f2_val(), create_ones_like_2_val(), rtol=RTOL, atol=ATOL)
    assert matching_types(f2_val(), create_ones_like_2_val())

    f3_shape = epyc_numpy_funcs_mod.ones_like_combined_args__create_ones_like_3_shape
    f3_val = epyc_numpy_funcs_mod.ones_like_combined_args__create_ones_like_3_val
    assert f3_shape() == create_ones_like_3_shape()
    assert isclose(f3_val(), create_ones_like_3_val(), rtol=RTOL, atol=ATOL)
    assert matching_types(f3_val(), create_ones_like_3_val())


def test_zeros_like_basic(epyc_numpy_funcs_mod):
    create_zeros_like_shape_1d = (
        numpy_funcs.zeros_like_basic__create_zeros_like_shape_1d
    )
    create_zeros_like_shape_2d = (
        numpy_funcs.zeros_like_basic__create_zeros_like_shape_2d
    )

    size = randint(1, 10)

    f_shape_1d = epyc_numpy_funcs_mod.zeros_like_basic__create_zeros_like_shape_1d
    assert f_shape_1d(size) == create_zeros_like_shape_1d(size)

    f_shape_2d = epyc_numpy_funcs_mod.zeros_like_basic__create_zeros_like_shape_2d
    assert f_shape_2d(size) == create_zeros_like_shape_2d(size)


def test_zeros_like_order(epyc_numpy_funcs_mod):
    create_zeros_like_shape_C = numpy_funcs.zeros_like_order__create_zeros_like_shape_C
    create_zeros_like_shape_F = numpy_funcs.zeros_like_order__create_zeros_like_shape_F

    size_1 = randint(1, 10)
    size_2 = randint(1, 10)

    f_shape_C = epyc_numpy_funcs_mod.zeros_like_order__create_zeros_like_shape_C
    assert f_shape_C(size_1, size_2) == create_zeros_like_shape_C(size_1, size_2)

    f_shape_F = epyc_numpy_funcs_mod.zeros_like_order__create_zeros_like_shape_F
    assert f_shape_F(size_1, size_2) == create_zeros_like_shape_F(size_1, size_2)


def test_zeros_like_dtype(epyc_numpy_funcs_mod):

    create_zeros_like_val_int = numpy_funcs.zeros_like_dtype__create_zeros_like_val_int
    create_zeros_like_val_float = (
        numpy_funcs.zeros_like_dtype__create_zeros_like_val_float
    )
    create_zeros_like_val_complex = (
        numpy_funcs.zeros_like_dtype__create_zeros_like_val_complex
    )
    create_zeros_like_val_int32 = (
        numpy_funcs.zeros_like_dtype__create_zeros_like_val_int32
    )
    create_zeros_like_val_float32 = (
        numpy_funcs.zeros_like_dtype__create_zeros_like_val_float32
    )
    create_zeros_like_val_float64 = (
        numpy_funcs.zeros_like_dtype__create_zeros_like_val_float64
    )
    create_zeros_like_val_complex64 = (
        numpy_funcs.zeros_like_dtype__create_zeros_like_val_complex64
    )
    create_zeros_like_val_complex128 = (
        numpy_funcs.zeros_like_dtype__create_zeros_like_val_complex128
    )

    f_int_int = epyc_numpy_funcs_mod.zeros_like_dtype__create_zeros_like_val_int
    assert f_int_int() == create_zeros_like_val_int()
    assert matching_types(f_int_int(), create_zeros_like_val_int())

    f_int_float = epyc_numpy_funcs_mod.zeros_like_dtype__create_zeros_like_val_float
    assert isclose(f_int_float(), create_zeros_like_val_float(), rtol=RTOL, atol=ATOL)
    assert matching_types(f_int_float(), create_zeros_like_val_float())

    f_int_complex = epyc_numpy_funcs_mod.zeros_like_dtype__create_zeros_like_val_complex
    assert isclose(
        f_int_complex(), create_zeros_like_val_complex(), rtol=RTOL, atol=ATOL
    )
    assert matching_types(f_int_complex(), create_zeros_like_val_complex())

    f_real_int32 = epyc_numpy_funcs_mod.zeros_like_dtype__create_zeros_like_val_int32
    assert f_real_int32() == create_zeros_like_val_int32()
    assert matching_types(f_real_int32(), create_zeros_like_val_int32())

    f_real_float32 = (
        epyc_numpy_funcs_mod.zeros_like_dtype__create_zeros_like_val_float32
    )
    assert isclose(
        f_real_float32(), create_zeros_like_val_float32(), rtol=RTOL, atol=ATOL
    )
    assert matching_types(f_real_float32(), create_zeros_like_val_float32())

    f_real_float64 = (
        epyc_numpy_funcs_mod.zeros_like_dtype__create_zeros_like_val_float64
    )
    assert isclose(
        f_real_float64(), create_zeros_like_val_float64(), rtol=RTOL, atol=ATOL
    )
    assert matching_types(f_real_float64(), create_zeros_like_val_float64())

    f_real_complex64 = (
        epyc_numpy_funcs_mod.zeros_like_dtype__create_zeros_like_val_complex64
    )
    assert isclose(
        f_real_complex64(), create_zeros_like_val_complex64(), rtol=RTOL, atol=ATOL
    )
    assert matching_types(f_real_complex64(), create_zeros_like_val_complex64())

    f_real_complex128 = (
        epyc_numpy_funcs_mod.zeros_like_dtype__create_zeros_like_val_complex128
    )
    assert isclose(
        f_real_complex128(), create_zeros_like_val_complex128(), rtol=RTOL, atol=ATOL
    )
    assert matching_types(f_real_complex128(), create_zeros_like_val_complex128())


def test_zeros_like_dtype_auto(epyc_numpy_funcs_mod):

    create_zeros_like_val_int_auto = numpy_funcs.zeros_like_dtype_auto__create_val_int
    create_zeros_like_val_float_auto = (
        numpy_funcs.zeros_like_dtype_auto__create_val_float
    )
    create_zeros_like_val_complex_auto = (
        numpy_funcs.zeros_like_dtype_auto__create_val_complex
    )
    create_zeros_like_val_int32_auto = (
        numpy_funcs.zeros_like_dtype_auto__create_val_int32
    )
    create_zeros_like_val_float32_auto = (
        numpy_funcs.zeros_like_dtype_auto__create_val_float32
    )
    create_zeros_like_val_float64_auto = (
        numpy_funcs.zeros_like_dtype_auto__create_val_float64
    )
    create_zeros_like_val_complex64_auto = (
        numpy_funcs.zeros_like_dtype_auto__create_val_complex64
    )
    create_zeros_like_val_complex128_auto = (
        numpy_funcs.zeros_like_dtype_auto__create_val_complex128
    )

    f_int_auto = epyc_numpy_funcs_mod.zeros_like_dtype_auto__create_val_int
    assert matching_types(f_int_auto(), create_zeros_like_val_int_auto())

    f_float_auto = epyc_numpy_funcs_mod.zeros_like_dtype_auto__create_val_float
    assert matching_types(f_float_auto(), create_zeros_like_val_float_auto())

    f_complex_auto = epyc_numpy_funcs_mod.zeros_like_dtype_auto__create_val_complex
    assert matching_types(f_complex_auto(), create_zeros_like_val_complex_auto())

    f_int32_auto = epyc_numpy_funcs_mod.zeros_like_dtype_auto__create_val_int32
    assert matching_types(f_int32_auto(), create_zeros_like_val_int32_auto())

    f_float32_auto = epyc_numpy_funcs_mod.zeros_like_dtype_auto__create_val_float32
    assert matching_types(f_float32_auto(), create_zeros_like_val_float32_auto())

    f_float64_auto = epyc_numpy_funcs_mod.zeros_like_dtype_auto__create_val_float64
    assert matching_types(f_float64_auto(), create_zeros_like_val_float64_auto())

    f_complex64_auto = epyc_numpy_funcs_mod.zeros_like_dtype_auto__create_val_complex64
    assert matching_types(f_complex64_auto(), create_zeros_like_val_complex64_auto())

    f_complex128_auto = (
        epyc_numpy_funcs_mod.zeros_like_dtype_auto__create_val_complex128
    )
    assert matching_types(f_complex128_auto(), create_zeros_like_val_complex128_auto())


def test_zeros_like_combined_args(epyc_numpy_funcs_mod):

    create_zeros_like_1_shape = (
        numpy_funcs.zeros_like_combined_args__create_zeros_like_1_shape
    )
    create_zeros_like_1_val = (
        numpy_funcs.zeros_like_combined_args__create_zeros_like_1_val
    )
    create_zeros_like_2_shape = (
        numpy_funcs.zeros_like_combined_args__create_zeros_like_2_shape
    )
    create_zeros_like_2_val = (
        numpy_funcs.zeros_like_combined_args__create_zeros_like_2_val
    )
    create_zeros_like_3_shape = (
        numpy_funcs.zeros_like_combined_args__create_zeros_like_3_shape
    )
    create_zeros_like_3_val = (
        numpy_funcs.zeros_like_combined_args__create_zeros_like_3_val
    )

    f1_shape = epyc_numpy_funcs_mod.zeros_like_combined_args__create_zeros_like_1_shape
    f1_val = epyc_numpy_funcs_mod.zeros_like_combined_args__create_zeros_like_1_val
    assert f1_shape() == create_zeros_like_1_shape()
    assert f1_val() == create_zeros_like_1_val()
    assert matching_types(f1_val(), create_zeros_like_1_val())

    f2_shape = epyc_numpy_funcs_mod.zeros_like_combined_args__create_zeros_like_2_shape
    f2_val = epyc_numpy_funcs_mod.zeros_like_combined_args__create_zeros_like_2_val
    assert f2_shape() == create_zeros_like_2_shape()
    assert isclose(f2_val(), create_zeros_like_2_val(), rtol=RTOL, atol=ATOL)
    assert matching_types(f2_val(), create_zeros_like_2_val())

    f3_shape = epyc_numpy_funcs_mod.zeros_like_combined_args__create_zeros_like_3_shape
    f3_val = epyc_numpy_funcs_mod.zeros_like_combined_args__create_zeros_like_3_val
    assert f3_shape() == create_zeros_like_3_shape()
    assert isclose(f3_val(), create_zeros_like_3_val(), rtol=RTOL, atol=ATOL)
    assert matching_types(f3_val(), create_zeros_like_3_val())


def test_numpy_real_scalar(epyc_numpy_funcs_mod):

    get_real = numpy_funcs.numpy_real_scalar

    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=np.int64)
    integer32 = randint(min_int32, max_int32, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2)
    fl32 = uniform(min_float32 / 2, max_float32 / 2)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2)

    cmplx_from_float32 = (
        uniform(low=min_float32 / 2, high=max_float32 / 2)
        + uniform(low=min_float32 / 2, high=max_float32 / 2) * 1j
    )
    cmplx_from_float64 = (
        uniform(low=min_float64 / 2, high=max_float64 / 2)
        + uniform(low=min_float64 / 2, high=max_float64 / 2) * 1j
    )
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx_from_float32)
    cmplx128 = np.complex128(cmplx_from_float64)

    epyccel_func = epyc_numpy_funcs_mod.numpy_real_scalar

    f_bl_true_output = epyccel_func(True)
    test_bool_true_output = get_real(True)

    f_bl_false_output = epyccel_func(False)
    test_bool_false_output = get_real(False)

    assert f_bl_true_output == test_bool_true_output
    assert f_bl_false_output == test_bool_false_output

    assert matching_types(f_bl_true_output, test_bool_true_output)
    assert matching_types(f_bl_false_output, test_bool_false_output)

    f_integer_output = epyccel_func(integer)
    test_int_output = get_real(integer)

    assert f_integer_output == test_int_output
    assert matching_types(f_integer_output, test_int_output)

    f_integer8_output = epyccel_func(integer8)
    test_int8_output = get_real(integer8)

    assert f_integer8_output == test_int8_output
    assert matching_types(f_integer8_output, test_int8_output)

    f_integer16_output = epyccel_func(integer16)
    test_int16_output = get_real(integer16)

    assert f_integer16_output == test_int16_output
    assert matching_types(f_integer16_output, test_int16_output)

    f_integer32_output = epyccel_func(integer32)
    test_int32_output = get_real(integer32)

    assert f_integer32_output == test_int32_output
    assert matching_types(f_integer32_output, test_int32_output)

    f_integer64_output = epyccel_func(integer64)
    test_int64_output = get_real(integer64)

    assert f_integer64_output == test_int64_output
    assert matching_types(f_integer64_output, test_int64_output)

    f_fl_output = epyccel_func(fl)
    test_float_output = get_real(fl)

    assert f_fl_output == test_float_output
    assert matching_types(f_fl_output, test_float_output)

    f_fl32_output = epyccel_func(fl32)
    test_float32_output = get_real(fl32)

    assert f_fl32_output == test_float32_output
    assert matching_types(f_fl32_output, test_float32_output)

    f_fl64_output = epyccel_func(fl64)
    test_float64_output = get_real(fl64)

    assert f_fl64_output == test_float64_output
    assert matching_types(f_fl64_output, test_float64_output)

    f_complex64_output = epyccel_func(cmplx64)
    test_complex64_output = get_real(cmplx64)

    assert f_complex64_output == test_complex64_output
    assert matching_types(f_complex64_output, test_complex64_output)

    f_complex128_output = epyccel_func(cmplx128)
    test_complex128_output = get_real(cmplx128)

    assert f_complex128_output == test_complex128_output
    assert matching_types(f_complex64_output, test_complex64_output)


def test_numpy_real_array_like_1d(epyc_numpy_funcs_mod):

    get_real = numpy_funcs.numpy_real_array_like_1d

    size = 5

    bl = randint(0, 2, size=size, dtype=bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=np.int64)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size=size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size=size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size=size)

    cmplx128_from_float32 = (
        uniform(low=min_float32 / 2, high=max_float32 / 2, size=size)
        + uniform(low=min_float32 / 2, high=max_float32 / 2, size=size) * 1j
    )
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = (
        uniform(low=min_float64 / 2, high=max_float64 / 2, size=size)
        + uniform(low=min_float64 / 2, high=max_float64 / 2, size=size) * 1j
    )

    epyccel_func = epyc_numpy_funcs_mod.numpy_real_array_like_1d

    assert epyccel_func(bl) == get_real(bl)
    assert epyccel_func(integer8) == get_real(integer8)
    assert epyccel_func(integer16) == get_real(integer16)
    assert epyccel_func(integer) == get_real(integer)
    assert epyccel_func(integer32) == get_real(integer32)
    assert epyccel_func(integer64) == get_real(integer64)
    assert epyccel_func(fl) == get_real(fl)
    assert epyccel_func(fl32) == get_real(fl32)
    assert epyccel_func(fl64) == get_real(fl64)
    assert epyccel_func(cmplx64) == get_real(cmplx64)
    assert epyccel_func(cmplx128) == get_real(cmplx128)


def test_numpy_real_array_like_2d(epyc_numpy_funcs_mod):

    get_real = numpy_funcs.numpy_real_array_like_2d

    size = (2, 5)

    bl = randint(0, 2, size=size, dtype=bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=np.int64)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size=size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size=size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size=size)

    cmplx128_from_float32 = (
        uniform(low=min_float32 / 2, high=max_float32 / 2, size=size)
        + uniform(low=min_float32 / 2, high=max_float32 / 2, size=size) * 1j
    )
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = (
        uniform(low=min_float64 / 2, high=max_float64 / 2, size=size)
        + uniform(low=min_float64 / 2, high=max_float64 / 2, size=size) * 1j
    )

    epyccel_func = epyc_numpy_funcs_mod.numpy_real_array_like_2d

    assert epyccel_func(bl) == get_real(bl)
    assert epyccel_func(integer8) == get_real(integer8)
    assert epyccel_func(integer16) == get_real(integer16)
    assert epyccel_func(integer) == get_real(integer)
    assert epyccel_func(integer32) == get_real(integer32)
    assert epyccel_func(integer64) == get_real(integer64)
    assert epyccel_func(fl) == get_real(fl)
    assert epyccel_func(fl32) == get_real(fl32)
    assert epyccel_func(fl64) == get_real(fl64)
    assert epyccel_func(cmplx64) == get_real(cmplx64)
    assert epyccel_func(cmplx128) == get_real(cmplx128)


def test_numpy_imag_scalar(epyc_numpy_funcs_mod):

    get_imag = numpy_funcs.numpy_imag_scalar

    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=np.int64)
    integer32 = randint(min_int32, max_int32, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2)
    fl32 = uniform(min_float32 / 2, max_float32 / 2)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2)

    cmplx_from_float32 = (
        uniform(low=min_float32 / 2, high=max_float32 / 2)
        + uniform(low=min_float32 / 2, high=max_float32 / 2) * 1j
    )
    cmplx_from_float64 = (
        uniform(low=min_float64 / 2, high=max_float64 / 2)
        + uniform(low=min_float64 / 2, high=max_float64 / 2) * 1j
    )
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx_from_float32)
    cmplx128 = np.complex128(cmplx_from_float64)

    epyccel_func = epyc_numpy_funcs_mod.numpy_imag_scalar

    f_bl_true_output = epyccel_func(True)
    test_bool_true_output = get_imag(True)

    f_bl_false_output = epyccel_func(False)
    test_bool_false_output = get_imag(False)

    assert f_bl_true_output == test_bool_true_output
    assert f_bl_false_output == test_bool_false_output

    assert matching_types(f_bl_true_output, test_bool_true_output)
    assert matching_types(f_bl_false_output, test_bool_false_output)

    f_integer_output = epyccel_func(integer)
    test_int_output = get_imag(integer)

    assert f_integer_output == test_int_output
    assert matching_types(f_integer_output, test_int_output)

    f_integer8_output = epyccel_func(integer8)
    test_int8_output = get_imag(integer8)

    assert f_integer8_output == test_int8_output
    assert matching_types(f_integer8_output, test_int8_output)

    f_integer16_output = epyccel_func(integer16)
    test_int16_output = get_imag(integer16)

    assert f_integer16_output == test_int16_output
    assert matching_types(f_integer16_output, test_int16_output)

    f_integer32_output = epyccel_func(integer32)
    test_int32_output = get_imag(integer32)

    assert f_integer32_output == test_int32_output
    assert matching_types(f_integer32_output, test_int32_output)

    f_integer64_output = epyccel_func(integer64)
    test_int64_output = get_imag(integer64)

    assert f_integer64_output == test_int64_output
    assert matching_types(f_integer64_output, test_int64_output)

    f_fl_output = epyccel_func(fl)
    test_float_output = get_imag(fl)

    assert f_fl_output == test_float_output
    assert matching_types(f_fl_output, test_float_output)

    f_fl32_output = epyccel_func(fl32)
    test_float32_output = get_imag(fl32)

    assert f_fl32_output == test_float32_output
    assert matching_types(f_fl32_output, test_float32_output)

    f_fl64_output = epyccel_func(fl64)
    test_float64_output = get_imag(fl64)

    assert f_fl64_output == test_float64_output
    assert matching_types(f_fl64_output, test_float64_output)

    f_complex64_output = epyccel_func(cmplx64)
    test_complex64_output = get_imag(cmplx64)

    assert f_complex64_output == test_complex64_output
    assert matching_types(f_complex64_output, test_complex64_output)

    f_complex128_output = epyccel_func(cmplx128)
    test_complex128_output = get_imag(cmplx128)

    assert f_complex128_output == test_complex128_output
    assert matching_types(f_complex64_output, test_complex64_output)


def test_numpy_imag_array_like_1d(epyc_numpy_funcs_mod):

    get_imag = numpy_funcs.numpy_imag_array_like_1d

    size = 5

    bl = randint(0, 2, size=size, dtype=bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=np.int64)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size=size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size=size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size=size)

    cmplx128_from_float32 = (
        uniform(low=min_float32 / 2, high=max_float32 / 2, size=size)
        + uniform(low=min_float32 / 2, high=max_float32 / 2, size=size) * 1j
    )
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = (
        uniform(low=min_float64 / 2, high=max_float64 / 2, size=size)
        + uniform(low=min_float64 / 2, high=max_float64 / 2, size=size) * 1j
    )

    epyccel_func = epyc_numpy_funcs_mod.numpy_imag_array_like_1d

    assert epyccel_func(bl) == get_imag(bl)
    assert epyccel_func(integer8) == get_imag(integer8)
    assert epyccel_func(integer16) == get_imag(integer16)
    assert epyccel_func(integer) == get_imag(integer)
    assert epyccel_func(integer32) == get_imag(integer32)
    assert epyccel_func(integer64) == get_imag(integer64)
    assert epyccel_func(fl) == get_imag(fl)
    assert epyccel_func(fl32) == get_imag(fl32)
    assert epyccel_func(fl64) == get_imag(fl64)
    assert epyccel_func(cmplx64) == get_imag(cmplx64)
    assert epyccel_func(cmplx128) == get_imag(cmplx128)


def test_numpy_imag_array_like_2d(epyc_numpy_funcs_mod):

    get_imag = numpy_funcs.numpy_imag_array_like_2d

    size = (2, 5)

    bl = randint(0, 2, size=size, dtype=bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=np.int64)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size=size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size=size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size=size)

    cmplx128_from_float32 = (
        uniform(low=min_float32 / 2, high=max_float32 / 2, size=size)
        + uniform(low=min_float32 / 2, high=max_float32 / 2, size=size) * 1j
    )
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = (
        uniform(low=min_float64 / 2, high=max_float64 / 2, size=size)
        + uniform(low=min_float64 / 2, high=max_float64 / 2, size=size) * 1j
    )

    epyccel_func = epyc_numpy_funcs_mod.numpy_imag_array_like_2d

    assert epyccel_func(bl) == get_imag(bl)
    assert epyccel_func(integer8) == get_imag(integer8)
    assert epyccel_func(integer16) == get_imag(integer16)
    assert epyccel_func(integer) == get_imag(integer)
    assert epyccel_func(integer32) == get_imag(integer32)
    assert epyccel_func(integer64) == get_imag(integer64)
    assert epyccel_func(fl) == get_imag(fl)
    assert epyccel_func(fl32) == get_imag(fl32)
    assert epyccel_func(fl64) == get_imag(fl64)
    assert epyccel_func(cmplx64) == get_imag(cmplx64)
    assert epyccel_func(cmplx128) == get_imag(cmplx128)


@pytest.mark.skipif_by_language(
    True,
    language="python",
    reason=(
        "mod has special treatment for bool so it "
        "cannot be used in a translated interface in python"
    ),
)
# Not all the arguments supported
@pytest.mark.xfail(
    os.environ.get("PYCCEL_DEFAULT_COMPILER", None) == "intel",
    reason="Rounding errors. See #1669",
)
def test_numpy_mod_scalar(epyc_numpy_funcs_mod):
    get_mod = numpy_funcs.numpy_mod_scalar
    epyccel_func = epyc_numpy_funcs_mod.numpy_mod_scalar

    f_bl_true_output = epyccel_func(True)
    test_bool_true_output = get_mod(True)

    assert f_bl_true_output == test_bool_true_output
    assert matching_types(f_bl_true_output, test_bool_true_output)

    def test_int(min_int, max_int, dtype):
        integer = dtype(randint(min_int, max_int, dtype=dtype) or 1)

        f_integer_output = epyccel_func(integer)
        test_int_output = get_mod(integer)

        assert f_integer_output == test_int_output
        assert matching_types(f_integer_output, test_int_output)

    test_int(min_int8, max_int8, np.int8)
    test_int(min_int16, max_int16, np.int16)
    test_int(min_int, max_int, int)
    test_int(min_int32, max_int32, np.int32)
    test_int(min_int64, max_int64, np.int64)

    fl = uniform(min_float / 2, max_float / 2)
    fl32 = uniform(min_float32 / 2, max_float32 / 2)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2)

    f_fl_output = epyccel_func(fl)
    test_float_output = get_mod(fl)

    assert f_fl_output == test_float_output
    assert matching_types(f_fl_output, test_float_output)

    f_fl32_output = epyccel_func(fl32)
    test_float32_output = get_mod(fl32)

    assert f_fl32_output == test_float32_output
    assert matching_types(f_fl32_output, test_float32_output)

    f_fl64_output = epyccel_func(fl64)
    test_float64_output = get_mod(fl64)

    assert f_fl64_output == test_float64_output
    assert matching_types(f_fl64_output, test_float64_output)


@pytest.mark.skipif_by_language(
    True,
    language="python",
    reason=(
        "mod has special treatment for bool so it "
        "cannot be used in a translated interface in python"
    ),
)
@pytest.mark.xfail(
    os.environ.get("PYCCEL_DEFAULT_COMPILER", None) == "intel",
    reason="Rounding errors. See #1669",
)
def test_numpy_mod_array_like_1d(epyc_numpy_funcs_mod):
    get_mod = numpy_funcs.numpy_mod_array_like_1d

    size = 5

    epyccel_func = epyc_numpy_funcs_mod.numpy_mod_array_like_1d

    bl = np.full(size, True, dtype=bool)
    assert epyccel_func(bl) == get_mod(bl)

    def test_int(min_int, max_int, dtype):
        integer = randint(min_int, max_int - 1, size=size, dtype=dtype)
        integer = np.where(integer == 0, 1, integer)
        assert epyccel_func(integer) == get_mod(integer)

    test_int(min_int8, max_int8, np.int8)
    test_int(min_int16, max_int16, np.int16)
    test_int(min_int, max_int, int)
    test_int(min_int32, max_int32, np.int32)
    test_int(min_int64, max_int64, np.int64)

    fl = uniform(min_float / 2, max_float / 2, size=size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size=size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size=size)

    assert epyccel_func(fl) == get_mod(fl)
    assert epyccel_func(fl32) == get_mod(fl32)
    assert epyccel_func(fl64) == get_mod(fl64)


@pytest.mark.skipif_by_language(
    True,
    language="python",
    reason=(
        "mod has special treatment for bool so it "
        "cannot be used in a translated interface in python"
    ),
)
@pytest.mark.xfail(
    os.environ.get("PYCCEL_DEFAULT_COMPILER", None) == "intel",
    reason="Rounding errors. See #1669",
)
def test_numpy_mod_array_like_2d(epyc_numpy_funcs_mod):
    get_mod = numpy_funcs.numpy_mod_array_like_2d

    size = (2, 5)

    epyccel_func = epyc_numpy_funcs_mod.numpy_mod_array_like_2d

    bl = np.full(size, True, dtype=bool)
    assert epyccel_func(bl) == get_mod(bl)

    def test_int(min_int, max_int, dtype):
        integer = randint(min_int, max_int - 1, size=size, dtype=dtype)
        integer = np.where(integer == 0, 1, integer)
        assert epyccel_func(integer) == get_mod(integer)

    test_int(min_int8, max_int8, np.int8)
    test_int(min_int16, max_int16, np.int16)
    test_int(min_int, max_int, int)
    test_int(min_int32, max_int32, np.int32)
    test_int(min_int64, max_int64, np.int64)

    fl = uniform(min_float / 2, max_float / 2, size=size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size=size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size=size)

    assert epyccel_func(fl) == get_mod(fl)
    assert epyccel_func(fl32) == get_mod(fl32)
    assert epyccel_func(fl64) == get_mod(fl64)


@pytest.mark.xfail(
    os.environ.get("PYCCEL_DEFAULT_COMPILER", None) == "intel",
    reason="Rounding errors. See #1669",
)
def test_numpy_mod_mixed_order(epyc_numpy_funcs_mod):

    get_mod = numpy_funcs.numpy_mod_mixed_order

    epyccel_func = epyc_numpy_funcs_mod.numpy_mod_mixed_order

    fl1 = uniform(min_float / 2, max_float / 2, size=(2, 5))
    fl2 = uniform(min_float / 2, max_float / 2, size=(5, 2)).T

    assert epyccel_func(fl1, fl2) == get_mod(fl1, fl2)


@pytest.mark.parametrize(
    "language",
    (
        pytest.param("fortran", marks=[pytest.mark.fortran]),
        pytest.param(
            "c",
            marks=[
                pytest.mark.skip(
                    reason="Needs a C printer see https://github.com/pyccel/pyccel/issues/791"
                ),
                pytest.mark.c,
            ],
        ),
        pytest.param("python", marks=pytest.mark.python),
    ),
)

# Not all arguments are supported


def test_numpy_prod_scalar(language):

    def get_prod(a: C):
        from numpy import prod

        b = prod(a)
        return b

    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=np.int64)
    integer32 = randint(min_int32, max_int32, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2)
    fl32 = uniform(min_float32 / 2, max_float32 / 2)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2)

    cmplx128_from_float32 = (
        uniform(low=min_float32 / 2, high=max_float32 / 2)
        + uniform(low=min_float32 / 2, high=max_float32 / 2) * 1j
    )
    cmplx128_from_float64 = (
        uniform(low=min_float64 / 2, high=max_float64 / 2)
        + uniform(low=min_float64 / 2, high=max_float64 / 2) * 1j
    )
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = np.complex128(cmplx128_from_float64)

    epyccel_func = epyccel(get_prod, language=language)

    f_bl_true_output = epyccel_func(True)
    test_bool_true_output = get_prod(True)

    f_bl_false_output = epyccel_func(False)
    test_bool_false_output = get_prod(False)

    assert f_bl_true_output == test_bool_true_output
    assert f_bl_false_output == test_bool_false_output

    assert matching_types(f_bl_true_output, test_bool_true_output)
    assert matching_types(f_bl_false_output, test_bool_false_output)

    f_integer_output = epyccel_func(integer)
    test_int_output = get_prod(integer)

    assert f_integer_output == test_int_output
    assert matching_types(f_integer_output, test_int_output)

    f_integer8_output = epyccel_func(integer8)
    test_int8_output = get_prod(integer8)

    assert f_integer8_output == test_int8_output
    assert matching_types(f_integer8_output, test_int8_output)

    f_integer16_output = epyccel_func(integer16)
    test_int16_output = get_prod(integer16)

    assert f_integer16_output == test_int16_output
    assert matching_types(f_integer16_output, test_int16_output)

    f_integer32_output = epyccel_func(integer32)
    test_int32_output = get_prod(integer32)

    assert f_integer32_output == test_int32_output
    assert matching_types(f_integer32_output, test_int32_output)

    f_integer64_output = epyccel_func(integer64)
    test_int64_output = get_prod(integer64)

    assert f_integer64_output == test_int64_output
    assert matching_types(f_integer64_output, test_int64_output)

    f_fl_output = epyccel_func(fl)
    test_float_output = get_prod(fl)

    assert f_fl_output == test_float_output
    assert matching_types(f_fl_output, test_float_output)

    f_fl32_output = epyccel_func(fl32)
    test_float32_output = get_prod(fl32)

    assert f_fl32_output == test_float32_output
    assert matching_types(f_fl32_output, test_float32_output)

    f_fl64_output = epyccel_func(fl64)
    test_float64_output = get_prod(fl64)

    assert f_fl64_output == test_float64_output
    assert matching_types(f_fl64_output, test_float64_output)

    f_complex64_output = get_prod(cmplx64)
    test_complex64_output = get_prod(cmplx64)

    assert f_complex64_output == test_complex64_output
    assert matching_types(f_complex64_output, test_complex64_output)

    f_complex128_output = get_prod(cmplx128)
    test_complex128_output = get_prod(cmplx128)

    assert f_complex128_output == test_complex128_output
    assert matching_types(f_complex64_output, test_complex64_output)


@pytest.mark.parametrize(
    "language",
    (
        pytest.param("fortran", marks=[pytest.mark.fortran]),
        pytest.param(
            "c",
            marks=[
                pytest.mark.skip(
                    reason="Needs a C printer see https://github.com/pyccel/pyccel/issues/791"
                ),
                pytest.mark.c,
            ],
        ),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_numpy_prod_array_like_1d(language):

    def get_prod(arr: "C[:]"):
        from numpy import prod

        a = prod(arr)
        return a

    size = 5

    bl = randint(0, 2, size=size, dtype=bool)

    max_ok_int = int(max_int64 ** (1 / 5))

    integer8 = randint(
        max(min_int8, -max_ok_int), min(max_ok_int, max_int8), size=size, dtype=np.int8
    )
    integer16 = randint(
        max(min_int16, -max_ok_int),
        min(max_ok_int, max_int16),
        size=size,
        dtype=np.int16,
    )
    integer = randint(
        max(min_int, -max_ok_int), min(max_ok_int, max_int), size=size, dtype=np.int64
    )
    integer32 = randint(
        max(min_int32, -max_ok_int),
        min(max_ok_int, max_int32),
        size=size,
        dtype=np.int32,
    )
    integer64 = randint(-max_ok_int, max_ok_int, size=size, dtype=np.int64)

    fl = uniform(-((-min_float) ** (1 / 5)), max_float ** (1 / 5), size=size)

    min_ok_float32 = -((-min_float32) ** (1 / 5))
    min_ok_float64 = -((-min_float64) ** (1 / 5))
    max_ok_float32 = max_float32 ** (1 / 5)
    max_ok_float64 = max_float64 ** (1 / 5)

    fl32 = uniform(min_ok_float32, max_ok_float32, size=size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_ok_float64, max_ok_float64, size=size)

    cmplx128_from_float32 = (
        uniform(low=min_ok_float32 / 2, high=max_ok_float32 / 2, size=size)
        + uniform(low=min_ok_float32 / 2, high=max_ok_float32 / 2, size=size) * 1j
    )
    cmplx128_from_float64 = (
        uniform(low=min_ok_float64 / 2, high=max_ok_float64 / 2, size=size)
        + uniform(low=min_ok_float64 / 2, high=max_ok_float64 / 2, size=size) * 1j
    )
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = np.complex128(cmplx128_from_float64)

    epyccel_func = epyccel(get_prod, language=language)

    assert epyccel_func(bl) == get_prod(bl)
    assert epyccel_func(integer8) == get_prod(integer8)
    assert epyccel_func(integer16) == get_prod(integer16)
    assert epyccel_func(integer) == get_prod(integer)
    assert epyccel_func(integer32) == get_prod(integer32)
    assert epyccel_func(integer64) == get_prod(integer64)
    assert np.isclose(epyccel_func(fl), get_prod(fl), rtol=RTOL, atol=ATOL)
    assert np.isclose(epyccel_func(fl32), get_prod(fl32), rtol=RTOL32, atol=ATOL32)
    assert np.isclose(epyccel_func(fl64), get_prod(fl64), rtol=RTOL, atol=ATOL)
    assert np.isclose(
        epyccel_func(cmplx64), get_prod(cmplx64), rtol=RTOL32, atol=ATOL32
    )
    assert np.isclose(epyccel_func(cmplx128), get_prod(cmplx128), rtol=RTOL, atol=ATOL)
    assert matching_types(epyccel_func(bl), get_prod(bl))
    assert matching_types(epyccel_func(integer8), get_prod(integer8))
    assert matching_types(epyccel_func(integer16), get_prod(integer16))
    assert matching_types(epyccel_func(integer), get_prod(integer))
    assert matching_types(epyccel_func(integer32), get_prod(integer32))
    assert matching_types(epyccel_func(integer64), get_prod(integer64))
    assert matching_types(epyccel_func(fl), get_prod(fl))
    assert matching_types(epyccel_func(fl32), get_prod(fl32))
    assert matching_types(epyccel_func(fl64), get_prod(fl64))
    assert matching_types(epyccel_func(cmplx64), get_prod(cmplx64))
    assert matching_types(epyccel_func(cmplx128), get_prod(cmplx128))


@pytest.mark.parametrize(
    "language",
    (
        pytest.param("fortran", marks=[pytest.mark.fortran]),
        pytest.param(
            "c",
            marks=[
                pytest.mark.skip(
                    reason="Needs a C printer see https://github.com/pyccel/pyccel/issues/791"
                ),
                pytest.mark.c,
            ],
        ),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_numpy_prod_array_like_2d(language):

    def get_prod(arr: "C[:,:]"):
        from numpy import prod

        a = prod(arr)
        return a

    size = (2, 5)

    bl = randint(0, 2, size=size, dtype=bool)

    max_ok_int = int(max_int64 ** (1 / 10))

    integer8 = randint(
        max(min_int8, -max_ok_int), min(max_ok_int, max_int8), size=size, dtype=np.int8
    )
    integer16 = randint(
        max(min_int16, -max_ok_int),
        min(max_ok_int, max_int16),
        size=size,
        dtype=np.int16,
    )
    integer = randint(
        max(min_int, -max_ok_int), min(max_ok_int, max_int), size=size, dtype=np.int64
    )
    integer32 = randint(
        max(min_int32, -max_ok_int),
        min(max_ok_int, max_int32),
        size=size,
        dtype=np.int32,
    )
    integer64 = randint(-max_ok_int, max_ok_int, size=size, dtype=np.int64)

    fl = uniform(-((-min_float) ** (1 / 10)), max_float ** (1 / 10), size=size)

    min_ok_float32 = -((-min_float32) ** (1 / 10))
    min_ok_float64 = -((-min_float64) ** (1 / 10))
    max_ok_float32 = max_float32 ** (1 / 10)
    max_ok_float64 = max_float64 ** (1 / 10)

    fl32 = uniform(min_ok_float32, max_ok_float32, size=size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_ok_float64, max_ok_float64, size=size)

    cmplx128_from_float32 = (
        uniform(low=min_ok_float32 / 2, high=max_ok_float32 / 2, size=size)
        + uniform(low=min_ok_float32 / 2, high=max_ok_float32 / 2, size=size) * 1j
    )
    cmplx128_from_float64 = (
        uniform(low=min_ok_float64 / 2, high=max_ok_float64 / 2, size=size)
        + uniform(low=min_ok_float64 / 2, high=max_ok_float64 / 2, size=size) * 1j
    )
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = np.complex128(cmplx128_from_float64)

    epyccel_func = epyccel(get_prod, language=language)

    assert epyccel_func(bl) == get_prod(bl)
    assert epyccel_func(integer8) == get_prod(integer8)
    assert epyccel_func(integer16) == get_prod(integer16)
    assert epyccel_func(integer) == get_prod(integer)
    assert epyccel_func(integer32) == get_prod(integer32)
    assert epyccel_func(integer64) == get_prod(integer64)
    assert np.isclose(epyccel_func(fl), get_prod(fl), rtol=RTOL, atol=ATOL)
    assert np.isclose(epyccel_func(fl32), get_prod(fl32), rtol=RTOL32, atol=ATOL32)
    assert np.isclose(epyccel_func(fl64), get_prod(fl64), rtol=RTOL, atol=ATOL)
    assert np.isclose(
        epyccel_func(cmplx64), get_prod(cmplx64), rtol=RTOL32, atol=ATOL32
    )
    assert np.isclose(epyccel_func(cmplx128), get_prod(cmplx128), rtol=RTOL, atol=ATOL)


def test_numpy_norm_scalar(epyc_numpy_funcs_mod):

    get_norm = numpy_funcs.numpy_norm_scalar

    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=np.int64)
    integer32 = randint(min_int32, max_int32, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, dtype=np.int64)

    fl = uniform(low=-(abs(min_float) ** (1 / 2)), high=abs(max_float) ** (1 / 2))
    fl32 = uniform(low=-(abs(min_float32) ** (1 / 2)), high=abs(max_float32) ** (1 / 2))
    fl32 = np.float32(fl32)
    fl64 = uniform(low=-(abs(min_float64) ** (1 / 2)), high=abs(max_float64) ** (1 / 2))

    cmplx128_from_float32 = (
        uniform(
            low=-((abs(min_float32) / 2) ** (1 / 2)),
            high=((abs(max_float32) / 2) ** (1 / 2)),
        )
        + uniform(
            low=-((abs(max_float32) / 2) ** (1 / 2)),
            high=((abs(max_float32) / 2) ** (1 / 2)),
        )
        * 1j
    )
    cmplx128_from_float64 = (
        uniform(
            low=-((abs(min_float64) / 2) ** (1 / 2)),
            high=((abs(max_float64) / 2) ** (1 / 2)),
        )
        + uniform(
            low=-((abs(max_float64) / 2) ** (1 / 2)),
            high=((abs(max_float64) / 2) ** (1 / 2)),
        )
        * 1j
    )
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = np.complex128(cmplx128_from_float64)

    epyccel_func = epyc_numpy_funcs_mod.numpy_norm_scalar

    f_bl_true_output = epyccel_func(True)
    test_bool_true_output = get_norm(True)

    f_bl_false_output = epyccel_func(False)
    test_bool_false_output = get_norm(False)

    assert f_bl_true_output == test_bool_true_output
    assert f_bl_false_output == test_bool_false_output

    assert matching_types(f_bl_false_output, test_bool_false_output)
    assert matching_types(f_bl_true_output, test_bool_true_output)

    f_integer_output = epyccel_func(integer)
    test_int_output = get_norm(integer)

    assert np.isclose(f_integer_output, test_int_output, rtol=RTOL, atol=ATOL)
    assert matching_types(f_integer_output, test_int_output)

    f_integer8_output = epyccel_func(integer8)
    test_int8_output = get_norm(integer8)

    assert np.isclose(f_integer8_output, test_int8_output, rtol=RTOL, atol=ATOL)
    assert matching_types(f_integer8_output, test_int8_output)

    f_integer16_output = epyccel_func(integer16)
    test_int16_output = get_norm(integer16)

    assert np.isclose(f_integer16_output, test_int16_output, rtol=RTOL, atol=ATOL)
    assert matching_types(f_integer16_output, test_int16_output)

    f_integer32_output = epyccel_func(integer32)
    test_int32_output = get_norm(integer32)

    assert np.isclose(f_integer32_output, test_int32_output, rtol=RTOL, atol=ATOL)
    assert matching_types(f_integer32_output, test_int32_output)

    f_integer64_output = epyccel_func(integer64)
    test_int64_output = get_norm(integer64)

    assert np.isclose(f_integer64_output, test_int64_output, rtol=RTOL, atol=ATOL)
    assert matching_types(f_integer64_output, test_int64_output)

    f_fl_output = epyccel_func(fl)
    test_float_output = get_norm(fl)

    assert np.isclose(f_fl_output, test_float_output, rtol=RTOL, atol=ATOL)
    assert matching_types(f_fl_output, test_float_output)

    f_fl32_output = epyccel_func(fl32)
    test_float32_output = get_norm(fl32)

    assert np.isclose(f_fl32_output, test_float32_output, rtol=RTOL32, atol=ATOL32)
    assert matching_types(f_fl32_output, test_float32_output)

    f_fl64_output = epyccel_func(fl64)
    test_float64_output = get_norm(fl64)

    assert np.isclose(f_fl64_output, test_float64_output, rtol=RTOL, atol=ATOL)
    assert matching_types(f_fl64_output, test_float64_output)

    f_complex64_output = epyccel_func(cmplx64)
    test_complex64_output = get_norm(cmplx64)

    assert np.isclose(
        f_complex64_output, test_complex64_output, rtol=RTOL32, atol=ATOL32
    )
    assert matching_types(f_complex64_output, test_complex64_output)

    f_complex128_output = epyccel_func(cmplx128)
    test_complex128_output = get_norm(cmplx128)

    assert np.isclose(f_complex128_output, test_complex128_output, rtol=RTOL, atol=ATOL)
    assert matching_types(f_complex128_output, test_complex128_output)


def test_numpy_norm_scalar_expr(epyc_numpy_funcs_mod):

    get_norm = numpy_funcs.numpy_norm_scalar_expr

    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=np.int64)
    integer32 = randint(min_int32, max_int32, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, dtype=np.int64)

    fl = uniform(low=-(abs(min_float) ** (1 / 2)), high=abs(max_float) ** (1 / 2))
    fl32 = uniform(low=-(abs(min_float32) ** (1 / 2)), high=abs(max_float32) ** (1 / 2))
    fl32 = np.float32(fl32)
    fl64 = uniform(low=-(abs(min_float64) ** (1 / 2)), high=abs(max_float64) ** (1 / 2))

    cmplx128_from_float32 = (
        uniform(
            low=-((abs(min_float32) / 2) ** (1 / 2)),
            high=((abs(max_float32) / 2) ** (1 / 2)),
        )
        + uniform(
            low=-((abs(max_float32) / 2) ** (1 / 2)),
            high=((abs(max_float32) / 2) ** (1 / 2)),
        )
        * 1j
    )
    cmplx128_from_float64 = (
        uniform(
            low=-((abs(min_float64) / 2) ** (1 / 2)),
            high=((abs(max_float64) / 2) ** (1 / 2)),
        )
        + uniform(
            low=-((abs(max_float64) / 2) ** (1 / 2)),
            high=((abs(max_float64) / 2) ** (1 / 2)),
        )
        * 1j
    )
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = np.complex128(cmplx128_from_float64)

    epyccel_func = epyc_numpy_funcs_mod.numpy_norm_scalar_expr

    f_bl_true_output = epyccel_func(True)
    test_bool_true_output = get_norm(True)

    f_bl_false_output = epyccel_func(False)
    test_bool_false_output = get_norm(False)

    assert f_bl_true_output == test_bool_true_output
    assert f_bl_false_output == test_bool_false_output

    assert matching_types(f_bl_false_output, test_bool_false_output)
    assert matching_types(f_bl_true_output, test_bool_true_output)

    f_integer_output = epyccel_func(integer)
    test_int_output = get_norm(integer)

    assert np.isclose(f_integer_output, test_int_output, rtol=RTOL, atol=ATOL)
    assert matching_types(f_integer_output, test_int_output)

    f_integer8_output = epyccel_func(integer8)
    test_int8_output = get_norm(integer8)

    assert np.isclose(f_integer8_output, test_int8_output, rtol=RTOL, atol=ATOL)
    assert matching_types(f_integer8_output, test_int8_output)

    f_integer16_output = epyccel_func(integer16)
    test_int16_output = get_norm(integer16)

    assert np.isclose(f_integer16_output, test_int16_output, rtol=RTOL, atol=ATOL)
    assert matching_types(f_integer16_output, test_int16_output)

    f_integer32_output = epyccel_func(integer32)
    test_int32_output = get_norm(integer32)

    assert np.isclose(f_integer32_output, test_int32_output, rtol=RTOL, atol=ATOL)
    assert matching_types(f_integer32_output, test_int32_output)

    f_integer64_output = epyccel_func(integer64)
    test_int64_output = get_norm(integer64)

    assert np.isclose(f_integer64_output, test_int64_output, rtol=RTOL, atol=ATOL)
    assert matching_types(f_integer64_output, test_int64_output)

    f_fl_output = epyccel_func(fl)
    test_float_output = get_norm(fl)

    assert np.isclose(f_fl_output, test_float_output, rtol=RTOL, atol=ATOL)
    assert matching_types(f_fl_output, test_float_output)

    f_fl32_output = epyccel_func(fl32)
    test_float32_output = get_norm(fl32)

    assert np.isclose(f_fl32_output, test_float32_output, rtol=RTOL32, atol=ATOL32)
    assert matching_types(f_fl32_output, test_float32_output)

    f_fl64_output = epyccel_func(fl64)
    test_float64_output = get_norm(fl64)

    assert np.isclose(f_fl64_output, test_float64_output, rtol=RTOL, atol=ATOL)
    assert matching_types(f_fl64_output, test_float64_output)

    f_complex64_output = epyccel_func(cmplx64)
    test_complex64_output = get_norm(cmplx64)

    assert np.isclose(
        f_complex64_output, test_complex64_output, rtol=RTOL32, atol=ATOL32
    )
    assert matching_types(f_complex64_output, test_complex64_output)

    f_complex128_output = epyccel_func(cmplx128)
    test_complex128_output = get_norm(cmplx128)

    assert np.isclose(f_complex128_output, test_complex128_output, rtol=RTOL, atol=ATOL)
    assert matching_types(f_complex128_output, test_complex128_output)


def test_numpy_norm_array_like_1d(epyc_numpy_funcs_mod):

    get_norm = numpy_funcs.numpy_norm_array_like_1d

    size = 5

    bl = randint(0, 2, size=size, dtype=bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=np.int64)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(
        low=-((abs(min_float) / size) ** (1 / 2)),
        high=(abs(max_float) / size) ** (1 / 2),
        size=size,
    )
    fl32 = uniform(
        low=-((abs(min_float32) / size) ** (1 / 2)),
        high=(abs(max_float32) / size) ** (1 / 2),
        size=size,
    )
    fl32 = np.float32(fl32)
    fl64 = uniform(
        low=-((abs(min_float64) / size) ** (1 / 2)),
        high=(abs(max_float64) / size) ** (1 / 2),
        size=size,
    )

    cmplx128_from_float32 = (
        uniform(
            low=-((abs(min_float32) / (size * 2)) ** (1 / 2)),
            high=(abs(max_float32) / (size * 2)) ** (1 / 2),
            size=size,
        )
        + uniform(
            low=-((abs(min_float32) / (size * 2)) ** (1 / 2)),
            high=(abs(max_float32) / (size * 2)) ** (1 / 2),
            size=size,
        )
        * 1j
    )
    cmplx128_from_float64 = (
        uniform(
            low=-((abs(min_float64) / (size * 2)) ** (1 / 2)),
            high=(abs(max_float64) / (size * 2)) ** (1 / 2),
            size=size,
        )
        + uniform(
            low=-((abs(min_float64) / (size * 2)) ** (1 / 2)),
            high=(abs(max_float64) / (size * 2)) ** (1 / 2),
            size=size,
        )
        * 1j
    )
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = np.complex128(cmplx128_from_float64)

    epyccel_func = epyc_numpy_funcs_mod.numpy_norm_array_like_1d

    assert np.isclose(epyccel_func(bl), get_norm(bl), rtol=RTOL, atol=ATOL)
    assert np.isclose(epyccel_func(integer8), get_norm(integer8), rtol=RTOL, atol=ATOL)
    assert np.isclose(
        epyccel_func(integer16), get_norm(integer16), rtol=RTOL, atol=ATOL
    )
    assert np.isclose(epyccel_func(integer), get_norm(integer), rtol=RTOL, atol=ATOL)
    assert np.isclose(
        epyccel_func(integer32), get_norm(integer32), rtol=RTOL, atol=ATOL
    )
    assert np.isclose(
        epyccel_func(integer64), get_norm(integer64), rtol=RTOL, atol=ATOL
    )
    assert np.isclose(epyccel_func(fl), get_norm(fl), rtol=RTOL, atol=ATOL)
    assert np.isclose(epyccel_func(fl32), get_norm(fl32), rtol=RTOL32, atol=ATOL32)
    assert np.isclose(epyccel_func(fl64), get_norm(fl64), rtol=RTOL, atol=ATOL)
    assert np.isclose(
        epyccel_func(cmplx64), get_norm(cmplx64), rtol=RTOL32, atol=ATOL32
    )
    assert np.isclose(epyccel_func(cmplx128), get_norm(cmplx128), rtol=RTOL, atol=ATOL)


def test_numpy_norm_array_like_2d(epyc_numpy_funcs_mod):

    get_norm = numpy_funcs.numpy_norm_array_like_2d

    size = (2, 5)

    bl = randint(0, 2, size=size, dtype=bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=np.int64)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(
        low=-((abs(min_float) / (size[0] * size[1])) ** (1 / 2)),
        high=(abs(max_float) / (size[0] * size[1])) ** (1 / 2),
        size=size,
    )
    fl32 = uniform(
        low=-((abs(min_float32) / (size[0] * size[1])) ** (1 / 2)),
        high=(abs(max_float32) / (size[0] * size[1])) ** (1 / 2),
        size=size,
    )
    fl32 = np.float32(fl32)
    fl64 = uniform(
        low=-((abs(min_float64) / (size[0] * size[1])) ** (1 / 2)),
        high=(abs(max_float64) / (size[0] * size[1])) ** (1 / 2),
        size=size,
    )

    cmplx128_from_float32 = (
        uniform(
            low=-((abs(min_float32) / (size[0] * size[1] * 2)) ** (1 / 2)),
            high=(abs(max_float32) / (size[0] * size[1] * 2)) ** (1 / 2),
            size=size,
        )
        + uniform(
            low=-((abs(min_float32) / (size[0] * size[1] * 2)) ** (1 / 2)),
            high=(abs(max_float32) / (size[0] * size[1] * 2)) ** (1 / 2),
            size=size,
        )
        * 1j
    )
    cmplx128_from_float64 = (
        uniform(
            low=-((abs(min_float64) / (size[0] * size[1] * 2)) ** (1 / 2)),
            high=(abs(max_float64) / (size[0] * size[1] * 2)) ** (1 / 2),
            size=size,
        )
        + uniform(
            low=-((abs(min_float64) / (size[0] * size[1] * 2)) ** (1 / 2)),
            high=(abs(max_float64) / (size[0] * size[1] * 2)) ** (1 / 2),
            size=size,
        )
        * 1j
    )
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = np.complex128(cmplx128_from_float64)

    epyccel_func = epyc_numpy_funcs_mod.numpy_norm_array_like_2d

    assert np.allclose(epyccel_func(bl), get_norm(bl), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(integer8), get_norm(integer8), rtol=RTOL, atol=ATOL)
    assert np.allclose(
        epyccel_func(integer16), get_norm(integer16), rtol=RTOL, atol=ATOL
    )
    assert np.allclose(epyccel_func(integer), get_norm(integer), rtol=RTOL, atol=ATOL)
    assert np.allclose(
        epyccel_func(integer32), get_norm(integer32), rtol=RTOL, atol=ATOL
    )
    assert np.allclose(
        epyccel_func(integer64), get_norm(integer64), rtol=RTOL, atol=ATOL
    )
    assert np.allclose(epyccel_func(fl), get_norm(fl), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(fl32), get_norm(fl32), rtol=RTOL32, atol=ATOL32)
    assert np.allclose(epyccel_func(fl64), get_norm(fl64), rtol=RTOL, atol=ATOL)
    assert np.allclose(
        epyccel_func(cmplx64), get_norm(cmplx64), rtol=RTOL32, atol=ATOL32
    )
    assert np.allclose(epyccel_func(cmplx128), get_norm(cmplx128), rtol=RTOL, atol=ATOL)


def test_numpy_norm_array_like_2d_fortran_order(epyc_numpy_funcs_mod):

    get_norm = numpy_funcs.numpy_norm_array_like_2d_fortran_order

    size = (2, 5)

    bl = randint(0, 2, size=size, dtype=bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=np.int64)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(
        low=-((abs(min_float) / (size[0] * size[1])) ** (1 / 2)),
        high=(abs(max_float) / (size[0] * size[1])) ** (1 / 2),
        size=size,
    )
    fl32 = uniform(
        low=-((abs(min_float32) / (size[0] * size[1])) ** (1 / 2)),
        high=(abs(max_float32) / (size[0] * size[1])) ** (1 / 2),
        size=size,
    )
    fl32 = np.float32(fl32)
    fl64 = uniform(
        low=-((abs(min_float64) / (size[0] * size[1])) ** (1 / 2)),
        high=(abs(max_float64) / (size[0] * size[1])) ** (1 / 2),
        size=size,
    )

    cmplx128_from_float32 = (
        uniform(
            low=-((abs(min_float32) / (size[0] * size[1] * 2)) ** (1 / 2)),
            high=(abs(max_float32) / (size[0] * size[1] * 2)) ** (1 / 2),
            size=size,
        )
        + uniform(
            low=-((abs(min_float32) / (size[0] * size[1] * 2)) ** (1 / 2)),
            high=(abs(max_float32) / (size[0] * size[1] * 2)) ** (1 / 2),
            size=size,
        )
        * 1j
    )
    cmplx128_from_float64 = (
        uniform(
            low=-((abs(min_float64) / (size[0] * size[1] * 2)) ** (1 / 2)),
            high=(abs(max_float64) / (size[0] * size[1] * 2)) ** (1 / 2),
            size=size,
        )
        + uniform(
            low=-((abs(min_float64) / (size[0] * size[1] * 2)) ** (1 / 2)),
            high=(abs(max_float64) / (size[0] * size[1] * 2)) ** (1 / 2),
            size=size,
        )
        * 1j
    )
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = np.complex128(cmplx128_from_float64)

    epyccel_func = epyc_numpy_funcs_mod.numpy_norm_array_like_2d_fortran_order

    # re-ordering to Fortran order
    bl = np.ndarray(size, buffer=bl, order="F", dtype=bool)
    integer8 = np.ndarray(size, buffer=integer8, order="F", dtype=np.int8)
    integer16 = np.ndarray(size, buffer=integer16, order="F", dtype=np.int16)
    integer = np.ndarray(size, buffer=integer, order="F", dtype=int)
    integer32 = np.ndarray(size, buffer=integer32, order="F", dtype=np.int32)
    integer64 = np.ndarray(size, buffer=integer64, order="F", dtype=np.int64)
    fl = np.ndarray(size, buffer=fl, order="F", dtype=float)
    fl32 = np.ndarray(size, buffer=fl32, order="F", dtype=np.float32)
    fl64 = np.ndarray(size, buffer=fl64, order="F", dtype=np.float64)
    cmplx64 = np.ndarray(size, buffer=cmplx64, order="F", dtype=np.complex64)
    cmplx128 = np.ndarray(size, buffer=cmplx128, order="F", dtype=np.complex128)

    assert np.allclose(epyccel_func(bl), get_norm(bl), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(integer8), get_norm(integer8), rtol=RTOL, atol=ATOL)
    assert np.allclose(
        epyccel_func(integer16), get_norm(integer16), rtol=RTOL, atol=ATOL
    )
    assert np.allclose(epyccel_func(integer), get_norm(integer), rtol=RTOL, atol=ATOL)
    assert np.allclose(
        epyccel_func(integer32), get_norm(integer32), rtol=RTOL, atol=ATOL
    )
    assert np.allclose(
        epyccel_func(integer64), get_norm(integer64), rtol=RTOL, atol=ATOL
    )
    assert np.allclose(epyccel_func(fl), get_norm(fl), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(fl32), get_norm(fl32), rtol=RTOL32, atol=ATOL32)
    assert np.allclose(epyccel_func(fl64), get_norm(fl64), rtol=RTOL, atol=ATOL)
    assert np.allclose(
        epyccel_func(cmplx64), get_norm(cmplx64), rtol=RTOL32, atol=ATOL32
    )
    assert np.allclose(epyccel_func(cmplx128), get_norm(cmplx128), rtol=RTOL, atol=ATOL)


def test_numpy_norm_array_like_3d(epyc_numpy_funcs_mod):

    get_norm = numpy_funcs.numpy_norm_array_like_3d

    size = (2, 5, 5)

    bl = randint(0, 2, size=size, dtype=bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=np.int64)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(
        low=-((abs(min_float) / (size[0] * size[1] * size[2])) ** (1 / 2)),
        high=(abs(max_float) / (size[0] * size[1] * size[2])) ** (1 / 2),
        size=size,
    )
    fl32 = uniform(
        low=-((abs(min_float32) / (size[0] * size[1] * size[2])) ** (1 / 2)),
        high=(abs(max_float32) / (size[0] * size[1] * size[2])) ** (1 / 2),
        size=size,
    )
    fl32 = np.float32(fl32)
    fl64 = uniform(
        low=-((abs(min_float64) / (size[0] * size[1] * size[2])) ** (1 / 2)),
        high=(abs(max_float64) / (size[0] * size[1] * size[2])) ** (1 / 2),
        size=size,
    )

    cmplx128_from_float32 = (
        uniform(
            low=-((abs(min_float32) / (size[0] * size[1] * size[2] * 2)) ** (1 / 2)),
            high=(abs(max_float32) / (size[0] * size[1] * size[2] * 2)) ** (1 / 2),
            size=size,
        )
        + uniform(
            low=-((abs(min_float32) / (size[0] * size[1] * size[2] * 2)) ** (1 / 2)),
            high=(abs(max_float32) / (size[0] * size[1] * size[2] * 2)) ** (1 / 2),
            size=size,
        )
        * 1j
    )
    cmplx128_from_float64 = (
        uniform(
            low=-((abs(min_float64) / (size[0] * size[1] * size[2] * 2)) ** (1 / 2)),
            high=(abs(max_float64) / (size[0] * size[1] * size[2] * 2)) ** (1 / 2),
            size=size,
        )
        + uniform(
            low=-((abs(min_float64) / (size[0] * size[1] * size[2] * 2)) ** (1 / 2)),
            high=(abs(max_float64) / (size[0] * size[1] * size[2] * 2)) ** (1 / 2),
            size=size,
        )
        * 1j
    )
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = np.complex128(cmplx128_from_float64)

    epyccel_func = epyc_numpy_funcs_mod.numpy_norm_array_like_3d

    assert np.allclose(epyccel_func(bl), get_norm(bl), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(integer8), get_norm(integer8), rtol=RTOL, atol=ATOL)
    assert np.allclose(
        epyccel_func(integer16), get_norm(integer16), rtol=RTOL, atol=ATOL
    )
    assert np.allclose(epyccel_func(integer), get_norm(integer), rtol=RTOL, atol=ATOL)
    assert np.allclose(
        epyccel_func(integer32), get_norm(integer32), rtol=RTOL, atol=ATOL
    )
    assert np.allclose(
        epyccel_func(integer64), get_norm(integer64), rtol=RTOL, atol=ATOL
    )
    assert np.allclose(epyccel_func(fl), get_norm(fl), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(fl32), get_norm(fl32), rtol=RTOL32, atol=ATOL32)
    assert np.allclose(epyccel_func(fl64), get_norm(fl64), rtol=RTOL, atol=ATOL)
    assert np.allclose(
        epyccel_func(cmplx64), get_norm(cmplx64), rtol=RTOL32, atol=ATOL32
    )
    assert np.allclose(epyccel_func(cmplx128), get_norm(cmplx128), rtol=RTOL, atol=ATOL)


def test_numpy_norm_array_like_3d_fortran_order(epyc_numpy_funcs_mod):

    get_norm = numpy_funcs.numpy_norm_array_like_3d_fortran_order

    size = (2, 5, 5)

    bl = randint(0, 2, size=size, dtype=bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=np.int64)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(
        low=-((abs(min_float) / (size[0] * size[1] * size[2])) ** (1 / 2)),
        high=(abs(max_float) / (size[0] * size[1] * size[2])) ** (1 / 2),
        size=size,
    )
    fl32 = uniform(
        low=-((abs(min_float32) / (size[0] * size[1] * size[2])) ** (1 / 2)),
        high=(abs(max_float32) / (size[0] * size[1] * size[2])) ** (1 / 2),
        size=size,
    )
    fl32 = np.float32(fl32)
    fl64 = uniform(
        low=-((abs(min_float64) / (size[0] * size[1] * size[2])) ** (1 / 2)),
        high=(abs(max_float64) / (size[0] * size[1] * size[2])) ** (1 / 2),
        size=size,
    )

    cmplx128_from_float32 = (
        uniform(
            low=-((abs(min_float32) / (size[0] * size[1] * size[2] * 2)) ** (1 / 2)),
            high=(abs(max_float32) / (size[0] * size[1] * size[2] * 2)) ** (1 / 2),
            size=size,
        )
        + uniform(
            low=-((abs(min_float32) / (size[0] * size[1] * size[2] * 2)) ** (1 / 2)),
            high=(abs(max_float32) / (size[0] * size[1] * size[2] * 2)) ** (1 / 2),
            size=size,
        )
        * 1j
    )
    cmplx128_from_float64 = (
        uniform(
            low=-((abs(min_float64) / (size[0] * size[1] * size[2] * 2)) ** (1 / 2)),
            high=(abs(max_float64) / (size[0] * size[1] * size[2] * 2)) ** (1 / 2),
            size=size,
        )
        + uniform(
            low=-((abs(min_float64) / (size[0] * size[1] * size[2] * 2)) ** (1 / 2)),
            high=(abs(max_float64) / (size[0] * size[1] * size[2] * 2)) ** (1 / 2),
            size=size,
        )
        * 1j
    )
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = np.complex128(cmplx128_from_float64)

    epyccel_func = epyc_numpy_funcs_mod.numpy_norm_array_like_3d_fortran_order

    # re-ordering to Fortran order
    bl = np.ndarray(size, buffer=bl, order="F", dtype=bool)
    integer8 = np.ndarray(size, buffer=integer8, order="F", dtype=np.int8)
    integer16 = np.ndarray(size, buffer=integer16, order="F", dtype=np.int16)
    integer = np.ndarray(size, buffer=integer, order="F", dtype=int)
    integer32 = np.ndarray(size, buffer=integer32, order="F", dtype=np.int32)
    integer64 = np.ndarray(size, buffer=integer64, order="F", dtype=np.int64)
    fl = np.ndarray(size, buffer=fl, order="F", dtype=float)
    fl32 = np.ndarray(size, buffer=fl32, order="F", dtype=np.float32)
    fl64 = np.ndarray(size, buffer=fl64, order="F", dtype=np.float64)
    cmplx64 = np.ndarray(size, buffer=cmplx64, order="F", dtype=np.complex64)
    cmplx128 = np.ndarray(size, buffer=cmplx128, order="F", dtype=np.complex128)

    assert np.allclose(epyccel_func(bl), get_norm(bl), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(integer8), get_norm(integer8), rtol=RTOL, atol=ATOL)
    assert np.allclose(
        epyccel_func(integer16), get_norm(integer16), rtol=RTOL, atol=ATOL
    )
    assert np.allclose(epyccel_func(integer), get_norm(integer), rtol=RTOL, atol=ATOL)
    assert np.allclose(
        epyccel_func(integer32), get_norm(integer32), rtol=RTOL, atol=ATOL
    )
    assert np.allclose(
        epyccel_func(integer64), get_norm(integer64), rtol=RTOL, atol=ATOL
    )
    assert np.allclose(epyccel_func(fl), get_norm(fl), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(fl32), get_norm(fl32), rtol=RTOL32, atol=ATOL32)
    assert np.allclose(epyccel_func(fl64), get_norm(fl64), rtol=RTOL, atol=ATOL)
    assert np.allclose(
        epyccel_func(cmplx64), get_norm(cmplx64), rtol=RTOL32, atol=ATOL32
    )
    assert np.allclose(epyccel_func(cmplx128), get_norm(cmplx128), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("order", [0, 1, 2, -1, np.inf, -np.inf, 10, 2.2])
def test_norm_vector_ord(language, order):
    def norm_call(x: "float[:]"):
        from numpy.linalg import norm

        return norm(x, ord=order)

    f1 = epyccel(norm_call, language=language)
    x = rand(12) * 200 - 100
    assert np.allclose(f1(x), norm_call(x), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("order", [0, 1, 2, -1, np.inf, -np.inf, 10, 2.2])
def test_norm_vector_ord_complex(language, order):
    def norm_call(x: "complex[:]"):
        from numpy.linalg import norm

        return norm(x, ord=order)

    f1 = epyccel(norm_call, language=language)
    x = rand(12) * 200 - 100 + rand(12) * 1j * 200 - 100j
    assert np.allclose(f1(x), norm_call(x), rtol=RTOL, atol=ATOL)


def test_norm_axis_2d(epyc_numpy_funcs_mod):
    norm_call = numpy_funcs.norm_axis_2d

    f1 = epyc_numpy_funcs_mod.norm_axis_2d
    x = rand(5, 7)
    assert np.allclose(f1(x), norm_call(x), rtol=RTOL, atol=ATOL)


def test_norm_axis_keepdims(epyc_numpy_funcs_mod):
    norm_call = numpy_funcs.norm_axis_keepdims

    f1 = epyc_numpy_funcs_mod.norm_axis_keepdims
    x = rand(6, 4)
    res_ref = norm_call(x)
    res_cc = f1(x)
    assert np.allclose(res_cc, res_ref, rtol=RTOL, atol=ATOL)
    assert res_cc.shape == res_ref.shape


@pytest.mark.xfail(
    os.environ.get("PYCCEL_DEFAULT_COMPILER", None) == "intel",
    reason="Boolean conversion. See #1670",
)
def test_numpy_matmul_array_like_1d(epyc_numpy_funcs_mod):

    get_matmul = numpy_funcs.numpy_matmul_array_like_1d

    size = 5

    integer = randint(min_int, max_int, size=size, dtype=np.int64)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(
        -((max_float / size) ** (1 / 2)), (max_float / size) ** (1 / 2), size=size
    )
    fl32 = uniform(
        -((max_float32 / size) ** (1 / 2)), (max_float32 / size) ** (1 / 2), size=size
    )
    fl32 = np.float32(fl32)
    fl64 = uniform(
        -((max_float64 / size) ** (1 / 2)), (max_float64 / size) ** (1 / 2), size=size
    )

    cmplx128_from_float32 = (
        uniform(
            low=-((max_float32 / (size * 2)) ** (1 / 2)),
            high=(max_float32 / (size * 2)) ** (1 / 2),
            size=size,
        )
        + uniform(
            low=-((max_float32 / (size * 2)) ** (1 / 2)),
            high=(max_float32 / (size * 2)) ** (1 / 2),
            size=size,
        )
        * 1j
    )
    cmplx128_from_float64 = (
        uniform(
            low=-((max_float64 / (size * 2)) ** (1 / 2)),
            high=(max_float64 / (size * 2)) ** (1 / 2),
            size=size,
        )
        + uniform(
            low=-((max_float64 / (size * 2)) ** (1 / 2)),
            high=(max_float64 / (size * 2)) ** (1 / 2),
            size=size,
        )
        * 1j
    )
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = np.complex128(cmplx128_from_float64)

    epyccel_func = epyc_numpy_funcs_mod.numpy_matmul_array_like_1d

    assert np.array_equal(epyccel_func(integer), get_matmul(integer))
    assert np.array_equal(epyccel_func(integer32), get_matmul(integer32))
    assert np.array_equal(epyccel_func(integer64), get_matmul(integer64))
    assert isclose(epyccel_func(fl), get_matmul(fl), rtol=RTOL, atol=ATOL)
    assert isclose(epyccel_func(fl32), get_matmul(fl32), rtol=RTOL32, atol=ATOL32)
    assert isclose(epyccel_func(fl64), get_matmul(fl64), rtol=RTOL, atol=ATOL)
    assert isclose(epyccel_func(cmplx64), get_matmul(cmplx64), rtol=RTOL32, atol=ATOL32)
    assert isclose(epyccel_func(cmplx128), get_matmul(cmplx128), rtol=RTOL, atol=ATOL)


def test_numpy_matmul_array_like_2x2d(epyc_numpy_funcs_mod):

    get_matmul = numpy_funcs.numpy_matmul_array_like_2x2d

    size = (2, 2)

    def calculate_max_values(min_for_type, max_for_type):
        cast = type(min_for_type)
        min_test = -np.sqrt(abs(min_for_type) / size[0])
        max_test = np.sqrt(abs(max_for_type) / size[0])
        return cast(min_test), cast(max_test)

    integer = randint(
        *calculate_max_values(min_int, max_int), size=size, dtype=np.int64
    )
    integer32 = randint(
        *calculate_max_values(min_int32, max_int32), size=size, dtype=np.int32
    )
    integer64 = randint(
        *calculate_max_values(min_int64, max_int64), size=size, dtype=np.int64
    )

    fl = uniform(*calculate_max_values(min_float, max_float), size=size)
    fl32 = uniform(*calculate_max_values(min_float32, max_float32), size=size)
    fl32 = np.float32(fl32)
    fl64 = uniform(*calculate_max_values(min_float64, max_float64), size=size)

    cmplx128_from_float32 = (
        uniform(*calculate_max_values(min_int, max_int), size=size)
        + uniform(*calculate_max_values(min_int, max_int), size=size) * 1j
    )
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = (
        uniform(*calculate_max_values(min_int, max_int), size=size)
        + uniform(*calculate_max_values(min_int, max_int), size=size) * 1j
    )

    integer = np.full(size, calculate_max_values(min_int, max_int)[1])
    integer32 = np.full(size, calculate_max_values(min_int32, max_int32)[1])
    integer64 = np.full(size, calculate_max_values(min_int64, max_int64)[1])

    fl = np.full(size, calculate_max_values(min_float, max_float)[1])
    fl32 = np.full(size, calculate_max_values(min_float32, max_float32)[1])
    fl64 = np.full(size, calculate_max_values(min_float64, max_float64)[1])

    cmplx64 = np.full(size, np.complex64(integer + integer * 1j))
    cmplx128 = np.full(size, integer + integer * 1j)

    epyccel_func = epyc_numpy_funcs_mod.numpy_matmul_array_like_2x2d

    assert np.allclose(epyccel_func(integer), get_matmul(integer), rtol=RTOL, atol=ATOL)
    assert np.allclose(
        epyccel_func(integer32), get_matmul(integer32), rtol=RTOL, atol=ATOL
    )
    assert np.allclose(
        epyccel_func(integer64), get_matmul(integer64), rtol=RTOL, atol=ATOL
    )
    assert np.allclose(epyccel_func(fl), get_matmul(fl), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(fl32), get_matmul(fl32), rtol=RTOL32, atol=ATOL32)
    assert np.allclose(epyccel_func(fl64), get_matmul(fl64), rtol=RTOL, atol=ATOL)
    assert np.allclose(
        epyccel_func(cmplx64), get_matmul(cmplx64), rtol=RTOL32, atol=ATOL32
    )
    assert np.allclose(
        epyccel_func(cmplx128), get_matmul(cmplx128), rtol=RTOL, atol=ATOL
    )


def test_matmul_4d_multi_batch(epyc_numpy_funcs_mod):
    matmul_call = numpy_funcs.matmul_4d_multi_batch

    f1 = epyc_numpy_funcs_mod.matmul_4d_multi_batch

    # Two batch dimensions
    a = rand(2, 3, 4, 6)
    b = rand(2, 3, 6, 5)

    res_ref = matmul_call(a, b)
    res_pycc = f1(a, b)

    assert res_pycc.shape == res_ref.shape
    assert np.allclose(res_pycc, res_ref, rtol=RTOL, atol=ATOL)


def test_matmul_3d_broadcast_batch(epyc_numpy_funcs_mod):
    matmul_call = numpy_funcs.matmul_3d_broadcast_batch

    f1 = epyc_numpy_funcs_mod.matmul_3d_broadcast_batch

    # a has batch dimension, b is shared
    a = rand(5, 4, 3)
    b = rand(3, 2)

    res_ref = matmul_call(a, b)
    res_pycc = f1(a, b)

    assert res_pycc.shape == res_ref.shape
    assert np.allclose(res_pycc, res_ref, rtol=RTOL, atol=ATOL)


def test_numpy_where_array_like_1d_with_condition(epyc_numpy_funcs_mod):

    get_chosen_elements = numpy_funcs.numpy_where_array_like_1d_with_condition

    size = 5

    bl = randint(0, 2, size=size, dtype=bool)

    integer8 = randint(min_int8 // 2, max_int8 // 2, size=size, dtype=np.int8)
    integer16 = randint(min_int16 // 2, max_int16 // 2, size=size, dtype=np.int16)
    integer = randint(min_int // 2, max_int // 2, size=size, dtype=np.int64)
    integer32 = randint(min_int32 // 2, max_int32 // 2, size=size, dtype=np.int32)
    integer64 = randint(min_int64 // 2, max_int64 // 2, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size=size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size=size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size=size)

    epyccel_func = epyc_numpy_funcs_mod.numpy_where_array_like_1d_with_condition

    assert epyccel_func(bl) == get_chosen_elements(bl)
    assert epyccel_func(integer8) == get_chosen_elements(integer8)
    assert epyccel_func(integer16) == get_chosen_elements(integer16)
    assert epyccel_func(integer) == get_chosen_elements(integer)
    assert epyccel_func(integer32) == get_chosen_elements(integer32)
    assert epyccel_func(integer64) == get_chosen_elements(integer64)
    assert epyccel_func(fl) == get_chosen_elements(fl)
    assert epyccel_func(fl32) == get_chosen_elements(fl32)
    assert epyccel_func(fl64) == get_chosen_elements(fl64)


@pytest.mark.parametrize(
    "language",
    (
        pytest.param("fortran", marks=pytest.mark.fortran),
        pytest.param(
            "c",
            marks=[pytest.mark.skip(reason="nonzero not implemented"), pytest.mark.c],
        ),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_numpy_where_array_like_1d_1_arg(language):

    def get_chosen_elements(arr: "S[:]"):
        from numpy import shape, where

        a = where(arr > 5)
        s = shape(a)
        return len(s), s[1], a[0][1], a[0][0]

    # Arrays must have at least 2 elements larger than 5 to avoid IndexError
    integer8 = np.array([6, 1, 8, 2, 3], dtype=np.int8)
    integer16 = np.array([6, 1, 8, 2, 3], dtype=np.int16)
    integer = np.array([6, 1, 8, 2, 3], dtype=int)
    integer32 = np.array([6, 1, 8, 2, 3], dtype=np.int32)
    integer64 = np.array([6, 1, 8, 2, 3], dtype=np.int64)

    fl = np.array([6, 22, 1, 8, 2, 3], dtype=float)
    fl32 = np.array([6, 22, 1, 8, 2, 3], dtype=np.float32)
    fl64 = np.array([6, 22, 1, 8, 2, 3], dtype=np.float64)

    epyccel_func = epyccel(get_chosen_elements, language=language)

    assert epyccel_func(integer8) == get_chosen_elements(integer8)
    assert epyccel_func(integer16) == get_chosen_elements(integer16)
    assert epyccel_func(integer) == get_chosen_elements(integer)
    assert epyccel_func(integer32) == get_chosen_elements(integer32)
    assert epyccel_func(integer64) == get_chosen_elements(integer64)
    assert epyccel_func(fl) == get_chosen_elements(fl)
    assert epyccel_func(fl32) == get_chosen_elements(fl32)
    assert epyccel_func(fl64) == get_chosen_elements(fl64)


def test_numpy_where_array_like_2d_with_condition(epyc_numpy_funcs_mod):

    get_chosen_elements = numpy_funcs.numpy_where_array_like_2d_with_condition

    size = (2, 5)

    bl = randint(0, 2, size=size, dtype=bool)

    integer8 = randint(min_int8, max_int8 - 1, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16 - 1, size=size, dtype=np.int16)
    integer = randint(min_int, max_int - 1, size=size, dtype=np.int64)
    integer32 = randint(min_int32, max_int32 - 1, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64 - 1, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size=size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size=size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size=size)

    epyccel_func = epyc_numpy_funcs_mod.numpy_where_array_like_2d_with_condition

    assert epyccel_func(bl) == get_chosen_elements(bl)
    assert epyccel_func(integer8) == get_chosen_elements(integer8)
    assert epyccel_func(integer16) == get_chosen_elements(integer16)
    assert epyccel_func(integer) == get_chosen_elements(integer)
    assert epyccel_func(integer32) == get_chosen_elements(integer32)
    assert epyccel_func(integer64) == get_chosen_elements(integer64)
    assert epyccel_func(fl) == get_chosen_elements(fl)
    assert epyccel_func(fl32) == get_chosen_elements(fl32)
    assert epyccel_func(fl64) == get_chosen_elements(fl64)


def test_numpy_where_complex(epyc_numpy_funcs_mod):
    where_wrapper = numpy_funcs.numpy_where_complex

    size = 7

    cond = randint(0, 1, size=size, dtype=bool)

    cmplx128_from_float32_1 = (
        uniform(low=min_float32 / 2, high=max_float32 / 2, size=size)
        + uniform(low=min_float32 / 2, high=max_float32 / 2, size=size) * 1j
    )
    cmplx128_from_float32_2 = (
        uniform(low=min_float32 / 2, high=max_float32 / 2, size=size)
        + uniform(low=min_float32 / 2, high=max_float32 / 2, size=size) * 1j
    )
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64_1 = np.complex64(cmplx128_from_float32_1)
    cmplx64_2 = np.complex64(cmplx128_from_float32_2)
    cmplx128_1 = (
        uniform(low=min_float64 / 2, high=max_float64 / 2, size=size)
        + uniform(low=min_float64 / 2, high=max_float64 / 2, size=size) * 1j
    )
    cmplx128_2 = (
        uniform(low=min_float64 / 2, high=max_float64 / 2, size=size)
        + uniform(low=min_float64 / 2, high=max_float64 / 2, size=size) * 1j
    )

    epyccel_func = epyc_numpy_funcs_mod.numpy_where_complex

    assert epyccel_func(cmplx64_1, cmplx64_2, cond) == where_wrapper(
        cmplx64_1, cmplx64_2, cond
    )
    assert epyccel_func(cmplx128_1, cmplx128_2, cond) == where_wrapper(
        cmplx128_1, cmplx128_2, cond
    )


def test_where_combined_types(epyc_numpy_funcs_mod):
    where_wrapper = numpy_funcs.where_combined_types

    size = 6

    cond = randint(0, 1, size=size, dtype=bool)

    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    float32 = uniform(min_float32, max_float32, size=size)
    float32 = np.float32(float32)
    float64 = uniform(min_float64 / 2, max_float64 / 2, size=size)

    complex128 = (
        uniform(low=min_float64 / 2, high=max_float64 / 2, size=size)
        + uniform(low=min_float64 / 2, high=max_float64 / 2, size=size) * 1j
    )

    epyccel_func = epyc_numpy_funcs_mod.where_combined_types

    res_pyc = epyccel_func(cond, integer32, integer64)
    res_pyt = where_wrapper(cond, integer32, integer64)
    assert res_pyc == res_pyt
    assert matching_types(res_pyc, res_pyt)
    res_pyc = epyccel_func(cond, integer32, float32)
    res_pyt = where_wrapper(cond, integer32, float32)
    assert res_pyc == res_pyt
    assert matching_types(res_pyc, res_pyt)
    res_pyc = epyccel_func(cond, float64, integer64)
    res_pyt = where_wrapper(cond, float64, integer64)
    assert res_pyc == res_pyt
    assert matching_types(res_pyc, res_pyt)
    res_pyc = epyccel_func(cond, complex128, integer64)
    res_pyt = where_wrapper(cond, complex128, integer64)
    assert res_pyc == res_pyt
    assert matching_types(res_pyc, res_pyt)


def test_numpy_linspace_scalar(epyc_numpy_funcs_mod):
    from numpy import linspace

    get_linspace = numpy_funcs.numpy_linspace_scalar__get_linspace
    test_linspace_int = numpy_funcs.numpy_linspace_scalar__test_linspace_int
    test_linspace = numpy_funcs.numpy_linspace_scalar__test_linspace

    integer8 = randint(min_int8, max_int8 // 2, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=np.int64)
    integer32 = randint(min_int32, max_int32, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, dtype=np.int64)

    fl = uniform(min_float / 200, max_float / 200)
    fl32 = uniform(min_float32 / 200, max_float32 / 200)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 200, max_float64 / 200)

    epyccel_func = epyc_numpy_funcs_mod.numpy_linspace_scalar__get_linspace
    epyccel_func_type = epyc_numpy_funcs_mod.numpy_linspace_scalar__test_linspace_type
    epyccel_func_type2 = epyc_numpy_funcs_mod.numpy_linspace_scalar__test_linspace_type2
    epyccel_func_int = epyc_numpy_funcs_mod.numpy_linspace_scalar__test_linspace_int

    x = linspace(0 + 4, 10, 15, dtype=np.int64)
    ret, ele = epyccel_func_type(0, 10, x)
    assert ret == 1
    assert ele.dtype == np.int64
    x = linspace(0, 10 * 2, 15, dtype="complex128")
    out = np.empty_like(x)
    epyccel_func_type2(0, 10, out)
    assert np.allclose(x, out)
    x = randint(1, 60)
    assert np.allclose(
        epyccel_func(integer8, x, 30),
        get_linspace(integer8, x, 30),
        rtol=RTOL * 10,
        atol=ATOL * 10,
    )
    assert matching_types(
        epyccel_func(integer8, x, 100)[0], get_linspace(integer8, x, 100)[0]
    )
    x = randint(100, 200)
    assert np.allclose(
        epyccel_func(integer, x, 30),
        get_linspace(integer, x, 30),
        rtol=RTOL * 10,
        atol=ATOL * 10,
    )
    assert matching_types(
        epyccel_func(integer, x, 100)[0], get_linspace(integer, x, 100)[0]
    )
    x = randint(100, 200)
    assert np.allclose(
        epyccel_func(integer16, x, 30),
        get_linspace(integer16, x, 30),
        rtol=RTOL * 10,
        atol=ATOL * 10,
    )
    assert matching_types(
        epyccel_func(integer16, x, 100)[0], get_linspace(integer16, x, 100)[0]
    )
    x = randint(100, 200)
    assert np.allclose(
        epyccel_func(integer32, x, 30),
        get_linspace(integer32, x, 30),
        rtol=RTOL * 10,
        atol=ATOL * 10,
    )
    assert matching_types(
        epyccel_func(integer32, x, 100)[0], get_linspace(integer32, x, 100)[0]
    )
    x = randint(100, 200)
    assert np.allclose(
        epyccel_func(integer64, x, 200),
        get_linspace(integer64, x, 200),
        rtol=RTOL * 10,
        atol=ATOL * 10,
    )
    assert matching_types(
        epyccel_func(integer64, x, 100)[0], get_linspace(integer64, x, 100)[0]
    )
    x = randint(100, 200)
    assert np.allclose(
        epyccel_func(fl, x, 100),
        get_linspace(fl, x, 100),
        rtol=RTOL * 10,
        atol=ATOL * 10,
    )
    assert matching_types(epyccel_func(fl, x, 100)[0], get_linspace(fl, x, 100)[0])
    x = randint(100, 200)
    assert np.allclose(
        epyccel_func(fl32, x, 200),
        get_linspace(fl32, x, 200),
        rtol=RTOL * 10,
        atol=ATOL * 10,
    )
    assert matching_types(epyccel_func(fl32, x, 100)[0], get_linspace(fl32, x, 100)[0])
    x = randint(100, 200)
    assert np.allclose(
        epyccel_func(fl64, x, 200),
        get_linspace(fl64, x, 200),
        rtol=RTOL * 10,
        atol=ATOL * 10,
    )
    assert matching_types(epyccel_func(fl64, x, 100)[0], get_linspace(fl64, x, 100)[0])

    assert np.allclose(
        epyccel_func_int(-393, 5, 7, False), test_linspace_int(-393, 5, 7, False)
    )
    assert matching_types(
        epyccel_func_int(-393, 5, 7, False), test_linspace_int(-393, 5, 7, False)
    )

    assert np.allclose(
        epyccel_func_int(-393.0, 5.0, 7, False),
        test_linspace_int(-393.0, 5.0, 7, False),
    )
    assert matching_types(
        epyccel_func_int(-393.0, 5.0, 7, False),
        test_linspace_int(-393.0, 5.0, 7, False),
    )

    epyccel_func1 = epyc_numpy_funcs_mod.numpy_linspace_scalar__test_linspace
    epyccel_func2 = epyc_numpy_funcs_mod.numpy_linspace_scalar__test_linspace2
    assert np.allclose(
        epyccel_func1(np.complex64(3 + 6j), np.complex64(5 + 1j)),
        test_linspace(np.complex64(3 + 6j), np.complex64(5 + 1j)),
        rtol=RTOL * 10,
        atol=ATOL * 10,
    )
    assert np.allclose(
        epyccel_func1(np.complex64(-3 + 6j), np.complex64(5 - 1j)),
        test_linspace(np.complex64(-3 + 6j), np.complex64(5 - 1j)),
        rtol=RTOL * 10,
        atol=ATOL * 10,
    )
    assert np.allclose(
        epyccel_func2(np.complex128(3 + 6j), np.complex128(5 + 1j)),
        test_linspace(np.complex128(3 + 6j), np.complex128(5 + 1j)),
        rtol=RTOL * 10,
        atol=ATOL * 10,
    )
    assert np.allclose(
        epyccel_func2(np.complex128(-3 + 6j), np.complex128(5 - 1j)),
        test_linspace(np.complex128(-3 + 6j), np.complex128(5 - 1j)),
        rtol=RTOL * 10,
        atol=ATOL * 10,
    )

    res_pyc = epyccel_func2(np.complex128(3 + 6j), np.complex128(5 + 1j))
    res_pyt = test_linspace(np.complex128(3 + 6j), np.complex128(5 + 1j))
    for pyc, pyt in zip(res_pyc, res_pyt):
        assert matching_types(pyc, pyt)


def test_numpy_linspace_array_like_1d(epyc_numpy_funcs_mod):
    from numpy import linspace

    size = 5
    integer8 = randint(min_int8 / 2, max_int8 / 2, size=size, dtype=np.int8)
    integer16 = randint(min_int16 / 2, max_int16 / 2, size=size, dtype=np.int16)
    integer = randint(-10000, 10000, size=size, dtype=np.int64)
    integer32 = randint(min_int32 / 2, max_int32 / 2, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl32 = np.array([1.5, 2.2, 3.3, 4.4, 5.5], dtype=np.float32)

    epyccel_func = epyc_numpy_funcs_mod.numpy_linspace_array_like_1d__test_linspace
    epyccel_func2 = epyc_numpy_funcs_mod.numpy_linspace_array_like_1d__test_linspace2

    epyccel_func_dtype = (
        epyc_numpy_funcs_mod.numpy_linspace_array_like_1d__test_linspace_dtype
    )

    arr = linspace(integer, 5, 7)
    out = epyccel_func(integer, 5, True)
    assert np.allclose(arr, out)
    assert isinstance(out[0, 0], type(arr[0, 0]))
    arr = linspace(integer, 5, 7, endpoint=False)
    out = epyccel_func(integer, 5, False)
    assert np.allclose(arr, out)
    assert isinstance(out[0, 0], type(arr[0, 0]))

    arr = linspace(integer8, 5, 7)
    out = epyccel_func(integer8, 5, True)
    assert np.allclose(arr, out)
    assert isinstance(out[0, 0], type(arr[0, 0]))
    arr = linspace(integer8, 5, 7, endpoint=False)
    out = epyccel_func(integer8, 5, False)
    assert np.allclose(arr, out)
    assert isinstance(out[0, 0], type(arr[0, 0]))

    arr = linspace(integer16, 5, 7)
    out = epyccel_func(integer16, 5, True)
    assert np.allclose(arr, out)
    assert isinstance(out[0, 0], type(arr[0, 0]))
    arr = linspace(integer16, 5, 7, endpoint=False)
    out = epyccel_func(integer16, 5, False)
    assert np.allclose(arr, out)
    assert isinstance(out[0, 0], type(arr[0, 0]))

    arr = linspace(integer32, 5, 7)
    out = epyccel_func(integer32, 5, True)
    assert np.allclose(arr, out)
    assert isinstance(out[0, 0], type(arr[0, 0]))
    arr = linspace(integer32, 5, 7, endpoint=False)
    out = epyccel_func(integer32, 5, False)
    assert np.allclose(arr, out)
    assert isinstance(out[0, 0], type(arr[0, 0]))

    if sys.platform != "win32":
        arr = linspace(integer64, 5, 7)
        out = epyccel_func(integer64, 5, True)
        assert np.allclose(arr, out)
        assert isinstance(out[0, 0], type(arr[0, 0]))
        arr = linspace(integer64, 5, 7, endpoint=False)
        out = epyccel_func(integer64, 5, False)
        assert np.allclose(arr, out)
        assert isinstance(out[0, 0], type(arr[0, 0]))

    arr = linspace(fl32, 5, 7)
    out = epyccel_func(fl32, 5, True)
    assert np.allclose(arr, out)
    assert isinstance(out[0, 0], type(arr[0, 0]))
    arr = linspace(fl32, 5, 7, endpoint=False)
    out = epyccel_func(fl32, 5, False)
    assert np.allclose(arr, out)
    assert isinstance(out[0, 0], type(arr[0, 0]))

    rng = np.random.default_rng()
    fl64 = rng.random((5,), dtype=np.float64)
    arr = linspace(fl64, 2, 7)
    out = epyccel_func(fl64, 2, True)
    assert np.allclose(arr, out)
    assert isinstance(out[0, 0], type(arr[0, 0]))
    arr = linspace(fl64, 5, 7, endpoint=False)
    out = epyccel_func(fl64, 5, False)
    assert np.allclose(arr, out)
    assert isinstance(out[0, 0], type(arr[0, 0]))

    cmplx = (np.random.random(5) * 75) + (np.random.random(5) * 50) * 1j
    arr = linspace(cmplx, 0, 7)
    out = np.empty_like(arr)
    epyccel_func2(cmplx, 0, out, True)
    assert np.allclose(arr, out)
    arr = linspace(cmplx, 0, 7, endpoint=False)
    out = np.empty_like(arr)
    epyccel_func2(cmplx, 0, out, False)
    assert np.allclose(arr, out)

    arr = linspace(fl64, 5, 7, dtype=np.int32)
    out = epyccel_func_dtype(fl64, 5, True)
    assert np.allclose(arr, out)
    assert isinstance(out[0, 0], type(arr[0, 0]))
    # Integer test does not work. See #2126
    # arr = linspace(integer, 5, 7, endpoint=False, dtype=np.int32)
    # out = epyccel_func_dtype(integer, 5, False)
    # assert isinstance(out[0,0], type(arr[0,0]))
    # assert np.allclose(arr, out)


def test_numpy_linspace_array_like_2d(epyc_numpy_funcs_mod):
    from numpy import linspace

    size = (2, 5)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=np.int64)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)
    fl32 = np.array(
        [[1.5, 2.2, 3.3, 4.4, 5.5], [5.4, 2.1, 7.1, 10.46, 11.0]], dtype=np.float32
    )
    cmplx = (np.random.random((2, 5)) * 75) + (np.random.random((2, 5)) * 50) * 1j

    epyccel_func = epyc_numpy_funcs_mod.numpy_linspace_array_like_2d__test_linspace
    epyccel_func3 = epyc_numpy_funcs_mod.numpy_linspace_array_like_2d__test_linspace3
    epyccel_func2 = epyc_numpy_funcs_mod.numpy_linspace_array_like_2d__test_linspace2
    epyccel_func4 = epyc_numpy_funcs_mod.numpy_linspace_array_like_2d__test_linspace4

    arr = linspace(integer, 5, 7)
    out = epyccel_func(integer, 5, True)
    assert np.allclose(arr, out)
    assert arr.dtype is out.dtype
    arr = linspace(integer, 5, 7, endpoint=False)
    out = epyccel_func(integer, 5, False)
    assert np.allclose(arr, out)
    assert arr.dtype is out.dtype
    arr = linspace(integer8, 5, 7)
    out = epyccel_func(integer8, 5, True)
    assert np.allclose(arr, out)
    assert arr.dtype is out.dtype
    arr = linspace(integer8, 5, 7, endpoint=False)
    out = epyccel_func(integer8, 5, False)
    assert np.allclose(arr, out)
    assert arr.dtype is out.dtype
    arr = linspace(integer16, 5, 7)
    out = epyccel_func(integer16, 5, True)
    assert np.allclose(arr, out)
    assert arr.dtype is out.dtype
    arr = linspace(integer16, 5, 7, endpoint=False)
    out = epyccel_func(integer16, 5, False)
    assert np.allclose(arr, out)
    assert arr.dtype is out.dtype
    arr = linspace(integer32, 5, 7)
    out = epyccel_func(integer32, 5, True)
    assert np.allclose(arr, out)
    assert arr.dtype is out.dtype
    arr = linspace(integer32, 5, 7, endpoint=False)
    out = epyccel_func(integer32, 5, False)
    assert np.allclose(arr, out)
    assert arr.dtype is out.dtype
    integer = randint(min_int / 2, max_int / 2, size=size, dtype=np.int64)
    integer_2 = np.array([[1, 2, 3, 4, 5], [5, 2, 7, 10, 11]], dtype=int)
    arr = linspace(integer, integer_2, 7, endpoint=False)
    out = np.empty_like(arr)
    epyccel_func2(integer, integer_2, out, False)
    assert np.allclose(arr, out)
    if sys.platform != "win32":
        arr = linspace(integer64, 5, 7)
        out = epyccel_func(integer64, 5, True)
        assert np.allclose(arr, out)
        assert arr.dtype is out.dtype
        arr = linspace(integer64, 5, 7, endpoint=False)
        out = epyccel_func(integer64, 5, False)
        assert np.allclose(arr, out)
        assert arr.dtype is out.dtype

    arr = linspace(fl32, 5, 7)
    out = epyccel_func(fl32, 5, True)
    assert np.allclose(arr, out)
    assert arr.dtype is out.dtype
    arr = linspace(fl32, 5, 7, endpoint=False)
    out = epyccel_func(fl32, 5, False)
    assert np.allclose(arr, out)
    assert arr.dtype is out.dtype
    rng = np.random.default_rng()
    fl64 = rng.random((2, 5), dtype=np.float64)
    arr = linspace(fl64, 5, 7)
    out = epyccel_func(fl64, 5, True)
    assert np.allclose(arr, out)
    assert arr.dtype is out.dtype
    arr = linspace(fl64, 5, 7, endpoint=False)
    out = epyccel_func(fl64, 5, False)
    assert np.allclose(arr, out)
    assert arr.dtype is out.dtype

    arr = linspace(cmplx, 5, 7)
    out = np.empty_like(arr)
    epyccel_func3(cmplx, 5, out, True)
    assert np.allclose(arr, out)
    arr = linspace(cmplx, 5, 7, endpoint=False)
    out = np.empty_like(arr)
    epyccel_func3(cmplx, 5, out, False)
    assert np.allclose(arr, out)
    cmplx = (np.random.random((2, 5)) * 55) + (np.random.random((2, 5)) * 50) * 1j
    cmplx2 = (np.random.random((2, 5)) * 14) + (np.random.random((2, 5)) * 15) * 1j
    arr = linspace(cmplx, cmplx2, 7)
    out = np.empty_like(arr)
    epyccel_func4(cmplx, cmplx2, out, True)
    assert np.allclose(arr, out)


@pytest.mark.parametrize(
    "language",
    (
        pytest.param("fortran", marks=pytest.mark.fortran),
        pytest.param(
            "c",
            marks=[
                pytest.mark.skip(reason="count_nonzero not implemented"),
                pytest.mark.c,
            ],
        ),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_numpy_count_non_zero_1d(language):
    def count(arr: "F[:]"):
        from numpy import count_nonzero

        return count_nonzero(arr)

    size = 5

    bl = randint(0, 2, size=size, dtype=bool)

    integer8 = randint(min_int8 // 2, max_int8 // 2, size=size, dtype=np.int8)
    integer16 = randint(min_int16 // 2, max_int16 // 2, size=size, dtype=np.int16)
    integer = randint(min_int // 2, max_int // 2, size=size, dtype=np.int64)
    integer32 = randint(min_int32 // 2, max_int32 // 2, size=size, dtype=np.int32)
    integer64 = randint(min_int64 // 2, max_int64 // 2, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size=size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size=size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size=size)

    epyccel_func = epyccel(count, language=language)

    assert epyccel_func(bl) == count(bl)
    assert epyccel_func(integer8) == count(integer8)
    assert epyccel_func(integer16) == count(integer16)
    assert epyccel_func(integer) == count(integer)
    assert epyccel_func(integer32) == count(integer32)
    assert epyccel_func(integer64) == count(integer64)
    assert epyccel_func(fl) == count(fl)
    assert epyccel_func(fl32) == count(fl32)
    assert epyccel_func(fl64) == count(fl64)


@pytest.mark.parametrize(
    "language",
    (
        pytest.param("fortran", marks=pytest.mark.fortran),
        pytest.param(
            "c",
            marks=[
                pytest.mark.skip(reason="count_nonzero not implemented"),
                pytest.mark.c,
            ],
        ),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_numpy_count_non_zero_2d(language):
    def count(arr: "F[:,:]"):
        from numpy import count_nonzero

        return count_nonzero(arr)

    size = (2, 5)

    bl = randint(0, 2, size=size, dtype=bool)

    integer8 = randint(min_int8, max_int8 - 1, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16 - 1, size=size, dtype=np.int16)
    integer = randint(min_int, max_int - 1, size=size, dtype=np.int64)
    integer32 = randint(min_int32, max_int32 - 1, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64 - 1, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size=size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size=size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size=size)

    epyccel_func = epyccel(count, language=language)

    assert epyccel_func(bl) == count(bl)
    assert epyccel_func(integer8) == count(integer8)
    assert epyccel_func(integer16) == count(integer16)
    assert epyccel_func(integer) == count(integer)
    assert epyccel_func(integer32) == count(integer32)
    assert epyccel_func(integer64) == count(integer64)
    assert epyccel_func(fl) == count(fl)
    assert epyccel_func(fl32) == count(fl32)
    assert epyccel_func(fl64) == count(fl64)


@pytest.mark.parametrize(
    "language",
    (
        pytest.param("fortran", marks=pytest.mark.fortran),
        pytest.param(
            "c",
            marks=[
                pytest.mark.skip(reason="count_nonzero not implemented"),
                pytest.mark.c,
            ],
        ),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_numpy_count_non_zero_1d_keep_dims(language):
    def count(arr: "F[:]"):
        from numpy import count_nonzero

        a = count_nonzero(arr, keepdims=True)
        s = a.shape
        return s[0], a[0]

    size = 5

    bl = randint(0, 2, size=size, dtype=bool)

    integer8 = randint(min_int8 // 2, max_int8 // 2, size=size, dtype=np.int8)
    integer16 = randint(min_int16 // 2, max_int16 // 2, size=size, dtype=np.int16)
    integer = randint(min_int // 2, max_int // 2, size=size, dtype=np.int64)
    integer32 = randint(min_int32 // 2, max_int32 // 2, size=size, dtype=np.int32)
    integer64 = randint(min_int64 // 2, max_int64 // 2, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size=size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size=size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size=size)

    epyccel_func = epyccel(count, language=language)

    assert epyccel_func(bl) == count(bl)
    assert epyccel_func(integer8) == count(integer8)
    assert epyccel_func(integer16) == count(integer16)
    assert epyccel_func(integer) == count(integer)
    assert epyccel_func(integer32) == count(integer32)
    assert epyccel_func(integer64) == count(integer64)
    assert epyccel_func(fl) == count(fl)
    assert epyccel_func(fl32) == count(fl32)
    assert epyccel_func(fl64) == count(fl64)


@pytest.mark.parametrize(
    "language",
    (
        pytest.param("fortran", marks=pytest.mark.fortran),
        pytest.param(
            "c",
            marks=[
                pytest.mark.skip(reason="count_nonzero not implemented"),
                pytest.mark.c,
            ],
        ),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_numpy_count_non_zero_2d_keep_dims(language):
    def count(arr: "F[:,:]"):
        from numpy import count_nonzero

        a = count_nonzero(arr, keepdims=True)
        s = a.shape
        return s[0], s[1], a[0, 0]

    size = (2, 5)

    bl = randint(0, 2, size=size, dtype=bool)

    integer8 = randint(min_int8, max_int8 - 1, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16 - 1, size=size, dtype=np.int16)
    integer = randint(min_int, max_int - 1, size=size, dtype=np.int64)
    integer32 = randint(min_int32, max_int32 - 1, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64 - 1, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size=size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size=size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size=size)

    epyccel_func = epyccel(count, language=language)

    assert epyccel_func(bl) == count(bl)
    assert epyccel_func(integer8) == count(integer8)
    assert epyccel_func(integer16) == count(integer16)
    assert epyccel_func(integer) == count(integer)
    assert epyccel_func(integer32) == count(integer32)
    assert epyccel_func(integer64) == count(integer64)
    assert epyccel_func(fl) == count(fl)
    assert epyccel_func(fl32) == count(fl32)
    assert epyccel_func(fl64) == count(fl64)


@pytest.mark.parametrize(
    "language",
    (
        pytest.param("fortran", marks=pytest.mark.fortran),
        pytest.param(
            "c",
            marks=[
                pytest.mark.skip(reason="count_nonzero not implemented"),
                pytest.mark.c,
            ],
        ),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_numpy_count_non_zero_axis(language):
    def count(arr: "F[:,:,:]"):
        from numpy import count_nonzero

        a = count_nonzero(arr, axis=1)
        s = a.shape
        return len(s), s[0], s[1], a[0, 0], a[0, -1]

    size = (2, 5, 3)

    bl = randint(0, 2, size=size, dtype=bool)

    integer8 = randint(min_int8, max_int8 - 1, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16 - 1, size=size, dtype=np.int16)
    integer = randint(min_int, max_int - 1, size=size, dtype=np.int64)
    integer32 = randint(min_int32, max_int32 - 1, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64 - 1, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size=size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size=size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size=size)

    epyccel_func = epyccel(count, language=language)

    assert epyccel_func(bl) == count(bl)
    assert epyccel_func(integer8) == count(integer8)
    assert epyccel_func(integer16) == count(integer16)
    assert epyccel_func(integer) == count(integer)
    assert epyccel_func(integer32) == count(integer32)
    assert epyccel_func(integer64) == count(integer64)
    assert epyccel_func(fl) == count(fl)
    assert epyccel_func(fl32) == count(fl32)
    assert epyccel_func(fl64) == count(fl64)


@pytest.mark.parametrize(
    "language",
    (
        pytest.param("fortran", marks=pytest.mark.fortran),
        pytest.param(
            "c",
            marks=[
                pytest.mark.skip(reason="count_nonzero not implemented"),
                pytest.mark.c,
            ],
        ),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_numpy_count_non_zero_axis_keep_dims(language):
    def count(arr: "F[:,:,:]"):
        from numpy import count_nonzero

        a = count_nonzero(arr, axis=0, keepdims=True)
        s = a.shape
        return len(s), s[0], s[1], s[2], a[0, 0, 0], a[0, 0, -1]

    size = (5, 2, 3)

    bl = randint(0, 2, size=size, dtype=bool)

    integer8 = randint(min_int8, max_int8 - 1, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16 - 1, size=size, dtype=np.int16)
    integer = randint(min_int, max_int - 1, size=size, dtype=np.int64)
    integer32 = randint(min_int32, max_int32 - 1, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64 - 1, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size=size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size=size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size=size)

    epyccel_func = epyccel(count, language=language)

    assert epyccel_func(bl) == count(bl)
    assert epyccel_func(integer8) == count(integer8)
    assert epyccel_func(integer16) == count(integer16)
    assert epyccel_func(integer) == count(integer)
    assert epyccel_func(integer32) == count(integer32)
    assert epyccel_func(integer64) == count(integer64)
    assert epyccel_func(fl) == count(fl)
    assert epyccel_func(fl32) == count(fl32)
    assert epyccel_func(fl64) == count(fl64)


@pytest.mark.parametrize(
    "language",
    (
        pytest.param("fortran", marks=pytest.mark.fortran),
        pytest.param(
            "c",
            marks=[
                pytest.mark.skip(reason="count_nonzero not implemented"),
                pytest.mark.c,
            ],
        ),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_numpy_count_non_zero_axis_keep_dims_F(language):
    def count(arr: "F[:,:,:](order=F)"):
        from numpy import count_nonzero

        a = count_nonzero(arr, axis=1, keepdims=True)
        s = a.shape
        return len(s), s[0], s[1], s[2], a[0, 0, 0], a[0, 0, -1]

    size = (2, 5, 3)

    bl = np.array(randint(0, 2, size=size), dtype=bool, order="F")

    integer8 = np.array(
        randint(min_int8, max_int8 - 1, size=size, dtype=np.int8), order="F"
    )
    integer16 = np.array(
        randint(min_int16, max_int16 - 1, size=size, dtype=np.int16), order="F"
    )
    integer = np.array(
        randint(min_int, max_int - 1, size=size), order="F", dtype=np.int64
    )
    integer32 = np.array(
        randint(min_int32, max_int32 - 1, size=size, dtype=np.int32), order="F"
    )
    integer64 = np.array(
        randint(min_int64, max_int64 - 1, size=size, dtype=np.int64), order="F"
    )

    fl = np.array(
        uniform(min_float / 2, max_float / 2, size=size), dtype=float, order="F"
    )
    fl32 = np.array(
        uniform(min_float32 / 2, max_float32 / 2, size=size),
        dtype=np.float32,
        order="F",
    )
    fl64 = np.array(
        uniform(min_float64 / 2, max_float64 / 2, size=size),
        dtype=np.float64,
        order="F",
    )

    epyccel_func = epyccel(count, language=language)

    assert epyccel_func(bl) == count(bl)
    assert epyccel_func(integer8) == count(integer8)
    assert epyccel_func(integer16) == count(integer16)
    assert epyccel_func(integer) == count(integer)
    assert epyccel_func(integer32) == count(integer32)
    assert epyccel_func(integer64) == count(integer64)
    assert epyccel_func(fl) == count(fl)
    assert epyccel_func(fl32) == count(fl32)
    assert epyccel_func(fl64) == count(fl64)


@pytest.mark.parametrize(
    "language",
    (
        pytest.param("fortran", marks=pytest.mark.fortran),
        pytest.param(
            "c",
            marks=[pytest.mark.skip(reason="nonzero not implemented"), pytest.mark.c],
        ),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_nonzero(language):

    def nonzero_func(a: "F[:]"):
        from numpy import nonzero

        b = nonzero(a)
        return len(b), b[0][0], b[0][1]

    # Arrays must have at least 2 non-zero elements to avoid IndexError
    bl = np.array([True, False, True, False, True])
    integer8 = np.array([6, 1, 8, 2, 3], dtype=np.int8)
    integer16 = np.array([6, 1, 8, 2, 3], dtype=np.int16)
    integer = np.array([6, 1, 8, 2, 3], dtype=int)
    integer32 = np.array([6, 1, 8, 2, 3], dtype=np.int32)
    integer64 = np.array([6, 1, 8, 2, 3], dtype=np.int64)

    fl = np.array([6, 22, 1, 8, 2, 3], dtype=float)
    fl32 = np.array([6, 22, 1, 8, 2, 3], dtype=np.float32)
    fl64 = np.array([6, 22, 1, 8, 2, 3], dtype=np.float64)

    epyccel_func = epyccel(nonzero_func, language=language)

    assert epyccel_func(bl) == nonzero_func(bl)
    assert epyccel_func(integer8) == nonzero_func(integer8)
    assert epyccel_func(integer16) == nonzero_func(integer16)
    assert epyccel_func(integer) == nonzero_func(integer)
    assert epyccel_func(integer32) == nonzero_func(integer32)
    assert epyccel_func(integer64) == nonzero_func(integer64)
    assert epyccel_func(fl) == nonzero_func(fl)
    assert epyccel_func(fl32) == nonzero_func(fl32)
    assert epyccel_func(fl64) == nonzero_func(fl64)


def test_dtype(epyc_numpy_funcs_mod):

    func = numpy_funcs.dtype

    bl = np.array([True, False, True, False, True])
    integer8 = np.array([6, 1, 8, 2, 3], dtype=np.int8)
    integer16 = np.array([6, 1, 8, 2, 3], dtype=np.int16)
    integer = np.array([6, 1, 8, 2, 3], dtype=int)
    integer32 = np.array([6, 1, 8, 2, 3], dtype=np.int32)
    integer64 = np.array([6, 1, 8, 2, 3], dtype=np.int64)

    fl = np.array([6, 22, 1, 8, 2, 3], dtype=float)
    fl32 = np.array([6, 22, 1, 8, 2, 3], dtype=np.float32)
    fl64 = np.array([6, 22, 1, 8, 2, 3], dtype=np.float64)

    epyccel_func = epyc_numpy_funcs_mod.dtype

    assert matching_types(epyccel_func(bl), func(bl))
    assert matching_types(epyccel_func(integer8), func(integer8))
    assert matching_types(epyccel_func(integer16), func(integer16))
    assert matching_types(epyccel_func(integer), func(integer))
    assert matching_types(epyccel_func(integer32), func(integer32))
    assert matching_types(epyccel_func(integer64), func(integer64))
    assert matching_types(epyccel_func(fl), func(fl))
    assert matching_types(epyccel_func(fl32), func(fl32))
    assert matching_types(epyccel_func(fl64), func(fl64))


def test_result_type(epyc_numpy_funcs_mod):
    int_vs_int_array = numpy_funcs.result_type__int_vs_int_array
    type_comparison = numpy_funcs.result_type__type_comparison
    type_comparison2 = numpy_funcs.result_type__type_comparison2
    value_types = numpy_funcs.result_type__value_types

    epyccel_int_vs_int_array = epyc_numpy_funcs_mod.result_type__int_vs_int_array
    epyccel_type_comparison = epyc_numpy_funcs_mod.result_type__type_comparison
    epyccel_type_comparison2 = epyc_numpy_funcs_mod.result_type__type_comparison2
    epyccel_value_types = epyc_numpy_funcs_mod.result_type__value_types

    assert matching_types(epyccel_int_vs_int_array(), int_vs_int_array())
    assert matching_types(epyccel_type_comparison(), type_comparison())
    assert matching_types(epyccel_type_comparison2(), type_comparison2())
    assert matching_types(epyccel_value_types(), value_types())


@pytest.mark.skipif_by_language(
    True, language="python", reason=("Template causes problems with order")
)
def test_copy(epyc_numpy_funcs_mod):

    arr_1d = randint(min_int, max_int, size=5, dtype=np.int64)
    arr_2d = uniform(min_float64 / 2, max_float64 / 2, size=(3, 4))
    arr_3d = (
        uniform(min_float64 / 2, max_float64 / 2, size=(3, 4, 5))
        + uniform(min_float64 / 2, max_float64 / 2, size=(3, 4, 5)) * 1j
    ).T

    funcs = [
        (numpy_funcs.test_copy__copy_array, epyc_numpy_funcs_mod.test_copy__copy_array),
        (
            numpy_funcs.test_copy__copy_array_to_F,
            epyc_numpy_funcs_mod.test_copy__copy_array_to_F,
        ),
        (
            numpy_funcs.test_copy__copy_array_to_C,
            epyc_numpy_funcs_mod.test_copy__copy_array_to_C,
        ),
    ]

    f, epyc_f = funcs[0]
    res_1d_pyt = f(arr_1d)
    res_1d_pyc = epyc_f(arr_1d)
    assert np.array_equal(res_1d_pyt, res_1d_pyc)
    assert res_1d_pyt.dtype is res_1d_pyc.dtype

    for f, epyc_f in funcs:
        res_2d_pyt = f(arr_2d)
        res_2d_pyc = epyc_f(arr_2d)
        assert np.array_equal(res_2d_pyt, res_2d_pyc)
        assert res_2d_pyt.dtype is res_2d_pyc.dtype
        assert res_2d_pyt.flags.c_contiguous == res_2d_pyc.flags.c_contiguous
        assert res_2d_pyt.flags.f_contiguous == res_2d_pyc.flags.f_contiguous

        res_3d_pyt = f(arr_3d)
        res_3d_pyc = epyc_f(arr_3d)
        assert np.array_equal(res_3d_pyt, res_3d_pyc)
        assert res_3d_pyt.dtype is res_3d_pyc.dtype
        assert res_3d_pyt.flags.c_contiguous == res_3d_pyc.flags.c_contiguous
        assert res_3d_pyt.flags.f_contiguous == res_3d_pyc.flags.f_contiguous


def test_true_divide(language):
    def basic_division(a: "int | float | complex", b: "int | float | complex"):
        from numpy import true_divide

        return true_divide(a, b)

    def basic_array_division(
        a: "int[:] | float[:]", b: "int | float | complex | int[:] | float[:]"
    ):
        from numpy import true_divide

        return true_divide(a, b)

    i = randint(1e6)
    f = max_float / 2
    c = max_float / 2 * (1 + 1j)
    i_arr_1d = randint(min_int, max_int, size=5, dtype=np.int64)
    f_arr_1d = uniform(min_float / 2, max_float / 2, size=5)

    # Avoid overflow on macOS
    if sys.platform == "darwin" and language == "c":
        c /= np.sqrt(2)

    epyccel_basic_division = epyccel(basic_division, language=language)
    epyccel_basic_array_division = epyccel(basic_array_division, language=language)

    assert np.isclose(
        basic_division(i, i), epyccel_basic_division(i, i), rtol=RTOL, atol=ATOL
    )
    assert np.isclose(
        basic_division(i, f), epyccel_basic_division(i, f), rtol=RTOL, atol=ATOL
    )
    assert np.isclose(
        basic_division(i, c), epyccel_basic_division(i, c), rtol=RTOL, atol=ATOL
    )
    assert np.isclose(
        basic_division(f, i), epyccel_basic_division(f, i), rtol=RTOL, atol=ATOL
    )
    assert np.isclose(
        basic_division(f, f), epyccel_basic_division(f, f), rtol=RTOL, atol=ATOL
    )
    assert np.isclose(
        basic_division(f, c), epyccel_basic_division(f, c), rtol=RTOL, atol=ATOL
    )
    assert np.isclose(
        basic_division(c, i), epyccel_basic_division(c, i), rtol=RTOL, atol=ATOL
    )
    assert np.isclose(
        basic_division(c, f), epyccel_basic_division(c, f), rtol=RTOL, atol=ATOL
    )
    assert np.isclose(
        basic_division(c, c), epyccel_basic_division(c, c), rtol=RTOL, atol=ATOL
    )
    assert np.allclose(
        basic_array_division(i_arr_1d, i_arr_1d),
        epyccel_basic_array_division(i_arr_1d, i_arr_1d),
        rtol=RTOL,
        atol=ATOL,
    )
    assert np.allclose(
        basic_array_division(i_arr_1d, f_arr_1d),
        epyccel_basic_array_division(i_arr_1d, f_arr_1d),
        rtol=RTOL,
        atol=ATOL,
    )
    assert np.allclose(
        basic_array_division(i_arr_1d, i),
        epyccel_basic_array_division(i_arr_1d, i),
        rtol=RTOL,
        atol=ATOL,
    )
    assert np.allclose(
        basic_array_division(i_arr_1d, f),
        epyccel_basic_array_division(i_arr_1d, f),
        rtol=RTOL,
        atol=ATOL,
    )
    assert np.allclose(
        basic_array_division(i_arr_1d, c),
        epyccel_basic_array_division(i_arr_1d, c),
        rtol=RTOL,
        atol=ATOL,
    )
    assert np.allclose(
        basic_array_division(f_arr_1d, i_arr_1d),
        epyccel_basic_array_division(f_arr_1d, i_arr_1d),
        rtol=RTOL,
        atol=ATOL,
    )
    assert np.allclose(
        basic_array_division(f_arr_1d, f_arr_1d),
        epyccel_basic_array_division(f_arr_1d, f_arr_1d),
        rtol=RTOL,
        atol=ATOL,
    )
    assert np.allclose(
        basic_array_division(f_arr_1d, i),
        epyccel_basic_array_division(f_arr_1d, i),
        rtol=RTOL,
        atol=ATOL,
    )
    assert np.allclose(
        basic_array_division(f_arr_1d, f),
        epyccel_basic_array_division(f_arr_1d, f),
        rtol=RTOL,
        atol=ATOL,
    )
    assert np.allclose(
        basic_array_division(f_arr_1d, c),
        epyccel_basic_array_division(f_arr_1d, c),
        rtol=RTOL,
        atol=ATOL,
    )
    with pytest.warns(RuntimeWarning, match="divide by zero encountered in divide"):
        assert basic_division(f, 0) == epyccel_basic_division(f, 0)


def test_cross_1d(epyc_numpy_funcs_mod):
    cross_call = numpy_funcs.cross_1d

    f1 = epyc_numpy_funcs_mod.cross_1d
    x = rand(3)
    y = rand(3)
    assert np.allclose(f1(x, y), cross_call(x, y), rtol=RTOL, atol=ATOL)


def test_cross_1d_expr(epyc_numpy_funcs_mod):
    cross_call = numpy_funcs.cross_1d_expr

    f1 = epyc_numpy_funcs_mod.cross_1d_expr
    x = rand(3)
    y = rand(3)
    assert np.allclose(f1(x, y), cross_call(x, y), rtol=RTOL, atol=ATOL)


def test_cross_2d_axis(epyc_numpy_funcs_mod):
    cross_call = numpy_funcs.cross_2d_axis

    f1 = epyc_numpy_funcs_mod.cross_2d_axis
    x = rand(5, 3)
    y = rand(5, 3)
    assert np.allclose(f1(x, y), cross_call(x, y), rtol=RTOL, atol=ATOL)
    assert f1(x, y).shape == cross_call(x, y).shape


def test_cross_mixed_dimensions(epyc_numpy_funcs_mod):
    cross_call = numpy_funcs.cross_mixed_dimensions

    f1 = epyc_numpy_funcs_mod.cross_mixed_dimensions
    x = np.array(rand(4, 3) * 10, dtype=int)
    assert np.allclose(f1(x), cross_call(x), rtol=RTOL, atol=ATOL)


def test_linalg_cross_1d(epyc_numpy_funcs_mod):
    cross_call = numpy_funcs.linalg_cross_1d

    f1 = epyc_numpy_funcs_mod.linalg_cross_1d
    x = rand(3)
    y = rand(3)
    assert np.allclose(f1(x, y), cross_call(x, y), rtol=RTOL, atol=ATOL)


def test_linalg_cross_1d_mixed_types(epyc_numpy_funcs_mod):
    cross_call = numpy_funcs.linalg_cross_1d_mixed_types

    f1 = epyc_numpy_funcs_mod.linalg_cross_1d_mixed_types
    x = rand(3)
    y = np.array(rand(3) * 10, dtype=int)
    assert np.allclose(f1(x, y), cross_call(x, y), rtol=RTOL, atol=ATOL)


def test_linalg_cross_axis(epyc_numpy_funcs_mod):
    cross_call = numpy_funcs.linalg_cross_axis

    f1 = epyc_numpy_funcs_mod.linalg_cross_axis
    x = rand(2, 3)
    y = rand(2, 3)
    assert np.allclose(f1(x, y), cross_call(x, y), rtol=RTOL, atol=ATOL)


def test_cross_axisa_axisb(epyc_numpy_funcs_mod):
    cross_call = numpy_funcs.cross_axisa_axisb

    f1 = epyc_numpy_funcs_mod.cross_axisa_axisb
    x = rand(5, 3)
    y = rand(5, 3)
    assert np.allclose(f1(x, y), cross_call(x, y), rtol=RTOL, atol=ATOL)
    assert f1(x, y).shape == cross_call(x, y).shape


def test_cross_axisc(epyc_numpy_funcs_mod):
    cross_call = numpy_funcs.cross_axisc

    f1 = epyc_numpy_funcs_mod.cross_axisc
    x = rand(5, 3)
    y = rand(5, 3)
    assert np.allclose(f1(x, y), cross_call(x, y), rtol=RTOL, atol=ATOL)
    assert f1(x, y).shape == cross_call(x, y).shape


def test_cross_axisa_axisb_axisc(epyc_numpy_funcs_mod):
    cross_call = numpy_funcs.cross_axisa_axisb_axisc

    f1 = epyc_numpy_funcs_mod.cross_axisa_axisb_axisc
    x = rand(4, 5, 3)
    y = rand(4, 3, 5)
    assert np.allclose(f1(x, y), cross_call(x, y), rtol=RTOL, atol=ATOL)
    assert f1(x, y).shape == cross_call(x, y).shape


def test_vecdot_1d_real(epyc_numpy_funcs_mod):
    vecdot_call = numpy_funcs.vecdot_1d_real

    f1 = epyc_numpy_funcs_mod.vecdot_1d_real
    x = rand(10)
    y = rand(10)
    assert np.allclose(f1(x, y), vecdot_call(x, y), rtol=RTOL, atol=ATOL)


def test_vecdot_1d_complex(epyc_numpy_funcs_mod):
    vecdot_call = numpy_funcs.vecdot_1d_complex

    f1 = epyc_numpy_funcs_mod.vecdot_1d_complex
    x = rand(8) + 1j * rand(8)
    y = rand(8) + 1j * rand(8)
    assert np.allclose(f1(x, y), vecdot_call(x, y), rtol=RTOL, atol=ATOL)


def test_vecdot_axis_2d(epyc_numpy_funcs_mod):
    vecdot_call = numpy_funcs.vecdot_axis_2d

    f1 = epyc_numpy_funcs_mod.vecdot_axis_2d
    x = rand(6, 5)
    y = rand(6, 5)
    assert np.allclose(f1(x, y), vecdot_call(x, y), rtol=RTOL, atol=ATOL)
    assert f1(x, y).shape == vecdot_call(x, y).shape


def test_vecdot_mixed_dimensions(epyc_numpy_funcs_mod):
    vecdot_call = numpy_funcs.vecdot_mixed_dimensions

    f1 = epyc_numpy_funcs_mod.vecdot_mixed_dimensions
    x = rand(4, 7)
    y = rand(7)
    assert np.allclose(f1(x, y), vecdot_call(x, y), rtol=RTOL, atol=ATOL)


def test_vecdot_out_axis_2d(epyc_numpy_funcs_mod):
    vecdot_call = numpy_funcs.vecdot_out_axis_2d

    f1 = epyc_numpy_funcs_mod.vecdot_out_axis_2d
    x = rand(5, 7)
    y = rand(5, 7)
    assert np.allclose(f1(x, y), vecdot_call(x, y), rtol=RTOL, atol=ATOL)


def test_vecdot_3d_axis_order(epyc_numpy_funcs_mod):
    vecdot_call = numpy_funcs.vecdot_3d_axis_order

    f1 = epyc_numpy_funcs_mod.vecdot_3d_axis_order
    x = rand(4, 5, 6)
    y = rand(4, 5, 6)
    res_ref = vecdot_call(x, y)
    res_cc = f1(x, y)

    assert np.allclose(res_cc, res_ref, rtol=RTOL, atol=ATOL)
    assert res_cc.shape == res_ref.shape
    assert res_cc.flags["C_CONTIGUOUS"] == res_ref.flags["C_CONTIGUOUS"]
    assert res_cc.flags["F_CONTIGUOUS"] == res_ref.flags["F_CONTIGUOUS"]


def test_vecdot_mixed_dimensions_expression(epyc_numpy_funcs_mod):
    vecdot_call = numpy_funcs.vecdot_mixed_dimensions_expression

    f1 = epyc_numpy_funcs_mod.vecdot_mixed_dimensions_expression
    x = rand(4, 7)
    y = rand(7)
    assert np.allclose(f1(x, y), vecdot_call(x, y), rtol=RTOL, atol=ATOL)


def test_vecdot_3d_axis_order_expression(epyc_numpy_funcs_mod):
    vecdot_call = numpy_funcs.vecdot_3d_axis_order_expression

    f1 = epyc_numpy_funcs_mod.vecdot_3d_axis_order_expression
    x = rand(4, 5, 6)
    y = rand(4, 5, 6)
    res_ref = vecdot_call(x, y)
    res_cc = f1(x, y)

    assert np.allclose(res_cc, res_ref, rtol=RTOL, atol=ATOL)
    assert res_cc.shape == res_ref.shape
    assert res_cc.flags["C_CONTIGUOUS"] == res_ref.flags["C_CONTIGUOUS"]
    assert res_cc.flags["F_CONTIGUOUS"] == res_ref.flags["F_CONTIGUOUS"]
