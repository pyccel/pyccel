# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np
import pytest
from modules import epyccel_return_arrays
from numpy.random import randint, uniform

from epyccel_utilities import epyccel_module_with_fallback
from tolerances import (
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
def epyc_epyccel_return_arrays_mod(language):
    return epyccel_module_with_fallback(epyccel_return_arrays, language)


def test_single_return(epyc_epyccel_return_arrays_mod):
    return_array = epyccel_return_arrays.return_array
    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=int)
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
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = np.complex128(
        uniform(low=min_float64 / 2, high=max_float64 / 2)
        + uniform(low=min_float64 / 2, high=max_float64 / 2) * 1j
    )

    epyccel_func = epyc_epyccel_return_arrays_mod.return_array

    f_bl_true_output = epyccel_func(True, True)
    test_bool_true_output = return_array(True, True)

    f_bl_false_output = epyccel_func(False, False)
    test_bool_false_output = return_array(False, False)

    assert np.array_equal(f_bl_true_output, test_bool_true_output)
    assert np.array_equal(f_bl_false_output, test_bool_false_output)

    assert f_bl_false_output.dtype == test_bool_false_output.dtype
    assert f_bl_true_output.dtype == test_bool_true_output.dtype

    f_integer_output = epyccel_func(integer, integer)
    test_int_output = return_array(integer, integer)

    assert np.array_equal(f_integer_output, test_int_output)
    assert f_integer_output.dtype == test_int_output.dtype

    f_integer8_output = epyccel_func(integer8, integer8)
    test_int8_output = return_array(integer8, integer8)

    assert np.array_equal(f_integer8_output, test_int8_output)
    assert f_integer8_output.dtype == test_int8_output.dtype

    f_integer16_output = epyccel_func(integer16, integer16)
    test_int16_output = return_array(integer16, integer16)

    assert np.array_equal(f_integer16_output, test_int16_output)
    assert f_integer16_output.dtype == test_int16_output.dtype

    f_integer32_output = epyccel_func(integer32, integer32)
    test_int32_output = return_array(integer32, integer32)

    assert np.array_equal(f_integer32_output, test_int32_output)
    assert f_integer32_output.dtype == test_int32_output.dtype

    f_integer64_output = epyccel_func(integer64, integer64)
    test_int64_output = return_array(integer64, integer64)

    assert np.array_equal(f_integer64_output, test_int64_output)
    assert f_integer64_output.dtype == test_int64_output.dtype

    f_fl_output = epyccel_func(fl, fl)
    test_float_output = return_array(fl, fl)

    assert np.array_equal(f_fl_output, test_float_output)
    assert f_fl_output.dtype == test_float_output.dtype

    f_fl32_output = epyccel_func(fl32, fl32)
    test_float32_output = return_array(fl32, fl32)

    assert np.array_equal(f_fl32_output, test_float32_output)
    assert f_fl32_output.dtype == test_float32_output.dtype

    f_fl64_output = epyccel_func(fl64, fl64)
    test_float64_output = return_array(fl64, fl64)

    assert np.array_equal(f_fl64_output, test_float64_output)
    assert f_fl64_output.dtype == test_float64_output.dtype

    f_cmplx64_output = epyccel_func(cmplx64, cmplx64)
    test_cmplx64_output = return_array(cmplx64, cmplx64)

    assert np.array_equal(f_cmplx64_output, test_cmplx64_output)
    assert f_cmplx64_output.dtype == test_cmplx64_output.dtype

    f_cmplx128_output = epyccel_func(cmplx128, cmplx128)
    test_cmplx128_output = return_array(cmplx128, cmplx128)

    assert np.array_equal(f_cmplx128_output, test_cmplx128_output)
    assert f_cmplx128_output.dtype == test_cmplx128_output.dtype


def test_multi_returns(epyc_epyccel_return_arrays_mod):
    return_array = epyccel_return_arrays.multi_returns
    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=int)
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
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = np.complex128(
        uniform(low=min_float64 / 2, high=max_float64 / 2)
        + uniform(low=min_float64 / 2, high=max_float64 / 2) * 1j
    )

    epyccel_func = epyc_epyccel_return_arrays_mod.multi_returns

    f_bl_true_output = epyccel_func(True, True)
    test_bool_true_output = return_array(True, True)

    f_bl_false_output = epyccel_func(False, False)
    test_bool_false_output = return_array(False, False)

    assert np.array_equal(f_bl_true_output, test_bool_true_output)
    assert np.array_equal(f_bl_false_output, test_bool_false_output)

    assert f_bl_false_output[0].dtype == test_bool_false_output[0].dtype
    assert f_bl_true_output[0].dtype == test_bool_true_output[0].dtype

    f_integer_output = epyccel_func(integer, integer)
    test_int_output = return_array(integer, integer)

    assert np.array_equal(f_integer_output, test_int_output)
    assert f_integer_output[0].dtype == test_int_output[0].dtype

    f_integer8_output = epyccel_func(integer8, integer8)
    test_int8_output = return_array(integer8, integer8)

    assert np.array_equal(f_integer8_output, test_int8_output)
    assert f_integer8_output[0].dtype == test_int8_output[0].dtype

    f_integer16_output = epyccel_func(integer16, integer16)
    test_int16_output = return_array(integer16, integer16)

    assert np.array_equal(f_integer16_output, test_int16_output)
    assert f_integer16_output[0].dtype == test_int16_output[0].dtype

    f_integer32_output = epyccel_func(integer32, integer32)
    test_int32_output = return_array(integer32, integer32)

    assert np.array_equal(f_integer32_output, test_int32_output)
    assert f_integer32_output[0].dtype == test_int32_output[0].dtype

    f_integer64_output = epyccel_func(integer64, integer64)
    test_int64_output = return_array(integer64, integer64)

    assert np.array_equal(f_integer64_output, test_int64_output)
    assert f_integer64_output[0].dtype == test_int64_output[0].dtype

    f_fl_output = epyccel_func(fl, fl)
    test_float_output = return_array(fl, fl)

    assert np.array_equal(f_fl_output, test_float_output)
    assert f_fl_output[0].dtype == test_float_output[0].dtype

    f_fl32_output = epyccel_func(fl32, fl32)
    test_float32_output = return_array(fl32, fl32)

    assert np.array_equal(f_fl32_output, test_float32_output)
    assert f_fl32_output[0].dtype == test_float32_output[0].dtype

    f_fl64_output = epyccel_func(fl64, fl64)
    test_float64_output = return_array(fl64, fl64)

    assert np.array_equal(f_fl64_output, test_float64_output)
    assert f_fl64_output[0].dtype == test_float64_output[0].dtype

    f_cmplx64_output = epyccel_func(cmplx64, cmplx64)
    test_cmplx64_output = return_array(cmplx64, cmplx64)

    assert np.array_equal(f_cmplx64_output, test_cmplx64_output)
    assert f_cmplx64_output[0].dtype == test_cmplx64_output[0].dtype

    f_cmplx128_output = epyccel_func(cmplx128, cmplx128)
    test_cmplx128_output = return_array(cmplx128, cmplx128)

    assert np.array_equal(f_cmplx128_output, test_cmplx128_output)
    assert f_cmplx128_output[0].dtype == test_cmplx128_output[0].dtype


def test_return_array_array_op(epyc_epyccel_return_arrays_mod):

    return_array = epyccel_return_arrays.return_array_array_op
    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=int)
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
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = np.complex128(
        uniform(low=min_float64 / 2, high=max_float64 / 2)
        + uniform(low=min_float64 / 2, high=max_float64 / 2) * 1j
    )

    epyccel_func = epyc_epyccel_return_arrays_mod.return_array_array_op

    f_integer_output = epyccel_func(integer, integer)
    test_int_output = return_array(integer, integer)

    assert np.array_equal(f_integer_output, test_int_output)
    assert f_integer_output[0].dtype == test_int_output[0].dtype

    f_integer8_output = epyccel_func(integer8, integer8)
    test_int8_output = return_array(integer8, integer8)

    assert np.array_equal(f_integer8_output, test_int8_output)
    assert f_integer8_output[0].dtype == test_int8_output[0].dtype

    f_integer16_output = epyccel_func(integer16, integer16)
    test_int16_output = return_array(integer16, integer16)

    assert np.array_equal(f_integer16_output, test_int16_output)
    assert f_integer16_output[0].dtype == test_int16_output[0].dtype

    f_integer32_output = epyccel_func(integer32, integer32)
    test_int32_output = return_array(integer32, integer32)

    assert np.array_equal(f_integer32_output, test_int32_output)
    assert f_integer32_output[0].dtype == test_int32_output[0].dtype

    f_integer64_output = epyccel_func(integer64, integer64)
    test_int64_output = return_array(integer64, integer64)

    assert np.array_equal(f_integer64_output, test_int64_output)
    assert f_integer64_output[0].dtype == test_int64_output[0].dtype

    f_fl_output = epyccel_func(fl, fl)
    test_float_output = return_array(fl, fl)

    assert np.array_equal(f_fl_output, test_float_output)
    assert f_fl_output[0].dtype == test_float_output[0].dtype

    f_fl32_output = epyccel_func(fl32, fl32)
    test_float32_output = return_array(fl32, fl32)

    assert np.array_equal(f_fl32_output, test_float32_output)
    assert f_fl32_output[0].dtype == test_float32_output[0].dtype

    f_fl64_output = epyccel_func(fl64, fl64)
    test_float64_output = return_array(fl64, fl64)

    assert np.array_equal(f_fl64_output, test_float64_output)
    assert f_fl64_output[0].dtype == test_float64_output[0].dtype

    f_cmplx64_output = epyccel_func(cmplx64, cmplx64)
    test_cmplx64_output = return_array(cmplx64, cmplx64)

    assert np.array_equal(f_cmplx64_output, test_cmplx64_output)
    assert f_cmplx64_output[0].dtype == test_cmplx64_output[0].dtype

    f_cmplx128_output = epyccel_func(cmplx128, cmplx128)
    test_cmplx128_output = return_array(cmplx128, cmplx128)

    assert np.array_equal(f_cmplx128_output, test_cmplx128_output)
    assert f_cmplx128_output[0].dtype == test_cmplx128_output[0].dtype


def test_return_multi_array_array_op(epyc_epyccel_return_arrays_mod):

    return_array = epyccel_return_arrays.return_multi_array_array_op
    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=int)
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
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = np.complex128(
        uniform(low=min_float64 / 2, high=max_float64 / 2)
        + uniform(low=min_float64 / 2, high=max_float64 / 2) * 1j
    )

    epyccel_func = epyc_epyccel_return_arrays_mod.return_multi_array_array_op

    f_integer_output = epyccel_func(integer, integer)
    test_int_output = return_array(integer, integer)

    assert np.array_equal(f_integer_output, test_int_output)
    assert f_integer_output[0].dtype == test_int_output[0].dtype

    f_integer8_output = epyccel_func(integer8, integer8)
    test_int8_output = return_array(integer8, integer8)

    assert np.array_equal(f_integer8_output, test_int8_output)
    assert f_integer8_output[0].dtype == test_int8_output[0].dtype

    f_integer16_output = epyccel_func(integer16, integer16)
    test_int16_output = return_array(integer16, integer16)

    assert np.array_equal(f_integer16_output, test_int16_output)
    assert f_integer16_output[0].dtype == test_int16_output[0].dtype

    f_integer32_output = epyccel_func(integer32, integer32)
    test_int32_output = return_array(integer32, integer32)

    assert np.array_equal(f_integer32_output, test_int32_output)
    assert f_integer32_output[0].dtype == test_int32_output[0].dtype

    f_integer64_output = epyccel_func(integer64, integer64)
    test_int64_output = return_array(integer64, integer64)

    assert np.array_equal(f_integer64_output, test_int64_output)
    assert f_integer64_output[0].dtype == test_int64_output[0].dtype

    f_fl_output = epyccel_func(fl, fl)
    test_float_output = return_array(fl, fl)

    assert np.array_equal(f_fl_output, test_float_output)
    assert f_fl_output[0].dtype == test_float_output[0].dtype

    f_fl32_output = epyccel_func(fl32, fl32)
    test_float32_output = return_array(fl32, fl32)

    assert np.array_equal(f_fl32_output, test_float32_output)
    assert f_fl32_output[0].dtype == test_float32_output[0].dtype

    f_fl64_output = epyccel_func(fl64, fl64)
    test_float64_output = return_array(fl64, fl64)

    assert np.array_equal(f_fl64_output, test_float64_output)
    assert f_fl64_output[0].dtype == test_float64_output[0].dtype

    f_cmplx64_output = epyccel_func(cmplx64, cmplx64)
    test_cmplx64_output = return_array(cmplx64, cmplx64)

    assert np.array_equal(f_cmplx64_output, test_cmplx64_output)
    assert f_cmplx64_output[0].dtype == test_cmplx64_output[0].dtype

    f_cmplx128_output = epyccel_func(cmplx128, cmplx128)
    test_cmplx128_output = return_array(cmplx128, cmplx128)

    assert np.array_equal(f_cmplx128_output, test_cmplx128_output)
    assert f_cmplx128_output[0].dtype == test_cmplx128_output[0].dtype


def test_return_array_scalar_op(epyc_epyccel_return_arrays_mod):

    return_array_scalar_op = epyccel_return_arrays.return_array_scalar_op
    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=int)
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
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = np.complex128(
        uniform(low=min_float64 / 2, high=max_float64 / 2)
        + uniform(low=min_float64 / 2, high=max_float64 / 2) * 1j
    )

    epyccel_func = epyc_epyccel_return_arrays_mod.return_array_scalar_op

    f_integer_output = epyccel_func(integer)
    test_int_output = return_array_scalar_op(integer)

    assert np.array_equal(f_integer_output, test_int_output)
    assert f_integer_output[0].dtype == test_int_output[0].dtype

    f_integer8_output = epyccel_func(integer8)
    test_int8_output = return_array_scalar_op(integer8)

    assert np.array_equal(f_integer8_output, test_int8_output)
    assert f_integer8_output[0].dtype == test_int8_output[0].dtype

    f_integer16_output = epyccel_func(integer16)
    test_int16_output = return_array_scalar_op(integer16)

    assert np.array_equal(f_integer16_output, test_int16_output)
    assert f_integer16_output[0].dtype == test_int16_output[0].dtype

    f_integer32_output = epyccel_func(integer32)
    test_int32_output = return_array_scalar_op(integer32)

    assert np.array_equal(f_integer32_output, test_int32_output)
    assert f_integer32_output[0].dtype == test_int32_output[0].dtype

    f_integer64_output = epyccel_func(integer64)
    test_int64_output = return_array_scalar_op(integer64)

    assert np.array_equal(f_integer64_output, test_int64_output)
    assert f_integer64_output[0].dtype == test_int64_output[0].dtype

    f_fl_output = epyccel_func(fl)
    test_float_output = return_array_scalar_op(fl)

    assert np.array_equal(f_fl_output, test_float_output)
    assert f_fl_output[0].dtype == test_float_output[0].dtype

    f_fl32_output = epyccel_func(fl32)
    test_float32_output = return_array_scalar_op(fl32)

    assert np.array_equal(f_fl32_output, test_float32_output)
    assert f_fl32_output[0].dtype == test_float32_output[0].dtype

    f_fl64_output = epyccel_func(fl64)
    test_float64_output = return_array_scalar_op(fl64)

    assert np.array_equal(f_fl64_output, test_float64_output)
    assert f_fl64_output[0].dtype == test_float64_output[0].dtype

    f_cmplx64_output = epyccel_func(cmplx64)
    test_cmplx64_output = return_array_scalar_op(cmplx64)

    assert np.array_equal(f_cmplx64_output, test_cmplx64_output)
    assert f_cmplx64_output[0].dtype == test_cmplx64_output[0].dtype

    f_cmplx128_output = epyccel_func(cmplx128)
    test_cmplx128_output = return_array_scalar_op(cmplx128)

    assert np.array_equal(f_cmplx128_output, test_cmplx128_output)
    assert f_cmplx128_output[0].dtype == test_cmplx128_output[0].dtype


def test_multi_return_array_scalar_op(epyc_epyccel_return_arrays_mod):

    return_multi_array_scalar_op = epyccel_return_arrays.return_multi_array_scalar_op
    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=int)
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
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = np.complex128(
        uniform(low=min_float64 / 2, high=max_float64 / 2)
        + uniform(low=min_float64 / 2, high=max_float64 / 2) * 1j
    )

    epyccel_func = epyc_epyccel_return_arrays_mod.return_multi_array_scalar_op

    f_integer_output = epyccel_func(integer)
    test_int_output = return_multi_array_scalar_op(integer)

    assert np.array_equal(f_integer_output, test_int_output)
    assert f_integer_output[0].dtype == test_int_output[0].dtype

    f_integer8_output = epyccel_func(integer8)
    test_int8_output = return_multi_array_scalar_op(integer8)

    assert np.array_equal(f_integer8_output, test_int8_output)
    assert f_integer8_output[0].dtype == test_int8_output[0].dtype

    f_integer16_output = epyccel_func(integer16)
    test_int16_output = return_multi_array_scalar_op(integer16)

    assert np.array_equal(f_integer16_output, test_int16_output)
    assert f_integer16_output[0].dtype == test_int16_output[0].dtype

    f_integer32_output = epyccel_func(integer32)
    test_int32_output = return_multi_array_scalar_op(integer32)

    assert np.array_equal(f_integer32_output, test_int32_output)
    assert f_integer32_output[0].dtype == test_int32_output[0].dtype

    f_integer64_output = epyccel_func(integer64)
    test_int64_output = return_multi_array_scalar_op(integer64)

    assert np.array_equal(f_integer64_output, test_int64_output)
    assert f_integer64_output[0].dtype == test_int64_output[0].dtype

    f_fl_output = epyccel_func(fl)
    test_float_output = return_multi_array_scalar_op(fl)

    assert np.array_equal(f_fl_output, test_float_output)
    assert f_fl_output[0].dtype == test_float_output[0].dtype

    f_fl32_output = epyccel_func(fl32)
    test_float32_output = return_multi_array_scalar_op(fl32)

    assert np.array_equal(f_fl32_output, test_float32_output)
    assert f_fl32_output[0].dtype == test_float32_output[0].dtype

    f_fl64_output = epyccel_func(fl64)
    test_float64_output = return_multi_array_scalar_op(fl64)

    assert np.array_equal(f_fl64_output, test_float64_output)
    assert f_fl64_output[0].dtype == test_float64_output[0].dtype

    f_cmplx64_output = epyccel_func(cmplx64)
    test_cmplx64_output = return_multi_array_scalar_op(cmplx64)

    assert np.array_equal(f_cmplx64_output, test_cmplx64_output)
    assert f_cmplx64_output[0].dtype == test_cmplx64_output[0].dtype

    f_cmplx128_output = epyccel_func(cmplx128)
    test_cmplx128_output = return_multi_array_scalar_op(cmplx128)

    assert np.array_equal(f_cmplx128_output, test_cmplx128_output)
    assert f_cmplx128_output[0].dtype == test_cmplx128_output[0].dtype


def test_multi_return_array_array_op(epyc_epyccel_return_arrays_mod):

    return_array_arg_array_op = epyccel_return_arrays.return_array_arg_array_op
    arr_integer8 = np.ones(7, dtype=np.int8)
    arr_integer16 = np.ones(7, dtype=np.int16)
    arr_integer = np.ones(7, dtype=int)
    arr_integer32 = np.ones(7, dtype=np.int32)
    arr_integer64 = np.ones(7, dtype=np.int64)

    arr_fl = np.ones(7, dtype=float)
    arr_fl32 = np.ones(7, dtype=np.float32)
    arr_fl64 = np.ones(7, dtype=np.float64)

    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.ones(7, dtype=np.float32) + np.ones(7, dtype=np.float32) * 1j
    cmplx128 = np.ones(7, dtype=np.float64) + np.ones(7, dtype=np.float64) * 1j

    epyccel_func = epyc_epyccel_return_arrays_mod.return_array_arg_array_op

    f_integer_output = epyccel_func(arr_integer)
    test_int_output = return_array_arg_array_op(arr_integer)

    assert np.array_equal(f_integer_output, test_int_output)
    assert f_integer_output[0].dtype == test_int_output[0].dtype

    f_integer8_output = epyccel_func(arr_integer8)
    test_int8_output = return_array_arg_array_op(arr_integer8)

    assert np.array_equal(f_integer8_output, test_int8_output)
    assert f_integer8_output[0].dtype == test_int8_output[0].dtype

    f_integer16_output = epyccel_func(arr_integer16)
    test_int16_output = return_array_arg_array_op(arr_integer16)

    assert np.array_equal(f_integer16_output, test_int16_output)
    assert f_integer16_output[0].dtype == test_int16_output[0].dtype

    f_integer32_output = epyccel_func(arr_integer32)
    test_int32_output = return_array_arg_array_op(arr_integer32)

    assert np.array_equal(f_integer32_output, test_int32_output)
    assert f_integer32_output[0].dtype == test_int32_output[0].dtype

    f_integer64_output = epyccel_func(arr_integer64)
    test_int64_output = return_array_arg_array_op(arr_integer64)

    assert np.array_equal(f_integer64_output, test_int64_output)
    assert f_integer64_output[0].dtype == test_int64_output[0].dtype

    f_fl_output = epyccel_func(arr_fl)
    test_float_output = return_array_arg_array_op(arr_fl)

    assert np.array_equal(f_fl_output, test_float_output)
    assert f_fl_output[0].dtype == test_float_output[0].dtype

    f_fl32_output = epyccel_func(arr_fl32)
    test_float32_output = return_array_arg_array_op(arr_fl32)

    assert np.array_equal(f_fl32_output, test_float32_output)
    assert f_fl32_output[0].dtype == test_float32_output[0].dtype

    f_fl64_output = epyccel_func(arr_fl64)
    test_float64_output = return_array_arg_array_op(arr_fl64)

    assert np.array_equal(f_fl64_output, test_float64_output)
    assert f_fl64_output[0].dtype == test_float64_output[0].dtype

    f_cmplx64_output = epyccel_func(cmplx64)
    test_cmplx64_output = return_array_arg_array_op(cmplx64)

    assert np.array_equal(f_cmplx64_output, test_cmplx64_output)
    assert f_cmplx64_output[0].dtype == test_cmplx64_output[0].dtype

    f_cmplx128_output = epyccel_func(cmplx128)
    test_cmplx128_output = return_array_arg_array_op(cmplx128)

    assert np.array_equal(f_cmplx128_output, test_cmplx128_output)
    assert f_cmplx128_output[0].dtype == test_cmplx128_output[0].dtype


def test_return_arrays_in_expression(epyc_epyccel_return_arrays_mod):
    return_arrays_in_expression = epyccel_return_arrays.return_arrays_in_expression
    epyccel_function = epyc_epyccel_return_arrays_mod.return_arrays_in_expression

    epyccel_function_output = epyccel_function()
    return_arrays_in_expression_output = return_arrays_in_expression()

    assert np.array_equal(epyccel_function_output, return_arrays_in_expression_output)
    assert epyccel_function_output.dtype == return_arrays_in_expression_output.dtype


def test_return_arrays_in_expression2(epyc_epyccel_return_arrays_mod):
    return_arrays_in_expression2 = epyccel_return_arrays.return_arrays_in_expression2
    epyccel_function = epyc_epyccel_return_arrays_mod.return_arrays_in_expression2

    n = randint(5)

    epyccel_function_output = epyccel_function(n)
    return_arrays_in_expression2_output = return_arrays_in_expression2(n)

    assert np.array_equal(epyccel_function_output, return_arrays_in_expression2_output)
    assert epyccel_function_output.dtype == return_arrays_in_expression2_output.dtype


def test_c_array_return(epyc_epyccel_return_arrays_mod):
    return_c_array = epyccel_return_arrays.return_c_array
    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=int)
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
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = np.complex128(
        uniform(low=min_float64 / 2, high=max_float64 / 2)
        + uniform(low=min_float64 / 2, high=max_float64 / 2) * 1j
    )

    epyccel_func = epyc_epyccel_return_arrays_mod.return_c_array

    for arg in (
        integer8,
        integer16,
        integer,
        integer32,
        integer64,
        fl,
        fl32,
        fl64,
        cmplx64,
        cmplx128,
    ):
        f_output = epyccel_func(arg)
        test_output = return_c_array(arg)

        assert np.array_equal(f_output, test_output)
        assert f_output.flags.c_contiguous == test_output.flags.c_contiguous
        assert f_output.flags.f_contiguous == test_output.flags.f_contiguous


def test_f_array_return(epyc_epyccel_return_arrays_mod):
    return_f_array = epyccel_return_arrays.return_f_array
    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=int)
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
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = np.complex128(
        uniform(low=min_float64 / 2, high=max_float64 / 2)
        + uniform(low=min_float64 / 2, high=max_float64 / 2) * 1j
    )

    epyccel_func = epyc_epyccel_return_arrays_mod.return_f_array

    for arg in (
        integer8,
        integer16,
        integer,
        integer32,
        integer64,
        fl,
        fl32,
        fl64,
        cmplx64,
        cmplx128,
    ):
        f_output = epyccel_func(arg)
        test_output = return_f_array(arg)

        assert np.array_equal(f_output, test_output)
        assert f_output.flags.c_contiguous == test_output.flags.c_contiguous
        assert f_output.flags.f_contiguous == test_output.flags.f_contiguous


def test_copy_f_to_f(epyc_epyccel_return_arrays_mod):
    copy_f_to_f = epyccel_return_arrays.copy_f_to_f
    epyccel_func = epyc_epyccel_return_arrays_mod.copy_f_to_f
    fl_3d = np.array(uniform(min_float / 2, max_float / 2, (2, 3, 4)), order="F")
    fl_2d = np.array(uniform(min_float / 2, max_float / 2, (3, 4)), order="F")

    for fl in (fl_2d, fl_3d):
        pyth_out = copy_f_to_f(fl)
        pycc_out = epyccel_func(fl)

        assert np.array_equal(pyth_out, pycc_out)
        assert pyth_out.dtype is pycc_out.dtype
        assert pyth_out.flags.c_contiguous == pycc_out.flags.c_contiguous
        assert pyth_out.flags.f_contiguous == pycc_out.flags.f_contiguous


def test_copy_f_to_c(epyc_epyccel_return_arrays_mod):
    copy_f_to_c = epyccel_return_arrays.copy_f_to_c
    epyccel_func = epyc_epyccel_return_arrays_mod.copy_f_to_c
    fl_3d = np.array(uniform(min_float / 2, max_float / 2, (2, 3, 4)), order="F")
    fl_2d = np.array(uniform(min_float / 2, max_float / 2, (3, 4)), order="F")

    for fl in (fl_2d, fl_3d):
        pyth_out = copy_f_to_c(fl)
        pycc_out = epyccel_func(fl)

        assert np.array_equal(pyth_out, pycc_out)
        assert pyth_out.dtype is pycc_out.dtype
        assert pyth_out.flags.c_contiguous == pycc_out.flags.c_contiguous
        assert pyth_out.flags.f_contiguous == pycc_out.flags.f_contiguous


def test_copy_c_to_c(epyc_epyccel_return_arrays_mod):
    copy_c_to_c = epyccel_return_arrays.copy_c_to_c
    epyccel_func = epyc_epyccel_return_arrays_mod.copy_c_to_c
    fl_3d = uniform(min_float / 2, max_float / 2, (2, 3, 4))
    fl_2d = uniform(min_float / 2, max_float / 2, (3, 4))

    for fl in (fl_2d, fl_3d):
        pyth_out = copy_c_to_c(fl)
        pycc_out = epyccel_func(fl)

        assert np.array_equal(pyth_out, pycc_out)
        assert pyth_out.dtype is pycc_out.dtype
        assert pyth_out.flags.c_contiguous == pycc_out.flags.c_contiguous
        assert pyth_out.flags.f_contiguous == pycc_out.flags.f_contiguous


def test_copy_c_to_f(epyc_epyccel_return_arrays_mod):
    copy_c_to_f = epyccel_return_arrays.copy_c_to_f
    epyccel_func = epyc_epyccel_return_arrays_mod.copy_c_to_f
    fl_3d = uniform(min_float / 2, max_float / 2, (2, 3, 4))
    fl_2d = uniform(min_float / 2, max_float / 2, (3, 4))

    for fl in (fl_2d, fl_3d):
        pyth_out = copy_c_to_f(fl)
        pycc_out = epyccel_func(fl)

        assert np.array_equal(pyth_out, pycc_out)
        assert pyth_out.dtype is pycc_out.dtype
        assert pyth_out.flags.c_contiguous == pycc_out.flags.c_contiguous
        assert pyth_out.flags.f_contiguous == pycc_out.flags.f_contiguous


def test_copy_c_to_default(epyc_epyccel_return_arrays_mod):
    copy_c_to_default = epyccel_return_arrays.copy_c_to_default
    epyccel_func = epyc_epyccel_return_arrays_mod.copy_c_to_default
    fl_3d = uniform(min_float / 2, max_float / 2, (2, 3, 4))
    fl_2d = uniform(min_float / 2, max_float / 2, (3, 4))

    for fl in (fl_2d, fl_3d):
        pyth_out = copy_c_to_default(fl)
        pycc_out = epyccel_func(fl)

        assert np.array_equal(pyth_out, pycc_out)
        assert pyth_out.dtype is pycc_out.dtype
        assert pyth_out.flags.c_contiguous == pycc_out.flags.c_contiguous
        assert pyth_out.flags.f_contiguous == pycc_out.flags.f_contiguous


def test_copy_f_to_default(epyc_epyccel_return_arrays_mod):
    copy_f_to_default = epyccel_return_arrays.copy_f_to_default
    epyccel_func = epyc_epyccel_return_arrays_mod.copy_f_to_default
    fl_3d = np.array(uniform(min_float / 2, max_float / 2, (2, 3, 4)), order="F")
    fl_2d = np.array(uniform(min_float / 2, max_float / 2, (3, 4)), order="F")

    for fl in (fl_2d, fl_3d):
        pyth_out = copy_f_to_default(fl)
        pycc_out = epyccel_func(fl)

        assert np.array_equal(pyth_out, pycc_out)
        assert pyth_out.dtype is pycc_out.dtype
        assert pyth_out.flags.c_contiguous == pycc_out.flags.c_contiguous
        assert pyth_out.flags.f_contiguous == pycc_out.flags.f_contiguous


def test_annotated_return(epyc_epyccel_return_arrays_mod):
    annotated_return = epyccel_return_arrays.annotated_return
    epyccel_func = epyc_epyccel_return_arrays_mod.annotated_return
    fl_b = np.array(uniform(min_float / 2, max_float / 2, (3, 4)))
    fl_c = np.array(uniform(min_float / 2, max_float / 2, (3, 4)))

    pyth_out = annotated_return(fl_b, fl_c)
    pycc_out = epyccel_func(fl_b, fl_c)

    assert np.array_equal(pyth_out, pycc_out)
    assert pyth_out.dtype is pycc_out.dtype
    assert pyth_out.flags.c_contiguous == pycc_out.flags.c_contiguous
    assert pyth_out.flags.f_contiguous == pycc_out.flags.f_contiguous


def test_unknown_size_array(epyc_epyccel_return_arrays_mod):
    unknown_size = epyccel_return_arrays.unknown_size
    epyccel_func = epyc_epyccel_return_arrays_mod.unknown_size

    assert np.array_equal(epyccel_func(True), unknown_size(True))
    assert np.array_equal(epyccel_func(False), unknown_size(False))
