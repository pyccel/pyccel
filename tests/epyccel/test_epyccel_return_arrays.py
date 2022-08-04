# pylint: disable=missing-function-docstring, missing-module-docstring/
from numpy.random import randint, uniform
import numpy as np
import pytest

from recognised_functions.test_numpy_funcs import (min_int, max_int, min_int8, max_int8,
                                min_int16, max_int16, min_int32, max_int32, max_int64, min_int64)
from recognised_functions.test_numpy_funcs import max_float, min_float, max_float32, min_float32,max_float64, min_float64
from pyccel.epyccel import epyccel
from pyccel.decorators import types


def test_single_return(language):
    @types('bool', 'bool')
    @types('int', 'int')
    @types('int8', 'int8')
    @types('int16', 'int16')
    @types('int32', 'int32')
    @types('int64', 'int64')
    @types('float', 'float')
    @types('float32', 'float32')
    @types('float64', 'float64')
    @types('complex64', 'complex64')
    @types('complex128', 'complex128')
    def return_array(a, b):
        from numpy import array
        x = array([a,b], dtype=type(a))
        return x

    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=int)
    integer32 = randint(min_int32, max_int32, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2)
    fl32 = uniform(min_float32 / 2, max_float32 / 2)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2)

    cmplx128_from_float32 = uniform(low=min_float32 / 2, high=max_float32 / 2) + uniform(low=min_float32 / 2, high=max_float32 / 2) * 1j
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = np.complex128(uniform(low=min_float64 / 2, high=max_float64 / 2) + uniform(low=min_float64 / 2, high=max_float64 / 2) * 1j)

    epyccel_func = epyccel(return_array, language=language)

    f_bl_true_output = epyccel_func(True, True)
    test_bool_true_output = return_array(True, True)

    f_bl_false_output = epyccel_func(False, False)
    test_bool_false_output = return_array(False, False)

    assert np.array_equal(f_bl_true_output, test_bool_true_output)
    assert np.array_equal(f_bl_false_output, test_bool_false_output)

    assert f_bl_false_output.dtype == test_bool_false_output.dtype
    assert f_bl_true_output.dtype == test_bool_true_output.dtype

    f_integer_output = epyccel_func(integer, integer)
    test_int_output  = return_array(integer, integer)

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


def test_multi_returns(language):
    @types('bool', 'bool')
    @types('int', 'int')
    @types('int8', 'int8')
    @types('int16', 'int16')
    @types('int32', 'int32')
    @types('int64', 'int64')
    @types('float', 'float')
    @types('float32', 'float32')
    @types('float64', 'float64')
    @types('complex64', 'complex64')
    @types('complex128', 'complex128')
    def return_array(a, b):
        from numpy import array
        x = array([a,b], dtype=type(a))
        y = array([a,b], dtype=type(a))
        return x, y

    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=int)
    integer32 = randint(min_int32, max_int32, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2)
    fl32 = uniform(min_float32 / 2, max_float32 / 2)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2)

    cmplx128_from_float32 = uniform(low=min_float32 / 2, high=max_float32 / 2) + uniform(low=min_float32 / 2, high=max_float32 / 2) * 1j
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = np.complex128(uniform(low=min_float64 / 2, high=max_float64 / 2) + uniform(low=min_float64 / 2, high=max_float64 / 2) * 1j)

    epyccel_func = epyccel(return_array, language=language)

    f_bl_true_output = epyccel_func(True, True)
    test_bool_true_output = return_array(True, True)

    f_bl_false_output = epyccel_func(False, False)
    test_bool_false_output = return_array(False, False)

    assert np.array_equal(f_bl_true_output, test_bool_true_output)
    assert np.array_equal(f_bl_false_output, test_bool_false_output)

    assert f_bl_false_output[0].dtype == test_bool_false_output[0].dtype
    assert f_bl_true_output[0].dtype == test_bool_true_output[0].dtype

    f_integer_output = epyccel_func(integer, integer)
    test_int_output  = return_array(integer, integer)

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

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="Function in function not implemented in C"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_return_arrays_in_expression(language):
    def return_arrays_in_expression():
        def single_return():
            from numpy import array
            return array([1,2,3,4])
        b = single_return()+1

        return b

    epyccel_function = epyccel(return_arrays_in_expression, language=language)

    epyccel_function_output = epyccel_function()
    return_arrays_in_expression_output = return_arrays_in_expression()

    assert np.array_equal(epyccel_function_output, return_arrays_in_expression_output)
    assert epyccel_function_output.dtype == return_arrays_in_expression_output.dtype

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="Function in function not implemented in C"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_return_arrays_in_expression2(language):
    def return_arrays_in_expression2(n : int):
        def single_return(n : int):
            from numpy import ones
            return ones(n)
        b = single_return(n)+1

        return b

    epyccel_function = epyccel(return_arrays_in_expression2, language=language)

    n = randint(5)

    epyccel_function_output = epyccel_function(n)
    return_arrays_in_expression2_output = return_arrays_in_expression2(n)

    assert np.array_equal(epyccel_function_output, return_arrays_in_expression2_output)
    assert epyccel_function_output.dtype == return_arrays_in_expression2_output.dtype
