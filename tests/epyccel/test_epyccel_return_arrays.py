# pylint: disable=wrong-import-position, disable=missing-function-docstring, missing-module-docstring/
import sys
from numpy.random import randint, uniform
import numpy as np
import pytest

sys.path.append('recognised_functions')

from recognised_functions.test_numpy_funcs import (min_int, max_int, min_int8, max_int8,
                                min_int16, max_int16, min_int32, max_int32, max_int64, min_int64)
from recognised_functions.test_numpy_funcs import max_float, min_float, max_float32, min_float32,max_float64, min_float64
from recognised_functions.test_numpy_funcs import matching_types
from pyccel.epyccel import epyccel
from pyccel.decorators import types


def test_return_arrays(language):
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
        x = array([a,b])
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

    assert np.allclose(f_bl_true_output, test_bool_true_output)
    assert np.allclose(f_bl_false_output, test_bool_false_output)

    assert matching_types(f_bl_true_output, test_bool_true_output)
    assert matching_types(f_bl_false_output, test_bool_false_output)

    f_integer_output = epyccel_func(integer, integer)
    test_int_output  = return_array(integer, integer)

    assert np.allclose(f_integer_output, test_int_output)
    assert matching_types(f_integer_output, test_int_output)

    f_integer8_output = epyccel_func(integer8, integer8)
    test_int8_output = return_array(integer8, integer8)

    assert np.allclose(f_integer8_output, test_int8_output)
    assert matching_types(f_integer8_output, test_int8_output)

    f_integer16_output = epyccel_func(integer16, integer16)
    test_int16_output = return_array(integer16, integer16)

    assert np.allclose(f_integer16_output, test_int16_output)
    assert matching_types(f_integer16_output, test_int16_output)

    f_integer32_output = epyccel_func(integer32, integer32)
    test_int32_output = return_array(integer32, integer32)

    assert np.allclose(f_integer32_output, test_int32_output)
    assert matching_types(f_integer32_output, test_int32_output)

    # the if block should be removed after resolving (https://github.com/pyccel/pyccel/issues/735).
    if sys.platform != 'win32':
        f_integer64_output = epyccel_func(integer64, integer64)
        test_int64_output = return_array(integer64, integer64)

        assert np.allclose(f_integer64_output, test_int64_output)
        assert matching_types(f_integer64_output, test_int64_output)

    f_fl_output = epyccel_func(fl, fl)
    test_float_output = return_array(fl, fl)

    assert np.allclose(f_fl_output, test_float_output)
    assert matching_types(f_fl_output, test_float_output)

    f_fl32_output = epyccel_func(fl32, fl32)
    test_float32_output = return_array(fl32, fl32)

    assert np.allclose(f_fl32_output, test_float32_output)
    assert matching_types(f_fl32_output, test_float32_output)

    f_fl64_output = epyccel_func(fl64, fl64)
    test_float64_output = return_array(fl64, fl64)

    assert np.allclose(f_fl64_output, test_float64_output)
    assert matching_types(f_fl64_output, test_float64_output)

    f_cmplx64_output = epyccel_func(cmplx64, cmplx64)
    test_cmplx64_output = return_array(cmplx64, cmplx64)

    assert np.allclose(f_cmplx64_output, test_cmplx64_output)
    assert matching_types(f_cmplx64_output, test_cmplx64_output)

    f_cmplx128_output = epyccel_func(cmplx128, cmplx128)
    test_cmplx128_output = return_array(cmplx128, cmplx128)

    assert np.allclose(f_cmplx128_output, test_cmplx128_output)
    assert matching_types(f_cmplx128_output, test_cmplx128_output)


@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="Unravelling loops is not working yet."),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_return_arrays_in_expression(language):
    def return_arrays_in_expression():
        def single_return():
            import numpy as np
            return np.array([1,2,3,4])
        b = single_return()+1

        return b

    epyccel_function = epyccel(return_arrays_in_expression, language=language)

    assert np.allclose(epyccel_function(), return_arrays_in_expression())
