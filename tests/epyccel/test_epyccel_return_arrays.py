# pylint: disable=wrong-import-position, disable=missing-function-docstring, missing-module-docstring/
import sys
import pytest
from numpy.random import randint, uniform
import numpy as np

sys.path.append('recognised_functions')

from test_numpy_funcs import (min_int, max_int, min_int8, max_int8,
                                min_int16, max_int16, min_int32, max_int32, max_int64, min_int64)
from test_numpy_funcs import max_float, min_float, max_float32, min_float32,max_float64, min_float64

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

    size = (1, 1)

    bl = randint(0, 1, size=size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=int)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(min_float, max_float, size = size)
    fl32 = uniform(min_float32, max_float32, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64, max_float64, size = size)

    cmplx128_from_float32 = uniform(low=min_float32, high=max_float32, size=size) + uniform(low=min_float32, high=max_float32, size=size) * 1j
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = uniform(low=min_float64, high=max_float64, size=size) + uniform(low=min_float64, high=max_float64, size=size) * 1j

    epyccel_func = epyccel(return_array, language=language)

    assert np.allclose(epyccel_func(bl, bl), return_array(bl, bl))
    assert np.allclose(epyccel_func(integer8, integer8), return_array(integer8, integer8))
    assert np.allclose(epyccel_func(integer16, integer16), return_array(integer16, integer16))
    assert np.allclose(epyccel_func(integer, integer), return_array(integer, integer))
    assert np.allclose(epyccel_func(integer32, integer32), return_array(integer32, integer32))
    assert np.allclose(epyccel_func(integer64, integer64), return_array(integer64, integer64))
    assert np.allclose(epyccel_func(fl, fl), return_array(fl, fl))
    assert np.allclose(epyccel_func(fl32, fl32), return_array(fl32, fl32))
    assert np.allclose(epyccel_func(fl64, fl64), return_array(fl64, fl64))
    assert np.allclose(epyccel_func(cmplx64, cmplx64), return_array(cmplx64, cmplx64))
    assert np.allclose(epyccel_func(cmplx128, cmplx128), return_array(cmplx128, cmplx128))
