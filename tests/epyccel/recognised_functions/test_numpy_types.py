# pylint: disable=missing-function-docstring, missing-module-docstring
import sys
import pytest
from numpy.random import rand, randint, uniform
from numpy import isclose, iinfo, finfo
import numpy as np

from pyccel.decorators import types
from pyccel.epyccel import epyccel

from test_numpy_funcs import (min_int, max_int, min_int8, max_int8,
                                min_int16, max_int16, min_int32, max_int32, max_int64, min_int64)
from test_numpy_funcs import max_float, min_float, max_float32, min_float32,max_float64, min_float64


from test_numpy_funcs import matching_types

def test_numpy_bool_scalar(language):

    @types('bool')
    @types('int')
    @types('int8')
    @types('int16')
    @types('int32')
    @types('int64')
    @types('float')
    @types('float32')
    @types('float64')
    def get_bool(a):
        from numpy import bool as NumpyBool
        b = NumpyBool(a)
        return b

    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=np.int)
    integer32 = randint(min_int32, max_int32, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2)
    fl32 = uniform(min_float32 / 2, max_float32 / 2)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2)


    epyccel_func = epyccel(get_bool, language=language)

    f_bl_true_output = epyccel_func(True)
    test_bool_true_output = get_bool(True)

    f_bl_false_output = epyccel_func(False)
    test_bool_false_output = get_bool(False)

    assert f_bl_true_output == test_bool_true_output
    assert f_bl_false_output == test_bool_false_output

    assert matching_types(f_bl_true_output, test_bool_true_output)
    assert matching_types(f_bl_false_output, test_bool_false_output)

    f_integer_output = epyccel_func(integer)
    test_int_output  = get_bool(integer)

    assert f_integer_output == test_int_output
    assert matching_types(f_integer_output, test_int_output)

    f_integer8_output = epyccel_func(integer8)
    test_int8_output = get_bool(integer8)

    assert f_integer8_output == test_int8_output
    assert matching_types(f_integer8_output, test_int8_output)

    f_integer16_output = epyccel_func(integer16)
    test_int16_output = get_bool(integer16)

    assert f_integer16_output == test_int16_output
    assert matching_types(f_integer16_output, test_int16_output)

    f_integer32_output = epyccel_func(integer32)
    test_int32_output = get_bool(integer32)

    assert f_integer32_output == test_int32_output
    assert matching_types(f_integer32_output, test_int32_output)

    # the if block should be removed after resolving (https://github.com/pyccel/pyccel/issues/735).
    if sys.platform != 'win32':
        f_integer64_output = epyccel_func(integer64)
        test_int64_output = get_bool(integer64)

        assert f_integer64_output == test_int64_output
        assert matching_types(f_integer64_output, test_int64_output)

    f_fl_output = epyccel_func(fl)
    test_float_output = get_bool(fl)

    assert f_fl_output == test_float_output
    assert matching_types(f_fl_output, test_float_output)

    f_fl32_output = epyccel_func(fl32)
    test_float32_output = get_bool(fl32)

    assert f_fl32_output == test_float32_output
    assert matching_types(f_fl32_output, test_float32_output)

    f_fl64_output = epyccel_func(fl64)
    test_float64_output = get_bool(fl64)

    assert f_fl64_output == test_float64_output
    assert matching_types(f_fl64_output, test_float64_output)

def test_numpy_int_scalar(language):

    @types('bool')
    @types('int')
    @types('int8')
    @types('int16')
    @types('int32')
    @types('int64')
    @types('float')
    @types('float32')
    @types('float64')
    def get_int(a):
        from numpy import int as NumpyInt
        b = NumpyInt(a)
        return b

    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=np.int)
    integer32 = randint(min_int32, max_int32, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, dtype=np.int64)

    fl = uniform(min_int, max_int)
    fl32 = uniform(min_int, max_int)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_int, max_int)

    epyccel_func = epyccel(get_int, language=language)

    f_bl_true_output = epyccel_func(True)
    test_bool_true_output = get_int(True)

    f_bl_false_output = epyccel_func(False)
    test_bool_false_output = get_int(False)

    assert f_bl_true_output == test_bool_true_output
    assert f_bl_false_output == test_bool_false_output

    assert matching_types(f_bl_true_output, test_bool_true_output)
    assert matching_types(f_bl_false_output, test_bool_false_output)

    f_integer_output = epyccel_func(integer)
    test_int_output  = get_int(integer)

    assert f_integer_output == test_int_output
    assert matching_types(f_integer_output, test_int_output)

    f_integer8_output = epyccel_func(integer8)
    test_int8_output = get_int(integer8)

    assert f_integer8_output == test_int8_output
    assert matching_types(f_integer8_output, test_int8_output)

    f_integer16_output = epyccel_func(integer16)
    test_int16_output = get_int(integer16)

    assert f_integer16_output == test_int16_output
    assert matching_types(f_integer16_output, test_int16_output)

    f_integer32_output = epyccel_func(integer32)
    test_int32_output = get_int(integer32)

    assert f_integer32_output == test_int32_output
    assert matching_types(f_integer32_output, test_int32_output)

    # the if block should be removed after resolving (https://github.com/pyccel/pyccel/issues/735).
    if sys.platform != 'win32':
        f_integer64_output = epyccel_func(integer64)
        test_int64_output = get_int(integer64)

        assert f_integer64_output == test_int64_output
        assert matching_types(f_integer64_output, test_int64_output)

    f_fl_output = epyccel_func(fl)
    test_float_output = get_int(fl)

    assert f_fl_output == test_float_output
    assert matching_types(f_fl_output, test_float_output)

    f_fl32_output = epyccel_func(fl32)
    test_float32_output = get_int(fl32)

    assert f_fl32_output == test_float32_output
    assert matching_types(f_fl32_output, test_float32_output)

    f_fl64_output = epyccel_func(fl64)
    test_float64_output = get_int(fl64)

    assert f_fl64_output == test_float64_output
    assert matching_types(f_fl64_output, test_float64_output)

def test_numpy_int8_scalar(language):

    @types('bool')
    @types('int')
    @types('int8')
    @types('int16')
    @types('int32')
    @types('int64')
    @types('float')
    @types('float32')
    @types('float64')
    def get_int8(a):
        from numpy import int8
        b = int8(a)
        return b

    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=np.int)
    integer32 = randint(min_int32, max_int32, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, dtype=np.int64)

    fl = uniform(min_int8, max_int8)
    fl32 = uniform(min_int8, max_int8)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_int8, max_int8)


    epyccel_func = epyccel(get_int8, language=language)

    f_bl_true_output = epyccel_func(True)
    test_bool_true_output = get_int8(True)

    f_bl_false_output = epyccel_func(False)
    test_bool_false_output = get_int8(False)

    assert f_bl_true_output == test_bool_true_output
    assert f_bl_false_output == test_bool_false_output

    assert matching_types(f_bl_true_output, test_bool_true_output)
    assert matching_types(f_bl_false_output, test_bool_false_output)

    f_integer_output = epyccel_func(integer)
    test_int_output  = get_int8(integer)

    assert f_integer_output == test_int_output
    assert matching_types(f_integer_output, test_int_output)

    f_integer8_output = epyccel_func(integer8)
    test_int8_output = get_int8(integer8)

    assert f_integer8_output == test_int8_output
    assert matching_types(f_integer8_output, test_int8_output)

    f_integer16_output = epyccel_func(integer16)
    test_int16_output = get_int8(integer16)

    assert f_integer16_output == test_int16_output
    assert matching_types(f_integer16_output, test_int16_output)

    f_integer32_output = epyccel_func(integer32)
    test_int32_output = get_int8(integer32)

    assert f_integer32_output == test_int32_output
    assert matching_types(f_integer32_output, test_int32_output)

    # the if block should be removed after resolving (https://github.com/pyccel/pyccel/issues/735).
    if sys.platform != 'win32':
        f_integer64_output = epyccel_func(integer64)
        test_int64_output = get_int8(integer64)

        assert f_integer64_output == test_int64_output
        assert matching_types(f_integer64_output, test_int64_output)

    f_fl_output = epyccel_func(fl)
    test_float_output = get_int8(fl)

    assert f_fl_output == test_float_output
    assert matching_types(f_fl_output, test_float_output)

    f_fl32_output = epyccel_func(fl32)
    test_float32_output = get_int8(fl32)

    assert f_fl32_output == test_float32_output
    assert matching_types(f_fl32_output, test_float32_output)

    f_fl64_output = epyccel_func(fl64)
    test_float64_output = get_int8(fl64)

    assert f_fl64_output == test_float64_output
    assert matching_types(f_fl64_output, test_float64_output)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Arrays not handled yet."),
            pytest.mark.c]
        ),
        pytest.param("python", marks = [
            pytest.mark.python]
        )
    )
)

def test_numpy_int8_array_like_1d(language):

    @types('bool[:]')
    @types('int[:]')
    @types('int8[:]')
    @types('int16[:]')
    @types('int32[:]')
    @types('int64[:]')
    @types('float[:]')
    @types('float32[:]')
    @types('float64[:]')
    def get_int8(arr):
        from numpy import int8, shape
        a = int8(arr)
        s = shape(a)
        return len(s), s[0], a[0]

    size = 5

    bl = randint(0, 1, size=size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=np.int)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size = size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size = size)

    epyccel_func = epyccel(get_int8, language=language)

    assert epyccel_func(bl) == get_int8(bl)
    assert epyccel_func(integer8) == get_int8(integer8)
    assert epyccel_func(integer16) == get_int8(integer16)
    assert epyccel_func(integer) == get_int8(integer)
    assert epyccel_func(integer32) == get_int8(integer32)
    # the if block should be removed after resolving (https://github.com/pyccel/pyccel/issues/735).
    if sys.platform != 'win32':
        assert epyccel_func(integer64) == get_int8(integer64)
        assert epyccel_func(fl) == get_int8(fl)
        assert epyccel_func(fl64) == get_int8(fl64)
    assert epyccel_func(fl32) == get_int8(fl32)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Arrays not handled yet."),
            pytest.mark.c]
        ),
        pytest.param("python", marks = [
            pytest.mark.python]
        )
    )
)

def test_numpy_int8_array_like_2d(language):

    @types('bool[:,:]')
    @types('int[:,:]')
    @types('int8[:,:]')
    @types('int16[:,:]')
    @types('int32[:,:]')
    @types('int64[:,:]')
    @types('float[:,:]')
    @types('float32[:,:]')
    @types('float64[:,:]')
    def get_int8(arr):
        from numpy import int8, shape
        a = int8(arr)
        s = shape(a)
        return len(s), s[0], s[1], a[0,0], a[0,1]

    size = (2, 5)

    bl = randint(0, 1, size=size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=np.int)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size = size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size = size)

    epyccel_func = epyccel(get_int8, language=language)

    assert epyccel_func(bl) == get_int8(bl)
    assert epyccel_func(integer8) == get_int8(integer8)
    assert epyccel_func(integer16) == get_int8(integer16)
    assert epyccel_func(integer) == get_int8(integer)
    assert epyccel_func(integer32) == get_int8(integer32)
    # the if block should be removed after resolving (https://github.com/pyccel/pyccel/issues/735).
    if sys.platform != 'win32':
        assert epyccel_func(integer64) == get_int8(integer64)
        assert epyccel_func(fl) == get_int8(fl)
        assert epyccel_func(fl64) == get_int8(fl64)
    assert epyccel_func(fl32) == get_int8(fl32)


def test_numpy_int16_scalar(language):

    @types('bool')
    @types('int')
    @types('int8')
    @types('int16')
    @types('int32')
    @types('int64')
    @types('float')
    @types('float32')
    @types('float64')
    def get_int16(a):
        from numpy import int16
        b = int16(a)
        return b

    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=np.int)
    integer32 = randint(min_int32, max_int32, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, dtype=np.int64)

    fl = uniform(min_int16, max_int16)
    fl32 = uniform(min_int16, max_int16)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_int16, max_int16)


    epyccel_func = epyccel(get_int16, language=language)

    f_bl_true_output = epyccel_func(True)
    test_bool_true_output = get_int16(True)

    f_bl_false_output = epyccel_func(False)
    test_bool_false_output = get_int16(False)

    assert f_bl_true_output == test_bool_true_output
    assert f_bl_false_output == test_bool_false_output

    assert matching_types(f_bl_true_output, test_bool_true_output)
    assert matching_types(f_bl_false_output, test_bool_false_output)

    f_integer_output = epyccel_func(integer)
    test_int_output  = get_int16(integer)

    assert f_integer_output == test_int_output
    assert matching_types(f_integer_output, test_int_output)

    f_integer8_output = epyccel_func(integer8)
    test_int8_output = get_int16(integer8)

    assert f_integer8_output == test_int8_output
    assert matching_types(f_integer8_output, test_int8_output)

    f_integer16_output = epyccel_func(integer16)
    test_int16_output = get_int16(integer16)

    assert f_integer16_output == test_int16_output
    assert matching_types(f_integer16_output, test_int16_output)

    f_integer32_output = epyccel_func(integer32)
    test_int32_output = get_int16(integer32)

    assert f_integer32_output == test_int32_output
    assert matching_types(f_integer32_output, test_int32_output)

    # the if block should be removed after resolving (https://github.com/pyccel/pyccel/issues/735).
    if sys.platform != 'win32':
        f_integer64_output = epyccel_func(integer64)
        test_int64_output = get_int16(integer64)

        assert f_integer64_output == test_int64_output
        assert matching_types(f_integer64_output, test_int64_output)

    f_fl_output = epyccel_func(fl)
    test_float_output = get_int16(fl)

    assert f_fl_output == test_float_output
    assert matching_types(f_fl_output, test_float_output)

    f_fl32_output = epyccel_func(fl32)
    test_float32_output = get_int16(fl32)

    assert f_fl32_output == test_float32_output
    assert matching_types(f_fl32_output, test_float32_output)

    f_fl64_output = epyccel_func(fl64)
    test_float64_output = get_int16(fl64)

    assert f_fl64_output == test_float64_output
    assert matching_types(f_fl64_output, test_float64_output)


@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Arrays not handled yet."),
            pytest.mark.c]
        ),
        pytest.param("python", marks = [
            pytest.mark.python]
        )
    )
)

def test_numpy_int16_array_like_1d(language):

    @types('bool[:]')
    @types('int[:]')
    @types('int8[:]')
    @types('int16[:]')
    @types('int32[:]')
    @types('int64[:]')
    @types('float[:]')
    @types('float32[:]')
    @types('float64[:]')
    def get_int16(arr):
        from numpy import int16, shape
        a = int16(arr)
        s = shape(a)
        return len(s), s[0], a[0]

    size = 5

    bl = randint(0, 1, size=size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=np.int)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size = size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size = size)

    epyccel_func = epyccel(get_int16, language=language)

    assert epyccel_func(bl) == get_int16(bl)
    assert epyccel_func(integer8) == get_int16(integer8)
    assert epyccel_func(integer16) == get_int16(integer16)
    assert epyccel_func(integer) == get_int16(integer)
    assert epyccel_func(integer32) == get_int16(integer32)
    # the if block should be removed after resolving (https://github.com/pyccel/pyccel/issues/735).
    if sys.platform != 'win32':
        assert epyccel_func(integer64) == get_int16(integer64)
        assert epyccel_func(fl) == get_int16(fl)
        assert epyccel_func(fl64) == get_int16(fl64)
    assert epyccel_func(fl32) == get_int16(fl32)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Arrays not handled yet."),
            pytest.mark.c]
        ),
        pytest.param("python", marks = [
            pytest.mark.python]
        )
    )
)

def test_numpy_int16_array_like_2d(language):

    @types('bool[:,:]')
    @types('int[:,:]')
    @types('int8[:,:]')
    @types('int16[:,:]')
    @types('int32[:,:]')
    @types('int64[:,:]')
    @types('float[:,:]')
    @types('float32[:,:]')
    @types('float64[:,:]')
    def get_int16(arr):
        from numpy import int16, shape
        a = int16(arr)
        s = shape(a)
        return len(s), s[0], s[1], a[0,0], a[0,1]

    size = (2, 5)

    bl = randint(0, 1, size=size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=np.int)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size = size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size = size)

    epyccel_func = epyccel(get_int16, language=language)

    assert epyccel_func(bl) == get_int16(bl)
    assert epyccel_func(integer8) == get_int16(integer8)
    assert epyccel_func(integer16) == get_int16(integer16)
    assert epyccel_func(integer) == get_int16(integer)
    assert epyccel_func(integer32) == get_int16(integer32)
    # the if block should be removed after resolving (https://github.com/pyccel/pyccel/issues/735).
    if sys.platform != 'win32':
        assert epyccel_func(integer64) == get_int16(integer64)
        assert epyccel_func(fl) == get_int16(fl)
        assert epyccel_func(fl64) == get_int16(fl64)
    assert epyccel_func(fl32) == get_int16(fl32)

def test_numpy_int32_scalar(language):

    @types('bool')
    @types('int')
    @types('int8')
    @types('int16')
    @types('int32')
    @types('int64')
    @types('float')
    @types('float32')
    @types('float64')
    def get_int32(a):
        from numpy import int32
        b = int32(a)
        return b

    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=np.int)
    integer32 = randint(min_int32, max_int32, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, dtype=np.int64)

    fl = uniform(min_int32 / 2, max_int32 / 2)
    fl32 = uniform(min_int32, max_int32)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_int32 / 2, max_int32 / 2)


    epyccel_func = epyccel(get_int32, language=language)

    f_bl_true_output = epyccel_func(True)
    test_bool_true_output = get_int32(True)

    f_bl_false_output = epyccel_func(False)
    test_bool_false_output = get_int32(False)

    assert f_bl_true_output == test_bool_true_output
    assert f_bl_false_output == test_bool_false_output

    assert matching_types(f_bl_true_output, test_bool_true_output)
    assert matching_types(f_bl_false_output, test_bool_false_output)

    f_integer_output = epyccel_func(integer)
    test_int_output  = get_int32(integer)

    assert f_integer_output == test_int_output
    assert matching_types(f_integer_output, test_int_output)

    f_integer8_output = epyccel_func(integer8)
    test_int8_output = get_int32(integer8)

    assert f_integer8_output == test_int8_output
    assert matching_types(f_integer8_output, test_int8_output)

    f_integer16_output = epyccel_func(integer16)
    test_int16_output = get_int32(integer16)

    assert f_integer16_output == test_int16_output
    assert matching_types(f_integer16_output, test_int16_output)

    f_integer32_output = epyccel_func(integer32)
    test_int32_output = get_int32(integer32)

    assert f_integer32_output == test_int32_output
    assert matching_types(f_integer32_output, test_int32_output)

    # the if block should be removed after resolving (https://github.com/pyccel/pyccel/issues/735).
    if sys.platform != 'win32':
        f_integer64_output = epyccel_func(integer64)
        test_int64_output = get_int32(integer64)

        assert f_integer64_output == test_int64_output
        assert matching_types(f_integer64_output, test_int64_output)

    f_fl_output = epyccel_func(fl)
    test_float_output = get_int32(fl)

    assert f_fl_output == test_float_output
    assert matching_types(f_fl_output, test_float_output)

    f_fl32_output = epyccel_func(fl32)
    test_float32_output = get_int32(fl32)

    assert f_fl32_output == test_float32_output
    assert matching_types(f_fl32_output, test_float32_output)

    f_fl64_output = epyccel_func(fl64)
    test_float64_output = get_int32(fl64)

    assert f_fl64_output == test_float64_output
    assert matching_types(f_fl64_output, test_float64_output)


@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Arrays not handled yet."),
            pytest.mark.c]
        ),
        pytest.param("python", marks = [
            pytest.mark.python]
        )
    )
)

def test_numpy_int32_array_like_1d(language):

    @types('bool[:]')
    @types('int[:]')
    @types('int8[:]')
    @types('int16[:]')
    @types('int32[:]')
    @types('int64[:]')
    @types('float[:]')
    @types('float32[:]')
    @types('float64[:]')
    def get_int32(arr):
        from numpy import int32, shape
        a = int32(arr)
        s = shape(a)
        return len(s), s[0], a[0]

    size = 5

    bl = randint(0, 1, size=size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=np.int)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size = size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size = size)

    epyccel_func = epyccel(get_int32, language=language)

    assert epyccel_func(bl) == get_int32(bl)
    assert epyccel_func(integer8) == get_int32(integer8)
    assert epyccel_func(integer16) == get_int32(integer16)
    assert epyccel_func(integer) == get_int32(integer)
    assert epyccel_func(integer32) == get_int32(integer32)
    # the if block should be removed after resolving (https://github.com/pyccel/pyccel/issues/735).
    if sys.platform != 'win32':
        assert epyccel_func(integer64) == get_int32(integer64)
        assert epyccel_func(fl) == get_int32(fl)
        assert epyccel_func(fl64) == get_int32(fl64)
    assert epyccel_func(fl32) == get_int32(fl32)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Arrays not handled yet."),
            pytest.mark.c]
        ),
        pytest.param("python", marks = [
            pytest.mark.python]
        )
    )
)

def test_numpy_int32_array_like_2d(language):

    @types('bool[:,:]')
    @types('int[:,:]')
    @types('int8[:,:]')
    @types('int16[:,:]')
    @types('int32[:,:]')
    @types('int64[:,:]')
    @types('float[:,:]')
    @types('float32[:,:]')
    @types('float64[:,:]')
    def get_int32(arr):
        from numpy import int32, shape
        a = int32(arr)
        s = shape(a)
        return len(s), s[0], s[1], a[0,0], a[0,1]

    size = (2, 5)

    bl = randint(0, 1, size=size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=np.int)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size = size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size = size)

    epyccel_func = epyccel(get_int32, language=language)

    assert epyccel_func(bl) == get_int32(bl)
    assert epyccel_func(integer8) == get_int32(integer8)
    assert epyccel_func(integer16) == get_int32(integer16)
    assert epyccel_func(integer) == get_int32(integer)
    assert epyccel_func(integer32) == get_int32(integer32)
    # the if block should be removed after resolving (https://github.com/pyccel/pyccel/issues/735).
    if sys.platform != 'win32':
        assert epyccel_func(integer64) == get_int32(integer64)
        assert epyccel_func(fl) == get_int32(fl)
        assert epyccel_func(fl64) == get_int32(fl64)
    assert epyccel_func(fl32) == get_int32(fl32)

def test_numpy_int64_scalar(language):

    @types('bool')
    @types('int')
    @types('int8')
    @types('int16')
    @types('int32')
    @types('int64')
    @types('float')
    @types('float32')
    @types('float64')
    def get_int64(a):
        from numpy import int64
        b = int64(a)
        return b

    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=np.int)
    integer32 = randint(min_int32, max_int32, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, dtype=np.int64)

    fl = uniform(min_int64, max_int64)
    fl32 = uniform(min_int64, max_int64)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_int64, max_int64)


    epyccel_func = epyccel(get_int64, language=language)

    f_bl_true_output = epyccel_func(True)
    test_bool_true_output = get_int64(True)

    f_bl_false_output = epyccel_func(False)
    test_bool_false_output = get_int64(False)

    assert f_bl_true_output == test_bool_true_output
    assert f_bl_false_output == test_bool_false_output

    assert matching_types(f_bl_true_output, test_bool_true_output)
    assert matching_types(f_bl_false_output, test_bool_false_output)

    f_integer_output = epyccel_func(integer)
    test_int_output  = get_int64(integer)

    assert f_integer_output == test_int_output
    assert matching_types(f_integer_output, test_int_output)

    f_integer8_output = epyccel_func(integer8)
    test_int8_output = get_int64(integer8)

    assert f_integer8_output == test_int8_output
    assert matching_types(f_integer8_output, test_int8_output)

    f_integer16_output = epyccel_func(integer16)
    test_int16_output = get_int64(integer16)

    assert f_integer16_output == test_int16_output
    assert matching_types(f_integer16_output, test_int16_output)

    f_integer32_output = epyccel_func(integer32)
    test_int32_output = get_int64(integer32)

    assert f_integer32_output == test_int32_output
    assert matching_types(f_integer32_output, test_int32_output)

    # the if block should be removed after resolving (https://github.com/pyccel/pyccel/issues/735).
    if sys.platform != 'win32':
        f_integer64_output = epyccel_func(integer64)
        test_int64_output = get_int64(integer64)

        assert f_integer64_output == test_int64_output
        assert matching_types(f_integer64_output, test_int64_output)

    f_fl_output = epyccel_func(fl)
    test_float_output = get_int64(fl)

    assert f_fl_output == test_float_output
    assert matching_types(f_fl_output, test_float_output)

    f_fl32_output = epyccel_func(fl32)
    test_float32_output = get_int64(fl32)

    assert f_fl32_output == test_float32_output
    assert matching_types(f_fl32_output, test_float32_output)

    f_fl64_output = epyccel_func(fl64)
    test_float64_output = get_int64(fl64)

    assert f_fl64_output == test_float64_output
    assert matching_types(f_fl64_output, test_float64_output)


@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Arrays not handled yet."),
            pytest.mark.c]
        ),
        pytest.param("python", marks = [
            pytest.mark.python]
        )
    )
)

def test_numpy_int64_array_like_1d(language):

    @types('bool[:]')
    @types('int[:]')
    @types('int8[:]')
    @types('int16[:]')
    @types('int32[:]')
    @types('int64[:]')
    @types('float[:]')
    @types('float32[:]')
    @types('float64[:]')
    def get_int64(arr):
        from numpy import int64, shape
        a = int64(arr)
        s = shape(a)
        return len(s), s[0], a[0]

    size = 5

    bl = randint(0, 1, size=size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=np.int)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size = size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size = size)

    epyccel_func = epyccel(get_int64, language=language)

    assert epyccel_func(bl) == get_int64(bl)
    assert epyccel_func(integer8) == get_int64(integer8)
    assert epyccel_func(integer16) == get_int64(integer16)
    assert epyccel_func(integer) == get_int64(integer)
    assert epyccel_func(integer32) == get_int64(integer32)
    # the if block should be removed after resolving (https://github.com/pyccel/pyccel/issues/735).
    if sys.platform != 'win32':
        assert epyccel_func(integer64) == get_int64(integer64)
        assert epyccel_func(fl) == get_int64(fl)
        assert epyccel_func(fl64) == get_int64(fl64)
    assert epyccel_func(fl32) == get_int64(fl32)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Arrays not handled yet."),
            pytest.mark.c]
        ),
        pytest.param("python", marks = [
            pytest.mark.python]
        )
    )
)

def test_numpy_int64_array_like_2d(language):

    @types('bool[:,:]')
    @types('int[:,:]')
    @types('int8[:,:]')
    @types('int16[:,:]')
    @types('int32[:,:]')
    @types('int64[:,:]')
    @types('float[:,:]')
    @types('float32[:,:]')
    @types('float64[:,:]')
    def get_int64(arr):
        from numpy import int64, shape
        a = int64(arr)
        s = shape(a)
        return len(s), s[0], s[1], a[0,0], a[0,1]

    size = (2, 5)

    bl = randint(0, 1, size=size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=np.int)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size = size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size = size)

    epyccel_func = epyccel(get_int64, language=language)

    assert epyccel_func(bl) == get_int64(bl)
    assert epyccel_func(integer8) == get_int64(integer8)
    assert epyccel_func(integer16) == get_int64(integer16)
    assert epyccel_func(integer) == get_int64(integer)
    assert epyccel_func(integer32) == get_int64(integer32)
    # the if block should be removed after resolving (https://github.com/pyccel/pyccel/issues/735).
    if sys.platform != 'win32':
        assert epyccel_func(integer64) == get_int64(integer64)
        assert epyccel_func(fl) == get_int64(fl)
        assert epyccel_func(fl64) == get_int64(fl64)
    assert epyccel_func(fl32) == get_int64(fl32)

def test_numpy_float_scalar(language):

    @types('bool')
    @types('int')
    @types('int8')
    @types('int16')
    @types('int32')
    @types('int64')
    @types('float')
    @types('float32')
    @types('float64')
    def get_float(a):
        from numpy import float as NumpyFloat
        b = NumpyFloat(a)
        return b

    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=np.int)
    integer32 = randint(min_int32, max_int32, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2)
    fl32 = uniform(min_float32 / 2, max_float32 / 2)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2)


    epyccel_func = epyccel(get_float, language=language)

    f_bl_true_output = epyccel_func(True)
    test_bool_true_output = get_float(True)

    f_bl_false_output = epyccel_func(False)
    test_bool_false_output = get_float(False)

    assert f_bl_true_output == test_bool_true_output
    assert f_bl_false_output == test_bool_false_output

    assert matching_types(f_bl_true_output, test_bool_true_output)
    assert matching_types(f_bl_false_output, test_bool_false_output)

    f_integer_output = epyccel_func(integer)
    test_int_output  = get_float(integer)

    assert f_integer_output == test_int_output
    assert matching_types(f_integer_output, test_int_output)

    f_integer8_output = epyccel_func(integer8)
    test_int8_output = get_float(integer8)

    assert f_integer8_output == test_int8_output
    assert matching_types(f_integer8_output, test_int8_output)

    f_integer16_output = epyccel_func(integer16)
    test_int16_output = get_float(integer16)

    assert f_integer16_output == test_int16_output
    assert matching_types(f_integer16_output, test_int16_output)

    f_integer32_output = epyccel_func(integer32)
    test_int32_output = get_float(integer32)

    assert f_integer32_output == test_int32_output
    assert matching_types(f_integer32_output, test_int32_output)

    # the if block should be removed after resolving (https://github.com/pyccel/pyccel/issues/735).
    if sys.platform != 'win32':
        f_integer64_output = epyccel_func(integer64)
        test_int64_output = get_float(integer64)

        assert f_integer64_output == test_int64_output
        assert matching_types(f_integer64_output, test_int64_output)

    f_fl_output = epyccel_func(fl)
    test_float_output = get_float(fl)

    assert f_fl_output == test_float_output
    assert matching_types(f_fl_output, test_float_output)

    f_fl32_output = epyccel_func(fl32)
    test_float32_output = get_float(fl32)

    assert f_fl32_output == test_float32_output
    assert matching_types(f_fl32_output, test_float32_output)

    f_fl64_output = epyccel_func(fl64)
    test_float64_output = get_float(fl64)

    assert f_fl64_output == test_float64_output
    assert matching_types(f_fl64_output, test_float64_output)

def test_numpy_float32_scalar(language):

    @types('bool')
    @types('int')
    @types('int8')
    @types('int16')
    @types('int32')
    @types('int64')
    @types('float')
    @types('float32')
    @types('float64')
    def get_float32(a):
        from numpy import float32
        b = float32(a)
        return b

    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=np.int)
    integer32 = randint(min_int32, max_int32, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2)
    fl32 = uniform(min_float32 / 2, max_float32 / 2)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2)


    epyccel_func = epyccel(get_float32, language=language)

    f_bl_true_output = epyccel_func(True)
    test_bool_true_output = get_float32(True)

    f_bl_false_output = epyccel_func(False)
    test_bool_false_output = get_float32(False)

    assert f_bl_true_output == test_bool_true_output
    assert f_bl_false_output == test_bool_false_output

    assert matching_types(f_bl_true_output, test_bool_true_output)
    assert matching_types(f_bl_false_output, test_bool_false_output)

    f_integer_output = epyccel_func(integer)
    test_int_output  = get_float32(integer)

    assert f_integer_output == test_int_output
    assert matching_types(f_integer_output, test_int_output)

    f_integer8_output = epyccel_func(integer8)
    test_int8_output = get_float32(integer8)

    assert f_integer8_output == test_int8_output
    assert matching_types(f_integer8_output, test_int8_output)

    f_integer16_output = epyccel_func(integer16)
    test_int16_output = get_float32(integer16)

    assert f_integer16_output == test_int16_output
    assert matching_types(f_integer16_output, test_int16_output)

    f_integer32_output = epyccel_func(integer32)
    test_int32_output = get_float32(integer32)

    assert f_integer32_output == test_int32_output
    assert matching_types(f_integer32_output, test_int32_output)

    # the if block should be removed after resolving (https://github.com/pyccel/pyccel/issues/735).
    if sys.platform != 'win32':
        f_integer64_output = epyccel_func(integer64)
        test_int64_output = get_float32(integer64)

        assert f_integer64_output == test_int64_output
        assert matching_types(f_integer64_output, test_int64_output)

    f_fl_output = epyccel_func(fl)
    test_float_output = get_float32(fl)

    assert f_fl_output == test_float_output
    assert matching_types(f_fl_output, test_float_output)

    f_fl32_output = epyccel_func(fl32)
    test_float32_output = get_float32(fl32)

    assert f_fl32_output == test_float32_output
    assert matching_types(f_fl32_output, test_float32_output)

    f_fl64_output = epyccel_func(fl64)
    test_float64_output = get_float32(fl64)

    assert f_fl64_output == test_float64_output
    assert matching_types(f_fl64_output, test_float64_output)


@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Arrays not handled yet."),
            pytest.mark.c]
        ),
        pytest.param("python", marks = [
            pytest.mark.python]
        )
    )
)

def test_numpy_float32_array_like_1d(language):

    @types('bool[:]')
    @types('int[:]')
    @types('int8[:]')
    @types('int16[:]')
    @types('int32[:]')
    @types('int64[:]')
    @types('float[:]')
    @types('float32[:]')
    @types('float64[:]')
    def get_float32(arr):
        from numpy import float32, shape
        a = float32(arr)
        s = shape(a)
        return len(s), s[0], a[0]

    size = 5

    bl = randint(0, 1, size=size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=np.int)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size = size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size = size)

    epyccel_func = epyccel(get_float32, language=language)

    assert epyccel_func(bl) == get_float32(bl)
    assert epyccel_func(integer8) == get_float32(integer8)
    assert epyccel_func(integer16) == get_float32(integer16)
    assert epyccel_func(integer) == get_float32(integer)
    assert epyccel_func(integer32) == get_float32(integer32)
    # the if block should be removed after resolving (https://github.com/pyccel/pyccel/issues/735).
    if sys.platform != 'win32':
        assert epyccel_func(integer64) == get_float32(integer64)
        assert epyccel_func(fl) == get_float32(fl)
        assert epyccel_func(fl64) == get_float32(fl64)
    assert epyccel_func(fl32) == get_float32(fl32)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Arrays not handled yet."),
            pytest.mark.c]
        ),
        pytest.param("python", marks = [
            pytest.mark.python]
        )
    )
)

def test_numpy_float32_array_like_2d(language):

    @types('bool[:,:]')
    @types('int[:,:]')
    @types('int8[:,:]')
    @types('int16[:,:]')
    @types('int32[:,:]')
    @types('int64[:,:]')
    @types('float[:,:]')
    @types('float32[:,:]')
    @types('float64[:,:]')
    def get_float32(arr):
        from numpy import float32, shape
        a = float32(arr)
        s = shape(a)
        return len(s), s[0], s[1], a[0,0], a[0,1]

    size = (2, 5)

    bl = randint(0, 1, size=size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=np.int)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size = size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size = size)

    epyccel_func = epyccel(get_float32, language=language)

    assert epyccel_func(bl) == get_float32(bl)
    assert epyccel_func(integer8) == get_float32(integer8)
    assert epyccel_func(integer16) == get_float32(integer16)
    assert epyccel_func(integer) == get_float32(integer)
    assert epyccel_func(integer32) == get_float32(integer32)
    # the if block should be removed after resolving (https://github.com/pyccel/pyccel/issues/735).
    if sys.platform != 'win32':
        assert epyccel_func(integer64) == get_float32(integer64)
        assert epyccel_func(fl) == get_float32(fl)
        assert epyccel_func(fl64) == get_float32(fl64)
    assert epyccel_func(fl32) == get_float32(fl32)

def test_numpy_float64_scalar(language):

    @types('bool')
    @types('int')
    @types('int8')
    @types('int16')
    @types('int32')
    @types('int64')
    @types('float')
    @types('float32')
    @types('float64')
    def get_float64(a):
        from numpy import float64
        b = float64(a)
        return b

    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=np.int)
    integer32 = randint(min_int32, max_int32, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2)
    fl32 = uniform(min_float32 / 2, max_float32 / 2)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2)


    epyccel_func = epyccel(get_float64, language=language)

    f_bl_true_output = epyccel_func(True)
    test_bool_true_output = get_float64(True)

    f_bl_false_output = epyccel_func(False)
    test_bool_false_output = get_float64(False)

    assert f_bl_true_output == test_bool_true_output
    assert f_bl_false_output == test_bool_false_output

    assert matching_types(f_bl_true_output, test_bool_true_output)
    assert matching_types(f_bl_false_output, test_bool_false_output)

    f_integer_output = epyccel_func(integer)
    test_int_output  = get_float64(integer)

    assert f_integer_output == test_int_output
    assert matching_types(f_integer_output, test_int_output)

    f_integer8_output = epyccel_func(integer8)
    test_int8_output = get_float64(integer8)

    assert f_integer8_output == test_int8_output
    assert matching_types(f_integer8_output, test_int8_output)

    f_integer16_output = epyccel_func(integer16)
    test_int16_output = get_float64(integer16)

    assert f_integer16_output == test_int16_output
    assert matching_types(f_integer16_output, test_int16_output)

    f_integer32_output = epyccel_func(integer32)
    test_int32_output = get_float64(integer32)

    assert f_integer32_output == test_int32_output
    assert matching_types(f_integer32_output, test_int32_output)

    # the if block should be removed after resolving (https://github.com/pyccel/pyccel/issues/735).
    if sys.platform != 'win32':
        f_integer64_output = epyccel_func(integer64)
        test_int64_output = get_float64(integer64)

        assert f_integer64_output == test_int64_output
        assert matching_types(f_integer64_output, test_int64_output)

    f_fl_output = epyccel_func(fl)
    test_float_output = get_float64(fl)

    assert f_fl_output == test_float_output
    assert matching_types(f_fl_output, test_float_output)

    f_fl32_output = epyccel_func(fl32)
    test_float32_output = get_float64(fl32)

    assert f_fl32_output == test_float32_output
    assert matching_types(f_fl32_output, test_float32_output)

    f_fl64_output = epyccel_func(fl64)
    test_float64_output = get_float64(fl64)

    assert f_fl64_output == test_float64_output
    assert matching_types(f_fl64_output, test_float64_output)


@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Arrays not handled yet."),
            pytest.mark.c]
        ),
        pytest.param("python", marks = [
            pytest.mark.python]
        )
    )
)

def test_numpy_float64_array_like_1d(language):

    @types('bool[:]')
    @types('int[:]')
    @types('int8[:]')
    @types('int16[:]')
    @types('int32[:]')
    @types('int64[:]')
    @types('float[:]')
    @types('float32[:]')
    @types('float64[:]')
    def get_float64(arr):
        from numpy import float64, shape
        a = float64(arr)
        s = shape(a)
        return len(s), s[0], a[0]

    size = 5

    bl = randint(0, 1, size=size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=np.int)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size = size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size = size)

    epyccel_func = epyccel(get_float64, language=language)

    assert epyccel_func(bl) == get_float64(bl)
    assert epyccel_func(integer8) == get_float64(integer8)
    assert epyccel_func(integer16) == get_float64(integer16)
    assert epyccel_func(integer) == get_float64(integer)
    assert epyccel_func(integer32) == get_float64(integer32)
    # the if block should be removed after resolving (https://github.com/pyccel/pyccel/issues/735).
    if sys.platform != 'win32':
        assert epyccel_func(integer64) == get_float64(integer64)
        assert epyccel_func(fl) == get_float64(fl)
        assert epyccel_func(fl64) == get_float64(fl64)
    assert epyccel_func(fl32) == get_float64(fl32)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Arrays not handled yet."),
            pytest.mark.c]
        ),
        pytest.param("python", marks = [
            pytest.mark.python]
        )
    )
)

def test_numpy_float64_array_like_2d(language):

    @types('bool[:,:]')
    @types('int[:,:]')
    @types('int8[:,:]')
    @types('int16[:,:]')
    @types('int32[:,:]')
    @types('int64[:,:]')
    @types('float[:,:]')
    @types('float32[:,:]')
    @types('float64[:,:]')
    def get_float64(arr):
        from numpy import float64, shape
        a = float64(arr)
        s = shape(a)
        return len(s), s[0], s[1], a[0,0], a[0,1]

    size = (2, 5)

    bl = randint(0, 1, size=size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=np.int)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size = size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size = size)

    epyccel_func = epyccel(get_float64, language=language)

    assert epyccel_func(bl) == get_float64(bl)
    assert epyccel_func(integer8) == get_float64(integer8)
    assert epyccel_func(integer16) == get_float64(integer16)
    assert epyccel_func(integer) == get_float64(integer)
    assert epyccel_func(integer32) == get_float64(integer32)
    # the if block should be removed after resolving (https://github.com/pyccel/pyccel/issues/735).
    if sys.platform != 'win32':
        assert epyccel_func(integer64) == get_float64(integer64)
        assert epyccel_func(fl) == get_float64(fl)
        assert epyccel_func(fl64) == get_float64(fl64)
    assert epyccel_func(fl32) == get_float64(fl32)

def test_numpy_double_scalar(language):

    @types('bool')
    @types('int')
    @types('int8')
    @types('int16')
    @types('int32')
    @types('int64')
    @types('float')
    @types('float32')
    @types('float64')
    def get_double(a):
        from numpy import double
        b = double(a)
        return b

    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=np.int)
    integer32 = randint(min_int32, max_int32, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2)
    fl32 = uniform(min_float32 / 2, max_float32 / 2)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2)


    epyccel_func = epyccel(get_double, language=language)

    f_bl_true_output = epyccel_func(True)
    test_bool_true_output = get_double(True)

    f_bl_false_output = epyccel_func(False)
    test_bool_false_output = get_double(False)

    assert f_bl_true_output == test_bool_true_output
    assert f_bl_false_output == test_bool_false_output

    assert matching_types(f_bl_true_output, test_bool_true_output)
    assert matching_types(f_bl_false_output, test_bool_false_output)

    f_integer_output = epyccel_func(integer)
    test_int_output  = get_double(integer)

    assert f_integer_output == test_int_output
    assert matching_types(f_integer_output, test_int_output)

    f_integer8_output = epyccel_func(integer8)
    test_int8_output = get_double(integer8)

    assert f_integer8_output == test_int8_output
    assert matching_types(f_integer8_output, test_int8_output)

    f_integer16_output = epyccel_func(integer16)
    test_int16_output = get_double(integer16)

    assert f_integer16_output == test_int16_output
    assert matching_types(f_integer16_output, test_int16_output)

    f_integer32_output = epyccel_func(integer32)
    test_int32_output = get_double(integer32)

    assert f_integer32_output == test_int32_output
    assert matching_types(f_integer32_output, test_int32_output)

    # the if block should be removed after resolving (https://github.com/pyccel/pyccel/issues/735).
    if sys.platform != 'win32':
        f_integer64_output = epyccel_func(integer64)
        test_int64_output = get_double(integer64)

        assert f_integer64_output == test_int64_output
        assert matching_types(f_integer64_output, test_int64_output)

    f_fl_output = epyccel_func(fl)
    test_float_output = get_double(fl)

    assert f_fl_output == test_float_output
    assert matching_types(f_fl_output, test_float_output)

    f_fl32_output = epyccel_func(fl32)
    test_float32_output = get_double(fl32)

    assert f_fl32_output == test_float32_output
    assert matching_types(f_fl32_output, test_float32_output)

    f_fl64_output = epyccel_func(fl64)
    test_float64_output = get_double(fl64)

    assert f_fl64_output == test_float64_output
    assert matching_types(f_fl64_output, test_float64_output)


@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Arrays not handled yet."),
            pytest.mark.c]
        ),
        pytest.param("python", marks = [
            pytest.mark.python]
        )
    )
)

def test_numpy_double_array_like_1d(language):

    @types('bool[:]')
    @types('int[:]')
    @types('int8[:]')
    @types('int16[:]')
    @types('int32[:]')
    @types('int64[:]')
    @types('float[:]')
    @types('float32[:]')
    @types('float64[:]')
    def get_double(arr):
        from numpy import double, shape
        a = double(arr)
        s = shape(a)
        return len(s), s[0], a[0]

    size = 5

    bl = randint(0, 1, size=size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=np.int)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size = size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size = size)

    epyccel_func = epyccel(get_double, language=language)

    assert epyccel_func(bl) == get_double(bl)
    assert epyccel_func(integer8) == get_double(integer8)
    assert epyccel_func(integer16) == get_double(integer16)
    assert epyccel_func(integer) == get_double(integer)
    assert epyccel_func(integer32) == get_double(integer32)
    # the if block should be removed after resolving (https://github.com/pyccel/pyccel/issues/735).
    if sys.platform != 'win32':
        assert epyccel_func(integer64) == get_double(integer64)
        assert epyccel_func(fl) == get_double(fl)
        assert epyccel_func(fl64) == get_double(fl64)
    assert epyccel_func(fl32) == get_double(fl32)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Arrays not handled yet."),
            pytest.mark.c]
        ),
        pytest.param("python", marks = [
            pytest.mark.python]
        )
    )
)

def test_numpy_double_array_like_2d(language):

    @types('bool[:,:]')
    @types('int[:,:]')
    @types('int8[:,:]')
    @types('int16[:,:]')
    @types('int32[:,:]')
    @types('int64[:,:]')
    @types('float[:,:]')
    @types('float32[:,:]')
    @types('float64[:,:]')
    def get_double(arr):
        from numpy import double, shape
        a = double(arr)
        s = shape(a)
        return len(s), s[0], s[1], a[0,0], a[0,1]

    size = (2, 5)

    bl = randint(0, 1, size=size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=np.int)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size = size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size = size)

    epyccel_func = epyccel(get_double, language=language)

    assert epyccel_func(bl) == get_double(bl)
    assert epyccel_func(integer8) == get_double(integer8)
    assert epyccel_func(integer16) == get_double(integer16)
    assert epyccel_func(integer) == get_double(integer)
    assert epyccel_func(integer32) == get_double(integer32)
    # the if block should be removed after resolving (https://github.com/pyccel/pyccel/issues/735).
    if sys.platform != 'win32':
        assert epyccel_func(integer64) == get_double(integer64)
        assert epyccel_func(fl) == get_double(fl)
        assert epyccel_func(fl64) == get_double(fl64)
    assert epyccel_func(fl32) == get_double(fl32)


@types('bool')
@types('int')
@types('int8')
@types('int16')
@types('int32')
@types('int64')
@types('float')
@types('float32')
@types('float64')
def get_complex64(a):
    from numpy import complex64
    b = complex64(a)
    return b

@types('bool')
@types('int')
@types('int8')
@types('int16')
@types('int32')
@types('int64')
@types('float')
@types('float32')
@types('float64')
def get_complex128(a):
    from numpy import complex128
    b = complex128(a)
    return b

@pytest.mark.parametrize( 'get_complex', [get_complex128, get_complex64])
def test_numpy_complex(language, get_complex):

    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=np.int)
    integer32 = randint(min_int32, max_int32, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2)
    fl32 = uniform(min_float32 / 2, max_float32 / 2)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2)


    epyccel_func = epyccel(get_complex, language=language)

    f_bl_true_output = epyccel_func(True)
    test_bool_true_output = get_complex(True)

    f_bl_false_output = epyccel_func(False)
    test_bool_false_output = get_complex(False)

    assert f_bl_true_output == test_bool_true_output
    assert f_bl_false_output == test_bool_false_output

    assert matching_types(f_bl_true_output, test_bool_true_output)
    assert matching_types(f_bl_false_output, test_bool_false_output)

    f_integer_output = epyccel_func(integer)
    test_int_output  = get_complex(integer)

    assert f_integer_output == test_int_output
    assert matching_types(f_integer_output, test_int_output)

    f_integer8_output = epyccel_func(integer8)
    test_int8_output = get_complex(integer8)

    assert f_integer8_output == test_int8_output
    assert matching_types(f_integer8_output, test_int8_output)

    f_integer16_output = epyccel_func(integer16)
    test_int16_output = get_complex(integer16)

    assert f_integer16_output == test_int16_output
    assert matching_types(f_integer16_output, test_int16_output)

    f_integer32_output = epyccel_func(integer32)
    test_int32_output = get_complex(integer32)

    assert f_integer32_output == test_int32_output
    assert matching_types(f_integer32_output, test_int32_output)

    # the if block should be removed after resolving (https://github.com/pyccel/pyccel/issues/735).
    if sys.platform != 'win32':
        f_integer64_output = epyccel_func(integer64)
        test_int64_output = get_complex(integer64)

        assert f_integer64_output == test_int64_output
        assert matching_types(f_integer64_output, test_int64_output)

    f_fl_output = epyccel_func(fl)
    test_float_output = get_complex(fl)

    assert f_fl_output == test_float_output
    assert matching_types(f_fl_output, test_float_output)

    f_fl32_output = epyccel_func(fl32)
    test_float32_output = get_complex(fl32)

    assert f_fl32_output == test_float32_output
    assert matching_types(f_fl32_output, test_float32_output)

    f_fl64_output = epyccel_func(fl64)
    test_float64_output = get_complex(fl64)

    assert f_fl64_output == test_float64_output
    assert matching_types(f_fl64_output, test_float64_output)

@types('bool[:,:]')
@types('int[:,:]')
@types('int8[:,:]')
@types('int16[:,:]')
@types('int32[:,:]')
@types('int64[:,:]')
@types('float[:,:]')
@types('float32[:,:]')
@types('float64[:,:]')
def get_complex128_arr(arr):
    from numpy import complex128, shape
    a = complex128(arr)
    s = shape(a)
    return len(s), s[0], s[1], a[0,0], a[0,1]

@types('bool[:,:]')
@types('int[:,:]')
@types('int8[:,:]')
@types('int16[:,:]')
@types('int32[:,:]')
@types('int64[:,:]')
@types('float[:,:]')
@types('float32[:,:]')
@types('float64[:,:]')
def get_complex64_arr(arr):
    from numpy import complex64, shape
    a = complex64(arr)
    s = shape(a)
    return len(s), s[0], s[1], a[0,0], a[0,1]

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Arrays not handled yet."),
            pytest.mark.c]
        ),
        pytest.param("python", marks = [
            pytest.mark.python]
        )
    )
)

@pytest.mark.parametrize( 'get_complex', [get_complex128_arr, get_complex64_arr])
def test_numpy_complex_array_like_1d(language, get_complex):

    size = 5

    bl = randint(0, 1, size=size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=np.int)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size = size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size = size)

    epyccel_func = epyccel(get_complex, language=language)

    assert epyccel_func(bl) == get_complex(bl)
    assert epyccel_func(integer8) == get_complex(integer8)
    assert epyccel_func(integer16) == get_complex(integer16)
    assert epyccel_func(integer) == get_complex(integer)
    assert epyccel_func(integer32) == get_complex(integer32)
    # the if block should be removed after resolving (https://github.com/pyccel/pyccel/issues/735).
    if sys.platform != 'win32':
        assert epyccel_func(integer64) == get_complex(integer64)
        assert epyccel_func(fl) == get_complex(fl)
        assert epyccel_func(fl64) == get_complex(fl64)
    assert epyccel_func(fl32) == get_complex(fl32)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Arrays not handled yet."),
            pytest.mark.c]
        ),
        pytest.param("python", marks = [
            pytest.mark.python]
        )
    )
)

@pytest.mark.parametrize( 'get_complex', [get_complex128_arr, get_complex64_arr])
def test_numpy_complex_array_like_2d(language, get_complex):

    size = (2, 5)

    bl = randint(0, 1, size=size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=np.int)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size = size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size = size)

    epyccel_func = epyccel(get_complex, language=language)

    assert epyccel_func(bl) == get_complex(bl)
    assert epyccel_func(integer8) == get_complex(integer8)
    assert epyccel_func(integer16) == get_complex(integer16)
    assert epyccel_func(integer) == get_complex(integer)
    assert epyccel_func(integer32) == get_complex(integer32)
    # the if block should be removed after resolving (https://github.com/pyccel/pyccel/issues/735).
    if sys.platform != 'win32':
        assert epyccel_func(integer64) == get_complex(integer64)
        assert epyccel_func(fl) == get_complex(fl)
        assert epyccel_func(fl64) == get_complex(fl64)
    assert epyccel_func(fl32) == get_complex(fl32)
