# pylint: disable=missing-function-docstring, missing-module-docstring
import sys
import pytest
from numpy.random import randint, uniform
import numpy as np

from test_numpy_funcs import (min_int, max_int, min_int8, max_int8,
                                min_int16, max_int16, min_int32, max_int32, max_int64, min_int64)
from test_numpy_funcs import max_float, min_float, max_float32, min_float32,max_float64, min_float64
from test_numpy_funcs import matching_types

from pyccel.decorators import template
from pyccel.epyccel import epyccel

numpy_basic_types_deprecated = tuple(int(v) for v in np.version.version.split('.'))>=(1,24,0)

def test_mult_numpy_python_type(language):

    def mult_on_array_int8():
        from numpy import ones, int8
        a = ones(5, dtype=int8)
        b = a * 2
        return b[0]

    def mult_on_array_int16():
        from numpy import ones, int16
        a = ones(5, dtype=int16)
        b = a * 2
        return b[0]

    def mult_on_array_int32():
        from numpy import ones, int32
        a = ones(5, dtype=int32)
        b = a * 2
        return b[0]

    def mult_on_array_int64():
        from numpy import ones, int64
        a = ones(5, dtype=int64)
        b = a * 2
        return b[0]

    def mult_on_array_float32():
        from numpy import ones, float32
        a = ones(5, dtype=float32)
        b = a * 2
        return b[0]

    def mult_on_array_float64():
        from numpy import ones, float64
        a = ones(5, dtype=float64)
        b = a * 2
        return b[0]

    epyccel_func = epyccel(mult_on_array_int8, language=language)
    python_result = mult_on_array_int8()
    pyccel_result = epyccel_func()
    assert python_result == pyccel_result
    assert matching_types(pyccel_result, python_result)

    epyccel_func = epyccel(mult_on_array_int16, language=language)
    python_result = mult_on_array_int16()
    pyccel_result = epyccel_func()
    assert python_result == pyccel_result
    assert matching_types(pyccel_result, python_result)

    epyccel_func = epyccel(mult_on_array_int32, language=language)
    python_result = mult_on_array_int32()
    pyccel_result = epyccel_func()
    assert python_result == pyccel_result
    assert matching_types(pyccel_result, python_result)

    epyccel_func = epyccel(mult_on_array_int64, language=language)
    python_result = mult_on_array_int64()
    pyccel_result = epyccel_func()
    assert python_result == pyccel_result
    assert matching_types(pyccel_result, python_result)

    epyccel_func = epyccel(mult_on_array_float32, language=language)
    python_result = mult_on_array_float32()
    pyccel_result = epyccel_func()
    assert python_result == pyccel_result
    assert matching_types(pyccel_result, python_result)

    epyccel_func = epyccel(mult_on_array_float64, language=language)
    python_result = mult_on_array_float64()
    pyccel_result = epyccel_func()
    assert python_result == pyccel_result
    assert matching_types(pyccel_result, python_result)

def test_numpy_scalar_promotion(language):

    @template(name='T', types=['int32', 'int64', 'float32', 'float64', 'complex64', 'complex128'])
    @template(name='D', types=['int32', 'int64', 'float32', 'float64', 'complex64', 'complex128'])
    def add_numpy_to_numpy_type(np_s_l : 'T', np_s_r : 'D'):
        rs = np_s_l + np_s_r
        return rs

    integer32   = randint(min_int32 // 2, max_int32 // 2, dtype=np.int32)
    integer64   = randint(min_int64 // 2, max_int64 // 2, dtype=np.int64)
    fl32        = np.float32(uniform(min_float32 / 2, max_float32 / 2))
    fl64        = np.float64(uniform(min_float64 / 2, max_float64 / 2))
    complex64   = np.complex64(uniform(min_float32 / 2, max_float32 / 2))
    complex128  = np.complex64(uniform(min_float32 / 2, max_float32 / 2))

    epyccel_func    = epyccel(add_numpy_to_numpy_type, language=language)

    pyccel_result   = epyccel_func(integer32, integer64)
    python_result   = add_numpy_to_numpy_type(integer32, integer64)

    assert pyccel_result == python_result
    assert isinstance(pyccel_result, type(python_result))

    pyccel_result = epyccel_func(integer64, fl32)
    python_result = add_numpy_to_numpy_type(integer64, fl32)
    assert pyccel_result == python_result
    assert isinstance(pyccel_result, type(python_result))

    pyccel_result = epyccel_func(integer64, fl64)
    python_result = add_numpy_to_numpy_type(integer64, fl64)
    assert pyccel_result == python_result
    assert isinstance(pyccel_result, type(python_result))

    pyccel_result = epyccel_func(fl64, complex64)
    python_result = add_numpy_to_numpy_type(fl64, complex64)
    assert pyccel_result == python_result
    assert isinstance(pyccel_result, type(python_result))

    pyccel_result = epyccel_func(complex128, fl64)
    python_result = add_numpy_to_numpy_type(complex128, fl64)
    assert pyccel_result == python_result
    assert isinstance(pyccel_result, type(python_result))

@pytest.mark.skipif(numpy_basic_types_deprecated, reason="Can't import bool from numpy")
def test_numpy_bool_scalar(language):

    @template('T', ['bool', 'int', 'int8', 'int16', 'int32', 'int64', 'float', 'float32', 'float64'])
    def get_bool(a : 'T'):
        from numpy import bool as NumpyBool
        b = NumpyBool(a)
        return b

    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=int)
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

@template('T', ['bool', 'int', 'int8', 'int16', 'int32', 'int64', 'float', 'float32', 'float64'])
def get_int(a : 'T'):
    from numpy import int as Numpyint
    b = Numpyint(a)
    return b

@template('T', ['bool', 'int', 'int8', 'int16', 'int32', 'int64', 'float', 'float32', 'float64'])
def get_int64(a : 'T'):
    from numpy import int64
    b = int64(a)
    return b

@template('T', ['bool', 'int', 'int8', 'int16', 'int32', 'int64', 'float', 'float32', 'float64'])
def get_int32(a : 'T'):
    from numpy import int32
    b = int32(a)
    return b

@template('T', ['bool', 'int', 'int8', 'int16', 'int32', 'int64', 'float', 'float32', 'float64'])
def get_int16(a : 'T'):
    from numpy import int16
    b = int16(a)
    return b

@template('T', ['bool', 'int', 'int8', 'int16', 'int32', 'int64', 'float', 'float32', 'float64'])
def get_int8(a : 'T'):
    from numpy import int8
    b = int8(a)
    return b

if numpy_basic_types_deprecated:
    int_functions_and_boundaries = [(get_int64, min_int64, max_int64), (get_int32, min_int32, max_int32),\
                                                 (get_int16, min_int16, max_int16), (get_int8, min_int8, max_int8)]
else:
    int_functions_and_boundaries = [(get_int, min_int, max_int), (get_int64, min_int64, max_int64), (get_int32, min_int32, max_int32),\
                                                 (get_int16, min_int16, max_int16), (get_int8, min_int8, max_int8)]

@pytest.mark.parametrize( 'function_boundaries', int_functions_and_boundaries)
def test_numpy_int_scalar(language, function_boundaries):

    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int8, max_int8, dtype=np.int16)
    integer = randint(min_int8, max_int8, dtype=int)
    integer32 = randint(min_int8, max_int8, dtype=np.int32)
    integer64 = randint(min_int8, max_int8, dtype=np.int64)

    get_int = function_boundaries[0]
    # Modifying a global variable in a scop will change it to a local variable, so it needs to be initialized.
    # we need to keep min_int/max_int as they are, and make different names for those that come from function_boundries
    # ffb stands for 'from function_boundaries'
    max_int_ffb = function_boundaries[1]
    min_int_ffb = function_boundaries[2]

    fl = uniform(min_int_ffb, max_int_ffb)
    fl32 = uniform(min_int_ffb, max_int_ffb)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_int_ffb, max_int_ffb)

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

        f_fl64_output = epyccel_func(fl64)
        test_float64_output = get_int(fl64)

        assert f_fl64_output == test_float64_output
        assert matching_types(f_fl64_output, test_float64_output)

        f_fl32_output = epyccel_func(fl32)
        test_float32_output = get_int(fl32)

        assert f_fl32_output == test_float32_output
        assert matching_types(f_fl32_output, test_float32_output)

@template('T', ['bool[:]', 'int[:]', 'int8[:]', 'int16[:]', 'int32[:]', 'int64[:]', 'float[:]', 'float32[:]', 'float64[:]'])
def get_int64_arr_1d(arr : 'T'):
    from numpy import int64, shape
    a = int64(arr)
    s = shape(a)
    return len(s), s[0], a[0], a[1]

@template('T', ['bool[:]', 'int[:]', 'int8[:]', 'int16[:]', 'int32[:]', 'int64[:]', 'float[:]', 'float32[:]', 'float64[:]'])
def get_int32_arr_1d(arr : 'T'):
    from numpy import int32, shape
    a = int32(arr)
    s = shape(a)
    return len(s), s[0], a[0], a[1]

@template('T', ['bool[:]', 'int[:]', 'int8[:]', 'int16[:]', 'int32[:]', 'int64[:]', 'float[:]', 'float32[:]', 'float64[:]'])
def get_int16_arr_1d(arr : 'T'):
    from numpy import int16, shape
    a = int16(arr)
    s = shape(a)
    return len(s), s[0], a[0], a[1]

@template('T', ['bool[:]', 'int[:]', 'int8[:]', 'int16[:]', 'int32[:]', 'int64[:]', 'float[:]', 'float32[:]', 'float64[:]'])
def get_int8_arr_1d(arr : 'T'):
    from numpy import int8, shape
    a = int8(arr)
    s = shape(a)
    return len(s), s[0], a[0], a[1]

@pytest.mark.parametrize( 'function_boundaries', [(get_int64_arr_1d, min_int64, max_int64), (get_int32_arr_1d, min_int32, max_int32),\
                                                 (get_int16_arr_1d, min_int16, max_int16), (get_int8_arr_1d, min_int8, max_int8)])
def test_numpy_int_array_like_1d(language, function_boundaries):

    size = 5

    bl = randint(0, 1, size=size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=int)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    get_int = function_boundaries[0]
    # Modifying a global variable in a scop will change it to a local variable, so it needs to be initialized.
    # we need to keep min_int/max_int as they are, and make different names for those that come from function_boundries
    # ffb stands for 'from function_boundaries'
    max_int_ffb = function_boundaries[1]
    min_int_ffb = function_boundaries[2]

    fl = uniform(min_int_ffb, max_int_ffb, size=size)
    fl32 = uniform(min_int_ffb, max_int_ffb, size=size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_int_ffb, max_int_ffb, size=size)

    epyccel_func = epyccel(get_int, language=language)

    assert epyccel_func(bl) == get_int(bl)
    assert epyccel_func(integer8) == get_int(integer8)
    assert epyccel_func(integer16) == get_int(integer16)
    assert epyccel_func(integer) == get_int(integer)
    assert epyccel_func(integer32) == get_int(integer32)
    # the if block should be removed after resolving (https://github.com/pyccel/pyccel/issues/735).
    if sys.platform != 'win32':
        assert epyccel_func(integer64) == get_int(integer64)
        assert epyccel_func(fl) == get_int(fl)
        assert epyccel_func(fl64) == get_int(fl64)
        assert epyccel_func(fl32) == get_int(fl32)

@template('T', ['bool[:,:]', 'int[:,:]', 'int8[:,:]', 'int16[:,:]', 'int32[:,:]', 'int64[:,:]', 'float[:,:]', 'float32[:,:]', 'float64[:,:]'])
def get_int64_arr_2d(arr : 'T'):
    from numpy import int64, shape
    a = int64(arr)
    s = shape(a)
    return len(s), s[0], s[1], a[0,0], a[1,0]

@template('T', ['bool[:,:]', 'int[:,:]', 'int8[:,:]', 'int16[:,:]', 'int32[:,:]', 'int64[:,:]', 'float[:,:]', 'float32[:,:]', 'float64[:,:]'])
def get_int32_arr_2d(arr : 'T'):
    from numpy import int32, shape
    a = int32(arr)
    s = shape(a)
    return len(s), s[0], s[1], a[0,0], a[1,0]

@template('T', ['bool[:,:]', 'int[:,:]', 'int8[:,:]', 'int16[:,:]', 'int32[:,:]', 'int64[:,:]', 'float[:,:]', 'float32[:,:]', 'float64[:,:]'])
def get_int16_arr_2d(arr : 'T'):
    from numpy import int16, shape
    a = int16(arr)
    s = shape(a)
    return len(s), s[0], s[1], a[0,0], a[1,0]

@template('T', ['bool[:,:]', 'int[:,:]', 'int8[:,:]', 'int16[:,:]', 'int32[:,:]', 'int64[:,:]', 'float[:,:]', 'float32[:,:]', 'float64[:,:]'])
def get_int8_arr_2d(arr : 'T'):
    from numpy import int8, shape
    a = int8(arr)
    s = shape(a)
    return len(s), s[0], s[1], a[0,0], a[1,0]

@pytest.mark.parametrize( 'function_boundaries', [(get_int64_arr_2d, min_int64, max_int64), (get_int32_arr_2d, min_int32, max_int32),\
                                                 (get_int16_arr_2d, min_int16, max_int16), (get_int8_arr_2d, min_int8, max_int8)])
def test_numpy_int_array_like_2d(language, function_boundaries):

    size = (2, 5)

    bl = randint(0, 1, size=size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=int)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    get_int = function_boundaries[0]
    # Modifying a global variable in a scop will change it to a local variable, so it needs to be initialized.
    # we need to keep min_int/max_int as they are, and make different names for those that come from function_boundries
    # ffb stands for 'from function_boundaries'
    max_int_ffb = function_boundaries[1]
    min_int_ffb = function_boundaries[2]

    fl = uniform(min_int_ffb, max_int_ffb, size=size)
    fl32 = uniform(min_int_ffb, max_int_ffb, size=size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_int_ffb, max_int_ffb, size=size)

    epyccel_func = epyccel(get_int, language=language)

    assert epyccel_func(bl) == get_int(bl)
    assert epyccel_func(integer8) == get_int(integer8)
    assert epyccel_func(integer16) == get_int(integer16)
    assert epyccel_func(integer) == get_int(integer)
    assert epyccel_func(integer32) == get_int(integer32)
    # the if block should be removed after resolving (https://github.com/pyccel/pyccel/issues/735).
    if sys.platform != 'win32':
        assert epyccel_func(integer64) == get_int(integer64)
        assert epyccel_func(fl) == get_int(fl)
        assert epyccel_func(fl64) == get_int(fl64)
        assert epyccel_func(fl32) == get_int(fl32)

@template('T', ['bool', 'int', 'int8', 'int16', 'int32', 'int64', 'float', 'float32', 'float64'])
def get_float(a : 'T'):
    from numpy import float as NumpyFloat
    b = NumpyFloat(a)
    return b

@template('T', ['bool', 'int', 'int8', 'int16', 'int32', 'int64', 'float', 'float32', 'float64'])
def get_float64(a : 'T'):
    from numpy import float64
    b = float64(a)
    return b

@template('T', ['bool', 'int', 'int8', 'int16', 'int32', 'int64', 'float', 'float32', 'float64'])
def get_float32(a : 'T'):
    from numpy import float32
    b = float32(a)
    return b

if numpy_basic_types_deprecated:
    float_functions = [get_float64, get_float32]
else:
    float_functions = [get_float64, get_float32, get_float]

@pytest.mark.parametrize( 'get_float', float_functions)
def test_numpy_float_scalar(language, get_float):

    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=int)
    integer32 = randint(min_int32, max_int32, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, dtype=np.int64)

    fl = uniform(min_float32 / 2, max_float32 / 2)
    fl32 = uniform(min_float32 / 2, max_float32 / 2)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float32 / 2, max_float32 / 2)


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


@template('T', ['bool[:]', 'int[:]', 'int8[:]', 'int16[:]', 'int32[:]', 'int64[:]', 'float[:]', 'float32[:]', 'float64[:]'])
def get_float64_arr_1d(arr : 'T'):
    from numpy import float64, shape
    a = float64(arr)
    s = shape(a)
    return len(s), s[0], a[0], a[1]

@template('T', ['bool[:]', 'int[:]', 'int8[:]', 'int16[:]', 'int32[:]', 'int64[:]', 'float[:]', 'float32[:]', 'float64[:]'])
def get_float32_arr_1d(arr : 'T'):
    from numpy import float32, shape
    a = float32(arr)
    s = shape(a)
    return len(s), s[0], a[0], a[1]

@pytest.mark.parametrize( 'get_float', [get_float64_arr_1d, get_float32_arr_1d])
def test_numpy_float_array_like_1d(language, get_float):

    size = 5

    bl = randint(0, 1, size=size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=int)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float32 / 2, max_float32 / 2, size = size)

    epyccel_func = epyccel(get_float, language=language)

    assert epyccel_func(bl) == get_float(bl)
    assert epyccel_func(integer8) == get_float(integer8)
    assert epyccel_func(integer16) == get_float(integer16)
    assert epyccel_func(integer) == get_float(integer)
    assert epyccel_func(integer32) == get_float(integer32)
    # the if block should be removed after resolving (https://github.com/pyccel/pyccel/issues/735).
    if sys.platform != 'win32':
        assert epyccel_func(integer64) == get_float(integer64)
        assert epyccel_func(fl) == get_float(fl)
        assert epyccel_func(fl64) == get_float(fl64)
    assert epyccel_func(fl32) == get_float(fl32)

@template('T', ['bool[:,:]', 'int[:,:]', 'int8[:,:]', 'int16[:,:]', 'int32[:,:]', 'int64[:,:]', 'float[:,:]', 'float32[:,:]', 'float64[:,:]'])
def get_float64_arr_2d(arr : 'T'):
    from numpy import float64, shape
    a = float64(arr)
    s = shape(a)
    return len(s), s[0], s[1], a[0,0], a[0,1]

@template('T', ['bool[:,:]', 'int[:,:]', 'int8[:,:]', 'int16[:,:]', 'int32[:,:]', 'int64[:,:]', 'float[:,:]', 'float32[:,:]', 'float64[:,:]'])
def get_float32_arr_2d(arr : 'T'):
    from numpy import float32, shape
    a = float32(arr)
    s = shape(a)
    return len(s), s[0], s[1], a[0,0], a[0,1]

@pytest.mark.parametrize( 'get_float', [get_float64_arr_2d, get_float32_arr_2d])
def test_numpy_float_array_like_2d(language, get_float):

    size = (2, 5)

    bl = randint(0, 1, size=size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=int)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float32 / 2, max_float32 / 2, size = size)

    epyccel_func = epyccel(get_float, language=language)

    assert epyccel_func(bl) == get_float(bl)
    assert epyccel_func(integer8) == get_float(integer8)
    assert epyccel_func(integer16) == get_float(integer16)
    assert epyccel_func(integer) == get_float(integer)
    assert epyccel_func(integer32) == get_float(integer32)
    # the if block should be removed after resolving (https://github.com/pyccel/pyccel/issues/735).
    if sys.platform != 'win32':
        assert epyccel_func(integer64) == get_float(integer64)
        assert epyccel_func(fl) == get_float(fl)
        assert epyccel_func(fl64) == get_float(fl64)
    assert epyccel_func(fl32) == get_float(fl32)

def test_numpy_double_scalar(language):

    @template('T', ['bool', 'int', 'int8', 'int16', 'int32', 'int64', 'float', 'float32', 'float64'])
    def get_double(a : 'T'):
        from numpy import double
        b = double(a)
        return b

    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=int)
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


def test_numpy_double_array_like_1d(language):

    @template('T', ['bool[:]', 'int[:]', 'int8[:]', 'int16[:]', 'int32[:]', 'int64[:]', 'float[:]', 'float32[:]', 'float64[:]'])
    def get_double(arr : 'T'):
        from numpy import double, shape
        a = double(arr)
        s = shape(a)
        return len(s), s[0], a[0]

    size = 5

    bl = randint(0, 1, size=size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=int)
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

def test_numpy_double_array_like_2d(language):

    @template('T', ['bool[:,:]', 'int[:,:]', 'int8[:,:]', 'int16[:,:]', 'int32[:,:]', 'int64[:,:]', 'float[:,:]', 'float32[:,:]', 'float64[:,:]'])
    def get_double(arr : 'T'):
        from numpy import double, shape
        a = double(arr)
        s = shape(a)
        return len(s), s[0], s[1], a[0,0], a[0,1]

    size = (2, 5)

    bl = randint(0, 1, size=size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=int)
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


@template('T', ['bool', 'int', 'int8', 'int16', 'int32', 'int64', 'float', 'float32', 'float64'])
def get_complex64(a : 'T'):
    from numpy import complex64
    b = complex64(a)
    return b

@template('T', ['bool', 'int', 'int8', 'int16', 'int32', 'int64', 'float', 'float32', 'float64'])
def get_complex128(a : 'T'):
    from numpy import complex128
    b = complex128(a)
    return b

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [pytest.mark.fortran]),
        pytest.param("c", marks = [pytest.mark.c]),
        pytest.param("python", marks = [
            pytest.mark.skip(reason=("complex handles types in __new__ so it "
                "cannot be used in a translated interface in python. See #802")),
            pytest.mark.python]
        )
    )
)
@pytest.mark.parametrize( 'get_complex', [get_complex128, get_complex64])
def test_numpy_complex_scalar(language, get_complex):

    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=int)
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



@template('T', ['bool[:]', 'int[:]', 'int8[:]', 'int16[:]', 'int32[:]', 'int64[:]', 'float[:]', 'float32[:]', 'float64[:]'])
def get_complex128_arr_1d(arr : 'T'):
    from numpy import complex128, shape
    a = complex128(arr)
    s = shape(a)
    return len(s), s[0], a[0], a[1]

@template('T', ['bool[:]', 'int[:]', 'int8[:]', 'int16[:]', 'int32[:]', 'int64[:]', 'float[:]', 'float32[:]', 'float64[:]'])
def get_complex64_arr_1d(arr : 'T'):
    from numpy import complex64, shape
    a = complex64(arr)
    s = shape(a)
    return len(s), s[0], a[0], a[1]

@template('T', ['bool[:,:]', 'int[:,:]', 'int8[:,:]', 'int16[:,:]', 'int32[:,:]', 'int64[:,:]', 'float[:,:]', 'float32[:,:]', 'float64[:,:]'])
def get_complex128_arr_2d(arr : 'T'):
    from numpy import complex128, shape
    a = complex128(arr)
    s = shape(a)
    return len(s), s[0], s[1], a[0,0], a[0,1]

@template('T', ['bool[:,:]', 'int[:,:]', 'int8[:,:]', 'int16[:,:]', 'int32[:,:]', 'int64[:,:]', 'float[:,:]', 'float32[:,:]', 'float64[:,:]'])
def get_complex64_arr_2d(arr : 'T'):
    from numpy import complex64, shape
    a = complex64(arr)
    s = shape(a)
    return len(s), s[0], s[1], a[0,0], a[0,1]


@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [pytest.mark.fortran]),
        pytest.param("c", marks = [pytest.mark.c]),
        pytest.param("python", marks = [
            pytest.mark.skip(reason=("complex handles types in __new__ so it "
                "cannot be used in a translated interface in python. See #802")),
            pytest.mark.python]
        )
    )
)
@pytest.mark.parametrize( 'get_complex', [get_complex128_arr_1d, get_complex64_arr_1d])
def test_numpy_complex_array_like_1d(language, get_complex):

    size = 5

    bl = randint(0, 1, size=size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=int)
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
        pytest.param("c", marks = [pytest.mark.c]),
        pytest.param("python", marks = [
            pytest.mark.skip(reason=("complex handles types in __new__ so it "
                "cannot be used in a translated interface in python. See #802")),
            pytest.mark.python]
        )
    )
)
@pytest.mark.parametrize( 'get_complex', [get_complex128_arr_2d, get_complex64_arr_2d])
def test_numpy_complex_array_like_2d(language, get_complex):

    size = (2, 5)

    bl = randint(0, 1, size=size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=int)
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

def test_literal_complex64(language):
    def get_complex64():
        from numpy import complex64
        compl = complex64(3+4j)
        return compl, compl.real, compl.imag

    epyccel_func = epyccel(get_complex64, language=language)

    pyth_res = get_complex64()
    pycc_res = epyccel_func()
    for pyth, pycc in zip(pyth_res, pycc_res):
        assert pyth == pycc
        assert isinstance(pycc, type(pyth))

def test_literal_complex128(language):
    def get_complex128():
        from numpy import complex128
        compl = complex128(3+4j)
        return compl, compl.real, compl.imag

    epyccel_func = epyccel(get_complex128, language=language)

    pyth_res = get_complex128()
    pycc_res = epyccel_func()
    for pyth, pycc in zip(pyth_res, pycc_res):
        assert pyth == pycc
        assert isinstance(pycc, type(pyth))
