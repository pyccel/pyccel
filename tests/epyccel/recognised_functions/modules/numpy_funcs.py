# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import TypeVar

import numpy as np

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
CNT = TypeVar("CNT", "complex64", "complex128")
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

FI = TypeVar("FI", float, int)


test_copy__X = TypeVar(
    "test_copy__X", "int[:]", "float[:,:]", "complex[:,:,:](order=F)"
)


test_copy__Y = TypeVar("test_copy__Y", "float[:,:]", "complex[:,:,:](order=F)")


def fabs_call_r(x: "float"):
    from numpy import fabs

    return fabs(x)


def fabs_call_i(x: "int"):
    from numpy import fabs

    return fabs(x)


def fabs_phrase_r_r(x: "float", y: "float"):
    from numpy import fabs

    a = fabs(x) * fabs(y)
    return a


def fabs_phrase_i_i(x: "int", y: "int"):
    from numpy import fabs

    a = fabs(x) * fabs(y)
    return a


def fabs_phrase_r_i(x: "float", y: "int"):
    from numpy import fabs

    a = fabs(x) * fabs(y)
    return a


def fabs_phrase_i_r(x: "int", y: "float"):
    from numpy import fabs

    a = fabs(x) * fabs(y)
    return a


def numpy_isnan__numpy_isnan_test(x: "float"):
    from numpy import isnan

    return isnan(x)


def numpy_isnan__numpy_isnan_array_test(x: "float[:]"):
    from numpy import isnan

    return isnan(x)


def numpy_isnan__numpy_isnan_expr_test(x: "float", y: "float"):
    from numpy import isnan

    return isnan(x + y)


def numpy_isinf__numpy_isinf_test(x: "float"):
    from numpy import isinf

    return isinf(x)


def numpy_isinf__numpy_isinf_array_test(x: "float[:]"):
    from numpy import isinf

    return isinf(x)


def numpy_isinf__numpy_isinf_expr_test(x: "float", y: "float"):
    from numpy import isinf

    return isinf(x + y)


def numpy_isfinite__numpy_isfinite_test(x: "float"):
    from numpy import isfinite

    return isfinite(x)


def numpy_isfinite__numpy_isfinite_array_test(x: "float[:]"):
    from numpy import isfinite

    return isfinite(x)


def numpy_isfinite__numpy_isfinite_expr_test(x: "float", y: "float"):
    from numpy import isfinite

    return isfinite(x + y)


def absolute_call_r(x: "float"):
    from numpy import absolute

    return absolute(x)


def absolute_call_i(x: "int"):
    from numpy import absolute

    return absolute(x)


def absolute_call_c(x: CT):
    from numpy import absolute

    return absolute(x)


def absolute_phrase_r_r(x: "float", y: "float"):
    from numpy import absolute

    a = absolute(x) * absolute(y)
    return a


def absolute_phrase_i_r(x: "int", y: "float"):
    from numpy import absolute

    a = absolute(x) * absolute(y)
    return a


def absolute_phrase_r_i(x: "float", y: "int"):
    from numpy import absolute

    a = absolute(x) * absolute(y)
    return a


def sin_call_r(x: "float"):
    from numpy import sin

    return sin(x)


def sin_call_i(x: "int"):
    from numpy import sin

    return sin(x)


def sin_phrase_r_r(x: "float", y: "float"):
    from numpy import sin

    a = sin(x) + sin(y)
    return a


def sin_phrase_i_i(x: "int", y: "int"):
    from numpy import sin

    a = sin(x) + sin(y)
    return a


def sin_phrase_i_r(x: "int", y: "float"):
    from numpy import sin

    a = sin(x) + sin(y)
    return a


def sin_phrase_r_i(x: "float", y: "int"):
    from numpy import sin

    a = sin(x) + sin(y)
    return a


def cos_call_i(x: "int"):
    from numpy import cos

    return cos(x)


def cos_call_r(x: "float"):
    from numpy import cos

    return cos(x)


def cos_call_out(x: "float[:]", y: "float[:]"):
    np.cos(x, out=y)


def cos_phrase_i_i(x: "int", y: "int"):
    from numpy import cos

    a = cos(x) + cos(y)
    return a


def cos_phrase_r_r(x: "float", y: "float"):
    from numpy import cos

    a = cos(x) + cos(y)
    return a


def cos_phrase_i_r(x: "int", y: "float"):
    from numpy import cos

    a = cos(x) + cos(y)
    return a


def cos_phrase_r_i(x: "float", y: "int"):
    from numpy import cos

    a = cos(x) + cos(y)
    return a


def tan_call_i(x: "int"):
    from numpy import tan

    return tan(x)


def tan_call_r(x: "float"):
    from numpy import tan

    return tan(x)


def tan_phrase_i_i(x: "int", y: "int"):
    from numpy import tan

    a = tan(x) + tan(y)
    return a


def tan_phrase_r_r(x: "float", y: "float"):
    from numpy import tan

    a = tan(x) + tan(y)
    return a


def tan_phrase_i_r(x: "int", y: "float"):
    from numpy import tan

    a = tan(x) + tan(y)
    return a


def tan_phrase_r_i(x: "float", y: "int"):
    from numpy import tan

    a = tan(x) + tan(y)
    return a


def exp_call_i(x: "int"):
    from numpy import exp

    return exp(x)


def exp_call_r(x: "float"):
    from numpy import exp

    return exp(x)


def exp_phrase_i_i(x: "int", y: "int"):
    from numpy import exp

    a = exp(x) + exp(y)
    return a


def exp_phrase_r_r(x: "float", y: "float"):
    from numpy import exp

    a = exp(x) + exp(y)
    return a


def exp_phrase_i_r(x: "int", y: "float"):
    from numpy import exp

    a = exp(x) + exp(y)
    return a


def exp_phrase_r_i(x: "float", y: "int"):
    from numpy import exp

    a = exp(x) + exp(y)
    return a


def expm1_call_i(x: "int"):
    from numpy import expm1

    return expm1(x)


def expm1_call_f(x: "float"):
    from numpy import expm1

    return expm1(x)


def expm1_call_c(x: complex):
    from numpy import expm1

    return expm1(x)


def expm1_call_f_array(x: "float[:]"):
    from numpy import expm1

    return expm1(x)


def expm1_call_c_array(x: "complex[:]"):
    from numpy import expm1

    return expm1(x)


def expm1_call_cast_f(x: "float32"):
    from numpy import expm1

    return expm1(x)


def expm1_call_cast_c(x: "complex64"):
    from numpy import expm1

    return expm1(x)


def expm1_phrase_i_i(x: "int", y: "int"):
    from numpy import expm1

    a = expm1(x) + expm1(y)
    return a


def expm1_phrase_f_f(x: "float", y: "float"):
    from numpy import expm1

    a = expm1(x) + expm1(y)
    return a


def expm1_phrase_i_f(x: "int", y: "float"):
    from numpy import expm1

    a = expm1(x) + expm1(y)
    return a


def expm1_phrase_f_i(x: "float", y: "int"):
    from numpy import expm1

    a = expm1(x) + expm1(y)
    return a


def expm1_phrase_i_c(x: int, y: complex):
    from numpy import expm1

    a = expm1(x) + expm1(y)
    return a


def log_call_i(x: "int"):
    from numpy import log

    return log(x)


def log_call_r(x: "float"):
    from numpy import log

    return log(x)


def log_phrase(x: "float", y: "float"):
    from numpy import log

    a = log(x) + log(y)
    return a


def arcsin_call_i(x: "int"):
    from numpy import arcsin

    return arcsin(x)


def arcsin_call_r(x: "float"):
    from numpy import arcsin

    return arcsin(x)


def arcsin_phrase(x: "float", y: "float"):
    from numpy import arcsin

    a = arcsin(x) + arcsin(y)
    return a


def arccos_call_i(x: "int"):
    from numpy import arccos

    return arccos(x)


def arccos_call_r(x: "float"):
    from numpy import arccos

    return arccos(x)


def arccos_phrase(x: "float", y: "float"):
    from numpy import arccos

    a = arccos(x) + arccos(y)
    return a


def arctan_call_i(x: "int"):
    from numpy import arctan

    return arctan(x)


def arctan_call_r(x: "float"):
    from numpy import arctan

    return arctan(x)


def arctan_phrase(x: "float", y: "float"):
    from numpy import arctan

    a = arctan(x) + arctan(y)
    return a


def sinh_call_i(x: "int"):
    from numpy import sinh

    return sinh(x)


def sinh_call_r(x: "float"):
    from numpy import sinh

    return sinh(x)


def sinh_phrase(x: "float", y: "float"):
    from numpy import sinh

    a = sinh(x) + sinh(y)
    return a


def cosh_call_i(x: "int"):
    from numpy import cosh

    return cosh(x)


def cosh_call_r(x: "float"):
    from numpy import cosh

    return cosh(x)


def cosh_phrase(x: "float", y: "float"):
    from numpy import cosh

    a = cosh(x) + cosh(y)
    return a


def tanh_call_i(x: "int"):
    from numpy import tanh

    return tanh(x)


def tanh_call_r(x: "float"):
    from numpy import tanh

    return tanh(x)


def tanh_phrase(x: "float", y: "float"):
    from numpy import tanh

    a = tanh(x) + tanh(y)
    return a


def arctan2_call_i_i(x: "int", y: "int"):
    from numpy import arctan2

    return arctan2(x, y)


def arctan2_call_i_r(x: "int", y: "float"):
    from numpy import arctan2

    return arctan2(x, y)


def arctan2_call_r_i(x: "float", y: "int"):
    from numpy import arctan2

    return arctan2(x, y)


def arctan2_call_r_r(x: "float", y: "float"):
    from numpy import arctan2

    return arctan2(x, y)


def arctan2_phrase(x: "float", y: "float", z: "float"):
    from numpy import arctan2

    a = arctan2(x, y) + arctan2(x, z)
    return a


def sqrt_call(x: "float"):
    from numpy import sqrt

    return sqrt(x)


def sqrt_phrase(x: "float", y: "float"):
    from numpy import sqrt

    a = sqrt(x) * sqrt(y)
    return a


def sqrt_return_type_r(x: "float"):
    from numpy import sqrt

    a = sqrt(x)
    return a


def sqrt_return_type_c(x: "complex"):
    from numpy import sqrt

    a = sqrt(x)
    return a


def floor_call_i(x: "int"):
    from numpy import floor

    return floor(x)


def floor_call_r(x: "float"):
    from numpy import floor

    return floor(x)


def floor_phrase(x: "float", y: "float"):
    from numpy import floor

    a = floor(x) * floor(y)
    return a


def shape_indexed__test_shape_1d(f: "int[:]"):
    from numpy import shape

    return shape(f)[0]


def shape_indexed__test_shape_2d(f: "int[:,:]"):
    from numpy import shape

    a = shape(f)
    return a[0], a[1]


def shape_indexed__test_shape_2d_f(f: "int[:,:](order=F)"):
    from numpy import shape

    a = shape(f)
    return a[0], a[1]


def shape_property__test_shape_1d(f: "int[:]"):
    return f.shape[0]


def shape_property__test_shape_2d(f: "int[:,:]"):
    a = f.shape
    return a[0], a[1]


def shape_tuple_output__test_shape_1d(f: "int[:]"):
    from numpy import shape

    s = shape(f)
    return s[0]


def shape_tuple_output__test_shape_1d_tuple(f: "int[:]"):
    from numpy import shape

    (s,) = shape(f)
    return s


def shape_tuple_output__test_shape_2d(f: "int[:,:]"):
    from numpy import shape

    a, b = shape(f)
    return a, b


def shape_real__test_shape_1d(f: "float[:]"):
    from numpy import shape

    b = shape(f)
    return b[0]


def shape_real__test_shape_2d(f: "float[:,:]"):
    from numpy import shape

    a = shape(f)
    return a[0], a[1]


def shape_int__test_shape_1d(f: "int[:]"):
    from numpy import shape

    b = shape(f)
    return b[0]


def shape_int__test_shape_2d(f: "int[:,:]"):
    from numpy import shape

    a = shape(f)
    return a[0], a[1]


def shape_bool__test_shape_1d(f: "bool[:]"):
    from numpy import shape

    b = shape(f)
    return b[0]


def shape_bool__test_shape_2d(f: "bool[:,:]"):
    from numpy import shape

    a = shape(f)
    return a[0], a[1]


def full_basic_int__create_full_shape_1d(n: "int"):
    from numpy import full, shape

    a = full(n, 4)
    s = shape(a)
    return len(s), s[0]


def full_basic_int__create_full_shape_2d(n: "int"):
    from numpy import full, shape

    a = full((n, n), 4)
    s = shape(a)
    return len(s), s[0], s[1]


def full_basic_int__create_full_val(val: "int"):
    from numpy import full

    a = full(3, val)
    return a[0], a[1], a[2]


def full_basic_int__create_full_arg_names(val: "int"):
    from numpy import full

    a = full(fill_value=val, shape=(2, 3))
    return a[0, 0], a[0, 1], a[0, 2], a[1, 0], a[1, 1], a[1, 2]


def size__test_size_1d(f: "int[:]"):
    from numpy import size

    return size(f)


def size__test_size_2d(f: "int[:,:]"):
    from numpy import size

    return size(f)


def size__test_size_axis_variable_2d(f: "int[:,:]", axis: "int"):
    from numpy import size

    return size(f, axis)


def size__test_size_axis_literal_3d(f: "int[:,:,:]"):
    from numpy import size

    return size(f, 2)


def size_property__test_size_1d(f: "int[:]"):
    return f.size


def size_property__test_size_2d(f: "int[:,:]"):
    return f.size


def size_property__test_size_3d(f: "int[:,:,:]"):
    return f.size


def size_property__test_slice_size_2d(f: "int[:,:,:]"):
    return f[0, :, :].size


def full_basic_real__create_full_shape_1d(n: "int"):
    from numpy import full, shape

    a = full(n, 4)
    s = shape(a)
    return len(s), s[0]


def full_basic_real__create_full_shape_2d(n: "int"):
    from numpy import full, shape

    a = full((n, n), 4)
    s = shape(a)
    return len(s), s[0], s[1]


def full_basic_real__create_full_val(val: "float"):
    from numpy import full

    a = full(3, val)
    return a[0], a[1], a[2]


def full_basic_real__create_full_arg_names(val: "float"):
    from numpy import full

    a = full(fill_value=val, shape=(2, 3))
    return a[0, 0], a[0, 1], a[0, 2], a[1, 0], a[1, 1], a[1, 2]


def full_basic_bool__create_full_shape_1d(n: "int"):
    from numpy import full, shape

    a = full(n, 4)
    s = shape(a)
    return len(s), s[0]


def full_basic_bool__create_full_shape_2d(n: "int"):
    from numpy import full, shape

    a = full((n, n), 4)
    s = shape(a)
    return len(s), s[0], s[1]


def full_basic_bool__create_full_val(val: "bool"):
    from numpy import full

    a = full(3, val)
    return a[0], a[1], a[2]


def full_basic_bool__create_full_arg_names(val: "bool"):
    from numpy import full

    a = full(fill_value=val, shape=(2, 3))
    return a[0, 0], a[0, 1], a[0, 2], a[1, 0], a[1, 1], a[1, 2]


def full_order__create_full_shape_C(n: "int", m: "int"):
    from numpy import full, shape

    a = full((n, m), 4, order="C")
    s = shape(a)
    return len(s), s[0], s[1]


def full_order__create_full_shape_F(n: "int", m: "int"):
    from numpy import full, shape

    a = full((n, m), 4, order="F")
    s = shape(a)
    return len(s), s[0], s[1]


def full_dtype__create_full_val_int_int(val: "int"):
    from numpy import full

    a = full(3, val, int)
    return a[0]


def full_dtype__create_full_val_int_float(val: "int"):
    from numpy import full

    a = full(3, val, float)
    return a[0]


def full_dtype__create_full_val_int_complex(val: "int"):
    from numpy import full

    a = full(3, val, complex)
    return a[0]


def full_dtype__create_full_val_real_int32(val: "float"):
    from numpy import full, int32

    a = full(3, val, int32)
    return a[0]


def full_dtype__create_full_val_real_float32(val: "float"):
    from numpy import float32, full

    a = full(3, val, float32)
    return a[0]


def full_dtype__create_full_val_real_float64(val: "float"):
    from numpy import float64, full

    a = full(3, val, float64)
    return a[0]


def full_dtype__create_full_val_real_complex64(val: "float"):
    from numpy import complex64, full

    a = full(3, val, complex64)
    return a[0]


def full_dtype__create_full_val_real_complex128(val: "float"):
    from numpy import complex128, full

    a = full(3, val, complex128)
    return a[0]


def full_dtype_auto(val: T):
    from numpy import full

    a = full(3, val)
    return a[0]


def full_combined_args__create_full_1_shape():
    from numpy import full, shape

    a = full((2, 1), 4.0, int, "F")
    s = shape(a)
    return len(s), s[0], s[1]


def full_combined_args__create_full_1_val():
    from numpy import full

    a = full((2, 1), 4.0, int, "F")
    return a[0, 0]


def full_combined_args__create_full_2_shape():
    from numpy import full, shape

    a = full((4, 2), dtype=float, fill_value=1)
    s = shape(a)
    return len(s), s[0], s[1]


def full_combined_args__create_full_2_val():
    from numpy import full

    a = full((4, 2), dtype=float, fill_value=1)
    return a[0, 0]


def full_combined_args__create_full_3_shape():
    from numpy import full, shape

    a = full(order="F", shape=(4, 2), dtype=complex, fill_value=1)
    s = shape(a)
    return len(s), s[0], s[1]


def full_combined_args__create_full_3_val():
    from numpy import full

    a = full(order="F", shape=(4, 2), dtype=complex, fill_value=1)
    return a[0, 0]


def empty_basic__create_empty_shape_1d(n: "int"):
    from numpy import empty, shape

    a = empty(n)
    s = shape(a)
    return len(s), s[0]


def empty_basic__create_empty_shape_2d(n: "int"):
    from numpy import empty, shape

    a = empty((n, n))
    s = shape(a)
    return len(s), s[0], s[1]


def empty_order__create_empty_shape_C(n: "int", m: "int"):
    from numpy import empty, shape

    a = empty((n, m), order="C")
    s = shape(a)
    return len(s), s[0], s[1]


def empty_order__create_empty_shape_F(n: "int", m: "int"):
    from numpy import empty, shape

    p = (n, m)
    a = empty(p, order="F")
    s = shape(a)
    return len(s), s[0], s[1], len(p), p[0], p[1]


def empty_dtype__create_empty_val_int():
    from numpy import empty

    a = empty(3, int)
    return a[0]


def empty_dtype__create_empty_val_float():
    from numpy import empty

    a = empty(3, float)
    return a[0]


def empty_dtype__create_empty_val_complex():
    from numpy import empty

    a = empty(3, complex)
    return a[0]


def empty_dtype__create_empty_val_int32():
    from numpy import empty, int32

    a = empty(3, int32)
    return a[0]


def empty_dtype__create_empty_val_float32():
    from numpy import empty, float32

    a = empty(3, float32)
    return a[0]


def empty_dtype__create_empty_val_float64():
    from numpy import empty, float64

    a = empty(3, float64)
    return a[0]


def empty_dtype__create_empty_val_complex64():
    from numpy import complex64, empty

    a = empty(3, complex64)
    return a[0]


def empty_dtype__create_empty_val_complex128():
    from numpy import complex128, empty

    a = empty(3, complex128)
    return a[0]


def empty_combined_args__create_empty_1_shape():
    from numpy import empty, shape

    a = empty((2, 1), int, "F")
    s = shape(a)
    return len(s), s[0], s[1]


def empty_combined_args__create_empty_1_val():
    from numpy import empty

    a = empty((2, 1), int, "F")
    return a[0, 0]


def empty_combined_args__create_empty_2_shape():
    from numpy import empty, shape

    a = empty((4, 2), dtype=float)
    s = shape(a)
    return len(s), s[0], s[1]


def empty_combined_args__create_empty_2_val():
    from numpy import empty

    a = empty((4, 2), dtype=float)
    return a[0, 0]


def empty_combined_args__create_empty_3_shape():
    from numpy import empty, shape

    a = empty(order="F", shape=(4, 2), dtype=complex)
    s = shape(a)
    return len(s), s[0], s[1]


def empty_combined_args__create_empty_3_val():
    from numpy import empty

    a = empty(order="F", shape=(4, 2), dtype=complex)
    return a[0, 0]


def ones_basic__create_ones_shape_1d(n: "int"):
    from numpy import ones, shape

    a = ones(n)
    s = shape(a)
    return len(s), s[0]


def ones_basic__create_ones_shape_2d(n: "int"):
    from numpy import ones, shape

    a = ones((n, n))
    s = shape(a)
    return len(s), s[0], s[1]


def ones_order__create_ones_shape_C(n: "int", m: "int"):
    from numpy import ones, shape

    a = ones((n, m), order="C")
    s = shape(a)
    return len(s), s[0], s[1]


def ones_order__create_ones_shape_F(n: "int", m: "int"):
    from numpy import ones, shape

    a = ones((n, m), order="F")
    s = shape(a)
    return len(s), s[0], s[1]


def ones_dtype__create_ones_val_int():
    from numpy import ones

    a = ones(3, int)
    return a[0]


def ones_dtype__create_ones_val_float():
    from numpy import ones

    a = ones(3, float)
    return a[0]


def ones_dtype__create_ones_val_complex():
    from numpy import ones

    a = ones(3, complex)
    return a[0]


def ones_dtype__create_ones_val_int32():
    from numpy import int32, ones

    a = ones(3, int32)
    return a[0]


def ones_dtype__create_ones_val_float32():
    from numpy import float32, ones

    a = ones(3, float32)
    return a[0]


def ones_dtype__create_ones_val_float64():
    from numpy import float64, ones

    a = ones(3, float64)
    return a[0]


def ones_dtype__create_ones_val_complex64():
    from numpy import complex64, ones

    a = ones(3, complex64)
    return a[0]


def ones_dtype__create_ones_val_complex128():
    from numpy import complex128, ones

    a = ones(3, complex128)
    return a[0]


def ones_combined_args__create_ones_1_shape():
    from numpy import ones, shape

    a = ones((2, 1), int, "F")
    s = shape(a)
    return len(s), s[0], s[1]


def ones_combined_args__create_ones_1_val():
    from numpy import ones

    a = ones((2, 1), int, "F")
    return a[0, 0]


def ones_combined_args__create_ones_2_shape():
    from numpy import ones, shape

    a = ones((4, 2), dtype=float)
    s = shape(a)
    return len(s), s[0], s[1]


def ones_combined_args__create_ones_2_val():
    from numpy import ones

    a = ones((4, 2), dtype=float)
    return a[0, 0]


def ones_combined_args__create_ones_3_shape():
    from numpy import ones, shape

    a = ones(order="F", shape=(4, 2), dtype=complex)
    s = shape(a)
    return len(s), s[0], s[1]


def ones_combined_args__create_ones_3_val():
    from numpy import ones

    a = ones(order="F", shape=(4, 2), dtype=complex)
    return a[0, 0]


def ones_in_expression__ones_plus_scalar():
    from numpy import ones

    a = ones(3, dtype=int) + 2
    return a.sum()


def ones_in_expression__ones_times_scalar():
    from numpy import ones

    a = ones(4) * 3.0
    return a.sum()


def zeros_basic__create_zeros_shape_1d(n: "int"):
    from numpy import shape, zeros

    a = zeros(n)
    s = shape(a)
    return len(s), s[0]


def zeros_basic__create_zeros_shape_2d(n: "int"):
    from numpy import shape, zeros

    a = zeros((n, n))
    s = shape(a)
    return len(s), s[0], s[1]


def zeros_order__create_zeros_shape_C(n: "int", m: "int"):
    from numpy import shape, zeros

    a = zeros((n, m), order="C")
    s = shape(a)
    return len(s), s[0], s[1]


def zeros_order__create_zeros_shape_F(n: "int", m: "int"):
    from numpy import shape, zeros

    a = zeros((n, m), order="F")
    s = shape(a)
    return len(s), s[0], s[1]


def zeros_dtype__create_zeros_val_int():
    from numpy import zeros

    a = zeros(3, int)
    return a[0]


def zeros_dtype__create_zeros_val_float():
    from numpy import zeros

    a = zeros(3, float)
    return a[0]


def zeros_dtype__create_zeros_val_complex():
    from numpy import zeros

    a = zeros(3, complex)
    return a[0]


def zeros_dtype__create_zeros_val_int32():
    from numpy import int32, zeros

    a = zeros(3, int32)
    return a[0]


def zeros_dtype__create_zeros_val_float32():
    from numpy import float32, zeros

    a = zeros(3, float32)
    return a[0]


def zeros_dtype__create_zeros_val_float64():
    from numpy import float64, zeros

    a = zeros(3, float64)
    return a[0]


def zeros_dtype__create_zeros_val_complex64():
    from numpy import complex64, zeros

    a = zeros(3, complex64)
    return a[0]


def zeros_dtype__create_zeros_val_complex128():
    from numpy import complex128, zeros

    a = zeros(3, complex128)
    return a[0]


def zeros_combined_args__create_zeros_1_shape():
    from numpy import shape, zeros

    a = zeros((2, 1), int, "F")
    s = shape(a)
    return len(s), s[0], s[1]


def zeros_combined_args__create_zeros_1_val():
    from numpy import zeros

    a = zeros((2, 1), int, "F")
    return a[0, 0]


def zeros_combined_args__create_zeros_2_shape():
    from numpy import shape, zeros

    a = zeros((4, 2), dtype=float)
    s = shape(a)
    return len(s), s[0], s[1]


def zeros_combined_args__create_zeros_2_val():
    from numpy import zeros

    a = zeros((4, 2), dtype=float)
    return a[0, 0]


def zeros_combined_args__create_zeros_3_shape():
    from numpy import shape, zeros

    a = zeros(order="F", shape=(4, 2), dtype=complex)
    s = shape(a)
    return len(s), s[0], s[1]


def zeros_combined_args__create_zeros_3_val():
    from numpy import zeros

    a = zeros(order="F", shape=(4, 2), dtype=complex)
    return a[0, 0]


def zeros_in_expression__zeros_plus_scalar():
    from numpy import zeros

    a = zeros(3, dtype=int) + 2
    return a.sum()


def zeros_in_expression__zeros_times_scalar():
    from numpy import zeros

    a = zeros(4) * 3.0
    return a.sum()


def zeros_in_expression__zeros_2d_plus_scalar():
    from numpy import zeros

    a = zeros((2, 3), dtype=int) + 1
    return a.sum()


def zeros_in_expression__zeros_plus_ones():
    from numpy import ones, zeros

    a = zeros(3) + ones(3)
    return a.sum()


def array__create_array_list_shape():
    from numpy import array, shape

    a = array([[1, 2, 3], [4, 5, 6]])
    s = shape(a)
    return len(s), s[0], s[1]


def array__create_array_list_val():
    from numpy import array

    a = array([[1, 2, 3], [4, 5, 6]])
    return a[0, 0]


def array__create_array_tuple_shape():
    from numpy import array, shape

    a = array(((1, 2, 3), (4, 5, 6)))
    s = shape(a)
    return len(s), s[0], s[1]


def array__create_array_tuple_val():
    from numpy import array

    a = array(((1, 2, 3), (4, 5, 6)))
    return a[0, 0]


def array__create_array_tuple_ref(a: "int[:,:]"):
    from numpy import array

    b = (a[0, :], a[1, :])
    c = array(b)
    return c


def array_in_expression():
    from numpy import array

    a = array([[1, 2, 3], [4, 5, 6]]) * 2
    return a


def array_new_dtype(a: "int[:,:]"):
    from numpy import array

    b = (a[0, :], a[1, :])
    c = array(b, dtype=float)
    return c


def sum_bool(x: "bool[:]"):
    from numpy import sum as np_sum

    return np_sum(x)


def sum_int(x: "int[:]"):
    from numpy import sum as np_sum

    return np_sum(x)


def sum_override_builtin(x: "int[:]"):
    from numpy import sum  # pylint: disable=redefined-builtin

    return sum(x)


def sum_real(x: "float[:]"):
    from numpy import sum as np_sum

    return np_sum(x)


def sum_type(x: "float32[:]"):
    from numpy import sum as np_sum

    return np_sum(x)


def sum_phrase(x: "float[:]", y: "float[:]"):
    from numpy import sum as np_sum

    a = np_sum(x) * np_sum(y)
    return a


def sum_property(x: "int[:]"):
    return x.sum()


def sum_3d(x: "float[:,:,:]"):
    return np.sum(x)


def sum_dtype(x: "int[:]"):
    return np.sum(x, dtype=float)


def sum_dtype_2(x: "float[:,:]"):
    return np.sum(x, dtype=int)


def sum_axis_2d(x: "int[:,:]"):
    return np.sum(x, axis=1)


def sum_keepdims(x: "float[:,:]"):
    return np.sum(x, axis=1, keepdims=True)


def sum_initial(x: "int[:]"):
    return np.sum(x, initial=10)


def sum_axis_keepdims_initial(x: "int[:,:]"):
    return np.sum(x, axis=0, keepdims=True, initial=5)


def sum_dtype_axis(x: "int[:,:]"):
    return np.sum(x, axis=1, dtype=float)


def sum_3d_multi_axis(x: "float[:,:,:]"):
    return np.sum(x, axis=(1, 2))


def sum_out_axis_2d(x: "int[:,:]"):
    out = np.empty(x.shape[0], dtype=x.dtype)
    np.sum(x, axis=1, out=out)
    return out


def sum_out_axis_keepdims(x: "float[:,:]"):
    out = np.empty((x.shape[0], 1), dtype=x.dtype)
    np.sum(x, axis=1, keepdims=True, out=out)
    return out


def sum_out_reference(x: "float[:,:]"):
    out = np.empty((x.shape[0], 1), dtype=x.dtype)
    y = np.sum(x, axis=1, keepdims=True, out=out)
    out[1, 0] = -out[1, 0]
    return out[0, 0], y[0, 0], out[1, 0], y[1, 0]


def min_int(x: "int[:]"):
    from numpy import amin

    return amin(x)


def min_real(x: "float[:]"):
    from numpy import amin

    return amin(x)


def min_complex(x: "complex128[:]"):
    from numpy import amin

    return amin(x)


def min_bool(x: "bool[:]"):
    from numpy import amin

    return amin(x)


def min_phrase(x: "float[:]", y: "float[:]"):
    from numpy import amin

    a = amin(x) * amin(y)
    return a


def min_property(x: "int[:]"):
    return x.min()


def amin_1d(x: "int[:]"):
    from numpy import amin

    return amin(x)


def amin_axis(x: "int[:,:]"):
    from numpy import amin

    return amin(x, axis=1)


def amin_keepdims(x: "float[:,:]"):
    from numpy import amin

    return amin(x, axis=1, keepdims=True)


def amin_initial(x: "int[:]"):
    from numpy import amin

    return amin(x, initial=50)


def amin_out_axis(x: "int[:,:]", out: "int[:]"):
    np.amin(x, axis=1, out=out)


def max_int(x: "int[:]"):
    from numpy import amax

    return amax(x)


def max_real(x: "float[:]"):
    from numpy import amax

    return amax(x)


def max_complex(x: "complex128[:]"):
    from numpy import amax

    return amax(x)


def max_bool(x: "bool[:]"):
    from numpy import amax

    return amax(x)


def max_phrase(x: "float[:]", y: "float[:]"):
    from numpy import amax

    a = amax(x) * amax(y)
    return a


def max_property(x: "int[:]"):
    return x.max()


def amax_1d(x: "int[:]"):
    from numpy import amax

    return amax(x)


def amax_axis(x: "int[:,:]"):
    from numpy import amax

    return amax(x, axis=0)


def amax_keepdims(x: "float[:,:]"):
    from numpy import amax

    return amax(x, axis=0, keepdims=True)


def amax_initial(x: "int[:]"):
    from numpy import amax

    return amax(x, initial=10)


def amax_out_axis(x: "int[:,:]", out: "int[:]"):
    np.amax(x, axis=1, out=out)


def full_like_basic_int__create_shape_1d(n: "int"):
    from numpy import array, full_like, shape

    arr = array([5, 1, 8, 0, 9])
    a = full_like(arr, n, int, "F")
    s = shape(a)
    return len(s), s[0]


def full_like_basic_int__create_shape_2d(n: "int"):
    from numpy import array, full_like, shape

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = full_like(arr, n, int, "F")
    s = shape(a)
    return len(s), s[0], s[1]


def full_like_basic_int__create_val(val: "int"):
    from numpy import array, full_like

    arr = array([5, 1, 8, 0, 9])
    a = full_like(arr, val, int, "F")
    return a[0], a[1], a[2]


def full_like_basic_int__create_arg_names(val: "int"):
    from numpy import array, full_like

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = full_like(arr, val, int, "F", shape=(2, 3))
    return a[0, 0], a[0, 1], a[0, 2], a[1, 0], a[1, 1], a[1, 2]


def full_like_basic_real__create_shape_1d(n: "float"):
    from numpy import array, full_like, shape

    arr = array([5, 1, 8, 0, 9])
    a = full_like(arr, n, float, "F")
    s = shape(a)
    return len(s), s[0]


def full_like_basic_real__create_shape_2d(n: "float"):
    from numpy import array, full_like, shape

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = full_like(arr, n, float, "F")
    s = shape(a)
    return len(s), s[0], s[1]


def full_like_basic_real__create_val(val: "float"):
    from numpy import array, full_like

    arr = array([5, 1, 8, 0, 9])
    a = full_like(arr, val, float, "F")
    return a[0], a[1], a[2]


def full_like_basic_real__create_arg_names(val: "float"):
    from numpy import array, full_like

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = full_like(arr, val, float, "F", shape=(2, 3))
    return a[0, 0], a[0, 1], a[0, 2], a[1, 0], a[1, 1], a[1, 2]


def full_like_order__create_shape_C(n: "int"):
    from numpy import array, full_like, shape

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = full_like(arr, 4, order="C")
    s = shape(a)
    return len(s), s[0], s[1]


def full_like_order__create_shape_F(n: "int"):
    from numpy import array, full_like, shape

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = full_like(arr, 4, order="F")
    s = shape(a)
    return len(s), s[0], s[1]


def full_like_dtype__create_val_int_int_auto(val: "int"):
    from numpy import array, full_like

    arr = array([5, 1, 8, 0, 9], int)
    a = full_like(arr, val)
    return a[0]


def full_like_dtype__create_val_int_int(val: "int"):
    from numpy import array, full_like

    arr = array([5, 1, 8, 0, 9])
    a = full_like(arr, val, int)
    return a[0]


def full_like_dtype__create_val_int_float_auto(val: "int"):
    from numpy import array, full_like

    arr = array([5, 1, 8, 0, 9], float)
    a = full_like(arr, val)
    return a[0]


def full_like_dtype__create_val_int_float(val: "int"):
    from numpy import array, full_like

    arr = array([5, 1, 8, 0, 9])
    a = full_like(arr, val, float)
    return a[0]


def full_like_dtype__create_val_int_complex_auto(val: "int"):
    from numpy import array, full_like

    arr = array([5, 1, 8, 0, 9], complex)
    a = full_like(arr, val)
    return a[0]


def full_like_dtype__create_val_int_complex(val: "int"):
    from numpy import array, full_like

    arr = array([5, 1, 8, 0, 9])
    a = full_like(arr, val, complex)
    return a[0]


def full_like_dtype__create_val_real_int32_auto(val: "float"):
    from numpy import array, full_like, int32

    arr = array([5, 1, 8, 0, 9], int32)
    a = full_like(arr, val)
    return a[0]


def full_like_dtype__create_val_real_int32(val: "float"):
    from numpy import array, full_like, int32

    arr = array([5, 1, 8, 0, 9])
    a = full_like(arr, val, int32)
    return a[0]


def full_like_dtype__create_val_real_float32_auto(val: "float"):
    from numpy import array, float32, full_like

    arr = array([5, 1, 8, 0, 9], float32)
    a = full_like(arr, val)
    return a[0]


def full_like_dtype__create_val_real_float32(val: "float"):
    from numpy import array, float32, full_like

    arr = array([5, 1, 8, 0, 9])
    a = full_like(arr, val, float32)
    return a[0]


def full_like_dtype__create_val_real_float64_auto(val: "float"):
    from numpy import array, float64, full_like

    arr = array([5, 1, 8, 0, 9], float64)
    a = full_like(arr, val)
    return a[0]


def full_like_dtype__create_val_real_float64(val: "float"):
    from numpy import array, float64, full_like

    arr = array([5, 1, 8, 0, 9])
    a = full_like(arr, val, float64)
    return a[0]


def full_like_dtype__create_val_real_complex64_auto(val: "float"):
    from numpy import array, complex64, full_like

    arr = array([5, 1, 8, 0, 9], complex64)
    a = full_like(arr, val)
    return a[0]


def full_like_dtype__create_val_real_complex64(val: "float"):
    from numpy import array, complex64, full_like

    arr = array([5, 1, 8, 0, 9])
    a = full_like(arr, val, complex64)
    return a[0]


def full_like_dtype__create_val_real_complex128_auto(val: "float"):
    from numpy import array, complex128, full_like

    arr = array([5, 1, 8, 0, 9], complex128)
    a = full_like(arr, val)
    return a[0]


def full_like_dtype__create_val_real_complex128(val: "float"):
    from numpy import array, complex128, full_like

    arr = array([5, 1, 8, 0, 9])
    a = full_like(arr, val, complex128)
    return a[0]


def full_like_combined_args__create_1_shape():
    from numpy import array, full_like, shape

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = full_like(arr, 5, int, "F")
    s = shape(a)
    return len(s), s[0], s[1]


def full_like_combined_args__create_1_val():
    from numpy import array, full_like

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = full_like(arr, 4.0, int, "F")
    return a[0, 0]


def full_like_combined_args__create_2_shape():
    from numpy import array, full_like, shape

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = full_like(arr, dtype=float, fill_value=1)
    s = shape(a)
    return len(s), s[0], s[1]


def full_like_combined_args__create_2_val():
    from numpy import array, full_like

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = full_like(arr, dtype=float, fill_value=1)
    return a[0, 0]


def full_like_combined_args__create_3_shape():
    from numpy import array, full_like, shape

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = full_like(arr, order="F", shape=(4, 2), dtype=complex, fill_value=1)
    s = shape(a)
    return len(s), s[0], s[1]


def full_like_combined_args__create_3_val():
    from numpy import array, full_like

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = full_like(arr, order="F", shape=(4, 2), dtype=complex, fill_value=1)
    return a[0, 0]


def empty_like_basic__create_empty_like_shape_1d(n: "int"):
    from numpy import array, empty_like, shape

    arr = array([5, 1, 8, 0, 9])
    a = empty_like(arr, int)
    s = shape(a)
    return len(s), s[0]


def empty_like_basic__create_empty_like_shape_2d(n: "int"):
    from numpy import array, empty_like, shape

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = empty_like(arr, int)
    s = shape(a)
    return len(s), s[0], s[1]


def empty_like_order__create_empty_like_shape_C(n: "int", m: "int"):
    from numpy import array, empty_like, shape

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = empty_like(arr, int, order="C")
    s = shape(a)
    return len(s), s[0], s[1]


def empty_like_order__create_empty_like_shape_F(n: "int", m: "int"):
    from numpy import array, empty_like, shape

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = empty_like(arr, int, order="F")
    s = shape(a)
    return len(s), s[0], s[1]


def empty_like_dtype__create_empty_like_val_int_auto():
    from numpy import array, empty_like

    arr = array([5, 1, 8, 0, 9], dtype=int)
    a = empty_like(arr)
    return a[0]


def empty_like_dtype__create_empty_like_val_int():
    from numpy import array, empty_like

    arr = array([5, 1, 8, 0, 9])
    a = empty_like(arr, int)
    return a[0]


def empty_like_dtype__create_empty_like_val_float_auto():
    from numpy import array, empty_like

    arr = array([5, 1, 8, 0, 9], dtype=float)
    a = empty_like(arr)
    return a[0]


def empty_like_dtype__create_empty_like_val_float():
    from numpy import array, empty_like

    arr = array([5, 1, 8, 0, 9])
    a = empty_like(arr, dtype=float)
    return a[0]


def empty_like_dtype__create_empty_like_val_complex_auto():
    from numpy import array, empty_like

    arr = array([5, 1, 8, 0, 9], dtype=complex)
    a = empty_like(arr)
    return a[0]


def empty_like_dtype__create_empty_like_val_complex():
    from numpy import array, empty_like

    arr = array([5, 1, 8, 0, 9])
    a = empty_like(arr, dtype=complex)
    return a[0]


def empty_like_dtype__create_empty_like_val_int32_auto():
    from numpy import array, empty_like, int32

    arr = array([5, 1, 8, 0, 9], dtype=int32)
    a = empty_like(arr)
    return a[0]


def empty_like_dtype__create_empty_like_val_int32():
    from numpy import array, empty_like, int32

    arr = array([5, 1, 8, 0, 9])
    a = empty_like(arr, dtype=int32)
    return a[0]


def empty_like_dtype__create_empty_like_val_float32_auto():
    from numpy import array, empty_like

    arr = array([5, 1, 8, 0, 9], dtype="float32")
    a = empty_like(arr)
    return a[0]


def empty_like_dtype__create_empty_like_val_float32():
    from numpy import array, empty_like, float32

    arr = array([5, 1, 8, 0, 9])
    a = empty_like(arr, dtype=float32)
    return a[0]


def empty_like_dtype__create_empty_like_val_float64_auto():
    from numpy import array, empty_like, float64

    arr = array([5, 1, 8, 0, 9], dtype=float64)
    a = empty_like(arr)
    return a[0]


def empty_like_dtype__create_empty_like_val_float64():
    from numpy import array, empty_like, float64

    arr = array([5, 1, 8, 0, 9])
    a = empty_like(arr, dtype=float64)
    return a[0]


def empty_like_dtype__create_empty_like_val_complex64_auto():
    from numpy import array, complex64, empty_like

    arr = array([5, 1, 8, 0, 9], dtype=complex64)
    a = empty_like(arr)
    return a[0]


def empty_like_dtype__create_empty_like_val_complex64():
    from numpy import array, complex64, empty_like

    arr = array([5, 1, 8, 0, 9])
    a = empty_like(arr, dtype=complex64)
    return a[0]


def empty_like_dtype__create_empty_like_val_complex128_auto():
    from numpy import array, complex128, empty_like

    arr = array([5, 1, 8, 0, 9], dtype=complex128)
    a = empty_like(arr)
    return a[0]


def empty_like_dtype__create_empty_like_val_complex128():
    from numpy import array, complex128, empty_like

    arr = array([5, 1, 8, 0, 9])
    a = empty_like(arr, dtype=complex128)
    return a[0]


def empty_like_combined_args__create_empty_like_1_shape():
    from numpy import array, empty_like, shape

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = empty_like(arr, dtype=int, order="F")
    s = shape(a)
    return len(s), s[0], s[1]


def empty_like_combined_args__create_empty_like_1_val():
    from numpy import array, empty_like

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = empty_like(arr, dtype=int, order="F")
    return a[0, 0]


def empty_like_combined_args__create_empty_like_2_shape():
    from numpy import array, empty_like, shape

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = empty_like(arr, dtype=float)
    s = shape(a)
    return len(s), s[0], s[1]


def empty_like_combined_args__create_empty_like_2_val():
    from numpy import array, empty_like

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = empty_like(arr, dtype=float)
    return a[0, 0]


def empty_like_combined_args__create_empty_like_3_shape():
    from numpy import array, empty_like, shape

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = empty_like(arr, shape=(4, 2), order="F", dtype=complex)
    s = shape(a)
    return len(s), s[0], s[1]


def empty_like_combined_args__create_empty_like_3_val():
    from numpy import array, empty_like

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = empty_like(arr, shape=(4, 2), order="F", dtype=complex)
    return a[0, 0]


def ones_like_basic__create_ones_like_shape_1d(n: "int"):
    from numpy import array, ones_like, shape

    arr = array([5, 1, 8, 0, 9])
    a = ones_like(arr)
    s = shape(a)
    return len(s), s[0]


def ones_like_basic__create_ones_like_shape_2d(n: "int"):
    from numpy import array, ones_like, shape

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = ones_like(arr)
    s = shape(a)
    return len(s), s[0], s[1]


def ones_like_order__create_ones_like_shape_C(n: "int", m: "int"):
    from numpy import array, ones_like, shape

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = ones_like(arr, order="C")
    s = shape(a)
    return len(s), s[0], s[1]


def ones_like_order__create_ones_like_shape_F(n: "int", m: "int"):
    from numpy import array, ones_like, shape

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = ones_like(arr, order="F")
    s = shape(a)
    return len(s), s[0], s[1]


def ones_like_dtype__create_ones_like_val_int():
    from numpy import array, ones_like

    arr = array([5, 1, 8, 0, 9])
    a = ones_like(arr, int)
    return a[0]


def ones_like_dtype__create_ones_like_val_float():
    from numpy import array, ones_like

    arr = array([5, 1, 8, 0, 9])
    a = ones_like(arr, float)
    return a[0]


def ones_like_dtype__create_ones_like_val_complex():
    from numpy import array, ones_like

    arr = array([5, 1, 8, 0, 9])
    a = ones_like(arr, complex)
    return a[0]


def ones_like_dtype__create_ones_like_val_int32():
    from numpy import array, int32, ones_like

    arr = array([5, 1, 8, 0, 9])
    a = ones_like(arr, int32)
    return a[0]


def ones_like_dtype__create_ones_like_val_float32():
    from numpy import array, float32, ones_like

    arr = array([5, 1, 8, 0, 9])
    a = ones_like(arr, float32)
    return a[0]


def ones_like_dtype__create_ones_like_val_float64():
    from numpy import array, float64, ones_like

    arr = array([5, 1, 8, 0, 9])
    a = ones_like(arr, float64)
    return a[0]


def ones_like_dtype__create_ones_like_val_complex64():
    from numpy import array, complex64, ones_like

    arr = array([5, 1, 8, 0, 9])
    a = ones_like(arr, complex64)
    return a[0]


def ones_like_dtype__create_ones_like_val_complex128():
    from numpy import array, complex128, ones_like

    arr = array([5, 1, 8, 0, 9])
    a = ones_like(arr, complex128)
    return a[0]


def ones_like_dtype__create_ones_like_val_int_auto():
    from numpy import array, ones_like

    arr = array([5, 1, 8, 0, 9], int)
    a = ones_like(arr)
    return a[0]


def ones_like_dtype__create_ones_like_val_float_auto():
    from numpy import array, ones_like

    arr = array([5, 1, 8, 0, 9], float)
    a = ones_like(arr)
    return a[0]


def ones_like_dtype__create_ones_like_val_complex_auto():
    from numpy import array, ones_like

    arr = array([5, 1, 8, 0, 9], complex)
    a = ones_like(arr)
    return a[0]


def ones_like_dtype__create_ones_like_val_int32_auto():
    from numpy import array, int32, ones_like

    arr = array([5, 1, 8, 0, 9], int32)
    a = ones_like(arr)
    return a[0]


def ones_like_dtype__create_ones_like_val_float32_auto():
    from numpy import array, float32, ones_like

    arr = array([5, 1, 8, 0, 9], float32)
    a = ones_like(arr)
    return a[0]


def ones_like_dtype__create_ones_like_val_float64_auto():
    from numpy import array, float64, ones_like

    arr = array([5, 1, 8, 0, 9], float64)
    a = ones_like(arr)
    return a[0]


def ones_like_dtype__create_ones_like_val_complex64_auto():
    from numpy import array, complex64, ones_like

    arr = array([5, 1, 8, 0, 9], complex64)
    a = ones_like(arr)
    return a[0]


def ones_like_dtype__create_ones_like_val_complex128_auto():
    from numpy import array, complex128, ones_like

    arr = array([5, 1, 8, 0, 9], complex128)
    a = ones_like(arr)
    return a[0]


def ones_like_combined_args__create_ones_like_1_shape():
    from numpy import array, ones_like, shape

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = ones_like(arr, int, "F")
    s = shape(a)
    return len(s), s[0], s[1]


def ones_like_combined_args__create_ones_like_1_val():
    from numpy import array, ones_like

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = ones_like(arr, int, "F")
    return a[0, 0]


def ones_like_combined_args__create_ones_like_2_shape():
    from numpy import array, ones_like, shape

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = ones_like(arr, dtype=float)
    s = shape(a)
    return len(s), s[0], s[1]


def ones_like_combined_args__create_ones_like_2_val():
    from numpy import array, ones_like

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = ones_like(arr, dtype=float)
    return a[0, 0]


def ones_like_combined_args__create_ones_like_3_shape():
    from numpy import array, ones_like, shape

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = ones_like(arr, shape=(4, 2), order="F", dtype=complex)
    s = shape(a)
    return len(s), s[0], s[1]


def ones_like_combined_args__create_ones_like_3_val():
    from numpy import array, ones_like

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = ones_like(arr, shape=(4, 2), order="F", dtype=complex)
    return a[0, 0]


def zeros_like_basic__create_zeros_like_shape_1d(n: "int"):
    from numpy import array, shape, zeros_like

    arr = array([5, 1, 8, 0, 9])
    a = zeros_like(arr, int)
    s = shape(a)
    return len(s), s[0]


def zeros_like_basic__create_zeros_like_shape_2d(n: "int"):
    from numpy import array, shape, zeros_like

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = zeros_like(arr, int)
    s = shape(a)
    return len(s), s[0], s[1]


def zeros_like_order__create_zeros_like_shape_C(n: "int", m: "int"):
    from numpy import array, shape, zeros_like

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = zeros_like(arr, order="C")
    s = shape(a)
    return len(s), s[0], s[1]


def zeros_like_order__create_zeros_like_shape_F(n: "int", m: "int"):
    from numpy import array, shape, zeros_like

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = zeros_like(arr, order="F")
    s = shape(a)
    return len(s), s[0], s[1]


def zeros_like_dtype__create_zeros_like_val_int():
    from numpy import array, zeros_like

    arr = array([5, 1, 8, 0, 9])
    a = zeros_like(arr, int)
    return a[0]


def zeros_like_dtype__create_zeros_like_val_float():
    from numpy import array, zeros_like

    arr = array([5, 1, 8, 0, 9])
    a = zeros_like(arr, float)
    return a[0]


def zeros_like_dtype__create_zeros_like_val_complex():
    from numpy import array, zeros_like

    arr = array([5, 1, 8, 0, 9])
    a = zeros_like(arr, complex)
    return a[0]


def zeros_like_dtype__create_zeros_like_val_int32():
    from numpy import array, int32, zeros_like

    arr = array([5, 1, 8, 0, 9])
    a = zeros_like(arr, int32)
    return a[0]


def zeros_like_dtype__create_zeros_like_val_float32():
    from numpy import array, float32, zeros_like

    arr = array([5, 1, 8, 0, 9])
    a = zeros_like(arr, float32)
    return a[0]


def zeros_like_dtype__create_zeros_like_val_float64():
    from numpy import array, float64, zeros_like

    arr = array([5, 1, 8, 0, 9])
    a = zeros_like(arr, float64)
    return a[0]


def zeros_like_dtype__create_zeros_like_val_complex64():
    from numpy import array, complex64, zeros_like

    arr = array([5, 1, 8, 0, 9])
    a = zeros_like(arr, complex64)
    return a[0]


def zeros_like_dtype__create_zeros_like_val_complex128():
    from numpy import array, complex128, zeros_like

    arr = array([5, 1, 8, 0, 9])
    a = zeros_like(arr, complex128)
    return a[0]


def zeros_like_dtype_auto__create_val_int():
    from numpy import array, zeros_like

    arr = array([5, 1, 8, 0, 9], dtype=int)
    a = zeros_like(arr)
    return a[0]


def zeros_like_dtype_auto__create_val_float():
    from numpy import array, zeros_like

    arr = array([5, 1, 8, 0, 9], dtype=float)
    a = zeros_like(arr)
    return a[0]


def zeros_like_dtype_auto__create_val_complex():
    from numpy import array, zeros_like

    arr = array([5, 1, 8, 0, 9], dtype=complex)
    a = zeros_like(arr)
    return a[0]


def zeros_like_dtype_auto__create_val_int32():
    from numpy import array, int32, zeros_like

    arr = array([5, 1, 8, 0, 9], dtype=int32)
    a = zeros_like(arr)
    return a[0]


def zeros_like_dtype_auto__create_val_float32():
    from numpy import array, zeros_like

    arr = array([5, 1, 8, 0, 9], dtype="float32")
    a = zeros_like(arr)
    return a[0]


def zeros_like_dtype_auto__create_val_float64():
    from numpy import array, float64, zeros_like

    arr = array([5, 1, 8, 0, 9], dtype=float64)
    a = zeros_like(arr)
    return a[0]


def zeros_like_dtype_auto__create_val_complex64():
    from numpy import array, complex64, zeros_like

    arr = array([5, 1, 8, 0, 9], dtype=complex64)
    a = zeros_like(arr)
    return a[0]


def zeros_like_dtype_auto__create_val_complex128():
    from numpy import array, complex128, zeros_like

    arr = array([5, 1, 8, 0, 9], dtype=complex128)
    a = zeros_like(arr)
    return a[0]


def zeros_like_combined_args__create_zeros_like_1_shape():
    from numpy import array, shape, zeros_like

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = zeros_like(arr, int, "F")
    s = shape(a)
    return len(s), s[0], s[1]


def zeros_like_combined_args__create_zeros_like_1_val():
    from numpy import array, zeros_like

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = zeros_like(arr, int, "F")
    return a[0, 0]


def zeros_like_combined_args__create_zeros_like_2_shape():
    from numpy import array, shape, zeros_like

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = zeros_like(arr, dtype=float)
    s = shape(a)
    return len(s), s[0], s[1]


def zeros_like_combined_args__create_zeros_like_2_val():
    from numpy import array, zeros_like

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = zeros_like(arr, dtype=float)
    return a[0, 0]


def zeros_like_combined_args__create_zeros_like_3_shape():
    from numpy import array, shape, zeros_like

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = zeros_like(arr, shape=(4, 2), order="F", dtype=complex)
    s = shape(a)
    return len(s), s[0], s[1]


def zeros_like_combined_args__create_zeros_like_3_val():
    from numpy import array, zeros_like

    arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
    a = zeros_like(arr, shape=(4, 2), order="F", dtype=complex)
    return a[0, 0]


def numpy_real_scalar(a: C):
    from numpy import real

    b = real(a)
    return b


def numpy_real_array_like_1d(arr: "C[:]"):
    from numpy import real, shape

    a = real(arr)
    s = shape(a)
    return len(s), s[0], a[0]


def numpy_real_array_like_2d(arr: "C[:,:]"):
    from numpy import real, shape

    a = real(arr)
    s = shape(a)
    return len(s), s[0], s[1], a[0, 1], a[1, 0]


def numpy_imag_scalar(a: C):
    from numpy import imag

    b = imag(a)
    return b


def numpy_imag_array_like_1d(arr: "C[:]"):
    from numpy import imag, shape

    a = imag(arr)
    s = shape(a)
    return len(s), s[0], a[0]


def numpy_imag_array_like_2d(arr: "C[:,:]"):
    from numpy import imag, shape

    a = imag(arr)
    s = shape(a)
    return len(s), s[0], s[1], a[0, 1], a[1, 0]


def numpy_mod_scalar(a: F):
    from numpy import mod

    b = mod(a, a)
    return b


def numpy_mod_array_like_1d(arr: "F[:]"):
    from numpy import mod, shape

    a = mod(arr, arr)
    s = shape(a)
    return len(s), s[0], a[0]


def numpy_mod_array_like_2d(arr: "F[:,:]"):
    from numpy import mod, shape

    a = mod(arr, arr)
    s = shape(a)
    return len(s), s[0], s[1], a[0, 1], a[1, 0]


def numpy_mod_mixed_order(arr1: "float[:,:]", arr2: "float[:,:](order=F)"):
    from numpy import mod, shape

    a = mod(arr1, arr2)
    s = shape(a)
    return len(s), s[0], s[1], a[0, 1], a[1, 0]


def numpy_norm_scalar(a: C):
    from numpy.linalg import norm

    b = norm(a)
    return b


def numpy_norm_scalar_expr(a: C):
    from numpy.linalg import norm

    b = norm(a) + 22
    return b


def numpy_norm_array_like_1d(arr: "C[:]"):
    from numpy.linalg import norm

    a = norm(arr)
    return a


def numpy_norm_array_like_2d(arr: "C[:,:]"):
    from numpy.linalg import norm

    a = norm(arr)
    return a


def numpy_norm_array_like_2d_fortran_order(arr: "C[:,:](order=F)"):
    from numpy import shape
    from numpy.linalg import norm

    a = norm(arr, axis=0)
    b = norm(arr, axis=1)
    sa = shape(a)
    sb = shape(b)
    return len(sb), sb[0], len(sa), sa[0], a[0], b[0]


def numpy_norm_array_like_3d(arr: "C[:,:,:]"):
    from numpy.linalg import norm

    a = norm(arr)
    return a


def numpy_norm_array_like_3d_fortran_order(arr: "C[:,:,:](order=F)"):
    from numpy import shape
    from numpy.linalg import norm

    a = norm(arr, axis=0)
    b = norm(arr, axis=1)
    c = norm(arr, axis=2)
    sa = shape(a)
    sb = shape(b)
    sc = shape(c)
    return len(sc), sc[0], len(sb), sb[0], len(sa), sa[0], a[0][0], b[0][0], c[0][0]


def norm_axis_2d(x: "float[:,:]"):
    from numpy.linalg import norm

    return norm(x, axis=(1,))


def norm_axis_keepdims(x: "float[:,:]"):
    from numpy.linalg import norm

    return norm(x, axis=1, keepdims=True)


def numpy_matmul_array_like_1d(arr: "T[:]"):
    from numpy import matmul

    a = matmul(arr, arr)
    return a


def numpy_matmul_array_like_2x2d(arr: "T[:,:]"):
    from numpy import matmul, shape

    a = matmul(arr, arr)
    s = shape(a)
    return len(s), s[0], s[1], a[0, 1], a[1, 0]


def matmul_4d_multi_batch(a: "float[:,:,:,:]", b: "float[:,:,:,:]"):
    from numpy import matmul

    return matmul(a, b)


def matmul_3d_broadcast_batch(a: "float[:,:,:]", b: "float[:,:]"):
    from numpy import matmul

    return matmul(a, b)


def numpy_where_array_like_1d_with_condition(arr: "F[:]"):
    from numpy import shape, where

    a = where(arr > 0, arr, arr * 2)
    s = shape(a)
    return len(s), s[0], a[1], a[0]


def numpy_where_array_like_2d_with_condition(arr: "F[:,:]"):
    from numpy import shape, where

    a = where(arr < 0, arr, arr + 1)
    s = shape(a)
    return len(s), s[0], a[0, 0], a[0, 1], a[1, 0], a[1, 1]


def numpy_where_complex(arr1: "CNT[:]", arr2: "CNT[:]", cond: "bool[:]"):
    from numpy import shape, where

    a = where(cond, arr1, arr2)
    s = shape(a)
    return len(s), s[0], a[1], a[0]


def where_combined_types(
    cond: "bool[:]",
    arr1: "int32[:] | float64[:] | complex128[:]",
    arr2: "int64[:] | float32[:]",
):
    from numpy import shape, where

    a = where(cond, arr1, arr2)
    s = shape(a)
    return len(s), s[0], a[1], a[0]


def numpy_linspace_scalar__get_linspace(start: S, steps: int, num: int):
    stop = start + steps
    b = np.linspace(start, stop, num)
    return b


def numpy_linspace_scalar__test_linspace_type(
    start: "int", end: "int", result: "int64[:]"
):
    x = np.linspace(start + 4, end, 15, dtype=np.int64)
    ret = 1
    for xi in enumerate(x):
        if xi != result[i]:
            ret = 0
    return ret, x[int(len(x) / 2)]


def numpy_linspace_scalar__test_linspace_type2(
    start: "int", end: "int", result: "complex128[:]"
):
    x = np.linspace(start, end * 2, 15, dtype="complex128")
    for i, xi in enumerate(x):
        result[i] = xi


def numpy_linspace_scalar__test_linspace_int(
    start: FI, end: FI, step: int, endpoint: bool
):
    return np.linspace(start, end, step, endpoint, dtype=np.int32)


def numpy_linspace_scalar__test_linspace(start: "complex64", end: "complex64"):
    x = np.linspace(start, end, 5)
    return x[0], x[1], x[2], x[3], x[4]


def numpy_linspace_scalar__test_linspace2(start: "complex128", end: "complex128"):
    x = np.linspace(start, end, 5)
    return x[0], x[1], x[2], x[3], x[4]


def numpy_linspace_array_like_1d__test_linspace(
    start: "S[:]", stop: int, endpoint: bool
):
    numberOfSamplesToGenerate = 7
    a = np.linspace(start, stop, numberOfSamplesToGenerate, endpoint=endpoint)
    return a


def numpy_linspace_array_like_1d__test_linspace2(
    start: "complex128[:]", stop: "int", out: "complex128[:,:]", endpoint: "bool"
):
    numberOfSamplesToGenerate = 7
    a = np.linspace(start, stop, numberOfSamplesToGenerate, endpoint=endpoint)
    for i in range(len(out)):
        for j in range(len(out[i])):
            out[i][j] = a[i][j]


def numpy_linspace_array_like_1d__test_linspace_dtype(
    start: "int[:] | float64[:]", stop: int, endpoint: bool
):
    numberOfSamplesToGenerate = 7
    a = np.linspace(
        start,
        stop,
        numberOfSamplesToGenerate,
        endpoint=endpoint,
        dtype=np.int32,
    )
    return a


def numpy_linspace_array_like_2d__test_linspace(
    start: "S[:,:]", stop: int, endpoint: bool
):
    numberOfSamplesToGenerate = 7
    a = np.linspace(start, stop, numberOfSamplesToGenerate, endpoint=endpoint)
    return a


def numpy_linspace_array_like_2d__test_linspace3(
    start: "complex128[:,:]",
    stop: "int",
    out: "complex128[:,:,:]",
    endpoint: "bool",
):
    numberOfSamplesToGenerate = 7
    a = np.linspace(start, stop, numberOfSamplesToGenerate, endpoint=endpoint)
    for i in range(len(out)):
        for j in range(len(out[i])):
            for k in range(len(out[i][j])):
                out[i][j][k] = a[i][j][k]


def numpy_linspace_array_like_2d__test_linspace2(
    start: "int[:,:]", stop: "int[:,:]", out: "float[:,:,:]", endpoint: "bool"
):
    numberOfSamplesToGenerate = 7
    a = np.linspace(start, stop, numberOfSamplesToGenerate, endpoint=endpoint)
    for i in range(len(out)):
        for j in range(len(out[i])):
            for k in range(len(out[i][j])):
                out[i][j][k] = a[i][j][k]


def numpy_linspace_array_like_2d__test_linspace4(
    start: "complex128[:,:]",
    stop: "complex128[:,:]",
    out: "complex128[:,:,:]",
    endpoint: "bool",
):
    numberOfSamplesToGenerate = 7
    a = np.linspace(start, stop, numberOfSamplesToGenerate, endpoint=endpoint)
    for i in range(len(out)):
        for j in range(len(out[i])):
            for k in range(len(out[i][j])):
                out[i][j][k] = a[i][j][k]


def dtype(a: "F[:]"):
    from numpy import zeros

    b = zeros(5, dtype=a.dtype)
    return b[0]


def result_type__int_vs_int_array():
    b = np.zeros(5, dtype=np.result_type(3, np.arange(7, dtype=np.int32)))
    return b[0]


def result_type__type_comparison():
    b = np.zeros(5, dtype=np.result_type(np.int32, np.int16))
    return b[0]


def result_type__type_comparison2():
    b = np.zeros(5, dtype=np.result_type(np.int32, np.complex64))
    return b[0]


def result_type__value_types():
    b = np.zeros(5, dtype=np.result_type(3.0, -2))
    return b[0]


def test_copy__copy_array(a: test_copy__X):
    b = a.copy()
    return b


def test_copy__copy_array_to_F(a: test_copy__Y):
    b = a.copy(order="F")
    return b


def test_copy__copy_array_to_C(a: test_copy__Y):
    b = a.copy(order="C")
    return b


def cross_1d(x: "float[:]", y: "float[:]"):
    return np.cross(x, y)


def cross_1d_expr(x: "float[:]", y: "float[:]"):
    return np.cross(x, y) + 2


def cross_2d_axis(x: "float[:,:]", y: "float[:,:]"):
    return np.cross(a=x, b=y, axis=1)


def cross_mixed_dimensions(x: "int[:,:]"):
    y = np.array(x[0:1, :])
    return np.cross(x, y)


def linalg_cross_1d(x: "float[:]", y: "float[:]"):
    return np.linalg.cross(x, y)


def linalg_cross_1d_mixed_types(x: "float[:]", y: "int[:]"):
    return np.linalg.cross(x, y)


def linalg_cross_axis(x: "float[:,:]", y: "float[:,:]"):
    return np.linalg.cross(x, y, axis=1)


def cross_axisa_axisb(x: "float[:,:]", y: "float[:,:]"):
    return np.cross(x, y, axisa=1, axisb=1)


def cross_axisc(x: "float[:,:]", y: "float[:,:]"):
    return np.cross(x, y, axisc=1)


def cross_axisa_axisb_axisc(x: "float[:,:,:]", y: "float[:,:,:]"):
    return np.cross(x, y, axisa=2, axisb=1, axisc=2)


def vecdot_1d_real(x: "float[:]", y: "float[:]"):
    return np.vecdot(x, y)


def vecdot_1d_complex(x: "complex[:]", y: "complex[:]"):
    return np.vecdot(x, y)


def vecdot_axis_2d(x: "float[:,:]", y: "float[:,:]"):
    return np.vecdot(x, y, axis=1)


def vecdot_mixed_dimensions(x: "float[:,:]", y: "float[:]"):
    return np.vecdot(x, y)


def vecdot_out_axis_2d(x: "float[:,:]", y: "float[:,:]"):
    out = np.empty(x.shape[0], dtype=x.dtype)
    np.vecdot(x, y, axis=1, out=out)
    return out


def vecdot_3d_axis_order(x: "float[:,:,:]", y: "float[:,:,:]"):
    return np.vecdot(x, y, axis=2)


def vecdot_mixed_dimensions_expression(x: "float[:,:]", y: "float[:]"):
    return np.vecdot(x, y) + 3.5


def vecdot_3d_axis_order_expression(x: "float[:,:,:]", y: "float[:,:,:]"):
    return np.vecdot(x, y, axis=2) - 7.2
