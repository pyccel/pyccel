# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import TypeVar

NT1 = TypeVar("NT1", "int32", "int64", "float32", "float64", "complex64", "complex128")
NT2 = TypeVar("NT2", "int32", "int64", "float32", "float64", "complex64", "complex128")
T = TypeVar(
    "T", "bool", "int", "int8", "int16", "int32", "int64", "float", "float32", "float64"
)


def mult_on_array_int8():
    from numpy import int8, ones

    a = ones(5, dtype=int8)
    b = a * 2
    return b[0]


def mult_on_array_int16():
    from numpy import int16, ones

    a = ones(5, dtype=int16)
    b = a * 2
    return b[0]


def mult_on_array_int32():
    from numpy import int32, ones

    a = ones(5, dtype=int32)
    b = a * 2
    return b[0]


def mult_on_array_int64():
    from numpy import int64, ones

    a = ones(5, dtype=int64)
    b = a * 2
    return b[0]


def mult_on_array_float32():
    from numpy import float32, ones

    a = ones(5, dtype=float32)
    b = a * 2
    return b[0]


def mult_on_array_float64():
    from numpy import float64, ones

    a = ones(5, dtype=float64)
    b = a * 2
    return b[0]


def add_numpy_to_numpy_type(np_s_l: NT1, np_s_r: NT2):
    rs = np_s_l + np_s_r
    return rs


def get_double(a: T):
    from numpy import double

    b = double(a)
    return b


def numpy_double_array_like_1d(arr: "T[:]"):
    from numpy import double, shape

    a = double(arr)
    s = shape(a)
    return len(s), s[0], a[0]


def numpy_double_array_like_2d(arr: "T[:,:]"):
    from numpy import double, shape

    a = double(arr)
    s = shape(a)
    return len(s), s[0], s[1], a[0, 0], a[0, 1]


def get_complex64():
    from numpy import complex64

    compl = complex64(3 + 4j)
    return compl, compl.real, compl.imag


def get_complex128():
    from numpy import complex128

    compl = complex128(3 + 4j)
    return compl, compl.real, compl.imag
