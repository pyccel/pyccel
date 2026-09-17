# pylint: disable=missing-function-docstring, missing-module-docstring
from pyccel.decorators import pure


@pure
def const_int_float():
    return 1, 3.4


@pure
def const_complex_bool_int():
    return 1 + 2j, False, 8


@pure
def expr_complex_int_bool(n: "int"):
    return 0.5 + n * 1j, 2 * n, n == 3
