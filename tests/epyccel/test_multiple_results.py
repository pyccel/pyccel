# pylint: disable=missing-function-docstring, missing-module-docstring
from epyccel_utilities import compare_epyccel

from pyccel.decorators import pure


# ==============================================================================
def test_const_int_float(language):

    @pure
    def const_int_float():
        return 1, 3.4

    compare_epyccel(const_int_float, language)


# ...
def test_const_complex_bool_int(language):

    @pure
    def const_complex_bool_int():
        return 1 + 2j, False, 8

    compare_epyccel(const_complex_bool_int, language)


# ...
def test_expr_float_int_bool(language):

    @pure
    def expr_complex_int_bool(n: "int"):
        return 0.5 + n * 1j, 2 * n, n == 3

    compare_epyccel(expr_complex_int_bool, language, 3)
