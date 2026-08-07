# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import TypeVar

import numpy as np

T = TypeVar("T", int, float)
T2 = TypeVar("T2", int, float, complex)


def abs_i(x: "int"):
    return abs(x)


def abs_r(x: "float"):
    return abs(x)


def abs_c(x: "complex"):
    return abs(x)


def min_2_args_i(x: "int", y: "int"):
    return min(x, y)


def min_2_args_i_adhoc(x: int):
    return min(x, 0)


def min_2_args_f_adhoc(x: float):
    return min(x, 0.0)


def min_2_args_f(x: "float", y: "float"):
    return min(x, y)


def min_3_args(x: T, y: T, z: T):
    return min(x, y, z)


def min_if(x: "int", y: "int"):
    if min(x + x + y, x + y + y) < (x + y):
        return x + y
    else:
        return x - y


def min_in_min(x: "int", y: "int"):
    if min(min(x + x, +y), min(x + y, y)) < (x + y):  # pylint: disable=nested-min-max
        return x + y
    else:
        return x - y


def min_list(x: T, y: T, z: T):
    return min([x, y, z])


def min_tuple(x: T, y: T, z: T):
    return min((x, y, z))


def min_list_var(x: T, y: T, z: T):
    w = [x, y, z]
    return min(w)


def min_tuple_var(x: T, y: T, z: T):
    w = (x, y, z)
    return min(w)


def min_expr(x: T, y: T):
    return min((x, y)) + 3, min(x, y) + 4


def min_temp_var_first_arg(x: "int", y: "int"):
    return min(x + 1, y)


def min_temp_var_second_arg(x: "int", y: "int"):
    return min(x, y + 2)


def max_2_args_i(x: "int", y: "int"):
    return max(x, y)


def max_2_args_f(x: "float", y: "float"):
    return max(x, y)


def max_3_args(x: T, y: T, z: T):
    return max(x, y, z)


def max_list(x: T, y: T, z: T):
    return max([x, y, z])


def max_tuple(x: T, y: T, z: T):
    return max((x, y, z))


def max_list_var(x: T, y: T, z: T):
    w = [x, y, z]
    return max(w)


def max_tuple_var(x: T, y: T, z: T):
    w = (x, y, z)
    return max(w)


def max_expr(x: T, y: T):
    return max((x, y)) + 3, max(x, y) + 4


def max_temp_var_first_arg(x: "int", y: "int"):
    return max(x + 1, y)


def max_temp_var_second_arg(x: "int", y: "int"):
    return max(x, y + 2)


def len_numpy():
    from numpy import ones

    a = ones((3, 4))
    b = ones((4, 3, 5))
    c = ones(4)
    return len(a), len(b), len(c)


def len_tuple():
    a = (3, 4)
    b = (4, 3, 5)
    c = b
    return len(a), len(b), len(c), len((1, 2))


def len_inhomog_tuple():
    a = (3, True)
    b = (4j, False, 5)
    c = b
    return len(a), len(b), len(c), len((1.5, 2))


def len_list_int():
    a = [1, 2, 3]
    return len(a)


def len_list_float():
    a = [1.4, 2.6, 3.5]
    b = len(a)
    return b


def len_list_complex():
    a = [1j, 2 + 1j, 3 + 1j]
    b = len(a)
    return b


def len_string():
    a = "abcdefghij"
    b = len(a)
    return b


def len_literal_string():
    b = len("abcd")
    return b


def round_int(x: float):
    return round(x)


def round_ndigits(x: float, i: int):
    return round(x, i)


def round_ndigits_int(x: int, i: int):
    return round(x, i)


def round_ndigits_bool():
    return round(True), round(False), round(True, 1), round(True, -1)


def isinstance_test(a: "bool | int | float | complex"):
    return (
        isinstance(a, bool),
        isinstance(a, int),
        isinstance(a, float),
        isinstance(a, complex),
    )


def isinstance_numpy(a: "int32 | int64 | int | float32"):
    return (
        isinstance(a, np.int32),
        isinstance(a, np.int64),
        isinstance(a, int),
        isinstance(a, np.float32),
    )


def isinstance_tuple(a: "bool | int | float | complex"):
    """
    Testing a case which should generate radically different functions.
    """
    return (
        isinstance(a, (bool, int)),
        isinstance(a, (bool, float)),
        isinstance(a, (int, complex)),
        isinstance(a, (tuple, list)),
    )
