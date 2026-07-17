# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np


def single_return_var_assign():
    y = 3
    return y


def divide_by(a: "float[:]", b: "float"):
    if abs(b) < 0.1:
        return
    for i, ai in enumerate(a):
        a[i] = ai / b


def return_None(  # pylint: disable=inconsistent-return-statements
    a: "float[:]", b: "float"
):
    if abs(b) < 0.1:
        return None
    for i, ai in enumerate(a):
        a[i] = ai / b


def assign_vars_return(a: "int", b: "int"):
    c = a + b
    d = a - b
    return c + d


def sum_in_single_return(a: "int", b: "int"):
    c = a + b
    return c


def return_expr(x: "int", y: "int"):
    return x + y


def return_single_var(x: "int"):
    return x


def return_scalare():
    return 5


def multi_return_scalare():
    return 5, 7


def multi_return_vars(a: "int", b: "int"):
    return a, b


def multi_return_vars_expr(a: "int", b: "int"):
    return (a - b), (a + b)


def scalare_multi_return_stmts(a: "int"):
    a = 7
    if a:
        return 1
    else:
        return 2
    a = 4
    return a


def create_arr(i: int):
    _ = np.ones(i)
    return True


def return_arr_element(i: int):
    a = np.ones(i)
    return a[0]


def create_multi_arrs(i: int):
    _ = np.ones(i)
    _ = np.zeros(i)
    _ = np.zeros(i)
    return True


def expr_arrs_elements(i: int):
    a = np.ones(i)
    b = np.zeros(i)
    return a[i - 1] + b[i - 1]


def complex_expr(i: int):
    a = np.ones(i)
    return ((4 + 5) / (6 - 3) * a[0]) % (9 - a[1])


def multi_allocs(i: int):
    a = np.ones(i)
    b = np.ones(i)
    c = np.ones(i)
    d = np.ones(i)
    e = np.ones(i)
    return ((4 + 5) / (d[0] + e[2]) * c[0]) % (b[2] + a[1]) - 4


def return_mult_arr_arg_element(i: "int", arg: "float[:]"):
    a = np.ones(i)
    return a[0] * arg[0]


def return_add_arr_arg_element(i: "int", arg: "float[:]"):
    a = np.ones(i)
    return a[0] + arg[0]


def return_op_arr_arg_element(i: "int", arg: "float[:]"):
    a = np.ones(i)
    return ((a[2] + arg[0]) * arg[2] - 2) / 4 * 2
