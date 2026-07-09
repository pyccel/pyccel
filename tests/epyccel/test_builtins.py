# pylint: disable=missing-function-docstring, missing-module-docstring
import sys

import numpy as np
import pytest
from numpy import finfo, iinfo
from numpy.random import randint, uniform

from pyccel import epyccel
from modules import builtins
from utilities import epyccel_module_with_fallback


@pytest.fixture(scope="module")
def epyc_builtins_mod(language):
    return epyccel_module_with_fallback(builtins, language)



ATOL = 1e-15
RTOL = 2e-14

# Use int32 for Windows compatibility
min_int = iinfo(np.int32).min
max_int = iinfo(np.int32).max

min_float = finfo(float).min
max_float = finfo(float).max


def test_abs_i(epyc_builtins_mod):
    f1 = builtins.abs_i
    f2 = epyc_builtins_mod.abs_i

    negative_test = randint(min_int, 0)
    positive_test = randint(0, max_int)

    assert np.isclose(f1(0), f2(0), rtol=RTOL, atol=ATOL)
    assert np.isclose(f1(negative_test), f2(negative_test), rtol=RTOL, atol=ATOL)
    assert np.isclose(f1(positive_test), f2(positive_test), rtol=RTOL, atol=ATOL)


def test_abs_r(epyc_builtins_mod):
    f1 = builtins.abs_r
    f2 = epyc_builtins_mod.abs_r

    negative_test = uniform(min_float, 0.0)
    positive_test = uniform(0.0, max_float)

    assert np.isclose(f1(0.00000), f2(0.00000), rtol=RTOL, atol=ATOL)
    assert np.isclose(f1(negative_test), f2(negative_test), rtol=RTOL, atol=ATOL)
    assert np.isclose(f1(positive_test), f2(positive_test), rtol=RTOL, atol=ATOL)


def test_abs_c(epyc_builtins_mod):
    f1 = builtins.abs_c
    f2 = epyc_builtins_mod.abs_c

    max_compl_abs = np.sqrt(max_float / 2)
    min_compl_abs = np.sqrt(-min_float / 2)

    pos_pos = uniform(0.0, max_compl_abs) + 1j * uniform(0.0, max_compl_abs)
    pos_neg = uniform(0.0, max_compl_abs) + 1j * uniform(min_compl_abs, 0.0)
    neg_pos = uniform(min_compl_abs, 0.0) + 1j * uniform(0.0, max_compl_abs)
    neg_neg = uniform(min_compl_abs, 0.0) + 1j * uniform(min_compl_abs, 0.0)
    zero_rand = 1j * uniform(min_compl_abs, max_compl_abs)
    rand_zero = uniform(min_compl_abs, max_compl_abs) + 0j

    assert np.isclose(f1(pos_pos), f2(pos_pos), rtol=RTOL, atol=ATOL)
    assert np.isclose(f1(pos_neg), f2(pos_neg), rtol=RTOL, atol=ATOL)
    assert np.isclose(f1(neg_pos), f2(neg_pos), rtol=RTOL, atol=ATOL)
    assert np.isclose(f1(neg_neg), f2(neg_neg), rtol=RTOL, atol=ATOL)
    assert np.isclose(f1(zero_rand), f2(zero_rand), rtol=RTOL, atol=ATOL)
    assert np.isclose(f1(rand_zero), f2(rand_zero), rtol=RTOL, atol=ATOL)
    assert np.isclose(f1(0j + 0), f2(0j + 0), rtol=RTOL, atol=ATOL)


def test_min_2_args_i(epyc_builtins_mod):
    f = builtins.min_2_args_i
    epyc_f = epyc_builtins_mod.min_2_args_i

    int_args = [randint(min_int, max_int) for _ in range(2)]

    assert epyc_f(*int_args) == f(*int_args)


def test_min_2_args_i_adhoc(epyc_builtins_mod):
    f = builtins.min_2_args_i_adhoc
    epyc_f = epyc_builtins_mod.min_2_args_i_adhoc

    int_arg = randint(min_int, max_int)

    assert epyc_f(int_arg) == f(int_arg)


def test_min_2_args_f_adhoc(epyc_builtins_mod):
    f = builtins.min_2_args_f_adhoc
    epyc_f = epyc_builtins_mod.min_2_args_f_adhoc

    float_arg = uniform(min_float / 2, max_float / 2)

    assert np.isclose(epyc_f(float_arg), f(float_arg), rtol=RTOL, atol=ATOL)


def test_min_2_args_f(epyc_builtins_mod):
    f = builtins.min_2_args_f
    epyc_f = epyc_builtins_mod.min_2_args_f

    float_args = [uniform(min_float / 2, max_float / 2) for _ in range(2)]

    assert np.isclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)


def test_min_3_args(epyc_builtins_mod):
    f = builtins.min_3_args
    epyc_f = epyc_builtins_mod.min_3_args

    int_args = [randint(min_int, max_int) for _ in range(3)]
    float_args = [uniform(min_float / 2, max_float / 2) for _ in range(3)]

    assert epyc_f(*int_args) == f(*int_args)
    assert np.isclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)


def test_min_if(epyc_builtins_mod):
    f = builtins.min_if
    epyc_f = epyc_builtins_mod.min_if

    int_args = [randint(min_int // 3, max_int // 3) for _ in range(2)]

    assert epyc_f(*int_args) == f(*int_args)


def test_min_in_min(epyc_builtins_mod):
    f = builtins.min_in_min
    epyc_f = epyc_builtins_mod.min_in_min

    int_args = [randint(min_int // 3, max_int // 3) for _ in range(2)]

    assert epyc_f(*int_args) == f(*int_args)


def test_min_list(epyc_builtins_mod):
    f = builtins.min_list
    epyc_f = epyc_builtins_mod.min_list

    int_args = [randint(min_int, max_int) for _ in range(3)]
    float_args = [uniform(min_float / 2, max_float / 2) for _ in range(3)]

    assert epyc_f(*int_args) == f(*int_args)
    assert np.isclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)


def test_min_tuple(epyc_builtins_mod):
    f = builtins.min_tuple
    epyc_f = epyc_builtins_mod.min_tuple

    int_args = [randint(min_int, max_int) for _ in range(3)]
    float_args = [uniform(min_float / 2, max_float / 2) for _ in range(3)]

    assert epyc_f(*int_args) == f(*int_args)
    assert np.isclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)


def test_min_list_var(epyc_builtins_mod):
    f = builtins.min_list_var
    epyc_f = epyc_builtins_mod.min_list_var

    int_args = [randint(min_int, max_int) for _ in range(3)]
    float_args = [uniform(min_float / 2, max_float / 2) for _ in range(3)]

    assert epyc_f(*int_args) == f(*int_args)
    assert np.isclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)


def test_min_tuple_var(epyc_builtins_mod):
    f = builtins.min_tuple_var
    epyc_f = epyc_builtins_mod.min_tuple_var

    int_args = [randint(min_int, max_int) for _ in range(3)]
    float_args = [uniform(min_float / 2, max_float / 2) for _ in range(3)]

    assert epyc_f(*int_args) == f(*int_args)
    assert np.isclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)


def test_min_expr(epyc_builtins_mod):
    f = builtins.min_expr
    epyc_f = epyc_builtins_mod.min_expr

    int_args = [randint(min_int, max_int) for _ in range(2)]
    float_args = [uniform(min_float / 2, max_float / 2) for _ in range(2)]

    assert np.array_equal(epyc_f(*int_args), f(*int_args))
    assert np.allclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)


def test_min_temp_var_first_arg(epyc_builtins_mod):
    f = builtins.min_temp_var_first_arg
    epyc_f = epyc_builtins_mod.min_temp_var_first_arg

    x, y = randint(min_int, max_int), randint(min_int, max_int)

    assert epyc_f(x, y) == f(x, y)


def test_min_temp_var_second_arg(epyc_builtins_mod):
    f = builtins.min_temp_var_second_arg
    epyc_f = epyc_builtins_mod.min_temp_var_second_arg

    x, y = randint(min_int, max_int), randint(min_int, max_int)

    assert epyc_f(x, y) == f(x, y)


def test_max_2_args_i(epyc_builtins_mod):
    f = builtins.max_2_args_i
    epyc_f = epyc_builtins_mod.max_2_args_i

    int_args = [randint(min_int, max_int) for _ in range(2)]

    assert epyc_f(*int_args) == f(*int_args)


def test_max_2_args_f(epyc_builtins_mod):
    f = builtins.max_2_args_f
    epyc_f = epyc_builtins_mod.max_2_args_f

    float_args = [uniform(min_float / 2, max_float / 2) for _ in range(2)]

    assert np.isclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)


def test_max_3_args(epyc_builtins_mod):
    f = builtins.max_3_args
    epyc_f = epyc_builtins_mod.max_3_args

    int_args = [randint(min_int, max_int) for _ in range(3)]
    float_args = [uniform(min_float / 2, max_float / 2) for _ in range(3)]

    assert epyc_f(*int_args) == f(*int_args)
    assert np.isclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)


def test_max_list(epyc_builtins_mod):
    f = builtins.max_list
    epyc_f = epyc_builtins_mod.max_list

    int_args = [randint(min_int, max_int) for _ in range(3)]
    float_args = [uniform(min_float / 2, max_float / 2) for _ in range(3)]

    assert epyc_f(*int_args) == f(*int_args)
    assert np.isclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)


def test_max_tuple(epyc_builtins_mod):
    f = builtins.max_tuple
    epyc_f = epyc_builtins_mod.max_tuple

    int_args = [randint(min_int, max_int) for _ in range(3)]
    float_args = [uniform(min_float / 2, max_float / 2) for _ in range(3)]

    assert epyc_f(*int_args) == f(*int_args)
    assert np.isclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)


def test_max_list_var(epyc_builtins_mod):
    f = builtins.max_list_var
    epyc_f = epyc_builtins_mod.max_list_var

    int_args = [randint(min_int, max_int) for _ in range(3)]
    float_args = [uniform(min_float / 2, max_float / 2) for _ in range(3)]

    assert epyc_f(*int_args) == f(*int_args)
    assert np.isclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)


def test_max_tuple_var(epyc_builtins_mod):
    f = builtins.max_tuple_var
    epyc_f = epyc_builtins_mod.max_tuple_var

    int_args = [randint(min_int, max_int) for _ in range(3)]
    float_args = [uniform(min_float / 2, max_float / 2) for _ in range(3)]

    assert epyc_f(*int_args) == f(*int_args)
    assert np.isclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)


def test_max_expr(epyc_builtins_mod):
    f = builtins.max_expr
    epyc_f = epyc_builtins_mod.max_expr

    int_args = [randint(min_int, max_int) for _ in range(2)]
    float_args = [uniform(min_float / 2, max_float / 2) for _ in range(2)]

    assert np.array_equal(epyc_f(*int_args), f(*int_args))
    assert np.allclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)


def test_max_temp_var_first_arg(epyc_builtins_mod):
    f = builtins.max_temp_var_first_arg
    epyc_f = epyc_builtins_mod.max_temp_var_first_arg

    x, y = randint(min_int, max_int), randint(min_int, max_int)

    assert epyc_f(x, y) == f(x, y)


def test_max_temp_var_second_arg(epyc_builtins_mod):
    f = builtins.max_temp_var_second_arg
    epyc_f = epyc_builtins_mod.max_temp_var_second_arg

    x, y = randint(min_int, max_int), randint(min_int, max_int)

    assert epyc_f(x, y) == f(x, y)


@pytest.mark.parametrize(
    "language",
    (
        pytest.param("fortran", marks=pytest.mark.fortran),
        pytest.param(
            "c",
            marks=[pytest.mark.skip(reason="sum not implemented in C"), pytest.mark.c],
        ),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_sum_matching_types(language):
    def f(x: builtins.T2, y: builtins.T2):
        return sum([x, y])

    epyc_f = epyccel(f, language=language)

    int_args = [randint(min_int // 2, max_int // 2) for _ in range(2)]
    float_args = [uniform(min_float / 2, max_float / 2) for _ in range(2)]
    complex_args = [
        uniform(min_float / 4, max_float / 4)
        + 1j * uniform(min_float / 4, max_float / 4)
        for _ in range(2)
    ]

    assert epyc_f(*int_args) == f(*int_args)
    assert np.isclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)
    assert np.isclose(epyc_f(*complex_args), f(*complex_args), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "language",
    (
        pytest.param("fortran", marks=pytest.mark.fortran),
        pytest.param(
            "c",
            marks=[pytest.mark.skip(reason="sum not implemented in C"), pytest.mark.c],
        ),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_sum_expr(language):
    def f(x: builtins.T, y: builtins.T):
        return sum((x, y)) + 3 

    epyc_f = epyccel(f, language=language)

    int_args = [randint(min_int // 3, max_int // 3) for _ in range(2)]
    float_args = [uniform(min_float / 2, max_float / 2) for _ in range(2)]

    assert epyc_f(*int_args) == f(*int_args)
    assert np.allclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)


def test_len_numpy(epyc_builtins_mod):
    f = builtins.len_numpy
    epyc_f = epyc_builtins_mod.len_numpy

    assert epyc_f() == f()


def test_len_tuple(epyc_builtins_mod):
    f = builtins.len_tuple
    epyc_f = epyc_builtins_mod.len_tuple

    assert epyc_f() == f()


def test_len_inhomog_tuple(epyc_builtins_mod):
    f = builtins.len_inhomog_tuple
    epyc_f = epyc_builtins_mod.len_inhomog_tuple

    assert epyc_f() == f()


def test_len_list_int(epyc_builtins_mod):
    f = builtins.len_list_int
    epyc_f = epyc_builtins_mod.len_list_int

    assert epyc_f() == f()


def test_len_list_float(epyc_builtins_mod):
    f = builtins.len_list_float
    epyc_f = epyc_builtins_mod.len_list_float

    assert epyc_f() == f()


def test_len_list_complex(epyc_builtins_mod):
    f = builtins.len_list_complex
    epyc_f = epyc_builtins_mod.len_list_complex

    assert epyc_f() == f()


def test_len_set_int(stc_language):
    def f():
        a = {1, 2, 3}
        return len(a)

    epyc_f = epyccel(f, language=stc_language)

    assert epyc_f() == f()


def test_len_set_float(stc_language):
    def f():
        a = {1.4, 2.6, 3.5}
        b = len(a)
        return b

    epyc_f = epyccel(f, language=stc_language)

    assert epyc_f() == f()


def test_len_set_complex(stc_language):
    def f():
        a = {1j, 2 + 1j, 3 + 1j}
        b = len(a)
        return b

    epyc_f = epyccel(f, language=stc_language)

    assert epyc_f() == f()


def test_len_dict_int_float(stc_language):
    def f():
        a = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
        b = len(a)
        return b

    epyc_f = epyccel(f, language=stc_language)

    assert epyc_f() == f()


def test_len_string(epyc_builtins_mod):
    f = builtins.len_string
    epyc_f = epyc_builtins_mod.len_string

    assert epyc_f() == f()


def test_len_literal_string(epyc_builtins_mod):
    f = builtins.len_literal_string
    epyc_f = epyc_builtins_mod.len_literal_string

    assert epyc_f() == f()


def test_len_multi_layer(stc_language):
    def f():
        x = [1, 2, 3]
        y = [x]
        return len(y), len(y[0])

    epyc_f = epyccel(f, language=stc_language)

    assert epyc_f() == f()


def test_round_int(epyc_builtins_mod):
    round_int = builtins.round_int
    f = epyc_builtins_mod.round_int
    x = randint(100) / 10

    f_output = f(x)
    round_int_output = round_int(x)
    assert round_int_output == f_output
    assert isinstance(f_output, type(round_int_output))

    # Round down
    x = 3.345

    f_output = f(x)
    round_int_output = round_int(x)
    assert round_int_output == f_output
    assert isinstance(f_output, type(round_int_output))

    # Round up
    x = 3.845

    f_output = f(x)
    round_int_output = round_int(x)
    assert round_int_output == f_output
    assert isinstance(f_output, type(round_int_output))

    # Round half
    x = 6.5

    f_output = f(x)
    round_int_output = round_int(x)
    assert round_int_output == f_output
    assert isinstance(f_output, type(round_int_output))


def test_negative_round_int(epyc_builtins_mod):
    round_int = builtins.negative_round_int
    f = epyc_builtins_mod.negative_round_int
    x = -randint(100) / 10

    f_output = f(x)
    round_int_output = round_int(x)
    assert round_int_output == f_output
    assert isinstance(f_output, type(round_int_output))

    # Round up
    x = -3.345

    f_output = f(x)
    round_int_output = round_int(x)
    assert round_int_output == f_output
    assert isinstance(f_output, type(round_int_output))

    # Round down
    x = -3.845

    f_output = f(x)
    round_int_output = round_int(x)
    assert round_int_output == f_output
    assert isinstance(f_output, type(round_int_output))

    # Round half
    x = -6.5

    f_output = f(x)
    round_int_output = round_int(x)
    assert round_int_output == f_output
    assert isinstance(f_output, type(round_int_output))


def test_round_ndigits(epyc_builtins_mod):
    round_ndigits = builtins.round_ndigits
    f = epyc_builtins_mod.round_ndigits
    x = randint(100) / 10

    f_output = f(x, 1)
    round_ndigits_output = round_ndigits(x, 1)
    assert np.isclose(round_ndigits_output, f_output)
    assert isinstance(f_output, type(round_ndigits_output))

    x = 3.343

    f_output = f(x, 2)
    round_ndigits_output = round_ndigits(x, 2)
    assert np.isclose(round_ndigits_output, f_output)
    assert isinstance(f_output, type(round_ndigits_output))

    x = 3323.0

    f_output = f(x, -2)
    round_ndigits_output = round_ndigits(x, -2)
    assert np.isclose(round_ndigits_output, f_output)
    assert isinstance(f_output, type(round_ndigits_output))

    x = -3390.0

    f_output = f(x, -2)
    round_ndigits_output = round_ndigits(x, -2)
    assert np.isclose(round_ndigits_output, f_output)
    assert isinstance(f_output, type(round_ndigits_output))


def test_round_ndigits_half(epyc_builtins_mod):
    round_ndigits = builtins.round_ndigits_half
    f = epyc_builtins_mod.round_ndigits_half
    x = randint(100) / 10

    f_output = f(x, 1)
    round_ndigits_output = round_ndigits(x, 1)
    assert np.isclose(round_ndigits_output, f_output)
    assert isinstance(f_output, type(round_ndigits_output))

    x = 3.345

    f_output = f(x, 2)
    round_ndigits_output = round_ndigits(x, 2)
    assert np.isclose(round_ndigits_output, f_output)
    assert isinstance(f_output, type(round_ndigits_output))

    x = -3350.0

    f_output = f(x, -2)
    round_ndigits_output = round_ndigits(x, -2)
    assert np.isclose(round_ndigits_output, f_output)
    assert isinstance(f_output, type(round_ndigits_output))

    x = 45.0

    f_output = f(x, -1)
    round_ndigits_output = round_ndigits(x, -1)
    assert np.isclose(round_ndigits_output, f_output)
    assert isinstance(f_output, type(round_ndigits_output))


def test_round_ndigits_int(epyc_builtins_mod):
    round_ndigits = builtins.round_ndigits_int
    f = epyc_builtins_mod.round_ndigits_int
    x = randint(100) // 10

    f_output = f(x, 1)
    round_ndigits_output = round_ndigits(x, 1)
    assert np.isclose(round_ndigits_output, f_output)
    assert isinstance(f_output, type(round_ndigits_output))

    x = 3

    f_output = f(x, 2)
    round_ndigits_output = round_ndigits(x, 2)
    assert np.isclose(round_ndigits_output, f_output)
    assert isinstance(f_output, type(round_ndigits_output))

    x = 3323

    f_output = f(x, -2)
    round_ndigits_output = round_ndigits(x, -2)
    assert np.isclose(round_ndigits_output, f_output)
    assert isinstance(f_output, type(round_ndigits_output))

    x = -3390

    f_output = f(x, -2)
    round_ndigits_output = round_ndigits(x, -2)
    assert np.isclose(round_ndigits_output, f_output)
    assert isinstance(f_output, type(round_ndigits_output))


def test_round_ndigits_bool(epyc_builtins_mod):
    round_ndigits = builtins.round_ndigits_bool
    f = epyc_builtins_mod.round_ndigits_bool

    f_output = f()
    round_ndigits_output = round_ndigits()
    assert all(o == r for o, r in zip(f_output, round_ndigits_output))
    assert all(isinstance(o, type(r)) for o, r in zip(f_output, round_ndigits_output))


def test_isinstance_native(epyc_builtins_mod):
    isinstance_test = builtins.isinstance_test
    f = epyc_builtins_mod.isinstance_test
    assert f(True) == isinstance_test(True)
    assert f(False) == isinstance_test(False)
    assert f(4) == isinstance_test(6)
    assert f(3.9) == isinstance_test(6.7)
    assert f(1 + 2j) == isinstance_test(6.5 + 8.3j)


def test_isinstance_containers(language):
    def isinstance_tup(a: int, b: int):
        container = (a, b)
        return (
            isinstance(container, tuple),
            isinstance(container, list),
            isinstance(container, set),
            isinstance(container, dict),
        )

    def isinstance_lst(a: int, b: int):
        container = [a, b]
        return (
            isinstance(container, tuple),
            isinstance(container, list),
            isinstance(container, set),
            isinstance(container, dict),
        )

    def isinstance_set(a: int, b: int):
        container = {a, b}
        return (
            isinstance(container, tuple),
            isinstance(container, list),
            isinstance(container, set),
            isinstance(container, dict),
        )

    def isinstance_dict(a: int, b: int):
        container = {a: False, b: True}
        return (
            isinstance(container, tuple),
            isinstance(container, list),
            isinstance(container, set),
            isinstance(container, dict),
        )

    test_funcs = (isinstance_tup, isinstance_lst, isinstance_set, isinstance_dict)

    for f in test_funcs:
        f_epyc = epyccel(f, language=language)

        assert f(2, 5) == f_epyc(2, 5)


def test_isinstance_numpy(epyc_builtins_mod):
    isinstance_test = builtins.isinstance_numpy
    f = epyc_builtins_mod.isinstance_numpy
    assert f(np.int32(4)) == isinstance_test(np.int32(4))
    assert f(np.int64(4)) == isinstance_test(np.int64(4))
    assert f(4) == isinstance_test(4)
    assert f(np.float32(4)) == isinstance_test(np.float32(4))


def test_isinstance_tuple(epyc_builtins_mod):
    isinstance_test = builtins.isinstance_tuple
    f = epyc_builtins_mod.isinstance_tuple
    assert f(True) == isinstance_test(True)
    assert f(False) == isinstance_test(False)
    assert f(4) == isinstance_test(6)
    assert f(3.9) == isinstance_test(6.7)
    assert f(1 + 2j) == isinstance_test(6.5 + 8.3j)


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="Union of types implemented in Python 3.10"
)
def test_isinstance_union(language):
    def isinstance_test(
        a: bool | int | float | complex,
    ):  # pylint: disable=unsupported-binary-operation
        return (
            isinstance(a, bool | int),
            isinstance(a, bool | float),
            isinstance(
                a, int | complex
            ),  # pylint: disable=unsupported-binary-operation
            isinstance(a, tuple | list),
        )  # pylint: disable=unsupported-binary-operation

    f = epyccel(isinstance_test, language=language)
    assert f(True) == isinstance_test(True)
    assert f(False) == isinstance_test(False)
    assert f(4) == isinstance_test(6)
    assert f(3.9) == isinstance_test(6.7)
    assert f(1 + 2j) == isinstance_test(6.5 + 8.3j)
