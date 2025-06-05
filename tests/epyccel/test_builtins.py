# pylint: disable=missing-function-docstring, missing-module-docstring
import sys
from typing import TypeVar
import pytest
from numpy.random import randint, uniform
from numpy import iinfo, finfo
import numpy as np

from pyccel import epyccel

ATOL = 1e-15
RTOL = 2e-14

# Use int32 for Windows compatibility
min_int = iinfo(np.int32).min
max_int = iinfo(np.int32).max

min_float = finfo(float).min
max_float = finfo(float).max

T = TypeVar('T', int, float)
T2 = TypeVar('T2', int, float, complex)

def test_abs_i(language):
    def f1(x : 'int'):
        return abs(x)

    f2 = epyccel(f1, language=language)

    negative_test = randint(min_int, 0)
    positive_test = randint(0, max_int)

    assert np.isclose(f1(0), f2(0), rtol=RTOL, atol=ATOL)
    assert np.isclose(f1(negative_test), f2(negative_test), rtol=RTOL, atol=ATOL)
    assert np.isclose(f1(positive_test), f2(positive_test), rtol=RTOL, atol=ATOL)

def test_abs_r(language):
    def f1(x : 'float'):
        return abs(x)

    f2 = epyccel(f1, language=language)

    negative_test = uniform(min_float, 0.0)
    positive_test = uniform(0.0, max_float)

    assert np.isclose(f1(0.00000), f2(0.00000), rtol=RTOL, atol=ATOL)
    assert np.isclose(f1(negative_test), f2(negative_test), rtol=RTOL, atol=ATOL)
    assert np.isclose(f1(positive_test), f2(positive_test), rtol=RTOL, atol=ATOL)



def test_abs_c(language):
    def f1(x : 'complex'):
        return abs(x)

    f2 = epyccel(f1, language=language)

    max_compl_abs = np.sqrt(max_float / 2)
    min_compl_abs = np.sqrt(-min_float / 2)

    pos_pos = uniform(0.0, max_compl_abs) + 1j*uniform(0.0, max_compl_abs)
    pos_neg = uniform(0.0, max_compl_abs) + 1j*uniform(min_compl_abs, 0.0)
    neg_pos = uniform(min_compl_abs, 0.0) + 1j*uniform(0.0, max_compl_abs)
    neg_neg = uniform(min_compl_abs, 0.0) + 1j*uniform(min_compl_abs, 0.0)
    zero_rand = 1j*uniform(min_compl_abs, max_compl_abs)
    rand_zero = uniform(min_compl_abs, max_compl_abs) + 0j

    assert np.isclose(f1(pos_pos), f2(pos_pos), rtol=RTOL, atol=ATOL)
    assert np.isclose(f1(pos_neg), f2(pos_neg), rtol=RTOL, atol=ATOL)
    assert np.isclose(f1(neg_pos), f2(neg_pos), rtol=RTOL, atol=ATOL)
    assert np.isclose(f1(neg_neg), f2(neg_neg), rtol=RTOL, atol=ATOL)
    assert np.isclose(f1(zero_rand), f2(zero_rand), rtol=RTOL, atol=ATOL)
    assert np.isclose(f1(rand_zero), f2(rand_zero), rtol=RTOL, atol=ATOL)
    assert np.isclose(f1(0j + 0), f2(0j + 0), rtol=RTOL, atol=ATOL)

def test_min_2_args_i(language):
    def f(x : 'int', y : 'int'):
        return min(x, y)

    epyc_f = epyccel(f, language=language)

    int_args = [randint(min_int, max_int) for _ in range(2)]

    assert epyc_f(*int_args) == f(*int_args)

def test_min_2_args_i_adhoc(language):
    def f(x:int):
        return min(x, 0)

    epyc_f = epyccel(f, language=language)

    int_arg = randint(min_int, max_int)

    assert epyc_f(int_arg) == f(int_arg)

def test_min_2_args_f_adhoc(language):
    def f(x:float):
        return min(x, 0.0)

    epyc_f = epyccel(f, language=language)

    float_arg = uniform(min_float /2, max_float/2)

    assert np.isclose(epyc_f(float_arg), f(float_arg), rtol=RTOL, atol=ATOL)

def test_min_2_args_f(language):
    def f(x : 'float', y : 'float'):
        return min(x, y)

    epyc_f = epyccel(f, language=language)

    float_args = [uniform(min_float/2, max_float/2) for _ in range(2)]

    assert np.isclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)

def test_min_3_args(language):
    def f(x : T, y : T, z : T):
        return min(x, y, z)

    epyc_f = epyccel(f, language=language)

    int_args = [randint(min_int, max_int) for _ in range(3)]
    float_args = [uniform(min_float/2, max_float/2) for _ in range(3)]

    assert epyc_f(*int_args) == f(*int_args)
    assert np.isclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)

def test_min_if(language):
    def f(x : 'int', y : 'int'):
        if min(x+x+y, x+y+y) < (x+y):
            return x+y
        else:
            return x-y

    epyc_f = epyccel(f, language=language)

    int_args = [randint(min_int//3, max_int//3) for _ in range(2)]

    assert epyc_f(*int_args) == f(*int_args)

def test_min_in_min(language):
    def f(x : 'int', y : 'int'):
        if min(min(x+x,+y), min(x+y,y)) < (x+y): #pylint: disable=nested-min-max
            return x+y
        else:
            return x-y

    epyc_f = epyccel(f, language=language)

    int_args = [randint(min_int//3, max_int//3) for _ in range(2)]

    assert epyc_f(*int_args) == f(*int_args)

def test_min_list(language):
    def f(x : T, y : T, z : T):
        return min([x, y, z])

    epyc_f = epyccel(f, language=language)

    int_args = [randint(min_int, max_int) for _ in range(3)]
    float_args = [uniform(min_float/2, max_float/2) for _ in range(3)]

    assert epyc_f(*int_args) == f(*int_args)
    assert np.isclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)

def test_min_tuple(language):
    def f(x : T, y : T, z : T):
        return min((x, y, z))

    epyc_f = epyccel(f, language=language)

    int_args = [randint(min_int, max_int) for _ in range(3)]
    float_args = [uniform(min_float/2, max_float/2) for _ in range(3)]

    assert epyc_f(*int_args) == f(*int_args)
    assert np.isclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)

def test_min_list_var(language):
    def f(x : T, y : T, z : T):
        w = [x, y, z]
        return min(w)

    epyc_f = epyccel(f, language=language)

    int_args = [randint(min_int, max_int) for _ in range(3)]
    float_args = [uniform(min_float/2, max_float/2) for _ in range(3)]

    assert epyc_f(*int_args) == f(*int_args)
    assert np.isclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)

def test_min_tuple_var(language):
    def f(x : T, y : T, z : T):
        w = (x, y, z)
        return min(w)

    epyc_f = epyccel(f, language=language)

    int_args = [randint(min_int, max_int) for _ in range(3)]
    float_args = [uniform(min_float/2, max_float/2) for _ in range(3)]

    assert epyc_f(*int_args) == f(*int_args)
    assert np.isclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)

def test_min_expr(language):
    def f(x : T, y : T):
        return min((x, y))+3, min(x, y)+4

    epyc_f = epyccel(f, language=language)

    int_args = [randint(min_int, max_int) for _ in range(2)]
    float_args = [uniform(min_float/2, max_float/2) for _ in range(2)]

    assert np.array_equal(epyc_f(*int_args), f(*int_args))
    assert np.allclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)

def test_min_temp_var_first_arg(language):
    def f(x: 'int', y: 'int'):
        return min(x + 1, y)

    epyc_f = epyccel(f, language=language)

    x, y = randint(min_int, max_int), randint(min_int, max_int)

    assert epyc_f(x, y) == f(x, y)

def test_min_temp_var_second_arg(language):
    def f(x: 'int', y: 'int'):
        return min(x, y + 2)

    epyc_f = epyccel(f, language=language)

    x, y = randint(min_int, max_int), randint(min_int, max_int)

    assert epyc_f(x, y) == f(x, y)

def test_max_2_args_i(language):
    def f(x : 'int', y : 'int'):
        return max(x, y)

    epyc_f = epyccel(f, language=language)

    int_args = [randint(min_int, max_int) for _ in range(2)]

    assert epyc_f(*int_args) == f(*int_args)

def test_max_2_args_f(language):
    def f(x : 'float', y : 'float'):
        return max(x, y)

    epyc_f = epyccel(f, language=language)

    float_args = [uniform(min_float/2, max_float/2) for _ in range(2)]

    assert np.isclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)

def test_max_3_args(language):
    def f(x : T, y : T, z : T):
        return max(x, y, z)

    epyc_f = epyccel(f, language=language)

    int_args = [randint(min_int, max_int) for _ in range(3)]
    float_args = [uniform(min_float/2, max_float/2) for _ in range(3)]

    assert epyc_f(*int_args) == f(*int_args)
    assert np.isclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)

def test_max_list(language):
    def f(x : T, y : T, z : T):
        return max([x, y, z])

    epyc_f = epyccel(f, language=language)

    int_args = [randint(min_int, max_int) for _ in range(3)]
    float_args = [uniform(min_float/2, max_float/2) for _ in range(3)]

    assert epyc_f(*int_args) == f(*int_args)
    assert np.isclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)

def test_max_tuple(language):
    def f(x : T, y : T, z : T):
        return max((x, y, z))

    epyc_f = epyccel(f, language=language)

    int_args = [randint(min_int, max_int) for _ in range(3)]
    float_args = [uniform(min_float/2, max_float/2) for _ in range(3)]

    assert epyc_f(*int_args) == f(*int_args)
    assert np.isclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)

def test_max_list_var(language):
    def f(x : T, y : T, z : T):
        w = [x, y, z]
        return max(w)

    epyc_f = epyccel(f, language=language)

    int_args = [randint(min_int, max_int) for _ in range(3)]
    float_args = [uniform(min_float/2, max_float/2) for _ in range(3)]

    assert epyc_f(*int_args) == f(*int_args)
    assert np.isclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)

def test_max_tuple_var(language):
    def f(x : T, y : T, z : T):
        w = (x, y, z)
        return max(w)

    epyc_f = epyccel(f, language=language)

    int_args = [randint(min_int, max_int) for _ in range(3)]
    float_args = [uniform(min_float/2, max_float/2) for _ in range(3)]

    assert epyc_f(*int_args) == f(*int_args)
    assert np.isclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)

def test_max_expr(language):
    def f(x : T, y : T):
        return max((x, y))+3, max(x, y)+4

    epyc_f = epyccel(f, language=language)

    int_args = [randint(min_int, max_int) for _ in range(2)]
    float_args = [uniform(min_float/2, max_float/2) for _ in range(2)]

    assert np.array_equal(epyc_f(*int_args), f(*int_args))
    assert np.allclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)

def test_max_temp_var_first_arg(language):
    def f(x: 'int', y: 'int'):
        return max(x + 1, y)

    epyc_f = epyccel(f, language=language)

    x, y = randint(min_int, max_int), randint(min_int, max_int)

    assert epyc_f(x, y) == f(x, y)

def test_max_temp_var_second_arg(language):
    def f(x: 'int', y: 'int'):
        return max(x, y + 2)

    epyc_f = epyccel(f, language=language)

    x, y = randint(min_int, max_int), randint(min_int, max_int)

    assert epyc_f(x, y) == f(x, y)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="sum not implemented in C"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_sum_matching_types(language):
    def f(x : T2, y : T2):
        return sum([x, y])

    epyc_f = epyccel(f, language=language)

    int_args = [randint(min_int//2, max_int//2) for _ in range(2)]
    float_args = [uniform(min_float/2, max_float/2) for _ in range(2)]
    complex_args = [uniform(min_float/4, max_float/4) + 1j*uniform(min_float/4, max_float/4)
                    for _ in range(2)]

    assert epyc_f(*int_args) == f(*int_args)
    assert np.isclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)
    assert np.isclose(epyc_f(*complex_args), f(*complex_args), rtol=RTOL, atol=ATOL)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="sum not implemented in C"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_sum_expr(language):
    def f(x : T, y : T):
        return sum((x, y))+3

    epyc_f = epyccel(f, language=language)

    int_args = [randint(min_int//3, max_int//3) for _ in range(2)]
    float_args = [uniform(min_float/2, max_float/2) for _ in range(2)]

    assert epyc_f(*int_args) == f(*int_args)
    assert np.allclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)

def test_len_numpy(language):
    def f():
        from numpy import ones
        a = ones((3,4))
        b = ones((4,3,5))
        c = ones(4)
        return len(a), len(b), len(c)

    epyc_f = epyccel(f, language=language)

    assert epyc_f() == f()


def test_len_tuple(language):
    def f():
        a = (3,4)
        b = (4,3,5)
        c = b
        return len(a), len(b), len(c), len((1,2))

    epyc_f = epyccel(f, language=language)

    assert epyc_f() == f()


def test_len_inhomog_tuple(language):
    def f():
        a = (3,True)
        b = (4j,False,5)
        c = b
        return len(a), len(b), len(c), len((1.5,2))

    epyc_f = epyccel(f, language=language)

    assert epyc_f() == f()

def test_len_list_int(language):
    def f():
        a = [1, 2, 3]
        return len(a)

    epyc_f = epyccel(f, language=language)

    assert epyc_f() == f()

def test_len_list_float(language):
    def f():
        a = [1.4, 2.6, 3.5]
        b = len(a)
        return b

    epyc_f = epyccel(f, language=language)

    assert epyc_f() == f()

def test_len_list_complex(language):
    def f():
        a = [1j, 2 + 1j, 3 + 1j]
        b = len(a)
        return b

    epyc_f = epyccel(f, language=language)

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
        a = {1:1.0, 2:2.0, 3:3.0, 4:4.0}
        b = len(a)
        return b

    epyc_f = epyccel(f, language=stc_language)

    assert epyc_f() == f()

def test_len_string(language):
    def f():
        a = 'abcdefghij'
        b = len(a)
        return b

    epyc_f = epyccel(f, language = language)

    assert epyc_f() == f()

def test_len_literal_string(language):
    def f():
        b = len('abcd')
        return b

    epyc_f = epyccel(f, language = language)

    assert epyc_f() == f()

def test_len_multi_layer(stc_language):
    def f():
        x = [1,2,3]
        y = [x]
        return len(y), len(y[0])

    epyc_f = epyccel(f, language = stc_language)

    assert epyc_f() == f()

def test_round_int(language):
    def round_int(x : float):
        return round(x)

    f = epyccel(round_int, language=language)
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

def test_negative_round_int(language):
    def round_int(x : float):
        return round(x)

    f = epyccel(round_int, language=language)
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

def test_round_ndigits(language):
    def round_ndigits(x : float, i : int):
        return round(x,i)

    f = epyccel(round_ndigits, language=language)
    x = randint(100) / 10

    f_output = f(x, 1)
    round_ndigits_output = round_ndigits(x, 1)
    assert np.isclose(round_ndigits_output, f_output)
    assert isinstance(f_output, type(round_ndigits_output))

    x = 3.343

    f_output = f(x,2)
    round_ndigits_output = round_ndigits(x,2)
    assert np.isclose(round_ndigits_output, f_output)
    assert isinstance(f_output, type(round_ndigits_output))

    x = 3323.0

    f_output = f(x,-2)
    round_ndigits_output = round_ndigits(x, -2)
    assert np.isclose(round_ndigits_output, f_output)
    assert isinstance(f_output, type(round_ndigits_output))

    x = -3390.0

    f_output = f(x,-2)
    round_ndigits_output = round_ndigits(x, -2)
    assert np.isclose(round_ndigits_output, f_output)
    assert isinstance(f_output, type(round_ndigits_output))

def test_round_ndigits_half(language):
    def round_ndigits(x : float, i : int):
        return round(x,i)

    f = epyccel(round_ndigits, language=language)
    x = randint(100) / 10

    f_output = f(x, 1)
    round_ndigits_output = round_ndigits(x, 1)
    assert np.isclose(round_ndigits_output, f_output)
    assert isinstance(f_output, type(round_ndigits_output))

    x = 3.345

    f_output = f(x,2)
    round_ndigits_output = round_ndigits(x,2)
    assert np.isclose(round_ndigits_output, f_output)
    assert isinstance(f_output, type(round_ndigits_output))

    x = -3350.0

    f_output = f(x,-2)
    round_ndigits_output = round_ndigits(x, -2)
    assert np.isclose(round_ndigits_output, f_output)
    assert isinstance(f_output, type(round_ndigits_output))

    x = 45.0

    f_output = f(x,-1)
    round_ndigits_output = round_ndigits(x,-1)
    assert np.isclose(round_ndigits_output, f_output)
    assert isinstance(f_output, type(round_ndigits_output))

def test_round_ndigits_int(language):
    def round_ndigits(x : int, i : int):
        return round(x,i)

    f = epyccel(round_ndigits, language=language)
    x = randint(100) // 10

    f_output = f(x, 1)
    round_ndigits_output = round_ndigits(x, 1)
    assert np.isclose(round_ndigits_output, f_output)
    assert isinstance(f_output, type(round_ndigits_output))

    x = 3

    f_output = f(x,2)
    round_ndigits_output = round_ndigits(x,2)
    assert np.isclose(round_ndigits_output, f_output)
    assert isinstance(f_output, type(round_ndigits_output))

    x = 3323

    f_output = f(x,-2)
    round_ndigits_output = round_ndigits(x, -2)
    assert np.isclose(round_ndigits_output, f_output)
    assert isinstance(f_output, type(round_ndigits_output))

    x = -3390

    f_output = f(x,-2)
    round_ndigits_output = round_ndigits(x, -2)
    assert np.isclose(round_ndigits_output, f_output)
    assert isinstance(f_output, type(round_ndigits_output))

def test_round_ndigits_bool(language):
    def round_ndigits():
        return round(True), round(False), round(True, 1), round(True, -1)

    f = epyccel(round_ndigits, language=language)

    f_output = f()
    round_ndigits_output = round_ndigits()
    assert all(o == r for o, r in zip(f_output, round_ndigits_output))
    assert all(isinstance(o, type(r)) for o, r in zip(f_output, round_ndigits_output))

def test_isinstance_native(language):
    def isinstance_test(a : 'bool | int | float | complex'):
        return isinstance(a, bool), isinstance(a, int), isinstance(a, float), isinstance(a, complex)

    f = epyccel(isinstance_test, language=language)
    assert f(True) == isinstance_test(True)
    assert f(False) == isinstance_test(False)
    assert f(4) == isinstance_test(6)
    assert f(3.9) == isinstance_test(6.7)
    assert f(1+2j) == isinstance_test(6.5+8.3j)

def test_isinstance_containers(language):
    def isinstance_tup(a : int, b : int):
        container = (a, b)
        return (isinstance(container, tuple), isinstance(container, list),
                isinstance(container, set), isinstance(container, dict))
    def isinstance_lst(a : int, b : int):
        container = [a, b]
        return (isinstance(container, tuple), isinstance(container, list),
                isinstance(container, set), isinstance(container, dict))
    def isinstance_set(a : int, b : int):
        container = {a, b}
        return (isinstance(container, tuple), isinstance(container, list),
                isinstance(container, set), isinstance(container, dict))
    def isinstance_dict(a : int, b : int):
        container = {a: False, b: True}
        return (isinstance(container, tuple), isinstance(container, list),
                isinstance(container, set), isinstance(container, dict))

    test_funcs = (isinstance_tup, isinstance_lst, isinstance_set, isinstance_dict)

    for f in test_funcs:
        f_epyc = epyccel(f, language=language)

        assert f(2, 5) == f_epyc(2, 5)

def test_isinstance_numpy(language):
    def isinstance_test(a : 'int32 | int64 | int | float32'):
        return isinstance(a, np.int32), isinstance(a, np.int64), isinstance(a, int), isinstance(a, np.float32)

    f = epyccel(isinstance_test, language=language)
    assert f(np.int32(4)) == isinstance_test(np.int32(4))
    assert f(np.int64(4)) == isinstance_test(np.int64(4))
    assert f(4) == isinstance_test(4)
    assert f(np.float32(4)) == isinstance_test(np.float32(4))

def test_isinstance_tuple(language):
    def isinstance_test(a : 'bool | int | float | complex'):
        """
        Testing a case which should generate radically different functions.
        """
        return (isinstance(a, (bool, int)), isinstance(a, (bool, float)), isinstance(a, (int, complex)),
                isinstance(a, (tuple, list)))

    f = epyccel(isinstance_test, language=language)
    assert f(True) == isinstance_test(True)
    assert f(False) == isinstance_test(False)
    assert f(4) == isinstance_test(6)
    assert f(3.9) == isinstance_test(6.7)
    assert f(1+2j) == isinstance_test(6.5+8.3j)

@pytest.mark.skipif(sys.version_info < (3, 10), reason="Union of types implemented in Python 3.10")
def test_isinstance_union(language):
    def isinstance_test(a : bool | int | float | complex): #pylint: disable=unsupported-binary-operation
        return (isinstance(a, bool | int), isinstance(a, bool | float), isinstance(a, int | complex), #pylint: disable=unsupported-binary-operation
                isinstance(a, tuple | list)) #pylint: disable=unsupported-binary-operation

    f = epyccel(isinstance_test, language=language)
    assert f(True) == isinstance_test(True)
    assert f(False) == isinstance_test(False)
    assert f(4) == isinstance_test(6)
    assert f(3.9) == isinstance_test(6.7)
    assert f(1+2j) == isinstance_test(6.5+8.3j)
