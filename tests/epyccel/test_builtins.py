# pylint: disable=missing-function-docstring, missing-module-docstring
import pytest
from numpy.random import randint, uniform
from numpy import iinfo, finfo
import numpy as np

from pyccel.epyccel import epyccel
from pyccel.decorators import template

ATOL = 1e-15
RTOL = 2e-14


min_int = iinfo('int').min
max_int = iinfo('int').max

min_float = finfo('float').min
max_float = finfo('float').max

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

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="min not implemented in C for more than 2 args"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_min_3_args(language):
    @template('T', [int, float])
    def f(x : 'T', y : 'T', z : 'T'):
        return min(x, y, z)

    epyc_f = epyccel(f, language=language)

    int_args = [randint(min_int, max_int) for _ in range(3)]
    float_args = [uniform(min_float/2, max_float/2) for _ in range(3)]

    assert epyc_f(*int_args) == f(*int_args)
    assert np.isclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="min not implemented in C for more than 2 args"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_min_list(language):
    @template('T', [int, float])
    def f(x : 'T', y : 'T', z : 'T'):
        return min([x, y, z])

    epyc_f = epyccel(f, language=language)

    int_args = [randint(min_int, max_int) for _ in range(3)]
    float_args = [uniform(min_float/2, max_float/2) for _ in range(3)]

    assert epyc_f(*int_args) == f(*int_args)
    assert np.isclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="min not implemented in C for more than 2 args"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_min_tuple(language):
    @template('T', [int, float])
    def f(x : 'T', y : 'T', z : 'T'):
        return min((x, y, z))

    epyc_f = epyccel(f, language=language)

    int_args = [randint(min_int, max_int) for _ in range(3)]
    float_args = [uniform(min_float/2, max_float/2) for _ in range(3)]

    assert epyc_f(*int_args) == f(*int_args)
    assert np.isclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)

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

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="max not implemented in C for more than 2 args"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_max_3_args(language):
    @template('T', [int, float])
    def f(x : 'T', y : 'T', z : 'T'):
        return min(x, y, z)

    epyc_f = epyccel(f, language=language)

    int_args = [randint(min_int, max_int) for _ in range(3)]
    float_args = [uniform(min_float/2, max_float/2) for _ in range(3)]

    assert epyc_f(*int_args) == f(*int_args)
    assert np.isclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="max not implemented in C for more than 2 args"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_max_list(language):
    @template('T', [int, float])
    def f(x : 'T', y : 'T', z : 'T'):
        return max([x, y, z])

    epyc_f = epyccel(f, language=language)

    int_args = [randint(min_int, max_int) for _ in range(3)]
    float_args = [uniform(min_float/2, max_float/2) for _ in range(3)]

    assert epyc_f(*int_args) == f(*int_args)
    assert np.isclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="max not implemented in C for more than 2 args"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_max_tuple(language):
    @template('T', [int, float])
    def f(x : 'T', y : 'T', z : 'T'):
        return max((x, y, z))

    epyc_f = epyccel(f, language=language)

    int_args = [randint(min_int, max_int) for _ in range(3)]
    float_args = [uniform(min_float/2, max_float/2) for _ in range(3)]

    assert epyc_f(*int_args) == f(*int_args)
    assert np.isclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)

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
    @template('T',['int','float','complex'])
    def f(x : 'T', y : 'T'):
        return sum([x, y])

    epyc_f = epyccel(f, language=language)

    int_args = [randint(min_int/2, max_int/2) for _ in range(2)]
    float_args = [uniform(min_float/2, max_float/2) for _ in range(2)]
    complex_args = [uniform(min_float/2, max_float/2) + 1j*uniform(min_float/2, max_float/2)
                    for _ in range(2)]

    assert epyc_f(*int_args) == f(*int_args)
    assert np.isclose(epyc_f(*float_args), f(*float_args), rtol=RTOL, atol=ATOL)
    assert np.isclose(epyc_f(*complex_args), f(*complex_args), rtol=RTOL, atol=ATOL)
