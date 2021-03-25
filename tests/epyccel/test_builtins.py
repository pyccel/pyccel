# pylint: disable=missing-function-docstring, missing-module-docstring/
import pytest
from numpy.random import randint
from numpy import isclose, iinfo, finfo

from pyccel.epyccel import epyccel
from pyccel.decorators import types, template

min_int = iinfo('int').min
max_int = iinfo('int').max

min_float = finfo('float').min
max_float = finfo('float').max

def test_abs_i(language):
    @types('int')
    def f1(x):
        return abs(x)

    f2 = epyccel(f1, language=language)

    negative_test = randint(min_int, 0)
    positive_test = randint(0, max_int)

    assert f1(0) == f2(0)
    assert f1(negative_test) == f2(negative_test)
    assert f1(positive_test) == f2(positive_test)

def test_abs_r(language):
    @types('real')
    def f1(x):
        return abs(x)

    f2 = epyccel(f1, language=language)

    negative_test = uniform(min_float, 0.0)
    positive_test = uniform(0.0, max_float)

    assert f1(0.00000) == f2(0.00000)
    assert f1(negative_test) == f2(negative_test)
    assert f1(positive_test) == f2(positive_test)



def test_abs_c(language):
    @types('complex')
    def f1(x):
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

    assert f1(pos_pos) == f2(pos_pos)
    assert f1(pos_neg) == f2(pos_neg)
    assert f1(neg_pos) == f2(neg_pos)
    assert f1(neg_neg) == f2(neg_neg)
    assert f1(zero_rand) == f2(zero_rand)
    assert f1(rand_zero) == f2(rand_zero)
    assert f1(0j + 0) == f2(0j + 0)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="min not implemented in C"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)

    )
)
def test_min_2_args(language):
    @types('int','int')
    @types('float','float')
    def f(x, y):
        return min(x, y)

    a = randint(100)
    b = randint(100)
    epyc_f = epyccel(f, language=language)
    assert epyc_f(a,b) == f(a,b)
    assert epyc_f(float(a),float(b)) == f(float(a),float(b))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="min not implemented in C"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_min_3_args(language):
    @types('int','int','int')
    @types('float','float','float')
    def f(x, y, z):
        return min(x, y, z)

    a = randint(100)
    b = randint(100)
    c = randint(100)
    epyc_f = epyccel(f, language=language)
    assert epyc_f(a,b,c) == f(a,b,c)
    assert epyc_f(float(a),float(b),float(c)) == f(float(a),float(b),float(c))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="max not implemented in C"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_max_2_args(language):
    @types('int','int')
    @types('float','float')
    def f(x, y):
        return max(x, y)

    a = randint(100)
    b = randint(100)
    epyc_f = epyccel(f, language=language)
    assert epyc_f(a,b) == f(a,b)
    assert epyc_f(float(a),float(b)) == f(float(a),float(b))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="max not implemented in C"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_max_3_args(language):
    @types('int','int','int')
    @types('float','float','float')
    def f(x, y, z):
        return min(x, y, z)

    a = randint(100)
    b = randint(100)
    c = randint(100)
    epyc_f = epyccel(f, language=language)
    assert epyc_f(a,b,c) == f(a,b,c)
    assert epyc_f(float(a),float(b),float(c)) == f(float(a),float(b),float(c))

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
    @types('T','T')
    def f(x, y):
        return sum([x, y])

    a = randint(100)
    b = randint(100)
    epyc_f = epyccel(f, language=language)
    assert epyc_f(a,b) == f(a,b)
    assert epyc_f(float(a),float(b)) == f(float(a),float(b))
    assert epyc_f(complex(a),complex(b)) == f(complex(a),complex(b))
