# pylint: disable=missing-function-docstring, missing-module-docstring/
import pytest
from numpy.random import randint

from pyccel.epyccel import epyccel
from pyccel.decorators import types, template

def test_abs_i(language):
    @types('int')
    def f1(x):
        return abs(x)

    f2 = epyccel(f1, language=language)

    assert f1(0) == f2(0)
    assert f1(-5) == f2(-5)
    assert f1(11) == f2(11)

def test_abs_r(language):
    @types('real')
    def f1(x):
        return abs(x)

    f2 = epyccel(f1, language=language)

    assert f1(0.00000) == f2(0.00000)
    assert f1(-3.1415) == f2(-3.1415)
    assert f1(2.71828) == f2(2.71828)



def test_abs_c(language):
    @types('complex')
    def f1(x):
        return abs(x)

    f2 = epyccel(f1, language=language)

    assert f1(3j + 4) == f2(3j + 4)
    assert f1(3j - 4) == f2(3j - 4)
    assert f1(5j + 0) == f2(5j + 0)
    assert f1(0j + 5) == f2(0j + 5)
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
