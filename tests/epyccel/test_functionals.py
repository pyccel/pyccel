# pylint: disable=missing-function-docstring, missing-module-docstring
from numpy.random import randint
from numpy import equal, array
import pytest


from pyccel import epyccel
from modules import functionals

np_default_int = array([1]).dtype

def compare_epyccel(f, language, *args):
    f2 = epyccel(f, language=language)
    out1 = f(*args)
    out2 = f2(*args)
    assert equal(out1, out2).all()

def test_functional_for_1d_range(language):
    compare_epyccel(functionals.functional_for_1d_range, language)

def test_functional_for_overwrite_1d_range(language):
    compare_epyccel(functionals.functional_for_overwrite_1d_range, language)

def test_functional_for_1d_var(language):
    y = randint(99, size = 4, dtype = np_default_int)
    compare_epyccel(functionals.functional_for_1d_var, language, y)

def test_functional_for_1d_const(language):
    y = randint(99, size = 4, dtype = np_default_int)
    z = randint(99)
    compare_epyccel(functionals.functional_for_1d_const, language, y, z)

def test_functional_for_1d_const2(language):
    compare_epyccel(functionals.functional_for_1d_const2, language)

def test_functional_for_2d_range(language):
    compare_epyccel(functionals.functional_for_2d_range, language)

def test_functional_for_2d_var_range(language):
    y = randint(99, size = 3, dtype = np_default_int)
    compare_epyccel(functionals.functional_for_2d_var_range, language, y)

def test_functional_for_2d_var_var(language):
    y = randint(99, size = 3, dtype = np_default_int)
    z = randint(99, size = 2, dtype = np_default_int)
    compare_epyccel(functionals.functional_for_2d_var_var, language, y, z)

def test_functional_for_2d_dependant_range(language):
    compare_epyccel(functionals.functional_for_2d_dependant_range_1, language)
    compare_epyccel(functionals.functional_for_2d_dependant_range_2, language)
    compare_epyccel(functionals.functional_for_2d_dependant_range_3, language)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason="lists of tuples are not yes supported"),
            pytest.mark.fortran]
        ),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="lists of tuples are not yes supported"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_functional_for_2d_array_range(language):
    idx = randint(28)
    compare_epyccel(functionals.functional_for_2d_array_range, language,idx)

def test_functional_for_2d_range_const(language):
    compare_epyccel(functionals.functional_for_2d_range_const, language)

def test_functional_for_3d_range(language):
    compare_epyccel(functionals.functional_for_3d_range, language)

def test_unknown_length_functional(language):
    y = randint(100, size = 20, dtype = np_default_int)
    compare_epyccel(functionals.unknown_length_functional, language, y)
