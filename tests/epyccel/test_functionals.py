# pylint: disable=missing-function-docstring, missing-module-docstring
from numpy.random import randint
from numpy import equal, array, arange
import pytest


from pyccel import epyccel
from modules import functionals


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
    y = array(randint(99, size = 4), dtype = int)
    compare_epyccel(functionals.functional_for_1d_var, language, y)

def test_functional_for_1d_const(language):
    y = array(randint(99, size = 4), dtype = int)
    z = randint(99)
    compare_epyccel(functionals.functional_for_1d_const, language, y, z)

def test_functional_for_1d_const2(language):
    compare_epyccel(functionals.functional_for_1d_const2, language)

def test_functional_for_2d_range(language):
    compare_epyccel(functionals.functional_for_2d_range, language)

def test_functional_for_2d_var_range(language):
    y = array(randint(99, size = 3), dtype = int)
    compare_epyccel(functionals.functional_for_2d_var_range, language, y)

def test_functional_for_2d_var_var(language):
    y = array(randint(99, size = 3), dtype = int)
    z = array(randint(99, size = 2), dtype = int)
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
    y = array(randint(100, size = 20), dtype = int)
    compare_epyccel(functionals.unknown_length_functional, language, y)

def test_functional_with_enumerate(language):
    compare_epyccel(functionals.functional_with_enumerate, language)

def test_functional_with_enumerate_with_start(language):
    compare_epyccel(functionals.functional_with_enumerate_with_start, language)

def test_functional_with_condition(language):
    compare_epyccel(functionals.functional_with_condition, language)

def test_functional_with_zip(language):
    compare_epyccel(functionals.functional_with_zip, language)

def test_functional_with_multiple_zips(language):
    compare_epyccel(functionals.functional_with_multiple_zips, language)

def test_functional_filter_and_transform(language):
    compare_epyccel(functionals.functional_with_condition, language)

def test_functional_with_multiple_conditions(language):
    compare_epyccel(functionals.functional_with_multiple_conditions, language)

def test_functional_negative_indices(language):
    compare_epyccel(functionals.functional_negative_indices, language, arange(10))

def test_functional_reverse(language):
    compare_epyccel(functionals.functional_reverse, language, arange(4))

def test_functional_reverse(language):
    compare_epyccel(functionals.functional_indexed_iterator, language, arange(10))
