# pylint: disable=missing-function-docstring, missing-module-docstring
from numpy.random import randint
from numpy import equal
import pytest


from pyccel import epyccel
from modules import functionals

@pytest.fixture( params=[
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason="Fortan support is disabled until lists are supported. See #1657"),
            pytest.mark.c]),
        pytest.param("python", marks = pytest.mark.python)
    ],
    scope = "module"
)
def language(request):
    return request.param

def compare_epyccel(f, language, *args):
    f2 = epyccel(f, language=language)
    out1 = f(*args)
    out2 = f2(*args)
    assert equal(out1, out2).all()

def test_functional_for_1d_range(language):
    compare_epyccel(functionals.functional_for_1d_range, language)

@pytest.mark.skipif(lambda lang: lang == 'c', reason="Skipping for C due to array redefinition issue. See #1979")
def test_functional_for_overwrite_1d_range(language):
    compare_epyccel(functionals.functional_for_overwrite_1d_range, language)

def test_functional_for_1d_var(language):
    y = randint(99,size = 4)
    compare_epyccel(functionals.functional_for_1d_var, language, y)

def test_functional_for_1d_const(language):
    y = randint(99,size = 4)
    z = randint(99)
    compare_epyccel(functionals.functional_for_1d_const, language, y, z)

def test_functional_for_1d_const2(language):
    compare_epyccel(functionals.functional_for_1d_const2, language)

def test_functional_for_2d_range(language):
    compare_epyccel(functionals.functional_for_2d_range, language)

def test_functional_for_2d_var_range(language):
    y = randint(99, size = 3)
    compare_epyccel(functionals.functional_for_2d_var_range, language, y)

def test_functional_for_2d_var_var(language):
    y = randint(99, size = 3)
    z = randint(99, size = 2)
    compare_epyccel(functionals.functional_for_2d_var_var, language, y, z)

def test_functional_for_2d_dependant_range(language):
    compare_epyccel(functionals.functional_for_2d_dependant_range_1, language)
    compare_epyccel(functionals.functional_for_2d_dependant_range_2, language)
    compare_epyccel(functionals.functional_for_2d_dependant_range_3, language)

@pytest.mark.skipif(lambda lang: lang == 'c', reason="Skipping for C due to list of lists initialization. See #1945")
def test_functional_for_2d_array_range(language):
    idx = randint(28)
    compare_epyccel(functionals.functional_for_2d_array_range, language,idx)

def test_functional_for_2d_array_range_const(language):
    compare_epyccel(functionals.functional_for_2d_range_const, language)

def test_functional_for_3d_range(language):
    compare_epyccel(functionals.functional_for_3d_range, language)

def test_unknown_length_functional(language):
    y = randint(100, size = 20)
    compare_epyccel(functionals.unknown_length_functional, language, y)

def test_functional_with_enumerate(language):
    compare_epyccel(functionals.functional_with_enumerate, language)

def test_functional_with_enumerate_with_start(language):
    compare_epyccel(functionals.functional_with_enumerate_with_start, language)

def test_functional_with_condition(language):
    compare_epyccel(functionals.functional_with_condition, language)

def test_functional_with_zip(language):
    compare_epyccel(functionals.functional_with_zip, language)
