# pylint: disable=missing-function-docstring, missing-module-docstring/
import pytest
from numpy.random import randint
from numpy import equal

from pyccel.epyccel import epyccel
from modules import functionals

#==============================================================================
class epyccel_test:
    """
    Class to pyccelize module then compare different results
    This avoids the need to pyccelize the file multiple times
    or write the arguments multiple times
    """
    def __init__(self, f, lang='fortran'):
        self._f  = f
        self._f2 = epyccel(f, language=lang)

    def compare_epyccel(self, *args):
        out1 = self._f(*args)
        out2 = self._f2(*args)
        assert equal(out1, out2 ).all()

#==============================================================================

def test_functional_for_1d_range(language):
    test = epyccel_test(functionals.functional_for_1d_range, lang=language)
    test.compare_epyccel()

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = [
            pytest.mark.python,
            pytest.mark.skip(reason = "Too many expressions introduced for 'for xi in x'")
        ])
    )
)
def test_functional_for_1d_var(language):
    y = randint(99,size = 4)
    test = epyccel_test(functionals.functional_for_1d_var, lang=language)
    test.compare_epyccel(y)

def test_functional_for_2d_range(language):
    test = epyccel_test(functionals.functional_for_2d_range, lang=language)
    test.compare_epyccel()

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = [
            pytest.mark.python,
            pytest.mark.skip(reason = "Too many expressions introduced for 'for xi in x'")
        ])
    )
)
def test_functional_for_2d_var_range(language):
    y = randint(99, size = 3)
    test = epyccel_test(functionals.functional_for_2d_var_range, lang=language)
    test.compare_epyccel(y)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = [
            pytest.mark.python,
            pytest.mark.skip(reason = "Too many expressions introduced for 'for xi in x'")
        ])
    )
)
def test_functional_for_2d_var_var(language):
    y = randint(99, size = 3)
    z = randint(99, size = 2)
    test = epyccel_test(functionals.functional_for_2d_var_var, lang=language)
    test.compare_epyccel(y, z)

def test_functional_for_2d_dependant_range(language):
    test = epyccel_test(functionals.functional_for_2d_dependant_range_1, lang=language)
    test.compare_epyccel()
    test.compare_epyccel()
    test.compare_epyccel()

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="Tuples not implemented in C"),
            pytest.mark.c]
        )
    )
)
def test_functional_for_2d_array_range(language):
    idx = randint(28)
    test = epyccel_test(functionals.functional_for_2d_array_range, lang=language)
    test.compare_epyccel(idx)

def test_functional_for_3d_range(language):
    test = epyccel_test(functionals.functional_for_3d_range, lang=language)
    test.compare_epyccel()
