# pylint: disable=missing-function-docstring, missing-module-docstring

import pytest
from numpy import ones

from pyccel.epyccel import epyccel

#------------------------------------------------------------------------------
@pytest.fixture(params=[
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python),
    ]
)
def language(request):
    return request.param

#==============================================================================

def test_import(language):
    def f1(x : 'int[:]'):
        import numpy
        s = numpy.shape(x)[0]
        return s

    f = epyccel(f1, language = language)
    x = ones(10, dtype=int)
    assert f(x) == f1(x)

def test_import_from(language):
    def f2(x : 'int[:]'):
        from numpy import shape
        s = shape(x)[0]
        return s


    f = epyccel(f2, language = language)
    x = ones(10, dtype=int)
    assert f(x) == f2(x)

def test_import_as(language):
    def f3(x : 'int[:]'):
        import numpy as np
        s = np.shape(x)[0]
        return s

    f = epyccel(f3, language = language)
    x = ones(10, dtype=int)
    assert f(x) == f3(x)

@pytest.mark.parametrize( 'language', [
        pytest.param("python", marks = pytest.mark.python),
    ]
)
def test_import_collision(language):
    def f4(x : 'int'):
        import modules.Module_3 as mod
        add_one = mod.add_one(x)
        return add_one

    f = epyccel(f4, language = language)
    assert f(5) == f4(5)

def test_import_method(language):
    def f5(x : 'int[:]'):
        s = x.shape[0]
        return s

    f = epyccel(f5, language = language)
    x = ones(10, dtype=int)
    assert f(x) == f5(x)
