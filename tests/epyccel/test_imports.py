# pylint: disable=missing-function-docstring, missing-module-docstring

import pytest
from numpy import ones

from pyccel.decorators import types
from pytest_teardown_tools import run_epyccel, clean_test

#==============================================================================

def test_import(language):
    @types('int[:]')
    def f1(x):
        import numpy
        s = numpy.shape(x)[0]
        return s

    f = run_epyccel(f1, language = language)
    x = ones(10, dtype=int)
    assert f(x) == f1(x)

def test_import_from(language):
    @types('int[:]')
    def f2(x):
        from numpy import shape
        s = shape(x)[0]
        return s


    f = run_epyccel(f2, language = language)
    x = ones(10, dtype=int)
    assert f(x) == f2(x)

def test_import_as(language):
    @types('int[:]')
    def f3(x):
        import numpy as np
        s = np.shape(x)[0]
        return s

    f = run_epyccel(f3, language = language)
    x = ones(10, dtype=int)
    assert f(x) == f3(x)

@pytest.mark.parametrize( 'language', [
        pytest.param("python", marks = pytest.mark.python),
    ]
)
def test_import_collision(language):
    @types('int')
    def f4(x):
        import modules.Module_3 as mod
        add_one = mod.add_one(x)
        return add_one

    f = run_epyccel(f4, language = language)
    assert f(5) == f4(5)

def test_import_method(language):
    @types('int[:]')
    def f5(x):
        s = x.shape[0]
        return s

    f = run_epyccel(f5, language = language)
    x = ones(10, dtype=int)
    assert f(x) == f5(x)

##==============================================================================
## CLEAN UP GENERATED FILES AFTER RUNNING TESTS
##==============================================================================

def teardown_module(module):
    clean_test()
