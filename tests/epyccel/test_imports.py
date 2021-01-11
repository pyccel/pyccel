# pylint: disable=missing-function-docstring, missing-module-docstring/

import pytest
import numpy as np

from pyccel.epyccel import epyccel
from pyccel.decorators import types

#------------------------------------------------------------------------------
@pytest.fixture(params=[
    pytest.param('python', marks = pytest.mark.python)]
)
def language(request):
    return request.param

#==============================================================================

def test_import(language):
    @types('int[:]')
    def f1(x):
        import numpy  # pylint: disable=import-error
        s = numpy.shape(x)[0]
        return s

    f = epyccel(f1, language = language)
    x = np.ones(10, dtype=int)
    assert f(x) == f1(x)

def test_import_from(language):
    @types('int[:]')
    def f2(x):
        from numpy import shape
        s = shape(x)[0]
        return s


    f = epyccel(f2, language = language)
    x = np.ones(10, dtype=int)
    assert f(x) == f2(x)

def test_import_as(language):
    @types('int[:]')
    def f3(x):
        import numpy as np  # pylint: disable=import-error
        s = np.shape(x)[0]
        return s

    f = epyccel(f3, language = language)
    x = np.ones(10, dtype=int)
    assert f(x) == f3(x)

def test_import_collision(language):
    @types('int')
    def f4(x):
        import modules.Module_3 as mod
        add_one = mod.add_one(x)
        return add_one

    f = epyccel(f4, language = language)
    assert f(5) == f4(5)
