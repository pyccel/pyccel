# pylint: disable=missing-function-docstring, missing-module-docstring

from numpy import ones

from pyccel import epyccel

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

def test_import_method(language):
    def f5(x : 'int[:]'):
        s = x.shape[0]
        return s

    f = epyccel(f5, language = language)
    x = ones(10, dtype=int)
    assert f(x) == f5(x)
