# pylint: disable=missing-function-docstring, missing-module-docstring
# coding: utf-8
import pytest
import numpy as np

from pyccel.epyccel import epyccel
from pyccel.decorators import types

#------------------------------------------------------------------------------
def test_f1(language):
    @types('int')
    def f1(x = 1):
        y = x - 1
        return y

    f = epyccel(f1, language = language)

    # ...
    assert f(2) == f1(2)
    assert f() == f1()
    # ...
#------------------------------------------------------------------------------
def test_f2(language):
    @types('real [:]', 'int')
    def f5(x, m1 = 2):
        x[:] = 0.
        for i in range(0, m1):
            x[i] = i * 1.

    f = epyccel(f5, language=language)

    # ...
    m1 = 3

    x = np.zeros(m1)
    f(x)

    x_expected = np.zeros(m1)
    f5(x_expected)

    assert np.allclose( x, x_expected, rtol=2e-14, atol=1e-15 )
    # ...

    f(x, m1 = m1)

    f5(x_expected, m1)

    assert np.allclose( x, x_expected, rtol=2e-14, atol=1e-15 )


#------------------------------------------------------------------------------
def test_f3(language):
    @types('real','real')
    def f3(x = 1.5, y = 2.5):
        return x+y

    f = epyccel(f3, language=language)

    # ...
    assert f(19.2,6.7) == f3(19.2,6.7)
    assert f(4.5) == f3(4.5)
    assert f(y = 8.2) == f3(y = 8.2)
    assert f() == f3()
    # ...

#------------------------------------------------------------------------------
def test_f4(language):
    @types('bool')
    def f4(x = True):
        if x:
            return 1
        else:
            return 2

    f = epyccel(f4, language = language)

    # ...
    assert f(True)  == f4(True)
    assert f(False) == f4(False)
    assert f()      == f4()
    # ...

#------------------------------------------------------------------------------
def test_f5(language):
    @types('complex')
    def f5(x = 1j):
        y = x - 1
        return y

    f = epyccel(f5, language = language)

    # ...
    assert f(2.9+3j) == f5(2.9+3j)
    assert f()       == f5()
    # ...

#------------------------------------------------------------------------------
def test_changed_precision_arguments(language):
    import modules.Module_8 as mod

    modnew = epyccel(mod, language=language)

    assert mod.get_f() == modnew.get_f()
    assert mod.get_g() == modnew.get_g()
