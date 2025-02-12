# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np
from pyccel import epyccel

def test_rank_differentiation_1(language):
    def f(a : 'int[:] | int[:,:]'):
        if len(a.shape) == 1:
            return a[0]
        else:
            return a[0,0]

    x = np.arange(10, dtype=int)
    y = np.array(np.reshape(x[::-1], (2,5)), dtype=int)

    f_epyc = epyccel(f, language=language)
    assert f_epyc(x) == f(x)
    assert f_epyc(y) == f(y)

def test_rank_differentiation_2(language):
    def f(a : 'int[:] | int[:,:]'):
        if len(a.shape) != 2:
            return a[0]
        else:
            return a[0,0]

    x = np.arange(10, dtype=int)
    y = np.array(np.reshape(x[::-1], (2,5)), dtype=int)

    f_epyc = epyccel(f, language=language)
    assert f_epyc(x) == f(x)
    assert f_epyc(y) == f(y)

def test_type_differentiation(language):
    def f(a : 'int | float'):
        if isinstance(a, int):
            return a*2
        else:
            return -a

    f_epyc = epyccel(f, language=language)
    assert f_epyc(3) == f(3)
    assert isinstance(f_epyc(3), type(f(3)))
    assert f_epyc(4.0) == f(4.0)
    assert isinstance(f_epyc(4.0), type(f(4.0)))
