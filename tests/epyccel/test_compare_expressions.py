# pylint: disable=missing-function-docstring, missing-module-docstring/
import numpy as np
from pyccel.epyccel import epyccel
from pyccel.decorators import types

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
        assert np.equal(out1, out2 )

#==============================================================================
@types('int, int, int')
def mod_eq_pow(a, m, n):
    return a%m == n**2

@types('int, int, int')
def mod_neq_pow(a, m, n):
    return a%m != n**2

@types('int, int, int')
def idiv_gt_add(a, m, n):
    return a//m > n+1

#==============================================================================
def test_mod_eq_pow():
    test = epyccel_test(mod_eq_pow)
    # True
    test.compare_epyccel(10, 3, 1)
    test.compare_epyccel(19, 10, 3)
    test.compare_epyccel(21, 3, 0)
    # False
    test.compare_epyccel(10, 5, 2)
    test.compare_epyccel(19, 10, 1)
    test.compare_epyccel(21, 3, 1)

def test_mod_neq_pow():
    test = epyccel_test(mod_neq_pow)
    # True
    test.compare_epyccel(10, 5, 2)
    test.compare_epyccel(19, 10, 1)
    test.compare_epyccel(21, 3, 1)
    # False
    test.compare_epyccel(10, 3, 1)
    test.compare_epyccel(19, 10, 3)
    test.compare_epyccel(21, 3, 0)

def test_idiv_gt_add():
    test = epyccel_test(idiv_gt_add)
    # True
    test.compare_epyccel(10, 3, 2)
    test.compare_epyccel(8, 2, 2)
    test.compare_epyccel(16, 3, 4)
    # False
    test.compare_epyccel(10, 3, 2)
    test.compare_epyccel(8, 2, 3)
    test.compare_epyccel(16, 3, 5)
