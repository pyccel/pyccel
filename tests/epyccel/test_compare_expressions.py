from pyccel.epyccel import epyccel
from pyccel.decorators import types

#==============================================================================
def compare_epyccel(f1, *args):
    f2 = epyccel(f1)
    out1 = f1(*args)
    out2 = f2(*args)
    assert out1 == out2

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
    # True
    compare_epyccel(mod_eq_pow, 10, 3, 1)
    compare_epyccel(mod_eq_pow, 19, 10, 3)
    compare_epyccel(mod_eq_pow, 21, 3, 0)
    # False
    compare_epyccel(mod_eq_pow, 10, 5, 2)
    compare_epyccel(mod_eq_pow, 19, 10, 1)
    compare_epyccel(mod_eq_pow, 21, 3, 1)

def test_mod_neq_pow():
    # True
    compare_epyccel(mod_neq_pow, 10, 5, 2)
    compare_epyccel(mod_neq_pow, 19, 10, 1)
    compare_epyccel(mod_neq_pow, 21, 3, 1)
    # False
    compare_epyccel(mod_neq_pow, 10, 3, 1)
    compare_epyccel(mod_neq_pow, 19, 10, 3)
    compare_epyccel(mod_neq_pow, 21, 3, 0)

def test_idiv_gt_add():
    # True
    compare_epyccel(idiv_gt_add, 10, 3, 2)
    compare_epyccel(idiv_gt_add, 8, 2, 2)
    compare_epyccel(idiv_gt_add, 16, 3, 4)
    # False
    compare_epyccel(idiv_gt_add, 10, 3, 2)
    compare_epyccel(idiv_gt_add, 8, 2, 3)
    compare_epyccel(idiv_gt_add, 16, 3, 5)
