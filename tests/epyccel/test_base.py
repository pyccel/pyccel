import pytest
import numpy as np

from pyccel.epyccel import epyccel
from modules import base
from conftest       import *


def compare_epyccel(f, *args, language='fortran'):
    f2 = epyccel(f, language=language)
    out1 = f(*args)
    out2 = f2(*args)
    assert np.equal(out1, out2)

def test_is_false():
    compare_epyccel(base.is_false, True)
    compare_epyccel(base.is_false, False)

def test_is_true():
    compare_epyccel(base.is_true, True)
    compare_epyccel(base.is_true, False)

def test_compare_is():
    compare_epyccel(base.compare_is, True, True)
    compare_epyccel(base.compare_is, True, False)
    compare_epyccel(base.compare_is, False, True)
    compare_epyccel(base.compare_is, False, False)

def test_compare_is_not():
    compare_epyccel(base.compare_is_not, True, True)
    compare_epyccel(base.compare_is_not, True, False)
    compare_epyccel(base.compare_is_not, False, True)
    compare_epyccel(base.compare_is_not, False, False)

def test_compare_is_int():
    compare_epyccel(base.compare_is_int, True, 1)
    compare_epyccel(base.compare_is_int, True, 0)
    compare_epyccel(base.compare_is_int, False, 1)
    compare_epyccel(base.compare_is_int, False, 0)

def test_compare_is_not_int():
    compare_epyccel(base.compare_is_not_int, True, 1)
    compare_epyccel(base.compare_is_not_int, True, 0)
    compare_epyccel(base.compare_is_not_int, False, 1)
    compare_epyccel(base.compare_is_not_int, False, 0)

def test_not_false():
    compare_epyccel(base.not_false, True)
    compare_epyccel(base.not_false, False)

def test_not_true():
    compare_epyccel(base.not_true, True)
    compare_epyccel(base.not_true, False)

def test_eq_false():
    compare_epyccel(base.eq_false, True)
    compare_epyccel(base.eq_false, False)

def test_eq_true():
    compare_epyccel(base.eq_true, True)
    compare_epyccel(base.eq_true, False)

def test_neq_false():
    compare_epyccel(base.eq_false, True)
    compare_epyccel(base.eq_false, False)

def test_neq_true():
    compare_epyccel(base.eq_true, True)
    compare_epyccel(base.eq_true, False)

def test_not():
    compare_epyccel(base.not_val, True)
    compare_epyccel(base.not_val, False)

@pytest.mark.xfail(reason="f2py does not support optional arguments https://github.com/numpy/numpy/issues/4013")
def test_compare_is_nil():
    compare_epyccel(base.is_nil, True, None)

@pytest.mark.xfail(reason="f2py does not support optional arguments https://github.com/numpy/numpy/issues/4013")
def test_compare_is_not_nil():
    compare_epyccel(base.is_not_nil, True, None)

def test_cast_int(language):
    compare_epyccel(base.cast_int, 4, language=language)
    compare_epyccel(base.cast_float_to_int, 4.5, language=language)

def test_cast_bool():
    compare_epyccel(base.cast_bool, True)

def test_compare_is_int_compare():
    compare_epyccel(base.compare_is_int_compare, 2, 2)
    compare_epyccel(base.compare_is_int_compare, 1, 2)

def test_cast_float(language):
    compare_epyccel(base.cast_float, 4.5, language=language)
    compare_epyccel(base.cast_int_to_float, 4, language=language)

def test_if_0_int():
    compare_epyccel(base.if_0_int, 22)
    compare_epyccel(base.if_0_int, 0)

def test_if_0_real():
    compare_epyccel(base.if_0_real, 22.3)
    compare_epyccel(base.if_0_real, 0.0)
