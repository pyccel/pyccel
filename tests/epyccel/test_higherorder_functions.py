# pylint: disable=missing-function-docstring, missing-module-docstring/
import pytest
import numpy as np
import modules.highorder_functions as mod
from pyccel.epyccel import epyccel

@pytest.fixture(scope="module")
def modnew(language):
    return epyccel(mod, language = language)

def test_int_1(modnew):
    x_expected = mod.test_int_1()
    x = modnew.test_int_1()
    assert x == x_expected

def test_int_int_1(modnew):
    x_expected = mod.test_int_int_1()
    x = modnew.test_int_int_1()
    assert x == x_expected

def test_real_1(modnew):
    x_expected = mod.test_real_1()
    x = modnew.test_real_1()
    assert x == x_expected

def test_real_2(modnew):
    x_expected = mod.test_real_2()
    x = modnew.test_real_2()
    assert x == x_expected

def test_real_3(modnew):
    x_expected = mod.test_real_3()
    x = modnew.test_real_3()
    assert x == x_expected

def test_valuedarg_1(modnew):
    x_expected = mod.test_valuedarg_1()
    x = modnew.test_valuedarg_1()
    assert x == x_expected

def test_real_real_int_1(modnew):
    x_expected = mod.test_real_real_int_1()
    x = modnew.test_real_real_int_1()
    assert x == x_expected

def test_real_4(modnew):
    x_expected = mod.test_real_4()
    x = modnew.test_real_4()
    assert x == x_expected

def test_valuedarg_2(modnew):
    x_expected = mod.test_valuedarg_2()
    x = modnew.test_valuedarg_2()
    assert x == x_expected

def test_real_real_int_2(modnew):
    x_expected = mod.test_real_real_int_2()
    x = modnew.test_real_real_int_2()
    assert x == x_expected

def test_euler(modnew):
    t0 = 0.0
    t1 = 2.0
    y0_l = np.array ( [ 5000., 100. ] )
    y0_p = y0_l.copy()
    n = 10

    modnew.euler_test(t0, t1, y0_l, n)
    mod.euler_test(t0, t1, y0_p, n)

    assert np.allclose(y0_l, y0_p)
