from pyccel.epyccel import epyccel
import numpy as np
from conftest       import *
import modules.highorder_functions as mod

def test_int_1():
    modnew = epyccel(mod, language = "c", verbose = True)

    x_expected = mod.test_int_1()
    x = modnew.test_int_1() 
    assert x == x_expected

def test_int_int_1():
    modnew = epyccel(mod, language = "c", verbose = True)

    x_expected = mod.test_int_int_1()
    x = modnew.test_int_int_1() 
    assert x == x_expected

def test_real_1():
    modnew = epyccel(mod, language = "c", verbose = True)

    x_expected = mod.test_real_1()
    x = modnew.test_real_1() 
    assert x == x_expected

def test_real_2():
    modnew = epyccel(mod, language = "c", verbose = True)

    x_expected = mod.test_real_2()
    x = modnew.test_real_2() 
    assert x == x_expected

def test_real_real_int():
    modnew = epyccel(mod, language = "c", verbose = True)

    x_expected = mod.test_real_real_int()
    x = modnew.test_real_real_int() 
    assert x == x_expected


