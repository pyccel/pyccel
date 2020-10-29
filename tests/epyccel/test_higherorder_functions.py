 # pylint: disable=missing-function-docstring, missing-module-docstring/
import pytest
from pyccel.epyccel import epyccel
import modules.highorder_functions as mod

@pytest.mark.c
def test_int_1():
    modnew = epyccel(mod, language = "c", verbose = True)

    x_expected = mod.test_int_1()
    x = modnew.test_int_1()
    assert x == x_expected

@pytest.mark.c
def test_int_int_1():
    modnew = epyccel(mod, language = "c", verbose = True)

    x_expected = mod.test_int_int_1()
    x = modnew.test_int_int_1()
    assert x == x_expected

@pytest.mark.c
def test_real_1():
    modnew = epyccel(mod, language = "c", verbose = True)

    x_expected = mod.test_real_1()
    x = modnew.test_real_1()
    assert x == x_expected

@pytest.mark.c
def test_real_2():
    modnew = epyccel(mod, language = "c", verbose = True)

    x_expected = mod.test_real_2()
    x = modnew.test_real_2()
    assert x == x_expected

@pytest.mark.c
def test_real_3():
    modnew = epyccel(mod, language = "c", verbose = True)

    x_expected = mod.test_real_3()
    x = modnew.test_real_3()
    assert x == x_expected

@pytest.mark.c
def test_valuedarg_1():
    modnew = epyccel(mod, language = "c", verbose = True)

    x_expected = mod.test_valuedarg_1()
    x = modnew.test_valuedarg_1()
    assert x == x_expected

@pytest.mark.c
def test_real_real_int_1():
    modnew = epyccel(mod, language = "c", verbose = True)

    x_expected = mod.test_real_real_int_1()
    x = modnew.test_real_real_int_1()
    assert x == x_expected

@pytest.mark.c
def test_real_4():
    modnew = epyccel(mod, language = "c", verbose = True)

    x_expected = mod.test_real_4()
    x = modnew.test_real_4()
    assert x == x_expected

@pytest.mark.c
def test_valuedarg_2():
    modnew = epyccel(mod, language = "c", verbose = True)

    x_expected = mod.test_valuedarg_2()
    x = modnew.test_valuedarg_2()
    assert x == x_expected

@pytest.mark.c
def test_real_real_int_2():
    modnew = epyccel(mod, language = "c", verbose = True)

    x_expected = mod.test_real_real_int_2()
    x = modnew.test_real_real_int_2()
    assert x == x_expected

