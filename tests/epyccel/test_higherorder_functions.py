# pylint: disable=missing-function-docstring, missing-module-docstring/
import pytest
import numpy as np
import modules.highorder_functions as mod
from pyccel.epyccel import epyccel

available_languages = {"fortran" : pytest.mark.fortran,
                       "c"       : pytest.mark.c,
                       "python"  : pytest.mark.python}

modules = {}

modparam = [
    pytest.param(l, marks=[pytest.mark.dependency(name="test_compile_module[%s]" % l), m])
    for l,m in available_languages.items()
]
testparam = [
    pytest.param(l, marks=[pytest.mark.dependency(depends=["test_compile_module[%s]" % l]), m])
    for l,m in available_languages.items()
]


@pytest.mark.parametrize( 'language', modparam )
def test_compile_module(language):
    modules[language] = epyccel(mod, language = language)

@pytest.mark.parametrize( 'language', testparam )
def test_int_1(language):
    modnew = modules[language]

    x_expected = mod.test_int_1()
    x = modnew.test_int_1()
    assert x == x_expected

@pytest.mark.parametrize( 'language', testparam )
def test_int_int_1(language):
    modnew = modules[language]

    x_expected = mod.test_int_int_1()
    x = modnew.test_int_int_1()
    assert x == x_expected

@pytest.mark.parametrize( 'language', testparam )
def test_real_1(language):
    modnew = modules[language]

    x_expected = mod.test_real_1()
    x = modnew.test_real_1()
    assert x == x_expected

@pytest.mark.parametrize( 'language', testparam )
def test_real_2(language):
    modnew = modules[language]

    x_expected = mod.test_real_2()
    x = modnew.test_real_2()
    assert x == x_expected

@pytest.mark.parametrize( 'language', testparam )
def test_real_3(language):
    modnew = modules[language]

    x_expected = mod.test_real_3()
    x = modnew.test_real_3()
    assert x == x_expected

@pytest.mark.parametrize( 'language', testparam )
def test_valuedarg_1(language):
    modnew = modules[language]

    x_expected = mod.test_valuedarg_1()
    x = modnew.test_valuedarg_1()
    assert x == x_expected

@pytest.mark.parametrize( 'language', testparam )
def test_real_real_int_1(language):
    modnew = modules[language]

    x_expected = mod.test_real_real_int_1()
    x = modnew.test_real_real_int_1()
    assert x == x_expected

@pytest.mark.parametrize( 'language', testparam )
def test_real_4(language):
    modnew = modules[language]

    x_expected = mod.test_real_4()
    x = modnew.test_real_4()
    assert x == x_expected

@pytest.mark.parametrize( 'language', testparam )
def test_valuedarg_2(language):
    modnew = modules[language]

    x_expected = mod.test_valuedarg_2()
    x = modnew.test_valuedarg_2()
    assert x == x_expected

@pytest.mark.parametrize( 'language', testparam )
def test_real_real_int_2(language):
    modnew = modules[language]

    x_expected = mod.test_real_real_int_2()
    x = modnew.test_real_real_int_2()
    assert x == x_expected

@pytest.mark.parametrize( 'language', testparam )
def test_euler(language):
    modnew = modules[language]

    t0 = 0.0
    t1 = 2.0
    y0_l = np.array ( [ 5000., 100. ] )
    y0_p = y0_l.copy()
    n = 10

    modnew.euler_test(t0, t1, y0_l, n)
    mod.euler_test(t0, t1, y0_p, n)

    assert np.allclose(y0_l, y0_p)
