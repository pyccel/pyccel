 # pylint: disable=missing-function-docstring, missing-module-docstring/
import pytest
import modules.highorder_functions as mod
from pyccel.epyccel import epyccel

@pytest.fixture( params=[
        pytest.param("fortran", marks = [
            pytest.mark.fortran,
            pytest.mark.skip]),
        pytest.param("c", marks = [
            pytest.mark.c]
        )
    ]
)
def language(request):
    return request.param

def test_int_1(language):
    modnew = epyccel(mod, language = language)

    x_expected = mod.test_int_1()
    x = modnew.test_int_1()
    assert x == x_expected

def test_int_int_1(language):
    modnew = epyccel(mod, language = language)

    x_expected = mod.test_int_int_1()
    x = modnew.test_int_int_1()
    assert x == x_expected

def test_real_1(language):
    modnew = epyccel(mod, language = language)

    x_expected = mod.test_real_1()
    x = modnew.test_real_1()
    assert x == x_expected

def test_real_2(language):
    modnew = epyccel(mod, language = language)

    x_expected = mod.test_real_2()
    x = modnew.test_real_2()
    assert x == x_expected

def test_real_3(language):
    modnew = epyccel(mod, language = language)

    x_expected = mod.test_real_3()
    x = modnew.test_real_3()
    assert x == x_expected

def test_valuedarg_1(language):
    modnew = epyccel(mod, language = language)

    x_expected = mod.test_valuedarg_1()
    x = modnew.test_valuedarg_1()
    assert x == x_expected

def test_real_real_int_1(language):
    modnew = epyccel(mod, language = language)

    x_expected = mod.test_real_real_int_1()
    x = modnew.test_real_real_int_1()
    assert x == x_expected

def test_real_4(language):
    modnew = epyccel(mod, language = language)

    x_expected = mod.test_real_4()
    x = modnew.test_real_4()
    assert x == x_expected

def test_valuedarg_2(language):
    modnew = epyccel(mod, language = language)

    x_expected = mod.test_valuedarg_2()
    x = modnew.test_valuedarg_2()
    assert x == x_expected

def test_real_real_int_2(language):
    modnew = epyccel(mod, language = language)

    x_expected = mod.test_real_real_int_2()
    x = modnew.test_real_real_int_2()
    assert x == x_expected

