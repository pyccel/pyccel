 # pylint: disable=missing-function-docstring, missing-module-docstring/
import pytest
import numpy as np
import modules.generic_functions as mod
from pyccel.epyccel import epyccel

@pytest.fixture( params=[
        pytest.param("fortran", marks = [
            pytest.mark.fortran,
            pytest.mark.skip(reason="issue #512")]),
        pytest.param("c", marks = [
            pytest.mark.c]
        )
    ]
)
def language(request):
    return request.param
'''
def test_gen_1(language):
    modnew = epyccel(mod, language = language)
    x_expected = mod.tst_gen_1()
    x = modnew.tst_gen_1()
    assert np.array_equal(x, x_expected)

def test_gen_2(language):
    modnew = epyccel(mod, language = language)
    x_expected = mod.tst_gen_2()
    x = modnew.tst_gen_2()
    assert np.array_equal(x ,x_expected)

def test_gen_3(language):
    modnew = epyccel(mod, language = language)
    x_expected = mod.tst_gen_3()
    x = modnew.tst_gen_3()
    assert np.array_equal(x, x_expected)

def test_gen_4(language):
    modnew = epyccel(mod, language = language)
    x_expected = mod.tst_gen_4()
    x = modnew.tst_gen_4()
    assert np.array_equal(x, x_expected)

def test_gen_5(language):
    modnew = epyccel(mod, language = language)
    x_expected = mod.tst_gen_5()
    x = modnew.tst_gen_5()
    assert np.array_equal(x, x_expected)

def test_gen_6(language):
    modnew = epyccel(mod, language = language)
    x_expected = mod.tst_gen_6()
    x = modnew.tst_gen_6()
    assert np.array_equal(x, x_expected)

def test_gen_7(language):
    modnew = epyccel(mod, language = language)
    x_expected = mod.tst_gen_7()
    x = modnew.tst_gen_7()
    assert np.array_equal(x, x_expected)

def test_multi_heads_1(language):
    modnew = epyccel(mod, language = language)
    x_expected = mod.tst_multi_heads_1()
    x = modnew.tst_multi_heads_1()
    assert np.array_equal(x, x_expected)

def test_tmplt_1(language):
    modnew = epyccel(mod, language = language)
    x_expected = mod.tst_tmplt_1()
    x = modnew.tst_tmplt_1()
    assert np.array_equal(x, x_expected)

def test_multi_tmplt_1(language):
    modnew = epyccel(mod, language = language)
    x_expected = mod.tst_multi_tmplt_1()
    x = modnew.tst_multi_tmplt_1()
    assert np.array_equal(x, x_expected)

def test_tmplt_head_1(language):
    modnew = epyccel(mod, language = language)
    x_expected = mod.tst_tmplt_head_1()
    x = modnew.tst_tmplt_head_1()
    assert np.array_equal(x, x_expected)

def test_local_overide_1(language):
    modnew = epyccel(mod, language = language)
    x_expected = mod.tst_local_overide_1()
    x = modnew.tst_local_overide_1()
    assert np.array_equal(x, x_expected)

def test_tmplt_tmplt_1(language):
    modnew = epyccel(mod, language = language)
    x_expected = mod.tst_tmplt_tmplt_1()
    x = modnew.tst_tmplt_tmplt_1()
    assert np.array_equal(x, x_expected)

def test_tmplt_2(language):
    modnew = epyccel(mod, language = language)
    x_expected = mod.tst_tmplt_2()
    x = modnew.tst_tmplt_2()
    assert np.array_equal(x, x_expected)

def test_multi_tmplt_2(language):
    modnew = epyccel(mod, language = language)
    x_expected = mod.tst_multi_tmplt_2()
    x = modnew.tst_multi_tmplt_2()
    assert np.array_equal(x, x_expected)

def test_default_var_1(language):
    modnew = epyccel(mod, language = language)
    x_expected = mod.tst_default_var_1()
    x = modnew.tst_default_var_1()
    assert np.array_equal(x, x_expected)

def test_default_var_2(language):
    modnew = epyccel(mod, language = language)
    x_expected = mod.tst_default_var_2()
    x = modnew.tst_default_var_2()
    assert np.array_equal(x, x_expected)

def test_default_var_3(language):
    modnew = epyccel(mod, language = language)
    x_expected = mod.tst_default_var_3()
    x = modnew.tst_default_var_3()
    assert np.array_equal(x, x_expected)

def test_default_var_4(language):
    modnew = epyccel(mod, language = language)
    x_expected = mod.tst_default_var_4()
    x = modnew.tst_default_var_4()
    assert np.array_equal(x, x_expected)

def test_optional_var_1(language):
    modnew = epyccel(mod, language = language)
    x_expected = mod.tst_optional_var_1()
    x = modnew.tst_optional_var_1()
    assert np.array_equal(x, x_expected)

def test_optional_var_2(language):
    modnew = epyccel(mod, language = language)
    x_expected = mod.tst_optional_var_2()
    x = modnew.tst_optional_var_2()
    assert np.array_equal(x, x_expected)

def test_optional_var_3(language):
    modnew = epyccel(mod, language = language)
    x_expected = mod.tst_optional_var_3()
    x = modnew.tst_optional_var_3()
    assert np.array_equal(x, x_expected)

def test_optional_var_4(language):
    modnew = epyccel(mod, language = language)
    x_expected = mod.tst_optional_var_4()
    x = modnew.tst_optional_var_4()
    assert np.array_equal(x, x_expected)
'''
def test_int_types(language):
    f1 = epyccel(mod.int_types , language = language, verbose=True)
    f2 = mod.int_types

    assert f1(10, 5) == f2(10, 5)
    assert f1(np.int(15) , np.int(10)) == f2(np.int(15) , np.int(10))
    assert f1(np.int16(5), np.int16(4)) == f2(np.int16(5), np.int16(4))
    assert f1(np.int8(4), np.int8(7)) == f2(np.int8(4), np.int8(7))
    assert f1(np.int32(155), np.int32(177)) == f2(np.int32(155), np.int32(177))
    assert f1(np.int64(166), np.int64(255)) == f2(np.int64(166), np.int64(255))
