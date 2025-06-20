# pylint: disable=missing-function-docstring, missing-module-docstring

import pytest
import numpy as np

from pyccel import epyccel

RTOL = 2e-14
ATOL = 1e-15

@pytest.fixture(scope="module")
def Module_5(language):
    import modules.Module_5 as mod

    modnew = epyccel(mod, language = language)
    return mod, modnew

#------------------------------------------------------------------------------
def test_f1(language):
    def f1(x  : 'int' =  None):
        if x is None :
            return 5
        return x + 5

    f = epyccel(f1, language = language)

    # ...
    assert f(2) == f1(2)
    assert f() == f1()
    assert f(None) == f1(None)
    assert f(0) == f1(0)
    if language != 'python':
        with pytest.raises(TypeError):
            f(3.5)
    # ...
#------------------------------------------------------------------------------
def test_f2(language):
    def f2(x  : 'float' =  None):
        if x is None :
            return 2.5
        return x + 2.5

    f = epyccel(f2, language = language)

    # ...
    assert np.isclose(f(2.0), f2(2.0), rtol=RTOL, atol=ATOL)
    assert np.isclose(f(), f2(), rtol=RTOL, atol=ATOL)
    assert np.isclose(f(None), f2(None), rtol=RTOL, atol=ATOL)
    assert np.isclose(f(0.0), f2(0.0), rtol=RTOL, atol=ATOL)
    if language != 'python':
        with pytest.raises(TypeError):
            f(3)
    # ...
#------------------------------------------------------------------------------
def test_f3(language):
    def f3(x  : 'complex' =  None):
        if x is None :
            return complex(2, 5.2)
        return x + complex(2.5, 2)

    f = epyccel(f3, language = language)

    # ...
    assert np.isclose(f(complex(1, 2.2)), f3(complex(1, 2.2)), rtol=RTOL, atol=ATOL)
    assert np.isclose(f(), f3(), rtol=RTOL, atol=ATOL)
    assert np.isclose(f(None), f3(None), rtol=RTOL, atol=ATOL)
    if language != 'python':
        with pytest.raises(TypeError):
            f(3.5)
    # ...
#------------------------------------------------------------------------------
def test_f4(language):
    def f4(x  : 'bool' =  None):
        if x is None :
            return True
        return False

    f = epyccel(f4, language = language)

    # ...
    assert f(True) == f4(True)
    assert f() == f4()
    assert f(None) == f4(None)
    assert f(False) == f4(False)
    if language != 'python':
        with pytest.raises(TypeError):
            f(3.5)
    # ...
#------------------------------------------------------------------------------
def test_f5(language):
    import modules.Module_3 as mod

    modnew = epyccel(mod, language = language)

    # ...
    assert mod.func(1) == modnew.func(1)
    assert mod.func() == modnew.func()
    assert mod.func(None) == modnew.func(None)
    assert mod.func(0) == modnew.func(0)

#------------------------------------------------------------------------------
def test_f6(language):
    import modules.Module_4 as mod

    modnew = epyccel(mod, language = language)

    # ...
    assert mod.call_optional_1() == modnew.call_optional_1()
    assert mod.call_optional_2(None) == modnew.call_optional_2(None)
    assert mod.call_optional_2(0) == modnew.call_optional_2(0)
    assert mod.call_optional_2() == modnew.call_optional_2()
    assert mod.optional_func_call() == modnew.optional_func_call()
#------------------------------------------------------------------------------
def test_f7(Module_5):
    mod, modnew = Module_5

    # ...
    assert mod.call_optional_1(3) == modnew.call_optional_1(3)
    assert mod.call_optional_2() == modnew.call_optional_2()
    assert mod.call_optional_3(3) == modnew.call_optional_3(3)

#------------------------------------------------------------------------------
def test_f9(Module_5):
    mod, modnew = Module_5

    # ...
    assert mod.call_optional_4(3) == modnew.call_optional_4(3)
    assert mod.call_optional_5(3) == modnew.call_optional_5(3)
    assert mod.call_optional_6() == modnew.call_optional_6()
    assert mod.call_optional_7() == modnew.call_optional_7()
    assert mod.call_optional_8() == modnew.call_optional_8()

#------------------------------------------------------------------------------
def test_f10(Module_5):
    mod, modnew = Module_5

    # ...
    assert mod.call_optional_9() == modnew.call_optional_9()
    assert mod.call_optional_10() == modnew.call_optional_10()

#------------------------------------------------------------------------------
def test_f11(Module_5):
    mod, modnew = Module_5

    # ...
    assert mod.call_optional_11() == modnew.call_optional_11()
    assert mod.call_optional_12() == modnew.call_optional_12()

#------------------------------------------------------------------------------
def test_optional_args_1d(language):
    def f12(x : 'int[:]', y  : 'int[:]' =  None):
        if y is None:
            x[:] *= 2
        else :
            x[:] = x // y
    f = epyccel(f12, language = language)

    x1 = np.array( [1,2,3], dtype=int )
    x2 = np.copy(x1)
    f(x1)
    f12(x2)

    # ...
    assert np.array_equal(x1, x2)

    if language != 'python':
        with pytest.raises(TypeError):
            f(x1, 3)

#------------------------------------------------------------------------------
def test_optional_2d_F(language):
    def f13(x : 'int32[:,:](order=F)', y  : 'int32[:,:](order=F)' =  None):
        if y is None:
            x[:] *= 2
        else :
            x[:] = x // y
    f = epyccel(f13, language = language)

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32, order='F' )
    x2 = np.copy(x1)
    f(x1)
    f13 (x2)

    # ...
    assert np.array_equal(x1, x2)

    if language != 'python':
        with pytest.raises(TypeError):
            f(x1, 3.5)

#------------------------------------------------------------------------------

def test_f14(language):
    def f14(x  : 'int' =  None, y  : 'int' =  None):
        if x is None :
            x = 3
        if y is not None :
            y = 4
        else:
            y = 5
        return x + y

    f = epyccel(f14, language = language)

    # ...
    assert f(2,7) == f14(2,7)
    assert f() == f14()
    assert f(6) == f14(6)
    assert f(y=0) == f14(y=0)
    # ...
