# pylint: disable=missing-function-docstring, missing-module-docstring/

import pytest

from pyccel.epyccel import epyccel
from pyccel.decorators import types

#------------------------------------------------------------------------------
def test_f1(language):
    @types('int')
    def f1(x = None):
        if x is None :
            return 5
        return x + 5

    f = epyccel(f1, language = language)

    # ...
    assert f(2) == f1(2)
    assert f() == f1()
    assert f(None) == f1(None)
    assert f(0) == f1(0)
    # ...
#------------------------------------------------------------------------------
def test_f2(language):
    @types('real')
    def f2(x = None):
        if x is None :
            return 2.5
        return x + 2.5

    f = epyccel(f2, language = language)

    # ...
    assert f(2.0) == f2(2.0)
    assert f() == f2()
    assert f(None) == f2(None)
    assert f(0.0) == f2(0.0)
    # ...
#------------------------------------------------------------------------------
def test_f3(language):
    @types('complex')
    def f3(x = None):
        if x is None :
            return complex(2, 5.2)
        return x + complex(2.5, 2)

    f = epyccel(f3, language = language)

    # ...
    assert f(complex(1, 2.2)) == f3(complex(1, 2.2))
    assert f() == f3()
    assert f(None) == f3(None)
    # ...
#------------------------------------------------------------------------------
def test_f4(language):
    @types('bool')
    def f4(x = None):
        if x is None :
            return True
        return False

    f = epyccel(f4, language = language)

    # ...
    assert f(True) == f4(True)
    assert f() == f4()
    assert f(None) == f4(None)
    assert f(False) == f4(False)
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
#------------------------------------------------------------------------------
def test_f7(language):
    import modules.Module_5 as mod

    modnew = epyccel(mod, language = language)

    # ...
    assert mod.call_optional_1(3) == modnew.call_optional_1(3)
    assert mod.call_optional_2() == modnew.call_optional_2()
    assert mod.call_optional_3(3) == modnew.call_optional_3(3)

#------------------------------------------------------------------------------
def test_f9(language):
    import modules.Module_5 as mod

    modnew = epyccel(mod, language = language)

    # ...
    assert mod.call_optional_4(3) == modnew.call_optional_4(3)
    assert mod.call_optional_5(3) == modnew.call_optional_5(3)
    assert mod.call_optional_6() == modnew.call_optional_6()
    assert mod.call_optional_7() == modnew.call_optional_7()
    assert mod.call_optional_8() == modnew.call_optional_8()

#------------------------------------------------------------------------------
def test_f10(language):
    import modules.Module_5 as mod

    modnew = epyccel(mod, language = language)

    # ...
    assert mod.call_optional_9() == modnew.call_optional_9()
    assert mod.call_optional_10() == modnew.call_optional_10()

#------------------------------------------------------------------------------
def test_f11(language):
    import modules.Module_5 as mod

    modnew = epyccel(mod, language = language)

    # ...
    assert mod.call_optional_11() == modnew.call_optional_11()
    assert mod.call_optional_12() == modnew.call_optional_12()
