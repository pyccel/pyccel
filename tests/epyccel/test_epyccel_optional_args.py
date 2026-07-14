# pylint: disable=missing-function-docstring, missing-module-docstring

import numpy as np
import pytest

from pyccel import epyccel
from modules import epyccel_optional_args
from limits import RTOL, ATOL
from utilities import epyccel_module_with_fallback


@pytest.fixture(scope="module")
def epyc_epyccel_optional_args_mod(language):
    return epyccel_module_with_fallback(epyccel_optional_args, language)



# ------------------------------------------------------------------------------
def test_f1(epyc_epyccel_optional_args_mod):
    f1 = epyccel_optional_args.f1
    f = epyc_epyccel_optional_args_mod.f1

    # ...
    assert f(2) == f1(2)
    assert f() == f1()
    assert f(None) == f1(None)
    assert f(0) == f1(0)
    if epyc_epyccel_optional_args_mod.language != "python":
        with pytest.raises(TypeError):
            f(3.5)
    # ...


# ------------------------------------------------------------------------------
def test_f2(epyc_epyccel_optional_args_mod):
    f2 = epyccel_optional_args.f2
    f = epyc_epyccel_optional_args_mod.f2

    # ...
    assert np.isclose(f(2.0), f2(2.0), rtol=RTOL, atol=ATOL)
    assert np.isclose(f(), f2(), rtol=RTOL, atol=ATOL)
    assert np.isclose(f(None), f2(None), rtol=RTOL, atol=ATOL)
    assert np.isclose(f(0.0), f2(0.0), rtol=RTOL, atol=ATOL)
    if epyc_epyccel_optional_args_mod.language != "python":
        with pytest.raises(TypeError):
            f(3)
    # ...


# ------------------------------------------------------------------------------
def test_f3(epyc_epyccel_optional_args_mod):
    f3 = epyccel_optional_args.f3
    f = epyc_epyccel_optional_args_mod.f3

    # ...
    assert np.isclose(f(complex(1, 2.2)), f3(complex(1, 2.2)), rtol=RTOL, atol=ATOL)
    assert np.isclose(f(), f3(), rtol=RTOL, atol=ATOL)
    assert np.isclose(f(None), f3(None), rtol=RTOL, atol=ATOL)
    if epyc_epyccel_optional_args_mod.language != "python":
        with pytest.raises(TypeError):
            f(3.5)
    # ...


# ------------------------------------------------------------------------------
def test_f4(epyc_epyccel_optional_args_mod):
    f4 = epyccel_optional_args.f4
    f = epyc_epyccel_optional_args_mod.f4

    # ...
    assert f(True) == f4(True)
    assert f() == f4()
    assert f(None) == f4(None)
    assert f(False) == f4(False)
    if epyc_epyccel_optional_args_mod.language != "python":
        with pytest.raises(TypeError):
            f(3.5)
    # ...


# ------------------------------------------------------------------------------
def test_f5(language):
    import modules.Module_3 as mod

    modnew = epyccel(mod, language=language)

    # ...
    assert mod.func(1) == modnew.func(1)
    assert mod.func() == modnew.func()
    assert mod.func(None) == modnew.func(None)
    assert mod.func(0) == modnew.func(0)


# ------------------------------------------------------------------------------
def test_f6(language):
    import modules.Module_4 as mod

    modnew = epyccel(mod, language=language)

    # ...
    assert mod.call_optional_1() == modnew.call_optional_1()
    assert mod.call_optional_2(None) == modnew.call_optional_2(None)
    assert mod.call_optional_2(0) == modnew.call_optional_2(0)
    assert mod.call_optional_2() == modnew.call_optional_2()
    assert mod.optional_func_call() == modnew.optional_func_call()


# ------------------------------------------------------------------------------
def test_f7(epyc_epyccel_optional_args_mod):
    assert epyccel_optional_args.call_optional_1(3) == epyc_epyccel_optional_args_mod.call_optional_1(3)
    assert epyccel_optional_args.call_optional_2()  == epyc_epyccel_optional_args_mod.call_optional_2()
    assert epyccel_optional_args.call_optional_3(3) == epyc_epyccel_optional_args_mod.call_optional_3(3)


# ------------------------------------------------------------------------------
def test_f9(epyc_epyccel_optional_args_mod):
    assert epyccel_optional_args.call_optional_4(3) == epyc_epyccel_optional_args_mod.call_optional_4(3)
    assert epyccel_optional_args.call_optional_5(3) == epyc_epyccel_optional_args_mod.call_optional_5(3)
    assert epyccel_optional_args.call_optional_6() == epyc_epyccel_optional_args_mod.call_optional_6()
    assert epyccel_optional_args.call_optional_7() == epyc_epyccel_optional_args_mod.call_optional_7()
    assert epyccel_optional_args.call_optional_8() == epyc_epyccel_optional_args_mod.call_optional_8()


# ------------------------------------------------------------------------------
def test_f10(epyc_epyccel_optional_args_mod):
    assert epyccel_optional_args.call_optional_9() == epyc_epyccel_optional_args_mod.call_optional_9()
    assert epyccel_optional_args.call_optional_10() == epyc_epyccel_optional_args_mod.call_optional_10()


# ------------------------------------------------------------------------------
def test_f11(epyc_epyccel_optional_args_mod):
    assert epyccel_optional_args.call_optional_11() == epyc_epyccel_optional_args_mod.call_optional_11()
    assert epyccel_optional_args.call_optional_12() == epyc_epyccel_optional_args_mod.call_optional_12()


# ------------------------------------------------------------------------------
def test_optional_args_1d(epyc_epyccel_optional_args_mod):
    f12 = epyccel_optional_args.f12
    f = epyc_epyccel_optional_args_mod.f12

    x1 = np.array([1, 2, 3], dtype=int)
    x2 = np.copy(x1)
    f(x1)
    f12(x2)

    # ...
    assert np.array_equal(x1, x2)

    if epyc_epyccel_optional_args_mod.language != "python":
        with pytest.raises(TypeError):
            f(x1, 3)


# ------------------------------------------------------------------------------
def test_optional_2d_F(epyc_epyccel_optional_args_mod):
    f13 = epyccel_optional_args.f13
    f = epyc_epyccel_optional_args_mod.f13

    x1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32, order="F")
    x2 = np.copy(x1)
    f(x1)
    f13(x2)

    # ...
    assert np.array_equal(x1, x2)

    if epyc_epyccel_optional_args_mod.language != "python":
        with pytest.raises(TypeError):
            f(x1, 3.5)


# ------------------------------------------------------------------------------


def test_f14(epyc_epyccel_optional_args_mod):
    f14 = epyccel_optional_args.f14
    f = epyc_epyccel_optional_args_mod.f14

    # ...
    assert f(2, 7) == f14(2, 7)
    assert f() == f14()
    assert f(6) == f14(6)
    assert f(y=0) == f14(y=0)
    # ...
