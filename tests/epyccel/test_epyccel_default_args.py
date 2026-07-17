# pylint: disable=missing-function-docstring, missing-module-docstring
# coding: utf-8
from typing import TypeVar

import numpy as np
import pytest
from modules import epyccel_default_args
from epyccel_utilities import epyccel_module_with_fallback

from pyccel import epyccel


@pytest.fixture(scope="module")
def epyc_epyccel_default_args_mod(language):
    return epyccel_module_with_fallback(epyccel_default_args, language)


# ------------------------------------------------------------------------------
def test_f1(epyc_epyccel_default_args_mod):
    f1 = epyccel_default_args.f1
    f = epyc_epyccel_default_args_mod.f1

    # ...
    assert f(2) == f1(2)
    assert f() == f1()
    # ...


# ------------------------------------------------------------------------------
def test_f2(epyc_epyccel_default_args_mod):
    f5 = epyccel_default_args.f2
    f = epyc_epyccel_default_args_mod.f2

    # ...
    m1 = 3

    x = np.zeros(m1)
    f(x)

    x_expected = np.zeros(m1)
    f5(x_expected)

    assert np.allclose(x, x_expected, rtol=2e-14, atol=1e-15)
    # ...

    f(x, m1=m1)

    f5(x_expected, m1)

    assert np.allclose(x, x_expected, rtol=2e-14, atol=1e-15)


# ------------------------------------------------------------------------------
def test_f3(epyc_epyccel_default_args_mod):
    f3 = epyccel_default_args.f3
    f = epyc_epyccel_default_args_mod.f3

    # ...
    assert f(19.2, 6.7) == f3(19.2, 6.7)
    assert f(4.5) == f3(4.5)
    assert f(y=8.2) == f3(y=8.2)
    assert f() == f3()
    # ...


# ------------------------------------------------------------------------------
def test_f4(epyc_epyccel_default_args_mod):
    f4 = epyccel_default_args.f4
    f = epyc_epyccel_default_args_mod.f4

    # ...
    assert f(True) == f4(True)
    assert f(False) == f4(False)
    assert f() == f4()
    # ...


# ------------------------------------------------------------------------------
def test_f5(epyc_epyccel_default_args_mod):
    f5 = epyccel_default_args.f5_f5
    f = epyc_epyccel_default_args_mod.f5_f5

    # ...
    assert f(2.9 + 3j) == f5(2.9 + 3j)
    assert f() == f5()
    # ...


# ------------------------------------------------------------------------------
def test_changed_precision_arguments(language):
    import modules.Module_8 as mod

    modnew = epyccel(mod, language=language)

    assert mod.get_f() == modnew.get_f()
    assert mod.get_g() == modnew.get_g()


# ------------------------------------------------------------------------------
def test_default_interface_value(epyc_epyccel_default_args_mod):
    max_abs = epyccel_default_args.max_abs
    f = epyc_epyccel_default_args_mod.max_abs

    # ...
    assert f(2.9) == max_abs(2.9)
    assert f(2.9, 2.3) == max_abs(2.9, 2.3)
    # ...
    assert f(2.9 + 3j) == max_abs(2.9 + 3j)
    assert f(2.9 + 3j, 2.3 + 4j) == max_abs(2.9 + 3j, 2.3 + 4j)
    # ...
