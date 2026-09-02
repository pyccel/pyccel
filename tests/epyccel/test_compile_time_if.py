# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np
import pytest
from modules import compile_time_if

from epyccel_utilities import epyccel_module_with_fallback


@pytest.fixture(scope="module")
def epyc_compile_time_if_mod(language):
    return epyccel_module_with_fallback(compile_time_if, language)


def test_rank_differentiation_1(epyc_compile_time_if_mod):
    f = compile_time_if.rank_differentiation_1
    x = np.arange(10, dtype=int)
    y = np.array(np.reshape(x[::-1], (2, 5)), dtype=int)

    f_epyc = epyc_compile_time_if_mod.rank_differentiation_1
    assert f_epyc(x) == f(x)
    assert f_epyc(y) == f(y)


def test_rank_differentiation_2(epyc_compile_time_if_mod):
    f = compile_time_if.rank_differentiation_2
    x = np.arange(10, dtype=int)
    y = np.array(np.reshape(x[::-1], (2, 5)), dtype=int)

    f_epyc = epyc_compile_time_if_mod.rank_differentiation_2
    assert f_epyc(x) == f(x)
    assert f_epyc(y) == f(y)


def test_type_differentiation(epyc_compile_time_if_mod):
    f = compile_time_if.type_differentiation
    f_epyc = epyc_compile_time_if_mod.type_differentiation
    assert f_epyc(3) == f(3)
    assert isinstance(f_epyc(3), type(f(3)))
    assert f_epyc(4.0) == f(4.0)
    assert isinstance(f_epyc(4.0), type(f(4.0)))
