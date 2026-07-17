# pylint: disable=missing-function-docstring, missing-module-docstring
import os

import pytest
from modules import epyccel_mod
from numpy import allclose
from numpy.random import randint, uniform

from epyccel_utilities import epyccel_module_with_fallback
from tolerances import ATOL, RTOL


@pytest.fixture(scope="module")
def epyc_modulo_mod(language):
    return epyccel_module_with_fallback(epyccel_mod, language)


# Relative and absolute tolerances for array comparisons in the form
# numpy.isclose(a, b, rtol, atol). Intel seems to use a different algorithm
if os.environ.get("PYCCEL_DEFAULT_COMPILER", "GNU") == "intel":
    RTOL = 1e-10
    ATOL = 1e-10


def test_modulo_int_int(epyc_modulo_mod):
    modulo_i_i = epyccel_mod.modulo_i_i
    f = epyc_modulo_mod.modulo_i_i
    x = randint(0, 1e6)
    y = randint(1, 1e6)

    f_output = f(x, y)
    modulo_i_i_output = modulo_i_i(x, y)
    assert modulo_i_i_output == f_output
    assert isinstance(f_output, type(modulo_i_i_output))


def test_modulo_real_real(epyc_modulo_mod):
    modulo_r_r = epyccel_mod.modulo_r_r
    f = epyc_modulo_mod.modulo_r_r
    x = uniform(low=0, high=1e6)
    y = uniform(low=1, high=1e2)

    f_output = f(x, y)
    modulo_r_r_output = modulo_r_r(x, y)
    assert allclose(f_output, modulo_r_r_output, rtol=RTOL, atol=ATOL)
    assert isinstance(f_output, type(modulo_r_r_output))


def test_modulo_real_int(epyc_modulo_mod):
    modulo_r_i = epyccel_mod.modulo_r_i
    f = epyc_modulo_mod.modulo_r_i
    x = uniform(low=0, high=1e6)
    y = randint(low=1, high=1e6)

    f_output = f(x, y)
    modulo_r_i_output = modulo_r_i(x, y)
    assert allclose(f_output, modulo_r_i_output, rtol=RTOL, atol=ATOL)
    assert isinstance(f_output, type(modulo_r_i_output))


def test_modulo_int_real(epyc_modulo_mod):
    modulo_i_r = epyccel_mod.modulo_i_r
    f = epyc_modulo_mod.modulo_i_r
    x = randint(0, 1e6)
    y = uniform(low=1, high=1e2)

    f_output = f(x, y)
    modulo_i_r_output = modulo_i_r(x, y)
    assert allclose(f_output, modulo_i_r_output, rtol=RTOL, atol=ATOL)
    assert isinstance(f_output, type(modulo_i_r_output))


def test_modulo_multiple(epyc_modulo_mod):
    modulo_multiple = epyccel_mod.modulo_multiple
    f = epyc_modulo_mod.modulo_multiple
    x = randint(0, 1e6)
    y = uniform(low=1, high=1e4)
    z = randint(low=1, high=1e2)

    assert allclose(f(x, y, z), modulo_multiple(x, y, z), rtol=RTOL, atol=ATOL)
    assert isinstance(f(x, y, z), type(modulo_multiple(x, y, z)))
