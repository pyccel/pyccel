# pylint: disable=missing-function-docstring, missing-module-docstring
import sys

import modules.complex_func as mod
import numpy as np
import pytest
from numpy.random import rand, randint
from utilities import epyccel_module_with_fallback

from pyccel import epyccel


@pytest.fixture(scope="module")
def epyc_mod(language):
    return epyccel_module_with_fallback(mod, language)


# ==============================================================================

ATOL = 1e-15
RTOL = 2e-14

# Determine whether some unit tests should be skipped with Python >= 3.14 to
# avoid annoying deprecation warnings dealing with the complex() constructor:
# https://docs.python.org/3/deprecations/index.html#pending-removal-in-future-versions
deprecation = sys.version_info >= (3, 14)
deprecation_reason = "Since Python 3.14 complex() requires real arguments"

# ==============================================================================


@pytest.mark.parametrize(
    "f_name",
    [
        "create_complex_literal__int_int",
        "create_complex_literal__int_float",
        "create_complex_literal__float_int",
        "create_complex_literal__float_float",
        "cast_complex_literal",
    ],
)
def test_create_complex_literal(f_name, epyc_mod):
    f = getattr(mod, f_name)
    f_epyc = getattr(epyc_mod, f_name)
    assert f_epyc() == f()


@pytest.mark.skipif(deprecation, reason=deprecation_reason)
@pytest.mark.parametrize(
    "f_name",
    [
        "create_complex_literal__int_complex",
        "create_complex_literal__float_complex",
        "create_complex_literal__complex_int",
        "create_complex_literal__complex_float",
        "create_complex_literal__complex_complex",
    ],
)
def test_create_complex_literal_old(f_name, epyc_mod):
    f = getattr(mod, f_name)
    f_epyc = getattr(epyc_mod, f_name)
    assert f_epyc() == f()


# ==============================================================================
def test_create_complex_var__int_int(epyc_mod):
    f = mod.create_complex_var__int_int
    f_epyc = epyc_mod.create_complex_var__int_int

    a = randint(100)
    b = randint(100)
    assert f_epyc(a, b) == f(a, b)


@pytest.mark.skipif(deprecation, reason=deprecation_reason)
def test_create_complex_var__int_complex(epyc_mod):
    f = mod.create_complex_var__int_complex
    f_epyc = epyc_mod.create_complex_var__int_complex

    a = randint(100)
    b = complex(randint(100), randint(100))
    assert f_epyc(a, b) == f(a, b)


@pytest.mark.skipif(deprecation, reason=deprecation_reason)
def test_create_complex_var__complex_float(epyc_mod):
    f = mod.create_complex_var__complex_float
    f_epyc = epyc_mod.create_complex_var__complex_float

    a = complex(randint(100), randint(100))
    b = rand() * 100
    assert np.allclose(f_epyc(a, b), f(a, b), rtol=RTOL, atol=ATOL)


@pytest.mark.skipif(deprecation, reason=deprecation_reason)
def test_create_complex_var__complex_complex(epyc_mod):
    f = mod.create_complex_var__complex_complex
    f_epyc = epyc_mod.create_complex_var__complex_complex

    a = complex(randint(100), randint(100))
    b = complex(randint(100), randint(100))
    assert np.allclose(f_epyc(a, b), f(a, b), rtol=RTOL, atol=ATOL)


def test_create_complex__int_int(epyc_mod):
    f = mod.create_complex__int_int
    f_epyc = epyc_mod.create_complex__int_int

    a = randint(100)
    assert f_epyc(a) == f(a)


def test_create_complex_0__int_int(epyc_mod):
    f = mod.create_complex_0__int_int
    f_epyc = epyc_mod.create_complex_0__int_int

    a = randint(100)
    assert f_epyc(a) == f(a)


def test_create_complex__float_float(epyc_mod):
    f = mod.create_complex__float_float
    f_epyc = epyc_mod.create_complex__float_float

    a = rand() * 100
    assert np.allclose(f_epyc(a), f(a), rtol=RTOL, atol=ATOL)


def test_create_complex_0__float_float(epyc_mod):
    f = mod.create_complex_0__float_float
    f_epyc = epyc_mod.create_complex_0__float_float

    a = rand() * 100
    assert np.allclose(f_epyc(a), f(a), rtol=RTOL, atol=ATOL)


@pytest.mark.skipif(deprecation, reason=deprecation_reason)
def test_create_complex__complex_complex(epyc_mod):
    f = mod.create_complex__complex_complex
    f_epyc = epyc_mod.create_complex__complex_complex

    a = complex(randint(100), randint(100))
    assert np.allclose(f_epyc(a), f(a), rtol=RTOL, atol=ATOL)


def test_cast_complex_1(epyc_mod):
    f = mod.cast_complex_1
    f_epyc = epyc_mod.cast_complex_1

    a = np.complex64(complex(randint(100), randint(100)))
    assert np.allclose(f_epyc(a), f(a), rtol=1e-7, atol=1e-8)


def test_cast_complex_2(epyc_mod):
    f = mod.cast_complex_2
    f_epyc = epyc_mod.cast_complex_2

    a = np.complex128(complex(randint(100), randint(100)))
    assert np.allclose(f_epyc(a), f(a), rtol=RTOL, atol=ATOL)


def test_cast_float_complex(epyc_mod):
    f = mod.cast_float_complex
    f_epyc = epyc_mod.cast_float_complex

    a = rand() * 100
    b = complex(randint(100), randint(100))
    assert np.allclose(f_epyc(a, b), f(a, b), rtol=RTOL, atol=ATOL)
