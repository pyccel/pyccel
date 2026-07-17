# pylint: disable=missing-function-docstring, missing-module-docstring
import pytest
from modules import epyccel_pyc_math
from numpy import isclose
from numpy.random import randint, uniform


from epyccel_utilities import epyccel_module_with_fallback
from tolerances import ATOL, RTOL


@pytest.fixture(scope="module")
def epyc_epyccel_pyc_math_mod(language):
    return epyccel_module_with_fallback(epyccel_pyc_math, language)


# -----------------------------------------------------------------------------


def test_call_gcd(epyc_epyccel_pyc_math_mod):
    call_gcd = epyccel_pyc_math.call_gcd
    f = epyc_epyccel_pyc_math_mod.call_gcd
    x = randint(0, 1e9)
    y = randint(0, 1e9)

    assert f(x, y) == call_gcd(x, y)
    assert f(-x, y) == call_gcd(-x, y)
    assert f(x, -y) == call_gcd(x, -y)
    assert f(-x, -y) == call_gcd(-x, -y)


# -----------------------------------------------------------------------------


def test_call_factorial(epyc_epyccel_pyc_math_mod):
    call_factorial = epyccel_pyc_math.call_factorial
    f = epyc_epyccel_pyc_math_mod.call_factorial
    x = randint(10)

    assert f(x) == call_factorial(x)


# -----------------------------------------------------------------------------


def test_call_lcm(epyc_epyccel_pyc_math_mod):
    call_lcm = epyccel_pyc_math.call_lcm
    f = epyc_epyccel_pyc_math_mod.call_lcm
    x = randint(0, 1e4)
    y = randint(0, 1e5)

    assert f(x, y) == call_lcm(x, y)
    assert f(-x, y) == call_lcm(-x, y)
    assert f(x, -y) == call_lcm(x, -y)
    assert f(-x, -y) == call_lcm(-x, -y)


# -----------------------------------------------------------------------------


def test_call_radians(epyc_epyccel_pyc_math_mod):
    call_radians = epyccel_pyc_math.call_radians
    f = epyc_epyccel_pyc_math_mod.call_radians
    x = uniform(low=0.0, high=1e6)

    assert isclose(f(x), call_radians(x), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x), call_radians(-x), rtol=RTOL, atol=ATOL)


# -----------------------------------------------------------------------------


def test_call_degrees(epyc_epyccel_pyc_math_mod):
    call_degrees = epyccel_pyc_math.call_degrees
    f = epyc_epyccel_pyc_math_mod.call_degrees
    x = uniform(low=0.0, high=1e6)

    assert isclose(f(x), call_degrees(x), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x), call_degrees(-x), rtol=RTOL, atol=ATOL)


# -----------------------------------------------------------------------------


def test_call_degrees_i(epyc_epyccel_pyc_math_mod):
    call_degrees_i = epyccel_pyc_math.call_degrees_i
    f = epyc_epyccel_pyc_math_mod.call_degrees_i
    x = randint(1e6)

    assert isclose(f(x), call_degrees_i(x), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x), call_degrees_i(-x), rtol=RTOL, atol=ATOL)


# -----------------------------------------------------------------------------
