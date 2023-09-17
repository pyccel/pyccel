# pylint: disable=missing-function-docstring, missing-module-docstring
import sys
import pytest
from numpy.random import randint, uniform
from numpy import isclose

from pyccel.epyccel import epyccel

RTOL = 2e-14
ATOL = 1e-15

# -----------------------------------------------------------------------------

def test_call_gcd(language):
    def call_gcd(x : int, y : int):
        from math import gcd
        return gcd(x, y)

    f = epyccel(call_gcd, language=language)
    x = randint(1e9)
    y = randint(1e9)

    assert(f(x,y) == call_gcd(x, y))

# -----------------------------------------------------------------------------

def test_call_factorial(language):
    def call_factorial(x : 'int'):
        from math import factorial
        return factorial(x)

    f = epyccel(call_factorial, language=language)
    x = randint(10)

    assert(f(x) == call_factorial(x))

# -----------------------------------------------------------------------------

# New in version 3.9.
@pytest.mark.skipif(sys.version_info < (3, 9), reason="requires python3.9 or higher")
def test_call_lcm(language):
    def call_lcm(x : int, y : int):
        from math import lcm
        return lcm(x, y)

    f = epyccel(call_lcm, language=language)
    x = randint(1e4)
    y = randint(1e5)

    assert(f(x,y) == call_lcm(x, y))

# -----------------------------------------------------------------------------

def test_call_radians(language):
    def call_radians(x : 'float'):
        from math import radians
        return radians(x)

    f = epyccel(call_radians, language=language)
    x = uniform(low=0.0, high=1e6)

    assert isclose(f(x), call_radians(x), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x), call_radians(-x), rtol=RTOL, atol=ATOL)

# -----------------------------------------------------------------------------

def test_call_degrees(language):
    def call_degrees(x : 'float'):
        from math import degrees
        return degrees(x)

    f = epyccel(call_degrees, language=language)
    x = uniform(low=0.0, high=1e6)

    assert isclose(f(x), call_degrees(x), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x), call_degrees(-x), rtol=RTOL, atol=ATOL)
# -----------------------------------------------------------------------------

def test_call_degrees_i(language):
    def call_degrees_i(x : 'int'):
        from math import degrees
        return degrees(x)

    f = epyccel(call_degrees_i, language=language)
    x = randint(1e6)

    assert isclose(f(x), call_degrees_i(x), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x), call_degrees_i(-x), rtol=RTOL, atol=ATOL)

# -----------------------------------------------------------------------------
