# pylint: disable=missing-function-docstring, missing-module-docstring
import sys

import pytest
from modules import epyccel_pow
from numpy import isclose
from numpy.random import rand, randint, uniform
from utilities import epyccel_module_with_fallback

from pyccel import epyccel


@pytest.fixture(scope="module")
def epyc_epyccel_pow_mod(language):
    return epyccel_module_with_fallback(epyccel_pow, language)


RTOL = 2e-14
ATOL = 1e-15

# this smallest positive float number
min_float = sys.float_info.min


def test_pow_int_int(epyc_epyccel_pow_mod):
    f_call = epyccel_pow.f_call
    f = epyc_epyccel_pow_mod.f_call
    x = randint(50)
    y = randint(5)

    assert f(x, y) == f_call(x, y)
    # negative base
    assert f(-x, y) == f_call(-x, y)

    assert isinstance(f(x, y), type(f_call(x, y)))


def test_pow_real_real(epyc_epyccel_pow_mod):
    pow_r_r = epyccel_pow.pow_r_r
    f = epyc_epyccel_pow_mod.pow_r_r
    x = uniform(low=min_float, high=50)
    y = uniform(high=5)

    assert isclose(f(x, y), pow_r_r(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(x, -y), pow_r_r(x, -y), rtol=RTOL, atol=ATOL)
    assert isinstance(f(x, y), type(pow_r_r(x, y)))


def test_pow_real_int(epyc_epyccel_pow_mod):
    pow_r_i = epyccel_pow.pow_r_i
    f = epyc_epyccel_pow_mod.pow_r_i
    x = uniform(low=min_float, high=50)
    y = randint(5)

    assert isclose(f(x, y), pow_r_i(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, y), pow_r_i(-x, y), rtol=RTOL, atol=ATOL)
    assert isinstance(f(x, y), type(pow_r_i(x, y)))


def test_pow_int_real(epyc_epyccel_pow_mod):
    pow_i_r = epyccel_pow.pow_i_r
    f = epyc_epyccel_pow_mod.pow_i_r
    x = randint(40)
    y = uniform()

    assert isclose(f(x, y), pow_i_r(x, y), rtol=RTOL, atol=ATOL)
    assert isinstance(f(x, y), type(pow_i_r(x, y)))


def test_pow_special_cases(epyc_epyccel_pow_mod):
    pow_sp = epyccel_pow.pow_sp
    f = epyc_epyccel_pow_mod.pow_sp
    e = uniform(high=1e6)
    assert isclose(f(0.0, e), pow_sp(0.0, e), rtol=RTOL, atol=ATOL)
    assert isclose(f(0.0, e), pow_sp(0.0, e), rtol=RTOL, atol=ATOL)


# ---------------------------- Complex numbers ----------------------------- #


def test_pow_c_c(epyc_epyccel_pow_mod):
    pow_c_c = epyccel_pow.pow_c_c
    f = epyc_epyccel_pow_mod.pow_c_c
    b = complex(rand(), rand())
    e = complex(rand(), rand())
    assert isclose(f(b, e), pow_c_c(b, e), rtol=RTOL, atol=ATOL)
    assert isclose(f(-b, e), pow_c_c(-b, e), rtol=RTOL, atol=ATOL)
    assert isclose(f(b, -e), pow_c_c(b, -e), rtol=RTOL, atol=ATOL)
    assert isclose(f(-b, -e), pow_c_c(-b, -e), rtol=RTOL, atol=ATOL)


def test_pow_c_i(epyc_epyccel_pow_mod):
    pow_c_i = epyccel_pow.pow_c_i
    f = epyc_epyccel_pow_mod.pow_c_i
    b = complex(rand(), rand())
    e = randint(10)
    assert isclose(f(b, e), pow_c_i(b, e), rtol=RTOL, atol=ATOL)
    assert isclose(f(-b, e), pow_c_i(-b, e), rtol=RTOL, atol=ATOL)
    assert isclose(f(b, -e), pow_c_i(b, -e), rtol=RTOL, atol=ATOL)
    assert isclose(f(-b, -e), pow_c_i(-b, -e), rtol=RTOL, atol=ATOL)


def test_pow_c_r(epyc_epyccel_pow_mod):
    pow_c_r = epyccel_pow.pow_c_r
    f = epyc_epyccel_pow_mod.pow_c_r
    b = complex(rand(), rand())
    e = rand()
    assert isclose(f(b, e), pow_c_r(b, e), rtol=RTOL, atol=ATOL)
    assert isclose(f(-b, e), pow_c_r(-b, e), rtol=RTOL, atol=ATOL)
    assert isclose(f(b, -e), pow_c_r(b, -e), rtol=RTOL, atol=ATOL)
    assert isclose(f(-b, -e), pow_c_r(-b, -e), rtol=RTOL, atol=ATOL)


def test_pow_r_c(epyc_epyccel_pow_mod):
    pow_r_c = epyccel_pow.pow_r_c
    f = epyc_epyccel_pow_mod.pow_r_c
    b = rand()
    e = complex(rand(), rand())
    assert isclose(f(b, e), pow_r_c(b, e), rtol=RTOL, atol=ATOL)
    assert isclose(f(-b, e), pow_r_c(-b, e), rtol=RTOL, atol=ATOL)
    assert isclose(f(b, -e), pow_r_c(b, -e), rtol=RTOL, atol=ATOL)
    assert isclose(f(-b, -e), pow_r_c(-b, -e), rtol=RTOL, atol=ATOL)


def test_pow_chain(epyc_epyccel_pow_mod):
    x = uniform(low=min_float, high=10)
    y = uniform(high=5)
    z = uniform(high=1.0)

    for c_name in ("chain_pow1", "chain_pow2", "chain_pow3"):
        c = getattr(epyccel_pow, c_name)
        f = getattr(epyc_epyccel_pow_mod, c_name)
        assert isclose(f(x, y, z), c(x, y, z), rtol=RTOL, atol=ATOL)
        assert isinstance(f(x, y, z), type(c(x, y, z)))


def test_square(epyc_epyccel_pow_mod):
    square = epyccel_pow.square
    f = epyc_epyccel_pow_mod.square
    x = randint(40)
    y = uniform()

    assert isclose(f(x), square(x), rtol=RTOL, atol=ATOL)
    assert isinstance(f(x), type(square(x)))
    assert isclose(f(y), square(y), rtol=RTOL, atol=ATOL)
    assert isinstance(f(y), type(square(y)))


def test_sqrt(epyc_epyccel_pow_mod):
    sqrt = epyccel_pow.sqrt
    f = epyc_epyccel_pow_mod.sqrt
    x = randint(40)
    y = uniform()

    assert isclose(f(x), sqrt(x), rtol=RTOL, atol=ATOL)
    assert isinstance(f(x), type(sqrt(x)))
    assert isclose(f(y), sqrt(y), rtol=RTOL, atol=ATOL)
    assert isinstance(f(y), type(sqrt(y)))


def test_fabs(epyc_epyccel_pow_mod):
    fabs = epyccel_pow.fabs
    f = epyc_epyccel_pow_mod.fabs
    x = randint(40)
    y = uniform()

    assert isclose(f(x), fabs(x), rtol=RTOL, atol=ATOL)
    assert isinstance(f(x), type(fabs(x)))
    assert isclose(f(y), fabs(y), rtol=RTOL, atol=ATOL)
    assert isinstance(f(y), type(fabs(y)))


def test_abs(epyc_epyccel_pow_mod):
    norm = epyccel_pow.norm
    f = epyc_epyccel_pow_mod.norm
    x = randint(40) + 1j * randint(40)
    y = randint(40) - 1j * randint(40)

    assert isclose(f(x), norm(x), rtol=RTOL, atol=ATOL)
    assert isinstance(f(x), type(norm(x)))
    assert isclose(f(y), norm(y), rtol=RTOL, atol=ATOL)
    assert isinstance(f(y), type(norm(y)))


def test_complicated_abs(epyc_epyccel_pow_mod):
    norm = epyccel_pow.complicated_abs
    f = epyc_epyccel_pow_mod.complicated_abs
    x = randint(40) + 1j * randint(40)
    y = randint(40) - 1j * randint(40)

    assert isclose(f(x), norm(x), rtol=RTOL, atol=ATOL)
    assert isinstance(f(x), type(norm(x)))
    assert isclose(f(y), norm(y), rtol=RTOL, atol=ATOL)
    assert isinstance(f(y), type(norm(y)))


def test_fcomplex_type_conversion(epyc_epyccel_pow_mod):
    fcomplex = epyccel_pow.fcomplex
    f = epyc_epyccel_pow_mod.fcomplex
    x = randint(40) + 1j * randint(40)
    y = randint(40) + 1j * randint(40)

    assert isclose(f(x, y), fcomplex(x, y), rtol=RTOL, atol=ATOL)
    assert isinstance(f(x, y), type(fcomplex(x, y)))
