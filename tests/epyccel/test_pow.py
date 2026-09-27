# pylint: disable=missing-function-docstring, missing-module-docstring
import pytest
from modules import pow as pow_mod
from numpy import isclose
from numpy.random import rand, randint, uniform

from epyccel_utilities import epyccel_module_with_fallback
from tolerances import ATOL, RTOL, min_abs_float


@pytest.fixture(scope="module")
def epyc_pow_mod(language):
    return epyccel_module_with_fallback(pow_mod, language)


def test_pow_int_int(epyc_pow_mod):
    f_call = pow_mod.f_call
    f = epyc_pow_mod.f_call
    x = randint(50)
    y = randint(5)

    assert f(x, y) == f_call(x, y)
    # negative base
    assert f(-x, y) == f_call(-x, y)

    assert isinstance(f(x, y), type(f_call(x, y)))


def test_pow_real_real(epyc_pow_mod):
    pow_r_r = pow_mod.pow_r_r
    f = epyc_pow_mod.pow_r_r
    x = uniform(low=min_abs_float, high=50)
    y = uniform(high=5)

    assert isclose(f(x, y), pow_r_r(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(x, -y), pow_r_r(x, -y), rtol=RTOL, atol=ATOL)
    assert isinstance(f(x, y), type(pow_r_r(x, y)))


def test_pow_real_int(epyc_pow_mod):
    pow_r_i = pow_mod.pow_r_i
    f = epyc_pow_mod.pow_r_i
    x = uniform(low=min_abs_float, high=50)
    y = randint(5)

    assert isclose(f(x, y), pow_r_i(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, y), pow_r_i(-x, y), rtol=RTOL, atol=ATOL)
    assert isinstance(f(x, y), type(pow_r_i(x, y)))


def test_pow_int_real(epyc_pow_mod):
    pow_i_r = pow_mod.pow_i_r
    f = epyc_pow_mod.pow_i_r
    x = randint(40)
    y = uniform()

    assert isclose(f(x, y), pow_i_r(x, y), rtol=RTOL, atol=ATOL)
    assert isinstance(f(x, y), type(pow_i_r(x, y)))


def test_pow_special_cases(epyc_pow_mod):
    pow_r_r = pow_mod.pow_r_r
    epyc_pow_r_r = epyc_pow_mod.pow_r_r
    e = uniform(high=1e6)
    assert isclose(epyc_pow_r_r(0.0, e), pow_r_r(0.0, e), rtol=RTOL, atol=ATOL)
    assert isclose(epyc_pow_r_r(0.0, e), pow_r_r(0.0, e), rtol=RTOL, atol=ATOL)


# ---------------------------- Complex numbers ----------------------------- #


def test_pow_c_c(epyc_pow_mod):
    pow_c_c = pow_mod.pow_c_c
    f = epyc_pow_mod.pow_c_c
    b = complex(rand(), rand())
    e = complex(rand(), rand())
    assert isclose(f(b, e), pow_c_c(b, e), rtol=RTOL, atol=ATOL)
    assert isclose(f(-b, e), pow_c_c(-b, e), rtol=RTOL, atol=ATOL)
    assert isclose(f(b, -e), pow_c_c(b, -e), rtol=RTOL, atol=ATOL)
    assert isclose(f(-b, -e), pow_c_c(-b, -e), rtol=RTOL, atol=ATOL)


def test_pow_c_i(epyc_pow_mod):
    pow_c_i = pow_mod.pow_c_i
    f = epyc_pow_mod.pow_c_i
    b = complex(rand(), rand())
    e = randint(10)
    assert isclose(f(b, e), pow_c_i(b, e), rtol=RTOL, atol=ATOL)
    assert isclose(f(-b, e), pow_c_i(-b, e), rtol=RTOL, atol=ATOL)
    assert isclose(f(b, -e), pow_c_i(b, -e), rtol=RTOL, atol=ATOL)
    assert isclose(f(-b, -e), pow_c_i(-b, -e), rtol=RTOL, atol=ATOL)


def test_pow_c_r(epyc_pow_mod):
    pow_c_r = pow_mod.pow_c_r
    f = epyc_pow_mod.pow_c_r
    b = complex(rand(), rand())
    e = rand()
    assert isclose(f(b, e), pow_c_r(b, e), rtol=RTOL, atol=ATOL)
    assert isclose(f(-b, e), pow_c_r(-b, e), rtol=RTOL, atol=ATOL)
    assert isclose(f(b, -e), pow_c_r(b, -e), rtol=RTOL, atol=ATOL)
    assert isclose(f(-b, -e), pow_c_r(-b, -e), rtol=RTOL, atol=ATOL)


def test_pow_r_c(epyc_pow_mod):
    pow_r_c = pow_mod.pow_r_c
    f = epyc_pow_mod.pow_r_c
    b = rand()
    e = complex(rand(), rand())
    assert isclose(f(b, e), pow_r_c(b, e), rtol=RTOL, atol=ATOL)
    assert isclose(f(-b, e), pow_r_c(-b, e), rtol=RTOL, atol=ATOL)
    assert isclose(f(b, -e), pow_r_c(b, -e), rtol=RTOL, atol=ATOL)
    assert isclose(f(-b, -e), pow_r_c(-b, -e), rtol=RTOL, atol=ATOL)


def test_pow_chain(epyc_pow_mod):
    x = uniform(low=min_abs_float, high=10)
    y = uniform(high=5)
    z = uniform(high=1.0)

    for c_name in ("chain_pow1", "chain_pow2", "chain_pow3"):
        c = getattr(pow_mod, c_name)
        f = getattr(epyc_pow_mod, c_name)
        assert isclose(f(x, y, z), c(x, y, z), rtol=RTOL, atol=ATOL)
        assert isinstance(f(x, y, z), type(c(x, y, z)))


def test_square(epyc_pow_mod):
    square = pow_mod.square
    f = epyc_pow_mod.square
    x = randint(40)
    y = uniform()

    assert isclose(f(x), square(x), rtol=RTOL, atol=ATOL)
    assert isinstance(f(x), type(square(x)))
    assert isclose(f(y), square(y), rtol=RTOL, atol=ATOL)
    assert isinstance(f(y), type(square(y)))


def test_sqrt(epyc_pow_mod):
    sqrt = pow_mod.sqrt
    f = epyc_pow_mod.sqrt
    x = randint(40)
    y = uniform()

    assert isclose(f(x), sqrt(x), rtol=RTOL, atol=ATOL)
    assert isinstance(f(x), type(sqrt(x)))
    assert isclose(f(y), sqrt(y), rtol=RTOL, atol=ATOL)
    assert isinstance(f(y), type(sqrt(y)))


def test_fabs(epyc_pow_mod):
    fabs = pow_mod.fabs
    f = epyc_pow_mod.fabs
    x = randint(40)
    y = uniform()

    assert isclose(f(x), fabs(x), rtol=RTOL, atol=ATOL)
    assert isinstance(f(x), type(fabs(x)))
    assert isclose(f(y), fabs(y), rtol=RTOL, atol=ATOL)
    assert isinstance(f(y), type(fabs(y)))


def test_abs(epyc_pow_mod):
    norm = pow_mod.norm
    f = epyc_pow_mod.norm
    x = randint(40) + 1j * randint(40)
    y = randint(40) - 1j * randint(40)

    assert isclose(f(x), norm(x), rtol=RTOL, atol=ATOL)
    assert isinstance(f(x), type(norm(x)))
    assert isclose(f(y), norm(y), rtol=RTOL, atol=ATOL)
    assert isinstance(f(y), type(norm(y)))


def test_complicated_abs(epyc_pow_mod):
    norm = pow_mod.complicated_abs
    f = epyc_pow_mod.complicated_abs
    x = randint(40) + 1j * randint(40)
    y = randint(40) - 1j * randint(40)

    assert isclose(f(x), norm(x), rtol=RTOL, atol=ATOL)
    assert isinstance(f(x), type(norm(x)))
    assert isclose(f(y), norm(y), rtol=RTOL, atol=ATOL)
    assert isinstance(f(y), type(norm(y)))


def test_fcomplex_sqrt(epyc_pow_mod):
    fcomplex_sqrt = pow_mod.fcomplex_sqrt
    epyc_fcomplex_sqrt = epyc_pow_mod.fcomplex_sqrt
    x = randint(40) + 1j * randint(40)
    y = randint(40) + 1j * randint(40)

    assert isclose(epyc_fcomplex_sqrt(x, y), fcomplex_sqrt(x, y), rtol=RTOL, atol=ATOL)
    assert isinstance(epyc_fcomplex_sqrt(x, y), type(fcomplex_sqrt(x, y)))
