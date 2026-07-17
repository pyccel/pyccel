# pylint: disable=missing-function-docstring, missing-module-docstring
import pytest
from modules import epyccel_division, epyccel_floor_division
from numpy import isclose
from numpy.random import randint, uniform
from epyccel_utilities import epyccel_module_with_fallback
from tolerances import ATOL, RTOL

from pyccel import epyccel

@pytest.fixture(scope="module")
def epyc_epyccel_division_mod(language):
    return epyccel_module_with_fallback(epyccel_division, language)


@pytest.fixture(scope="module")
def epyc_epyccel_floor_division_mod(language):
    return epyccel_module_with_fallback(
        epyccel_floor_division, language, flags="-Werror -Wconversion"
    )



# -------------------- simple division ---------------------- #


def test_call_div_i_i(epyc_epyccel_division_mod):
    div_i_i = epyccel_division.div_i_i
    f = epyc_epyccel_division_mod.div_i_i
    x = randint(1e9)
    y = randint(low=1, high=1e3)

    assert isclose(f(x, y), div_i_i(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, y), div_i_i(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(x, -y), div_i_i(x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, -y), div_i_i(-x, -y), rtol=RTOL, atol=ATOL)


def test_call_div_i_r(epyc_epyccel_division_mod):
    div_i_r = epyccel_division.div_i_r
    f = epyc_epyccel_division_mod.div_i_r
    x = randint(1e9)
    y = uniform(low=1, high=1e3)
    assert isclose(f(x, y), div_i_r(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, y), div_i_r(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(x, -y), div_i_r(x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, -y), div_i_r(-x, -y), rtol=RTOL, atol=ATOL)


def test_call_div_r_i(epyc_epyccel_division_mod):
    div_r_i = epyccel_division.div_r_i
    f = epyc_epyccel_division_mod.div_r_i
    x = uniform(high=1e9)
    y = randint(low=1, high=1e3)
    assert isclose(f(x, y), div_r_i(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, y), div_r_i(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(x, -y), div_r_i(x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, -y), div_r_i(-x, -y), rtol=RTOL, atol=ATOL)


def test_call_div_r_r(epyc_epyccel_division_mod):
    div_r_r = epyccel_division.div_r_r
    f = epyc_epyccel_division_mod.div_r_r
    x = uniform(high=1e9)
    y = uniform(low=1e-14, high=1e3)
    assert isclose(f(x, y), div_r_r(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, y), div_r_r(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(x, -y), div_r_r(x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, -y), div_r_r(-x, -y), rtol=RTOL, atol=ATOL)


# -------------------- Complex division ---------------------- #


def test_call_div_c_c(epyc_epyccel_division_mod):
    div_c_c = epyccel_division.div_c_c
    f = epyc_epyccel_division_mod.div_c_c
    x = complex(uniform(high=1e5), uniform(high=1e5))
    y = complex(uniform(low=1, high=1e2), uniform(low=1, high=1e2))
    assert isclose(f(x, y), div_c_c(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, y), div_c_c(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(x, -y), div_c_c(x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, -y), div_c_c(-x, -y), rtol=RTOL, atol=ATOL)


def test_call_div_i_c(epyc_epyccel_division_mod):
    div_i_c = epyccel_division.div_i_c
    f = epyc_epyccel_division_mod.div_i_c
    x = randint(1e5)
    y = complex(uniform(low=1, high=1e2), uniform(low=1, high=1e2))
    assert isclose(f(x, y), div_i_c(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, y), div_i_c(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(x, -y), div_i_c(x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, -y), div_i_c(-x, -y), rtol=RTOL, atol=ATOL)


def test_call_div_c_i(epyc_epyccel_division_mod):
    div_c_i = epyccel_division.div_c_i
    f = epyc_epyccel_division_mod.div_c_i
    x = complex(uniform(high=1e5), uniform(high=1e5))
    y = randint(low=1, high=1e2)
    assert isclose(f(x, y), div_c_i(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, y), div_c_i(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(x, -y), div_c_i(x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, -y), div_c_i(-x, -y), rtol=RTOL, atol=ATOL)


def test_call_div_r_c(epyc_epyccel_division_mod):
    div_r_c = epyccel_division.div_r_c
    f = epyc_epyccel_division_mod.div_r_c
    x = uniform(high=1e9)
    y = complex(uniform(low=1, high=1e2), uniform(low=1, high=1e2))
    assert isclose(f(x, y), div_r_c(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, y), div_r_c(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(x, -y), div_r_c(x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, -y), div_r_c(-x, -y), rtol=RTOL, atol=ATOL)


def test_call_div_c_r(epyc_epyccel_division_mod):
    div_c_r = epyccel_division.div_c_r
    f = epyc_epyccel_division_mod.div_c_r
    x = complex(uniform(high=1e5), uniform(high=1e5))
    y = uniform(low=1e-14, high=1e3)
    assert isclose(f(x, y), div_c_r(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, y), div_c_r(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(x, -y), div_c_r(x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, -y), div_c_r(-x, -y), rtol=RTOL, atol=ATOL)


# -------------------- floor division ---------------------- #


def test_call_fdiv_i_i_8(epyc_epyccel_floor_division_mod):
    fdiv_i_i = epyccel_floor_division.fdiv_i_i_8
    f = epyc_epyccel_floor_division_mod.fdiv_i_i_8

    x = randint(120, dtype="int8")
    y = randint(low=1, high=100, dtype="int8")

    assert f(x, y) == fdiv_i_i(x, y)
    assert isinstance(f(x, y), type(fdiv_i_i(x, y)))


def test_call_fdiv_i_i_16(epyc_epyccel_floor_division_mod):
    fdiv_i_i = epyccel_floor_division.fdiv_i_i_16
    f = epyc_epyccel_floor_division_mod.fdiv_i_i_16

    x = randint(32000, dtype="int16")
    y = randint(low=1, high=30000, dtype="int16")

    assert f(x, y) == fdiv_i_i(x, y)
    assert f(-x, y) == fdiv_i_i(-x, y)
    assert f(x, -y) == fdiv_i_i(x, -y)
    assert f(-x, -y) == fdiv_i_i(-x, -y)
    assert isinstance(f(x, y), type(fdiv_i_i(x, y)))


def test_call_fdiv_i_i_32(epyc_epyccel_floor_division_mod):
    fdiv_i_i = epyccel_floor_division.fdiv_i_i_32
    f = epyc_epyccel_floor_division_mod.fdiv_i_i_32

    x = randint(1e4, dtype="int32")
    y = randint(low=1, high=1e2, dtype="int32")

    assert f(x, y) == fdiv_i_i(x, y)
    assert f(-x, y) == fdiv_i_i(-x, y)
    assert f(x, -y) == fdiv_i_i(x, -y)
    assert f(-x, -y) == fdiv_i_i(-x, -y)
    assert isinstance(f(x, y), type(fdiv_i_i(x, y)))


def test_call_fdiv_i_i_i(epyc_epyccel_floor_division_mod):
    fdiv_i_i_i = epyccel_floor_division.fdiv_i_i_i
    f = epyc_epyccel_floor_division_mod.fdiv_i_i_i

    x = randint(1e9)
    y = randint(low=1, high=1e3)
    z = randint(low=1, high=1e2)

    assert f(x, y, z) == fdiv_i_i_i(x, y, z)
    assert f(-x, y, z) == fdiv_i_i_i(-x, y, z)
    assert f(x, -y, z) == fdiv_i_i_i(x, -y, z)
    assert f(-x, -y, z) == fdiv_i_i_i(-x, -y, z)
    assert isinstance(f(x, y, z), type(fdiv_i_i_i(x, y, z)))


def test_call_fdiv_b_b(epyc_epyccel_floor_division_mod):
    fdiv_b_b = epyccel_floor_division.fdiv_b_b
    f = epyc_epyccel_floor_division_mod.fdiv_b_b

    assert f(True, True) == fdiv_b_b(True, True)
    assert f(False, True) == fdiv_b_b(False, True)
    assert isinstance(f(True, True), type(fdiv_b_b(True, True)))


def test_call_fdiv_i_r(epyc_epyccel_floor_division_mod):
    fdiv_i_r = epyccel_floor_division.fdiv_i_r
    f = epyc_epyccel_floor_division_mod.fdiv_i_r

    x = randint(1e9)
    y = uniform(low=1, high=1e3)
    assert f(x, y) == fdiv_i_r(x, y)
    assert f(-x, y) == fdiv_i_r(-x, y)
    assert f(x, -y) == fdiv_i_r(x, -y)
    assert f(-x, -y) == fdiv_i_r(-x, -y)
    assert isinstance(f(x, y), type(fdiv_i_r(x, y)))


def test_call_fdiv_r_i(epyc_epyccel_floor_division_mod):
    fdiv_r_i = epyccel_floor_division.fdiv_r_i
    f = epyc_epyccel_floor_division_mod.fdiv_r_i

    x = uniform(high=1e9)
    y = randint(low=1, high=1e3)
    assert f(x, y) == fdiv_r_i(x, y)
    assert f(-x, y) == fdiv_r_i(-x, y)
    assert f(x, -y) == fdiv_r_i(x, -y)
    assert f(-x, -y) == fdiv_r_i(-x, -y)
    assert isinstance(f(x, y), type(fdiv_r_i(x, y)))


def test_call_fdiv_r_r(epyc_epyccel_floor_division_mod):
    fdiv_r_r = epyccel_floor_division.fdiv_r_r
    f = epyc_epyccel_floor_division_mod.fdiv_r_r

    x = uniform(high=1e9)
    y = uniform(low=1e-14, high=1e3)
    assert f(x, y) == fdiv_r_r(x, y)
    assert f(-x, y) == fdiv_r_r(-x, y)
    assert f(x, -y) == fdiv_r_r(x, -y)
    assert f(-x, -y) == fdiv_r_r(-x, -y)
    assert isinstance(f(x, y), type(fdiv_r_r(x, y)))
