# pylint: disable=missing-function-docstring, missing-module-docstring
from numpy.random import randint, uniform
from numpy import isclose

from pyccel.epyccel import epyccel
from pyccel.decorators import types

RTOL = 2e-14
ATOL = 1e-15

# -------------------- simple division ---------------------- #

def test_call_div_i_i(language):
    def div_i_i():
        return x / y

    f = epyccel(div_i_i, language=language)
    x = randint(1e9)
    y = randint(low=1, high= 1e3)

    assert isclose(f(x, y), div_i_i(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, y), div_i_i(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(x, -y), div_i_i(x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, -y), div_i_i(-x, -y), rtol=RTOL, atol=ATOL)

def test_call_div_i_r(language):
    def div_i_r(x : int):
        return x / y

    f = epyccel(div_i_r, language=language)
    x = randint(1e9)
    y = uniform(low=1, high= 1e3)
    assert isclose(f(x, y), div_i_r(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, y), div_i_r(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(x, -y), div_i_r(x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, -y), div_i_r(-x, -y), rtol=RTOL, atol=ATOL)

def test_call_div_r_i(language):
    def div_r_i(x : 'float'):
        return x / y

    f = epyccel(div_r_i, language=language)
    x = uniform(high=1e9)
    y = randint(low=1, high= 1e3)
    assert isclose(f(x, y), div_r_i(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, y), div_r_i(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(x, -y), div_r_i(x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, -y), div_r_i(-x, -y), rtol=RTOL, atol=ATOL)

def test_call_div_r_r(language):
    def div_r_r(x : 'float', y : 'float'):
        return x / y

    f = epyccel(div_r_r, language=language)
    x = uniform(high=1e9)
    y = uniform(low=1e-14, high= 1e3)
    assert isclose(f(x, y), div_r_r(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, y), div_r_r(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(x, -y), div_r_r(x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, -y), div_r_r(-x, -y), rtol=RTOL, atol=ATOL)

# -------------------- Complex division ---------------------- #

def test_call_div_c_c(language):
    def div_c_c(x : 'complex', y : 'complex'):
        return x / y

    f = epyccel(div_c_c, language=language)
    x = complex(uniform(high= 1e5), uniform(high= 1e5))
    y = complex(uniform(low=1, high= 1e2), uniform(low=1, high= 1e2))
    assert isclose(f(x, y), div_c_c(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, y), div_c_c(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(x, -y), div_c_c(x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, -y), div_c_c(-x, -y), rtol=RTOL, atol=ATOL)

def test_call_div_i_c(language):
    def div_i_c(x : int):
        return x / y

    f = epyccel(div_i_c, language=language)
    x = randint(1e5)
    y = complex(uniform(low=1, high= 1e2), uniform(low=1, high= 1e2))
    assert isclose(f(x, y), div_i_c(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, y), div_i_c(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(x, -y), div_i_c(x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, -y), div_i_c(-x, -y), rtol=RTOL, atol=ATOL)

def test_call_div_c_i(language):
    def div_c_i(x : 'complex'):
        return x / y

    f = epyccel(div_c_i, language=language)
    x = complex(uniform(high= 1e5), uniform(high= 1e5))
    y = randint(low=1, high= 1e2)
    assert isclose(f(x, y), div_c_i(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, y), div_c_i(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(x, -y), div_c_i(x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, -y), div_c_i(-x, -y), rtol=RTOL, atol=ATOL)

def test_call_div_r_c(language):
    def div_r_c(x : 'float', y : 'complex'):
        return x / y

    f = epyccel(div_r_c, language=language)
    x = uniform(high=1e9)
    y = complex(uniform(low=1, high= 1e2), uniform(low=1, high= 1e2))
    assert isclose(f(x, y), div_r_c(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, y), div_r_c(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(x, -y), div_r_c(x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, -y), div_r_c(-x, -y), rtol=RTOL, atol=ATOL)

def test_call_div_c_r(language):
    def div_c_r(x : 'complex', y : 'float'):
        return x / y

    f = epyccel(div_c_r, language=language)
    x = complex(uniform(high= 1e5), uniform(high= 1e5))
    y = uniform(low=1e-14, high= 1e3)
    assert isclose(f(x, y), div_c_r(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, y), div_c_r(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f(x, -y), div_c_r(x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f(-x, -y), div_c_r(-x, -y), rtol=RTOL, atol=ATOL)

# -------------------- floor division ---------------------- #

def test_call_fdiv_i_i_8(language):
    def fdiv_i_i(x : 'int8', y : 'int8'):
        return x // y

    fflags = "-Werror -Wconversion"

    f = epyccel(fdiv_i_i, language=language, fflags=fflags)
    x = randint(120, dtype='int8')
    y = randint(low=1, high= 100, dtype='int8')

    assert (f(x, y) == fdiv_i_i(x, y))
    assert isinstance(f(x, y), type(fdiv_i_i(x, y)))

def test_call_fdiv_i_i_16(language):
    def fdiv_i_i(x : 'int16', y : 'int16'):
        return x // y

    fflags = "-Werror -Wconversion"

    f = epyccel(fdiv_i_i, language=language, fflags=fflags)
    x = randint(32000, dtype='int16')
    y = randint(low=1, high= 30000, dtype='int16')

    assert (f(x, y) == fdiv_i_i(x, y))
    assert (f(-x, y) == fdiv_i_i(-x, y))
    assert (f(x, -y) == fdiv_i_i(x, -y))
    assert (f(-x, -y) == fdiv_i_i(-x, -y))
    assert isinstance(f(x, y), type(fdiv_i_i(x, y)))

def test_call_fdiv_i_i_32(language):
    def fdiv_i_i(x : 'int32', y : 'int32'):
        return x // y

    fflags = "-Werror -Wconversion"

    f = epyccel(fdiv_i_i, language=language, fflags=fflags)
    x = randint(1e4, dtype='int32')
    y = randint(low=1, high= 1e2, dtype='int32')

    assert (f(x, y) == fdiv_i_i(x, y))
    assert (f(-x, y) == fdiv_i_i(-x, y))
    assert (f(x, -y) == fdiv_i_i(x, -y))
    assert (f(-x, -y) == fdiv_i_i(-x, -y))
    assert isinstance(f(x, y), type(fdiv_i_i(x, y)))

def test_call_fdiv_i_i_i(language):
    def fdiv_i_i_i():
        return x // y // z

    fflags = "-Werror -Wconversion"

    f = epyccel(fdiv_i_i_i, language=language, fflags=fflags)
    x = randint(1e9)
    y = randint(low=1, high= 1e3)
    z = randint(low=1, high= 1e2)

    assert (f(x, y, z) == fdiv_i_i_i(x, y, z))
    assert (f(-x, y, z) == fdiv_i_i_i(-x, y, z))
    assert (f(x, -y, z) == fdiv_i_i_i(x, -y, z))
    assert (f(-x, -y, z) == fdiv_i_i_i(-x, -y, z))
    assert isinstance(f(x, y, z), type(fdiv_i_i_i(x, y, z)))

def test_call_fdiv_i_r(language):
    def fdiv_i_r(x : int):
        return x // y

    fflags = "-Werror -Wconversion"

    f = epyccel(fdiv_i_r, language=language, fflags=fflags)
    x = randint(1e9)
    y = uniform(low=1, high= 1e3)
    assert (f(x, y) == fdiv_i_r(x, y))
    assert (f(-x, y) == fdiv_i_r(-x, y))
    assert (f(x, -y) == fdiv_i_r(x, -y))
    assert (f(-x, -y) == fdiv_i_r(-x, -y))
    assert isinstance(f(x, y), type(fdiv_i_r(x, y)))

def test_call_fdiv_r_i(language):
    def fdiv_r_i(x : 'float'):
        return x // y

    fflags = "-Werror -Wconversion"

    f = epyccel(fdiv_r_i, language=language, fflags=fflags)
    x = uniform(high=1e9)
    y = randint(low=1, high= 1e3)
    assert (f(x, y) == fdiv_r_i(x, y))
    assert (f(-x, y) == fdiv_r_i(-x, y))
    assert (f(x, -y) == fdiv_r_i(x, -y))
    assert (f(-x, -y) == fdiv_r_i(-x, -y))
    assert isinstance(f(x, y), type(fdiv_r_i(x, y)))

def test_call_fdiv_r_r(language):
    def fdiv_r_r(x : 'float', y : 'float'):
        return x // y

    fflags = "-Werror -Wconversion"

    f = epyccel(fdiv_r_r, language=language, fflags=fflags)
    x = uniform(high=1e9)
    y = uniform(low=1e-14, high= 1e3)
    assert (f(x, y) == fdiv_r_r(x, y))
    assert (f(-x, y) == fdiv_r_r(-x, y))
    assert (f(x, -y) == fdiv_r_r(x, -y))
    assert (f(-x, -y) == fdiv_r_r(-x, -y))
    assert isinstance(f(x, y), type(fdiv_r_r(x, y)))
