# pylint: disable=missing-function-docstring, missing-module-docstring
import sys
from numpy.random import rand, randint, uniform
from numpy import isclose

from pyccel.decorators import types
from pyccel.epyccel import epyccel

RTOL = 2e-14
ATOL = 1e-15

# this smallest positive float number
min_float = sys.float_info.min

def test_pow_int_int(language):
    def f_call():
        return x ** y

    f = epyccel(f_call, language=language)
    x = randint(50)
    y = randint(5)

    assert f(x, y) == f_call(x, y)
    # negative base
    assert f(-x, y) == f_call(-x, y)

    assert isinstance(f(x, y), type(f_call(x, y)))

def test_pow_real_real(language):
    def pow_r_r(x : 'float', y : 'float'):
        return x ** y

    f = epyccel(pow_r_r, language=language)
    x = uniform(low=min_float, high=50)
    y = uniform(high=5)

    assert(isclose(f(x, y), pow_r_r(x, y), rtol=RTOL, atol=ATOL))
    assert(isclose(f(x, -y), pow_r_r(x, -y), rtol=RTOL, atol=ATOL))
    assert isinstance(f(x, y), type(pow_r_r(x, y)))

def test_pow_real_int(language):
    def pow_r_i(x : 'float', y : 'int'):
        return x ** y

    f = epyccel(pow_r_i, language=language)
    x = uniform(low=min_float, high=50)
    y = randint(5)

    assert(isclose(f(x, y), pow_r_i(x, y), rtol=RTOL, atol=ATOL))
    assert(isclose(f(-x, y), pow_r_i(-x, y), rtol=RTOL, atol=ATOL))
    assert isinstance(f(x, y), type(pow_r_i(x, y)))

def test_pow_int_real(language):
    def pow_i_r(x : 'int', y : 'float'):
        return x ** y

    f = epyccel(pow_i_r, language=language)
    x = randint(40)
    y = uniform()

    assert(isclose(f(x, y), pow_i_r(x, y), rtol=RTOL, atol=ATOL))
    assert isinstance(f(x, y), type(pow_i_r(x, y)))

def test_pow_special_cases(language):
    def pow_sp(x : 'float', y : 'float'):
        return x ** y

    f = epyccel(pow_sp, language=language)
    e = uniform(high=1e6)
    assert(isclose(f(0.0, e), pow_sp(0.0, e), rtol=RTOL, atol=ATOL))
    assert(isclose(f(0.0, e), pow_sp(0.0, e), rtol=RTOL, atol=ATOL))

# ---------------------------- Complex numbers ----------------------------- #

def test_pow_c_c(language):
    def pow_c_c(x : 'complex', y : 'complex'):
        return x ** y

    f = epyccel(pow_c_c, language=language)
    b = complex(rand(), rand())
    e = complex(rand(), rand())
    assert(isclose(f(b, e), pow_c_c(b, e), rtol=RTOL, atol=ATOL))
    assert(isclose(f(-b, e), pow_c_c(-b, e), rtol=RTOL, atol=ATOL))
    assert(isclose(f(b, -e), pow_c_c(b, -e), rtol=RTOL, atol=ATOL))
    assert(isclose(f(-b, -e), pow_c_c(-b, -e), rtol=RTOL, atol=ATOL))

def test_pow_c_i(language):
    def pow_c_i(x : 'complex', y : 'int'):
        return x ** y

    f = epyccel(pow_c_i, language=language)
    b = complex(rand(), rand())
    e = randint(10)
    assert(isclose(f(b, e), pow_c_i(b, e), rtol=RTOL, atol=ATOL))
    assert(isclose(f(-b, e), pow_c_i(-b, e), rtol=RTOL, atol=ATOL))
    assert(isclose(f(b, -e), pow_c_i(b, -e), rtol=RTOL, atol=ATOL))
    assert(isclose(f(-b, -e), pow_c_i(-b, -e), rtol=RTOL, atol=ATOL))

def test_pow_c_r(language):
    def pow_c_r(x : 'complex', y : 'float'):
        return x ** y

    f = epyccel(pow_c_r, language=language)
    b = complex(rand(), rand())
    e = rand()
    assert(isclose(f(b, e), pow_c_r(b, e), rtol=RTOL, atol=ATOL))
    assert(isclose(f(-b, e), pow_c_r(-b, e), rtol=RTOL, atol=ATOL))
    assert(isclose(f(b, -e), pow_c_r(b, -e), rtol=RTOL, atol=ATOL))
    assert(isclose(f(-b, -e), pow_c_r(-b, -e), rtol=RTOL, atol=ATOL))

def test_pow_r_c(language):
    def pow_r_c(x : 'float', y : 'complex'):
        return x ** y

    f = epyccel(pow_r_c, language=language)
    b = rand()
    e = complex(rand(), rand())
    assert(isclose(f(b, e), pow_r_c(b, e), rtol=RTOL, atol=ATOL))
    assert(isclose(f(-b, e), pow_r_c(-b, e), rtol=RTOL, atol=ATOL))
    assert(isclose(f(b, -e), pow_r_c(b, -e), rtol=RTOL, atol=ATOL))
    assert(isclose(f(-b, -e), pow_r_c(-b, -e), rtol=RTOL, atol=ATOL))

def test_pow_chain(language):
    def chain_pow1(x : float, y : float, z : float):
        return x ** y ** z
    def chain_pow2(x : float, y : float, z : float):
        return (x ** y) ** z
    def chain_pow3(x : float, y : float, z : float):
        return x ** (y ** z)

    x = uniform(low=min_float, high=10)
    y = uniform(high=5)
    z = uniform(high=1.0)

    for c in (chain_pow1, chain_pow2, chain_pow3):
        f = epyccel(c, language=language)
        assert(isclose(f(x, y, z), c(x, y, z), rtol=RTOL, atol=ATOL))
        assert isinstance(f(x, y, z), type(c(x, y, z)))

def test_square(language):
    @types('float')
    @types('int')
    def square(x):
        return x**2

    f = epyccel(square, language=language)
    x = randint(40)
    y = uniform()

    assert isclose(f(x), square(x), rtol=RTOL, atol=ATOL)
    assert isinstance(f(x), type(square(x)))
    assert isclose(f(y), square(y), rtol=RTOL, atol=ATOL)
    assert isinstance(f(y), type(square(y)))

def test_sqrt(language):
    @types('float')
    @types('int')
    def sqrt(x):
        return x**0.5

    f = epyccel(sqrt, language=language)
    x = randint(40)
    y = uniform()

    assert isclose(f(x), sqrt(x), rtol=RTOL, atol=ATOL)
    assert isinstance(f(x), type(sqrt(x)))
    assert isclose(f(y), sqrt(y), rtol=RTOL, atol=ATOL)
    assert isinstance(f(y), type(sqrt(y)))

def test_fabs(language):
    @types('float')
    @types('int')
    def fabs(x):
        return (x*x)**0.5

    f = epyccel(fabs, language=language)
    x = randint(40)
    y = uniform()

    assert isclose(f(x), fabs(x), rtol=RTOL, atol=ATOL)
    assert isinstance(f(x), type(fabs(x)))
    assert isclose(f(y), fabs(y), rtol=RTOL, atol=ATOL)
    assert isinstance(f(y), type(fabs(y)))

def test_abs(language):
    def norm(x : 'complex'):
        return (x*x.conjugate())**0.5

    f = epyccel(norm, language=language)
    x = randint(40) + 1j * randint(40)
    y = randint(40) - 1j * randint(40)

    assert isclose(f(x), norm(x), rtol=RTOL, atol=ATOL)
    assert isinstance(f(x), type(norm(x)))
    assert isclose(f(y), norm(y), rtol=RTOL, atol=ATOL)
    assert isinstance(f(y), type(norm(y)))

def test_complicated_abs(language):
    def norm(x : 'complex'):
        return ((x*x.conjugate()).real**2)**0.5

    f = epyccel(norm, language=language)
    x = randint(40) + 1j * randint(40)
    y = randint(40) - 1j * randint(40)

    assert isclose(f(x), norm(x), rtol=RTOL, atol=ATOL)
    assert isinstance(f(x), type(norm(x)))
    assert isclose(f(y), norm(y), rtol=RTOL, atol=ATOL)
    assert isinstance(f(y), type(norm(y)))
