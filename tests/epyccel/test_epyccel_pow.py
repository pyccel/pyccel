import pytest
from numpy.random import rand, randint, uniform
from numpy import isclose

from pyccel.decorators import types
from pyccel.epyccel import epyccel
from conftest import *

def test_pow_int_int(language):
    @types(int, int)
    def f_call(x, y):
        return x ** y

    f = epyccel(f_call, language=language)
    x = randint(10)
    y = randint(10)

    assert f(x, y) == f_call(x, y)
    # negative base
    assert f(-x, y) == f_call(-x, y)

    assert isinstance(f(x, y), type(f_call(x, y)))

def test_pow_real_real(language):
    @types('real', 'real')
    def pow_r_r(x, y):
        return x ** y

    f = epyccel(pow_r_r, language=language)
    x = rand()
    y = rand()

    assert(isclose(f(x, y), pow_r_r(x, y), rtol=1e-15, atol=1e-15))

    assert isinstance(f(x, y), type(pow_r_r(x, y)))

def test_pow_real_int(language):
    @types('real', 'int')
    def pow_r_i(x, y):
        return x ** y

    f = epyccel(pow_r_i, language=language)
    x = uniform()
    y = randint(5)

    assert(isclose(f(x, y), pow_r_i(x, y), rtol=1e-15, atol=1e-15))
    assert isinstance(f(x, y), type(pow_r_i(x, y)))

def test_pow_int_real(language):
    @types('int', 'real')
    def pow_i_r(x, y):
        return x ** y

    f = epyccel(pow_i_r, language=language)
    x = randint(40)
    y = uniform()

    assert(isclose(f(x, y), pow_i_r(x, y), rtol=1e-15, atol=1e-15))
    assert isinstance(f(x, y), type(pow_i_r(x, y)))

def test_pow_special_cases(language):
    @types('real', 'real')
    def pow_sp(x, y):
        return x ** y

    f = epyccel(pow_sp, language=language)
    e = uniform()
    assert(isclose(f(0.0, e), pow_sp(0.0, e), rtol=1e-15, atol=1e-15))
    assert(isclose(f(0.0, e), pow_sp(0.0, e), rtol=1e-15, atol=1e-15))

# ---------------------------- Complex numbers ----------------------------- #

def test_pow_c_c(language):
    @types('complex', 'complex')
    def pow_c_c(x, y):
        return x ** y

    f = epyccel(pow_c_c, language=language)
    b = complex(rand(), rand())
    e = complex(rand(), rand())
    assert(isclose(f(b, e), pow_c_c(b, e), rtol=1e-15, atol=1e-15))
    assert(isclose(f(-b, e), pow_c_c(-b, e), rtol=1e-15, atol=1e-15))
    assert(isclose(f(b, -e), pow_c_c(b, -e), rtol=1e-15, atol=1e-15))
    assert(isclose(f(-b, -e), pow_c_c(-b, -e), rtol=1e-15, atol=1e-15))

def test_pow_c_i(language):
    @types('complex', 'int')
    def pow_c_i(x, y):
        return x ** y

    f = epyccel(pow_c_i, language=language)
    b = complex(rand(), rand())
    e = randint(10)
    assert(isclose(f(b, e), pow_c_i(b, e), rtol=1e-15, atol=1e-15))
    assert(isclose(f(-b, e), pow_c_i(-b, e), rtol=1e-15, atol=1e-15))
    assert(isclose(f(b, -e), pow_c_i(b, -e), rtol=1e-15, atol=1e-15))
    assert(isclose(f(-b, -e), pow_c_i(-b, -e), rtol=1e-15, atol=1e-15))

def test_pow_c_r(language):
    @types('complex', 'real')
    def pow_c_r(x, y):
        return x ** y

    f = epyccel(pow_c_r, language=language)
    b = complex(rand(), rand())
    e = rand()
    assert(isclose(f(b, e), pow_c_r(b, e), rtol=1e-15, atol=1e-15))
    assert(isclose(f(-b, e), pow_c_r(-b, e), rtol=1e-15, atol=1e-15))
    assert(isclose(f(b, -e), pow_c_r(b, -e), rtol=1e-15, atol=1e-15))
    assert(isclose(f(-b, -e), pow_c_r(-b, -e), rtol=1e-15, atol=1e-15))