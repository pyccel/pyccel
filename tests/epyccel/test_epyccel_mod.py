# pylint: disable=missing-function-docstring, missing-module-docstring/
from numpy.random import randint, uniform
from numpy import isclose

from pyccel.decorators import types
from pyccel.epyccel import epyccel

def test_modulo_int_int(language):
    @types(int, int)
    def modulo_i_i(x, y):
        return x % y

    f = epyccel(modulo_i_i, language=language)
    x = randint(1e6)
    y = randint(low=1, high=100) # low=1 for avoid zero-division error

    assert f(x, y) == modulo_i_i(x, y)
    assert isinstance(f(x, y), type(modulo_i_i(x, y)))

def test_modulo_real_real(language):
    @types('real', 'real')
    def modulo_r_r(x, y):
        return x % y

    f = epyccel(modulo_r_r, language=language)
    x = uniform(high=1e6)
    y = uniform(low=1, high=1e2) # low=1 for avoid zero-division error

    assert(isclose(f(x, y), modulo_r_r(x, y), rtol=1e-15, atol=1e-15))
    assert isinstance(f(x, y), type(modulo_r_r(x, y)))

def test_modulo_real_int(language):
    @types('real', 'int')
    def modulo_r_i(x, y):
        return x % y

    f = epyccel(modulo_r_i, language=language)
    x = uniform(high=1e6)
    y = randint(low=1, high=100) # low=1 for avoid zero-division error

    assert(isclose(f(x, y), modulo_r_i(x, y), rtol=1e-15, atol=1e-15))
    assert isinstance(f(x, y), type(modulo_r_i(x, y)))

def test_modulo_int_real(language):
    @types('int', 'real')
    def modulo_i_r(x, y):
        return x % y

    f = epyccel(modulo_i_r, language=language)
    x = randint(1e6)
    y = uniform(low=1, high=1e2) # low=1 for avoid zero-division error

    assert(isclose(f(x, y), modulo_i_r(x, y), rtol=1e-15, atol=1e-15))
    assert isinstance(f(x, y), type(modulo_i_r(x, y)))

def test_modulo_multiple(language):
    @types('int', 'real', 'int')
    def modulo_multiple(x, y, z):
        return x % y % z

    f = epyccel(modulo_multiple, language=language)
    x = randint(1e6)
    y = uniform(low=1e2, high=1e4)
    z = randint(low=1, high=1e2)

    assert(isclose(f(x, y, z), modulo_multiple(x, y, z), rtol=1e-15, atol=1e-15))
    assert isinstance(f(x, y, z), type(modulo_multiple(x, y, z)))
