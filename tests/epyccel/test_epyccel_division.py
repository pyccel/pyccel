import pytest

from pyccel.epyccel import epyccel
from pyccel.decorators import types
from conftest       import *

from numpy.random import rand, randint, uniform

# -------------------- simple division ---------------------- #

def test_call_div_i_i(language):
    @types(int, int)
    def div_i_i(x, y):
        return x / y

    f = epyccel(div_i_i, language=language)
    x = randint(1e9)
    y = randint(low=1, high= 1e3)

    assert (f(x, y) == div_i_i(x, y))
    assert (f(-x, y) == div_i_i(-x, y))
    assert (f(x, -y) == div_i_i(x, -y))
    assert (f(-x, -y) == div_i_i(-x, -y))

def test_call_div_i_r(language):
    @types(int, 'real')
    def div_i_r(x, y):
        return x / y

    f = epyccel(div_i_r, language=language)
    x = randint(1e9)
    y = uniform(low=1, high= 1e3)
    assert (f(x, y) == div_i_r(x, y))
    assert (f(-x, y) == div_i_r(-x, y))
    assert (f(x, -y) == div_i_r(x, -y))
    assert (f(-x, -y) == div_i_r(-x, -y))

def test_call_div_r_i(language):
    @types('real', int)
    def div_r_i(x, y):
        return x / y

    f = epyccel(div_r_i, language=language)
    x = uniform(high=1e9)
    y = randint(low=1, high= 1e3)
    assert (f(x, y) == div_r_i(x, y))
    assert (f(-x, y) == div_r_i(-x, y))
    assert (f(x, -y) == div_r_i(x, -y))
    assert (f(-x, -y) == div_r_i(-x, -y))

def test_call_div_r_r(language):
    @types('real', 'real')
    def div_r_r(x, y):
        return x / y

    f = epyccel(div_r_r, language=language)
    x = uniform(high=1e9)
    y = uniform(low=1e-14, high= 1e3)
    assert (f(x, y) == div_r_r(x, y))
    assert (f(-x, y) == div_r_r(-x, y))
    assert (f(x, -y) == div_r_r(x, -y))
    assert (f(-x, -y) == div_r_r(-x, -y))

# -------------------- floor division ---------------------- #

def test_call_fdiv_i_i(language):
    @types(int, int)
    def fdiv_i_i(x, y):
        return x // y

    f = epyccel(fdiv_i_i, language=language)
    x = randint(1e9)
    y = randint(low=1, high= 1e3)

    assert (f(x, y) == fdiv_i_i(x, y))
    assert (f(-x, y) == fdiv_i_i(-x, y))
    assert (f(x, -y) == fdiv_i_i(x, -y))
    assert (f(-x, -y) == fdiv_i_i(-x, -y))

def test_call_fdiv_i_r(language):
    @types(int, 'real')
    def fdiv_i_r(x, y):
        return x // y

    f = epyccel(fdiv_i_r, language=language)
    x = randint(1e9)
    y = uniform(low=1, high= 1e3)
    assert (f(x, y) == fdiv_i_r(x, y))
    assert (f(-x, y) == fdiv_i_r(-x, y))
    assert (f(x, -y) == fdiv_i_r(x, -y))
    assert (f(-x, -y) == fdiv_i_r(-x, -y))

def test_call_fdiv_r_i(language):
    @types('real', int)
    def fdiv_r_i(x, y):
        return x // y

    f = epyccel(fdiv_r_i, language=language)
    x = uniform(high=1e9)
    y = randint(low=1, high= 1e3)
    assert (f(x, y) == fdiv_r_i(x, y))
    assert (f(-x, y) == fdiv_r_i(-x, y))
    assert (f(x, -y) == fdiv_r_i(x, -y))
    assert (f(-x, -y) == fdiv_r_i(-x, -y))

def test_call_fdiv_r_r(language):
    @types('real', 'real')
    def fdiv_r_r(x, y):
        return x // y

    f = epyccel(fdiv_r_r, language=language)
    x = uniform(high=1e9)
    y = uniform(low=1e-14, high= 1e3)
    assert (f(x, y) == fdiv_r_r(x, y))
    assert (f(-x, y) == fdiv_r_r(-x, y))
    assert (f(x, -y) == fdiv_r_r(x, -y))
    assert (f(-x, -y) == fdiv_r_r(-x, -y))
