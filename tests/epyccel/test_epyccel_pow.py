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
    # negative cases
    assert f(-x, -y) == f_call(-x, -y)
    assert f(x, -y) == f_call(x, -y)
    assert f(-x, y) == f_call(-x, y)

    assert isinstance(f(x, y), type(f_call(x, y)))

# def test_pow_real_real(language):
#     @types('real', 'real')
#     def pow_r_r(x, y):
#         return x ** y
    
#     f = epyccel(pow_r_r, language=language)
#     x = rand()
#     y = rand()

#     assert(isclose(f(x, y), pow_r_r(x, y), rtol=1e-15, atol=1e-15))
#     # negative cases
#     assert(isclose(f(-x, -y), pow_r_r(-x, -y), rtol=1e-15, atol=1e-15))
#     assert(isclose(f(-x, y), pow_r_r(-x, y), rtol=1e-15, atol=1e-15))
#     assert(isclose(f(x, -y), pow_r_r(x, -y), rtol=1e-15, atol=1e-15))

#     assert isinstance(f(x, y), type(pow_r_r(x, y)))

# def test_pow_real_int(language):
#     @types('real', 'int')
#     def pow_r_i(x, y):
#         return x ** y
    
#     f = epyccel(pow_r_i, language=language)
#     x = uniform()
#     y = randint()

#     assert(isclose(f(x, y), pow_r_i(x, y), rtol=1e-15, atol=1e-15))
#     # negative cases
#     assert(isclose(f(-x, -y), pow_r_r(-x, -y), rtol=1e-15, atol=1e-15))
#     assert(isclose(f(-x, y), pow_r_r(-x, y), rtol=1e-15, atol=1e-15))
#     assert(isclose(f(x, -y), pow_r_r(x, -y), rtol=1e-15, atol=1e-15))

#     assert isinstance(f(x, y), type(pow_r_i(x, y)))

# def test_pow_int_real(language):
#     @types('int', 'real')
#     def pow_i_r(x, y):
#         return x ** y
    
#     f = epyccel(pow_i_r, language=language)
#     x = randint(40)
#     y = uniform()

#     assert(isclose(f(x, y), pow_i_r(x, y), rtol=1e-15, atol=1e-15))
#     # negative cases
#     assert(isclose(f(-x, -y), pow_r_r(-x, -y), rtol=1e-15, atol=1e-15))
#     assert(isclose(f(-x, y), pow_r_r(-x, y), rtol=1e-15, atol=1e-15))
#     assert(isclose(f(x, -y), pow_r_r(x, -y), rtol=1e-15, atol=1e-15))

#     assert isinstance(f(x, y), type(pow_i_r(x, y)))