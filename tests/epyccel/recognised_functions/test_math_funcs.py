import pytest
from numpy.random import rand, randint
from numpy import isclose

from pyccel.decorators import types
from pyccel.epyccel import epyccel
from conftest import *

def test_fabs_call(language):
    @types('real')
    def fabs_call(x):
        from math import fabs
        return fabs(x)

    f1 = epyccel(fabs_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  fabs_call(x), rtol=1e-15, atol=1e-15))

def test_fabs_phrase(language):
    @types('real','real')
    def fabs_phrase(x,y):
        from math import fabs
        a = fabs(x)*fabs(y)
        return a

    f2 = epyccel(fabs_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  fabs_phrase(x,y), rtol=1e-15, atol=1e-15))

def test_fabs_return_type(language):
    @types('int')
    def fabs_return_type(x):
        from math import fabs
        a = fabs(x)
        return a

    f1 = epyccel(fabs_return_type, language = language)
    x = randint(100)
    assert(isclose(f1(x) ,  fabs_return_type(x), rtol=1e-15, atol=1e-15))
    assert(type(f1(x))  == type(fabs_return_type(x))) # pylint: disable=unidiomatic-typecheck

def test_sqrt_call(language):
    @types('real')
    def sqrt_call(x):
        from math import sqrt
        return sqrt(x)

    f1 = epyccel(sqrt_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  sqrt_call(x), rtol=1e-15, atol=1e-15))

def test_sqrt_phrase(language):
    @types('real','real')
    def sqrt_phrase(x,y):
        from math import sqrt
        a = sqrt(x)*sqrt(y)
        return a

    f2 = epyccel(sqrt_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  sqrt_phrase(x,y), rtol=1e-15, atol=1e-15))

def test_sqrt_return_type(language):
    @types('real')
    def sqrt_return_type_real(x):
        from math import sqrt
        a = sqrt(x)
        return a

    f1 = epyccel(sqrt_return_type_real, language = language)
    x = rand()
    assert(isclose(f1(x) ,  sqrt_return_type_real(x), rtol=1e-15, atol=1e-15))
    assert(type(f1(x))  == type(sqrt_return_type_real(x))) # pylint: disable=unidiomatic-typecheck

def test_sin_call(language):
    @types('real')
    def sin_call(x):
        from math import sin
        return sin(x)

    f1 = epyccel(sin_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  sin_call(x), rtol=1e-15, atol=1e-15))

def test_sin_phrase(language):
    @types('real','real')
    def sin_phrase(x,y):
        from math import sin
        a = sin(x)+sin(y)
        return a

    f2 = epyccel(sin_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  sin_phrase(x,y), rtol=1e-15, atol=1e-15))

def test_cos_call(language):
    @types('real')
    def cos_call(x):
        from math import cos
        return cos(x)

    f1 = epyccel(cos_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  cos_call(x), rtol=1e-15, atol=1e-15))

def test_cos_phrase(language):
    @types('real','real')
    def cos_phrase(x,y):
        from math import cos
        a = cos(x)+cos(y)
        return a

    f2 = epyccel(cos_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  cos_phrase(x,y), rtol=1e-15, atol=1e-15))


def test_tan_call(language):
    @types('real')
    def tan_call(x):
        from math import tan
        return tan(x)

    f1 = epyccel(tan_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  tan_call(x), rtol=1e-15, atol=1e-15))


def test_tan_phrase(language):
    @types('real','real')
    def tan_phrase(x,y):
        from math import tan
        a = tan(x)+tan(y)
        return a

    f2 = epyccel(tan_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  tan_phrase(x,y), rtol=1e-15, atol=1e-15))

def test_exp_call(language):
    @types('real')
    def exp_call(x):
        from math import exp
        return exp(x)

    f1 = epyccel(exp_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  exp_call(x), rtol=1e-15, atol=1e-15))

def test_exp_phrase(language):
    @types('real','real')
    def exp_phrase(x,y):
        from math import exp
        a = exp(x)+exp(y)
        return a

    f2 = epyccel(exp_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  exp_phrase(x,y), rtol=1e-15, atol=1e-15))

def test_log_call(language):
    @types('real')
    def log_call(x):
        from math import log
        return log(x)

    f1 = epyccel(log_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  log_call(x), rtol=1e-15, atol=1e-15))

def test_log_phrase(language):
    @types('real','real')
    def log_phrase(x,y):
        from math import log
        a = log(x)+log(y)
        return a

    f2 = epyccel(log_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  log_phrase(x,y), rtol=1e-15, atol=1e-15))

def test_asin_call(language):
    @types('real')
    def asin_call(x):
        from math import asin
        return asin(x)

    f1 = epyccel(asin_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  asin_call(x), rtol=1e-15, atol=1e-15))

def test_asin_phrase(language):
    @types('real','real')
    def asin_phrase(x,y):
        from math import asin
        a = asin(x)+asin(y)
        return a

    f2 = epyccel(asin_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  asin_phrase(x,y), rtol=1e-15, atol=1e-15))

def test_acos_call(language):
    @types('real')
    def acos_call(x):
        from math import acos
        return acos(x)

    f1 = epyccel(acos_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  acos_call(x), rtol=1e-15, atol=1e-15))

def test_acos_phrase(language):
    @types('real','real')
    def acos_phrase(x,y):
        from math import acos
        a = acos(x)+acos(y)
        return a

    f2 = epyccel(acos_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  acos_phrase(x,y), rtol=1e-15, atol=1e-15))

def test_atan_call(language):
    @types('real')
    def atan_call(x):
        from math import atan
        return atan(x)

    f1 = epyccel(atan_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  atan_call(x), rtol=1e-15, atol=1e-15))

def test_atan_phrase(language):
    @types('real','real')
    def atan_phrase(x,y):
        from math import atan
        a = atan(x)+atan(y)
        return a

    f2 = epyccel(atan_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  atan_phrase(x,y), rtol=1e-15, atol=1e-15))

def test_sinh_call(language):
    @types('real')
    def sinh_call(x):
        from math import sinh
        return sinh(x)

    f1 = epyccel(sinh_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  sinh_call(x), rtol=1e-15, atol=1e-15))

def test_sinh_phrase(language):
    @types('real','real')
    def sinh_phrase(x,y):
        from math import sinh
        a = sinh(x)+sinh(y)
        return a

    f2 = epyccel(sinh_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  sinh_phrase(x,y), rtol=1e-15, atol=1e-15))

def test_cosh_call(language):
    @types('real')
    def cosh_call(x):
        from math import cosh
        return cosh(x)

    f1 = epyccel(cosh_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  cosh_call(x), rtol=1e-15, atol=1e-15))

def test_cosh_phrase(language):
    @types('real','real')
    def cosh_phrase(x,y):
        from math import cosh
        a = cosh(x)+cosh(y)
        return a

    f2 = epyccel(cosh_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  cosh_phrase(x,y), rtol=1e-15, atol=1e-15))

def test_tanh_call(language):
    @types('real')
    def tanh_call(x):
        from math import tanh
        return tanh(x)

    f1 = epyccel(tanh_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  tanh_call(x), rtol=1e-15, atol=1e-15))

def test_tanh_phrase(language):
    @types('real','real')
    def tanh_phrase(x,y):
        from math import tanh
        a = tanh(x)+tanh(y)
        return a

    f2 = epyccel(tanh_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  tanh_phrase(x,y), rtol=1e-15, atol=1e-15))

@pytest.mark.xfail(reason = "scipy translation error (see issue #207)")
def test_atan2_call(language):
    @types('real')
    def atan2_call(x,y):
        from math import atan2
        return atan2(x)

    f1 = epyccel(atan2_call, language = language)
    x = rand()
    y = rand()
    assert(isclose(f1(x,y) ,  atan2_call(x,y), rtol=1e-15, atol=1e-15))

@pytest.mark.xfail(reason = "scipy translation error (see issue #207)")
def test_atan2_phrase(language):
    @types('real','real')
    def atan2_phrase(x,y,z):
        from math import atan2
        a = atan2(x,y)+atan2(x,y,z)
        return a

    f2 = epyccel(atan2_phrase, language = language)
    x = rand()
    y = rand()
    z = rand()
    assert(isclose(f2(x,y,z) ,  atan2_phrase(x,y,z), rtol=1e-15, atol=1e-15))

def test_floor_call(language):
    @types('real')
    def floor_call(x):
        from math import floor
        return floor(x)

    f1 = epyccel(floor_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  floor_call(x), rtol=1e-15, atol=1e-15))

def test_floor_phrase(language):
    @types('real','real')
    def floor_phrase(x,y):
        from math import floor
        a = floor(x)*floor(y)
        return a

    f2 = epyccel(floor_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  floor_phrase(x,y), rtol=1e-15, atol=1e-15))

def test_floor_return_type(language):
    @types('int')
    def floor_return_type_int(x):
        from math import floor
        a = floor(x)
        return a

    @types('real')
    def floor_return_type_real(x):
        from math import floor
        a = floor(x)
        return a

    f1 = epyccel(floor_return_type_int, language = language)
    x = randint(100)
    assert(isclose(f1(x) ,  floor_return_type_int(x), rtol=1e-15, atol=1e-15))
    assert(type(f1(x))  == type(floor_return_type_int(x))) # pylint: disable=unidiomatic-typecheck

    f1 = epyccel(floor_return_type_real, language = language)
    x = randint(100)
    assert(isclose(f1(x) ,  floor_return_type_real(x), rtol=1e-15, atol=1e-15))
    assert(type(f1(x))  == type(floor_return_type_real(x))) # pylint: disable=unidiomatic-typecheck
