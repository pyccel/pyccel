# pylint: disable=missing-function-docstring, missing-module-docstring
import pytest
from numpy.random import rand, randint, uniform
from numpy import isclose

from pyccel.epyccel import epyccel
from pyccel.decorators import types

import sys

RTOL = 1e-13
ATOL = 1e-14

max_float = 3.40282e5        # maximum positive float
min_float = sys.float_info.min  # Minimum positive float

def test_sqrt_call(language):
    def sqrt_call(x : 'float'):
        from cmath import sqrt
        return sqrt(x)

    f1 = epyccel(sqrt_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  sqrt_call(x), rtol=RTOL, atol=ATOL))

def test_sqrt_phrase(language):
    def sqrt_phrase(x : 'float', y : 'float'):
        from cmath import sqrt
        a = sqrt(x)*sqrt(y)
        return a

    f2 = epyccel(sqrt_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  sqrt_phrase(x,y), rtol=RTOL, atol=ATOL))

def test_sqrt_return_type(language):
    def sqrt_return_type_real(x : 'float'):
        from cmath import sqrt
        a = sqrt(x)
        return a

    f1 = epyccel(sqrt_return_type_real, language = language)
    x = rand()
    assert(isclose(f1(x) ,  sqrt_return_type_real(x), rtol=RTOL, atol=ATOL))
    assert(type(f1(x))  == type(sqrt_return_type_real(x))) # pylint: disable=unidiomatic-typecheck

def test_sin_call(language):
    def sin_call(x : 'float'):
        from cmath import sin
        return sin(x)

    f1 = epyccel(sin_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  sin_call(x), rtol=RTOL, atol=ATOL))

def test_sin_phrase(language):
    def sin_phrase(x : 'float', y : 'float'):
        from cmath import sin
        a = sin(x)+sin(y)
        return a

    f2 = epyccel(sin_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  sin_phrase(x,y), rtol=RTOL, atol=ATOL))

def test_cos_call(language):
    def cos_call(x : 'float'):
        from cmath import cos
        return cos(x)

    f1 = epyccel(cos_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  cos_call(x), rtol=RTOL, atol=ATOL))

def test_cos_phrase(language):
    def cos_phrase(x : 'float', y : 'float'):
        from cmath import cos
        a = cos(x)+cos(y)
        return a

    f2 = epyccel(cos_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  cos_phrase(x,y), rtol=RTOL, atol=ATOL))


def test_tan_call(language):
    def tan_call(x : 'float'):
        from cmath import tan
        return tan(x)

    f1 = epyccel(tan_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  tan_call(x), rtol=RTOL, atol=ATOL))


def test_tan_phrase(language):
    def tan_phrase(x : 'float', y : 'float'):
        from cmath import tan
        a = tan(x)+tan(y)
        return a

    f2 = epyccel(tan_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  tan_phrase(x,y), rtol=RTOL, atol=ATOL))

def test_exp_call(language):
    def exp_call(x : 'float'):
        from cmath import exp
        return exp(x)

    f1 = epyccel(exp_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  exp_call(x), rtol=RTOL, atol=ATOL))

def test_exp_phrase(language):
    def exp_phrase(x : 'float', y : 'float'):
        from cmath import exp
        a = exp(x)+exp(y)
        return a

    f2 = epyccel(exp_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  exp_phrase(x,y), rtol=RTOL, atol=ATOL))

def test_asin_call(language):
    def asin_call(x : 'float'):
        from cmath import asin
        return asin(x)

    f1 = epyccel(asin_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  asin_call(x), rtol=RTOL, atol=ATOL))

def test_asin_phrase(language):
    def asin_phrase(x : 'float', y : 'float'):
        from cmath import asin
        a = asin(x)+asin(y)
        return a

    f2 = epyccel(asin_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  asin_phrase(x,y), rtol=RTOL, atol=ATOL))

def test_acos_call(language):
    def acos_call(x : 'float'):
        from cmath import acos
        return acos(x)

    f1 = epyccel(acos_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  acos_call(x), rtol=RTOL, atol=ATOL))

def test_acos_phrase(language):
    def acos_phrase(x : 'float', y : 'float'):
        from cmath import acos
        a = acos(x)+acos(y)
        return a

    f2 = epyccel(acos_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  acos_phrase(x,y), rtol=RTOL, atol=ATOL))

def test_atan_call(language):
    def atan_call(x : 'float'):
        from cmath import atan
        return atan(x)

    f1 = epyccel(atan_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  atan_call(x), rtol=RTOL, atol=ATOL))

def test_atan_phrase(language):
    def atan_phrase(x : 'float', y : 'float'):
        from cmath import atan
        a = atan(x)+atan(y)
        return a

    f2 = epyccel(atan_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  atan_phrase(x,y), rtol=RTOL, atol=ATOL))

def test_sinh_call(language):
    def sinh_call(x : 'float'):
        from cmath import sinh
        return sinh(x)

    f1 = epyccel(sinh_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  sinh_call(x), rtol=RTOL, atol=ATOL))

def test_sinh_phrase(language):
    def sinh_phrase(x : 'float', y : 'float'):
        from cmath import sinh
        a = sinh(x)+sinh(y)
        return a

    f2 = epyccel(sinh_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  sinh_phrase(x,y), rtol=RTOL, atol=ATOL))

def test_cosh_call(language):
    def cosh_call(x : 'float'):
        from cmath import cosh
        return cosh(x)

    f1 = epyccel(cosh_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  cosh_call(x), rtol=RTOL, atol=ATOL))

def test_cosh_phrase(language):
    def cosh_phrase(x : 'float', y : 'float'):
        from cmath import cosh
        a = cosh(x)+cosh(y)
        return a

    f2 = epyccel(cosh_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  cosh_phrase(x,y), rtol=RTOL, atol=ATOL))

def test_tanh_call(language):
    def tanh_call(x : 'float'):
        from cmath import tanh
        return tanh(x)

    f1 = epyccel(tanh_call, language = language)
    x = rand()
    assert(isclose(f1(x) ,  tanh_call(x), rtol=RTOL, atol=ATOL))

def test_tanh_phrase(language):
    def tanh_phrase(x : 'float', y : 'float'):
        from cmath import tanh
        a = tanh(x)+tanh(y)
        return a

    f2 = epyccel(tanh_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y) ,  tanh_phrase(x,y), rtol=RTOL, atol=ATOL))

#----------------------------- isfinite function -----------------------------#
@pytest.mark.parametrize( 'language', (
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="isfinite not implemented"),
            pytest.mark.fortran]
        )
    )
)
def test_isfinite_call(language): # isfinite
    def isfinite_call(x : 'float'):
        from cmath import isfinite
        return isfinite(x)

    f1 = epyccel(isfinite_call, language = language)
    x = rand()

    assert(isfinite_call(x) == f1(x))

    from cmath import nan, inf
    # Test not a number
    assert(isfinite_call(nan) == f1(nan))
    # Test infinite number
    assert(isfinite_call(inf) == f1(inf))
    # Test negative infinite number
    assert(isfinite_call(-inf) == f1(-inf))

#------------------------------- isinf function ------------------------------#
@pytest.mark.parametrize( 'language', (
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="isinf not implemented"),
            pytest.mark.fortran]
        )
    )
)
def test_isinf_call(language): # isinf
    def isinf_call(x : 'float'):
        from cmath import isinf
        return isinf(x)

    f1 = epyccel(isinf_call, language = language)
    x = rand()

    assert(isinf_call(x) == f1(x))

    from cmath import nan, inf
    # Test not a number
    assert(isinf_call(nan) == f1(nan))
    # Test infinite number
    assert(isinf_call(inf) == f1(inf))
    # Test negative infinite number
    assert(isinf_call(-inf) == f1(-inf))

#------------------------------- isnan function ------------------------------#

def test_isnan_call(language): # isnan
    def isnan_call(x : 'float'):
        from cmath import isnan
        return isnan(x)

    f1 = epyccel(isnan_call, language = language)
    x = rand()

    assert(isnan_call(x) == f1(x))

    from cmath import nan, inf
    # Test not a number
    assert(isnan_call(nan) == f1(nan))
    # Test infinite number
    assert(isnan_call(inf) == f1(inf))
    # Test negative infinite number
    assert(isnan_call(-inf) == f1(-inf))

#------------------------------- Acosh function ------------------------------#

def test_acosh_call(language):
    def acosh_call(x : 'float'):
        from cmath import acosh
        return acosh(x)

    f1 = epyccel(acosh_call, language = language)

    x = uniform(low=1, high=max_float)
    assert(isclose(f1(x) ,  acosh_call(x), rtol=RTOL, atol=ATOL))
    assert isinstance(f1(x), type(acosh_call(x)))

def test_acosh_phrase(language):
    def acosh_phrase(x : 'float', y : 'float'):
        from cmath import acosh
        a = acosh(x) + acosh(y)
        return a

    f2 = epyccel(acosh_phrase, language = language)

    x = uniform(low=1, high=max_float)
    y = uniform(low=1, high=max_float)
    assert(isclose(f2(x,y) , acosh_phrase(x,y), rtol=RTOL, atol=ATOL))


#------------------------------- Asinh function ------------------------------#

def test_asinh_call(language):
    def asinh_call(x : 'float'):
        from cmath import asinh
        return asinh(x)

    f1 = epyccel(asinh_call, language = language)

    x = uniform(high=max_float)
    assert(isclose(f1(x) , asinh_call(x), rtol=RTOL, atol=ATOL))
    assert isinstance(f1(x), type(asinh_call(x)))

    # Negative value
    assert(isclose(f1(-x) , asinh_call(-x), rtol=RTOL, atol=ATOL))

def test_asinh_phrase(language):
    def asinh_phrase(x : 'float', y : 'float'):
        from cmath import asinh
        a = asinh(x)+ asinh(y)
        return a

    f2 = epyccel(asinh_phrase, language = language)
    x = uniform(high=max_float)
    y = uniform(high=max_float)
    assert(isclose(f2(x,y), asinh_phrase(x,y), rtol=RTOL, atol=ATOL))
    # Negative value
    assert(isclose(f2(-x,-y), asinh_phrase(-x,-y), rtol=RTOL, atol=ATOL))

#------------------------------- Atanh function ------------------------------#

def test_atanh_call(language):
    def atanh_call(x : 'float'):
        from cmath import atanh
        return atanh(x)

    f1 = epyccel(atanh_call, language = language)
    low = -1 + min_float
    high = 1 - min_float
    x = uniform(low=low, high=high)
    assert(isclose(f1(x) , atanh_call(x), rtol=RTOL, atol=ATOL))
    assert isinstance(f1(x), type(atanh_call(x)))

def test_atanh_phrase(language):
    def atanh_phrase(x : 'float', y : 'float'):
        from cmath import atanh
        a = atanh(x)+ atanh(y)
        return a

    f2 = epyccel(atanh_phrase, language = language)

    # Domain ]-1, 1[
    low = -1 + min_float
    high = 1 - min_float
    x = uniform(low=low, high=high)
    y = uniform(low=low, high=high)
    assert(isclose(f2(x, y), atanh_phrase(x, y), rtol=RTOL, atol=ATOL))
