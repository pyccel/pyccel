# pylint: disable=missing-function-docstring, missing-module-docstring
import sys
import pytest
from numpy.random import rand, randint, uniform
from numpy import isclose, iinfo, finfo
import numpy as np

from pyccel.decorators import template, types
from pyccel.epyccel import epyccel

min_int8 = iinfo('int8').min
max_int8 = iinfo('int8').max

min_int16 = iinfo('int16').min
max_int16 = iinfo('int16').max

min_int = iinfo('int').min
max_int = iinfo('int').max

min_int32 = iinfo('int32').min
max_int32 = iinfo('int32').max

min_int64 = iinfo('int64').min
max_int64 = iinfo('int64').max

min_float = finfo('float').min
max_float = finfo('float').max

min_float32 = finfo('float32').min
max_float32 = finfo('float32').max

min_float64 = finfo('float64').min
max_float64 = finfo('float64').max

# Functions still to be tested:
#    diag
#    cross
#    # ---

# Relative and absolute tolerances for array comparisons in the form
# numpy.isclose(a, b, rtol, atol). Windows has larger round-off errors.
if sys.platform == 'win32':
    RTOL = 1e-13
    ATOL = 1e-14
else:
    RTOL = 2e-14
    ATOL = 1e-15

RTOL32 = 1e-5
ATOL32 = 1e-6

def matching_types(pyccel_result, python_result):
    """  Returns True if the types match, False otherwise
    """
    if type(pyccel_result) is type(python_result):
        return True
    return (isinstance(pyccel_result, bool) and isinstance(python_result, np.bool_)) \
            or \
           (isinstance(pyccel_result, np.int32) and isinstance(python_result, np.intc))

#-------------------------------- Fabs function ------------------------------#
def test_fabs_call_r(language):
    def fabs_call_r(x : 'float'):
        from numpy import fabs
        return fabs(x)

    f1 = epyccel(fabs_call_r, language = language)
    x = uniform(high=1e6)
    assert(isclose(f1(x), fabs_call_r(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), fabs_call_r(-x), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x), fabs_call_r(x))

def test_fabs_call_i(language):
    def fabs_call_i(x : 'int'):
        from numpy import fabs
        return fabs(x)

    f1 = epyccel(fabs_call_i, language = language)
    x = randint(1e6)
    assert(isclose(f1(x), fabs_call_i(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), fabs_call_i(-x), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x), fabs_call_i(x))

def test_fabs_phrase_r_r(language):
    def fabs_phrase_r_r(x : 'float', y : 'float'):
        from numpy import fabs
        a = fabs(x)*fabs(y)
        return a

    f2 = epyccel(fabs_phrase_r_r, language = language)
    x = uniform(high=1e6)
    y = uniform(high=1e6)
    assert(isclose(f2(x,y), fabs_phrase_r_r(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), fabs_phrase_r_r(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), fabs_phrase_r_r(x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), fabs_phrase_r_r(-x,y), rtol=RTOL, atol=ATOL))

def test_fabs_phrase_i_i(language):
    def fabs_phrase_i_i(x : 'int', y : 'int'):
        from numpy import fabs
        a = fabs(x)*fabs(y)
        return a

    f2 = epyccel(fabs_phrase_i_i, language = language)
    x = randint(1e6)
    y = randint(1e6)
    assert(isclose(f2(x,y), fabs_phrase_i_i(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), fabs_phrase_i_i(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), fabs_phrase_i_i(x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), fabs_phrase_i_i(-x,y), rtol=RTOL, atol=ATOL))

def test_fabs_phrase_r_i(language):
    def fabs_phrase_r_i(x : 'float', y : 'int'):
        from numpy import fabs
        a = fabs(x)*fabs(y)
        return a

    f2 = epyccel(fabs_phrase_r_i, language = language)
    x = uniform(high=1e6)
    y = randint(1e6)
    assert(isclose(f2(x,y), fabs_phrase_r_i(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), fabs_phrase_r_i(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), fabs_phrase_r_i(x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), fabs_phrase_r_i(-x,y), rtol=RTOL, atol=ATOL))

def test_fabs_phrase_i_r(language):
    def fabs_phrase_r_i(x : 'int', y : 'float'):
        from numpy import fabs
        a = fabs(x)*fabs(y)
        return a

    f2 = epyccel(fabs_phrase_r_i, language = language)
    x = randint(1e6)
    y = uniform(high=1e6)
    assert(isclose(f2(x,y), fabs_phrase_r_i(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), fabs_phrase_r_i(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), fabs_phrase_r_i(x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), fabs_phrase_r_i(-x,y), rtol=RTOL, atol=ATOL))

#------------------------------ absolute function ----------------------------#
def test_absolute_call_r(language):
    def absolute_call_r(x : 'float'):
        from numpy import absolute
        return absolute(x)

    f1 = epyccel(absolute_call_r, language = language)
    x = uniform(high=1e6)
    assert f1(x) == absolute_call_r(x)
    assert f1(-x) == absolute_call_r(-x)
    assert matching_types(f1(x), absolute_call_r(x))

def test_absolute_call_i(language):
    def absolute_call_i(x : 'int'):
        from numpy import absolute
        return absolute(x)

    f1 = epyccel(absolute_call_i, language = language)
    x = randint(1e6)
    assert f1(x) == absolute_call_i(x)
    assert f1(-x) == absolute_call_i(-x)
    assert matching_types(f1(x), absolute_call_i(x))

def test_absolute_call_c(language):
    @template(name='T', types=['complex','complex64','complex128'])
    def absolute_call_c(x : 'T'):
        from numpy import absolute
        return absolute(x)

    f1 = epyccel(absolute_call_c, language = language)
    x = uniform(high=1e6)+1j*uniform(high=1e6)
    assert(isclose(f1(x), absolute_call_c(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), absolute_call_c(-x), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x), absolute_call_c(x))

    x = np.complex64(uniform(high=1e6)-1j*uniform(high=1e6))
    assert(isclose(f1(x), absolute_call_c(x), rtol=RTOL32, atol=ATOL32))
    assert matching_types(f1(x), absolute_call_c(x))

    x = np.complex128(uniform(high=1e6)-1j*uniform(high=1e6))
    assert(isclose(f1(x), absolute_call_c(x), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x), absolute_call_c(x))

def test_absolute_phrase_r_r(language):
    def absolute_phrase_r_r(x : 'float', y : 'float'):
        from numpy import absolute
        a = absolute(x)*absolute(y)
        return a

    f2 = epyccel(absolute_phrase_r_r, language = language)
    x = uniform(high=1e6)
    y = uniform(high=1e6)
    assert(isclose(f2(x,y), absolute_phrase_r_r(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), absolute_phrase_r_r(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), absolute_phrase_r_r(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), absolute_phrase_r_r(x,-y), rtol=RTOL, atol=ATOL))

def test_absolute_phrase_i_r(language):
    def absolute_phrase_i_r(x : 'int', y : 'float'):
        from numpy import absolute
        a = absolute(x)*absolute(y)
        return a

    f2 = epyccel(absolute_phrase_i_r, language = language)
    x = randint(1e6)
    y = uniform(high=1e6)
    assert(isclose(f2(x,y), absolute_phrase_i_r(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), absolute_phrase_i_r(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), absolute_phrase_i_r(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), absolute_phrase_i_r(x,-y), rtol=RTOL, atol=ATOL))

def test_absolute_phrase_r_i(language):
    def absolute_phrase_r_i(x : 'float', y : 'int'):
        from numpy import absolute
        a = absolute(x)*absolute(y)
        return a

    f2 = epyccel(absolute_phrase_r_i, language = language)
    x = uniform(high=1e6)
    y = randint(1e6)
    assert(isclose(f2(x,y), absolute_phrase_r_i(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), absolute_phrase_r_i(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), absolute_phrase_r_i(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), absolute_phrase_r_i(x,-y), rtol=RTOL, atol=ATOL))

#--------------------------------- sin function ------------------------------#
def test_sin_call_r(language):
    def sin_call_r(x : 'float'):
        from numpy import sin
        return sin(x)

    f1 = epyccel(sin_call_r, language = language)
    x = uniform(high=1e6)
    assert(isclose(f1(x), sin_call_r(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), sin_call_r(-x), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x), sin_call_r(x))

def test_sin_call_i(language):
    def sin_call_i(x : 'int'):
        from numpy import sin
        return sin(x)

    f1 = epyccel(sin_call_i, language = language)
    x = randint(1e6)
    assert(isclose(f1(x), sin_call_i(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), sin_call_i(-x), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x), sin_call_i(x))

def test_sin_phrase_r_r(language):
    def sin_phrase_r_r(x : 'float', y : 'float'):
        from numpy import sin
        a = sin(x)+sin(y)
        return a

    f2 = epyccel(sin_phrase_r_r, language = language)
    x = uniform(high=1e6)
    y = uniform(high=1e6)
    assert(isclose(f2(x,y), sin_phrase_r_r(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), sin_phrase_r_r(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), sin_phrase_r_r(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), sin_phrase_r_r(x,-y), rtol=RTOL, atol=ATOL))

def test_sin_phrase_i_i(language):
    def sin_phrase_i_i(x : 'int', y : 'int'):
        from numpy import sin
        a = sin(x)+sin(y)
        return a

    f2 = epyccel(sin_phrase_i_i, language = language)
    x = randint(1e6)
    y = randint(1e6)
    assert(isclose(f2(x,y), sin_phrase_i_i(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), sin_phrase_i_i(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), sin_phrase_i_i(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), sin_phrase_i_i(x,-y), rtol=RTOL, atol=ATOL))

def test_sin_phrase_i_r(language):
    def sin_phrase_i_r(x : 'int', y : 'float'):
        from numpy import sin
        a = sin(x)+sin(y)
        return a

    f2 = epyccel(sin_phrase_i_r, language = language)
    x = randint(1e6)
    y = uniform(high=1e6)
    assert(isclose(f2(x,y), sin_phrase_i_r(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), sin_phrase_i_r(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), sin_phrase_i_r(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), sin_phrase_i_r(x,-y), rtol=RTOL, atol=ATOL))

def test_sin_phrase_r_i(language):
    def sin_phrase_r_i(x : 'float', y : 'int'):
        from numpy import sin
        a = sin(x)+sin(y)
        return a

    f2 = epyccel(sin_phrase_r_i, language = language)
    x = uniform(high=1e6)
    y = randint(1e6)
    assert(isclose(f2(x,y), sin_phrase_r_i(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), sin_phrase_r_i(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), sin_phrase_r_i(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), sin_phrase_r_i(x,-y), rtol=RTOL, atol=ATOL))

#--------------------------------- cos function ------------------------------#
def test_cos_call_i(language):
    def cos_call_i(x : 'int'):
        from numpy import cos
        return cos(x)

    f1 = epyccel(cos_call_i, language = language)
    x = randint(1e6)
    assert(isclose(f1(x), cos_call_i(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), cos_call_i(-x), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x), cos_call_i(x))

def test_cos_call_r(language):
    def cos_call_r(x : 'float'):
        from numpy import cos
        return cos(x)

    f1 = epyccel(cos_call_r, language = language)
    x = uniform(high=1e6)
    assert(isclose(f1(x), cos_call_r(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), cos_call_r(-x), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x), cos_call_r(x))


def test_cos_phrase_i_i(language):
    def cos_phrase_i_i(x : 'int', y : 'int'):
        from numpy import cos
        a = cos(x)+cos(y)
        return a

    f2 = epyccel(cos_phrase_i_i, language = language)
    x = randint(1e6)
    y = randint(1e6)
    assert(isclose(f2(x,y), cos_phrase_i_i(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), cos_phrase_i_i(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), cos_phrase_i_i(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), cos_phrase_i_i(x,-y), rtol=RTOL, atol=ATOL))

def test_cos_phrase_r_r(language):
    def cos_phrase_r_r(x : 'float', y : 'float'):
        from numpy import cos
        a = cos(x)+cos(y)
        return a

    f2 = epyccel(cos_phrase_r_r, language = language)
    x = uniform(high=1e6)
    y = uniform(high=1e6)
    assert(isclose(f2(x,y), cos_phrase_r_r(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), cos_phrase_r_r(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), cos_phrase_r_r(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), cos_phrase_r_r(x,-y), rtol=RTOL, atol=ATOL))

def test_cos_phrase_i_r(language):
    def cos_phrase_i_r(x : 'int', y : 'float'):
        from numpy import cos
        a = cos(x)+cos(y)
        return a

    f2 = epyccel(cos_phrase_i_r, language = language)
    x = randint(1e6)
    y = uniform(high=1e6)
    assert(isclose(f2(x,y), cos_phrase_i_r(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), cos_phrase_i_r(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), cos_phrase_i_r(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), cos_phrase_i_r(x,-y), rtol=RTOL, atol=ATOL))

def test_cos_phrase_r_i(language):
    def cos_phrase_r_i(x : 'float', y : 'int'):
        from numpy import cos
        a = cos(x)+cos(y)
        return a

    f2 = epyccel(cos_phrase_r_i, language = language)
    x = uniform(high=1e6)
    y = randint(1e6)
    assert(isclose(f2(x,y), cos_phrase_r_i(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), cos_phrase_r_i(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), cos_phrase_r_i(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), cos_phrase_r_i(x,-y), rtol=RTOL, atol=ATOL))

#--------------------------------- tan function ------------------------------#
def test_tan_call_i(language):
    def tan_call_i(x : 'int'):
        from numpy import tan
        return tan(x)

    f1 = epyccel(tan_call_i, language = language)
    x = randint(1e6)
    assert(isclose(f1(x), tan_call_i(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), tan_call_i(-x), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x), tan_call_i(x))

def test_tan_call_r(language):
    def tan_call_r(x : 'float'):
        from numpy import tan
        return tan(x)

    f1 = epyccel(tan_call_r, language = language)
    x = uniform(high=1e6)
    assert(isclose(f1(x), tan_call_r(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), tan_call_r(-x), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x), tan_call_r(x))

def test_tan_phrase_i_i(language):
    def tan_phrase_i_i(x : 'int', y : 'int'):
        from numpy import tan
        a = tan(x)+tan(y)
        return a

    f2 = epyccel(tan_phrase_i_i, language = language)
    x = randint(1e6)
    y = randint(1e6)
    assert(isclose(f2(x,y), tan_phrase_i_i(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), tan_phrase_i_i(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), tan_phrase_i_i(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), tan_phrase_i_i(x,-y), rtol=RTOL, atol=ATOL))

def test_tan_phrase_r_r(language):
    def tan_phrase_r_r(x : 'float', y : 'float'):
        from numpy import tan
        a = tan(x)+tan(y)
        return a

    f2 = epyccel(tan_phrase_r_r, language = language)
    x = uniform(high=1e6)
    y = uniform(high=1e6)
    assert(isclose(f2(x,y), tan_phrase_r_r(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), tan_phrase_r_r(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), tan_phrase_r_r(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), tan_phrase_r_r(x,-y), rtol=RTOL, atol=ATOL))

def test_tan_phrase_i_r(language):
    def tan_phrase_i_r(x : 'int', y : 'float'):
        from numpy import tan
        a = tan(x)+tan(y)
        return a

    f2 = epyccel(tan_phrase_i_r, language = language)
    x = randint(1e6)
    y = uniform(high=1e6)
    assert(isclose(f2(x,y), tan_phrase_i_r(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), tan_phrase_i_r(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), tan_phrase_i_r(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), tan_phrase_i_r(x,-y), rtol=RTOL, atol=ATOL))

def test_tan_phrase_r_i(language):
    def tan_phrase_r_i(x : 'float', y : 'int'):
        from numpy import tan
        a = tan(x)+tan(y)
        return a

    f2 = epyccel(tan_phrase_r_i, language = language)
    x = uniform(high=1e6)
    y = randint(1e6)
    assert(isclose(f2(x,y), tan_phrase_r_i(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), tan_phrase_r_i(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), tan_phrase_r_i(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), tan_phrase_r_i(x,-y), rtol=RTOL, atol=ATOL))

#--------------------------------- exp function ------------------------------#
def test_exp_call_i(language):
    def exp_call_i(x : 'int'):
        from numpy import exp
        return exp(x)

    f1 = epyccel(exp_call_i, language = language)
    x = randint(1e2)
    assert(isclose(f1(x), exp_call_i(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), exp_call_i(-x), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x), exp_call_i(x))

def test_exp_call_r(language):
    def exp_call_r(x : 'float'):
        from numpy import exp
        return exp(x)

    f1 = epyccel(exp_call_r, language = language)
    x = uniform(high=1e2)
    assert(isclose(f1(x), exp_call_r(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), exp_call_r(-x), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x), exp_call_r(x))

def test_exp_phrase_i_i(language):
    def exp_phrase_i_i(x : 'int', y : 'int'):
        from numpy import exp
        a = exp(x)+exp(y)
        return a

    f2 = epyccel(exp_phrase_i_i, language = language)
    x = randint(1e2)
    y = randint(1e2)
    assert(isclose(f2(x,y), exp_phrase_i_i(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), exp_phrase_i_i(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), exp_phrase_i_i(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), exp_phrase_i_i(x,-y), rtol=RTOL, atol=ATOL))

def test_exp_phrase_r_r(language):
    def exp_phrase_r_r(x : 'float', y : 'float'):
        from numpy import exp
        a = exp(x)+exp(y)
        return a

    f2 = epyccel(exp_phrase_r_r, language = language)
    x = uniform(high=1e2)
    y = uniform(high=1e2)
    assert(isclose(f2(x,y), exp_phrase_r_r(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), exp_phrase_r_r(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), exp_phrase_r_r(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), exp_phrase_r_r(x,-y), rtol=RTOL, atol=ATOL))

def test_exp_phrase_i_r(language):
    def exp_phrase_i_r(x : 'int', y : 'float'):
        from numpy import exp
        a = exp(x)+exp(y)
        return a

    f2 = epyccel(exp_phrase_i_r, language = language)
    x = randint(1e2)
    y = uniform(high=1e2)
    assert(isclose(f2(x,y), exp_phrase_i_r(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), exp_phrase_i_r(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), exp_phrase_i_r(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), exp_phrase_i_r(x,-y), rtol=RTOL, atol=ATOL))

def test_exp_phrase_r_i(language):
    def exp_phrase_r_i(x : 'float', y : 'int'):
        from numpy import exp
        a = exp(x)+exp(y)
        return a

    f2 = epyccel(exp_phrase_r_i, language = language)
    x = uniform(high=1e2)
    y = randint(1e2)
    assert(isclose(f2(x,y), exp_phrase_r_i(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,y), exp_phrase_r_i(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,y), exp_phrase_r_i(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), exp_phrase_r_i(x,-y), rtol=RTOL, atol=ATOL))

#--------------------------------- log function ------------------------------#
def test_log_call_i(language):
    def log_call_i(x : 'int'):
        from numpy import log
        return log(x)

    f1 = epyccel(log_call_i, language = language)
    x = randint(low=sys.float_info.min, high=1e6)
    assert(isclose(f1(x), log_call_i(x), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x), log_call_i(x))

def test_log_call_r(language):
    def log_call_r(x : 'float'):
        from numpy import log
        return log(x)

    f1 = epyccel(log_call_r, language = language)
    x = uniform(low=sys.float_info.min, high=max_float)
    assert(isclose(f1(x), log_call_r(x), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x), log_call_r(x))

def test_log_phrase(language):
    def log_phrase(x : 'float', y : 'float'):
        from numpy import log
        a = log(x)+log(y)
        return a

    f2 = epyccel(log_phrase, language = language)
    x = uniform(low=sys.float_info.min, high=1e6)
    y = uniform(low=sys.float_info.min, high=1e6)
    assert(isclose(f2(x,y), log_phrase(x,y), rtol=RTOL, atol=ATOL))

#----------------------------- arcsin function -------------------------------#
def test_arcsin_call_i(language):
    def arcsin_call_i(x : 'int'):
        from numpy import arcsin
        return arcsin(x)

    f1 = epyccel(arcsin_call_i, language = language)
    x = randint(2)
    assert(isclose(f1(x), arcsin_call_i(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), arcsin_call_i(-x), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x), arcsin_call_i(x))

def test_arcsin_call_r(language):
    def arcsin_call_r(x : 'float'):
        from numpy import arcsin
        return arcsin(x)

    f1 = epyccel(arcsin_call_r, language = language)
    x = rand()
    assert(isclose(f1(x), arcsin_call_r(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), arcsin_call_r(-x), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x), arcsin_call_r(x))

def test_arcsin_phrase(language):
    def arcsin_phrase(x : 'float', y : 'float'):
        from numpy import arcsin
        a = arcsin(x)+arcsin(y)
        return a

    f2 = epyccel(arcsin_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y), arcsin_phrase(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), arcsin_phrase(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), arcsin_phrase(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), arcsin_phrase(x,-y), rtol=RTOL, atol=ATOL))

#----------------------------- arccos function -------------------------------#

def test_arccos_call_i(language):
    def arccos_call_i(x : 'int'):
        from numpy import arccos
        return arccos(x)

    f1 = epyccel(arccos_call_i, language = language)
    x = randint(2)
    assert(isclose(f1(x), arccos_call_i(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), arccos_call_i(-x), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x), arccos_call_i(x))

def test_arccos_call_r(language):
    def arccos_call_r(x : 'float'):
        from numpy import arccos
        return arccos(x)

    f1 = epyccel(arccos_call_r, language = language)
    x = rand()
    assert(isclose(f1(x), arccos_call_r(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), arccos_call_r(-x), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x), arccos_call_r(x))

def test_arccos_phrase(language):
    def arccos_phrase(x : 'float', y : 'float'):
        from numpy import arccos
        a = arccos(x)+arccos(y)
        return a

    f2 = epyccel(arccos_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y), arccos_phrase(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), arccos_phrase(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), arccos_phrase(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), arccos_phrase(x,-y), rtol=RTOL, atol=ATOL))

#----------------------------- arctan function -------------------------------#
def test_arctan_call_i(language):
    def arctan_call_i(x : 'int'):
        from numpy import arctan
        return arctan(x)

    f1 = epyccel(arctan_call_i, language = language)
    x = randint(1e6)
    assert(isclose(f1(x), arctan_call_i(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), arctan_call_i(-x), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x), arctan_call_i(x))

def test_arctan_call_r(language):
    def arctan_call_r(x : 'float'):
        from numpy import arctan
        return arctan(x)

    f1 = epyccel(arctan_call_r, language = language)
    x = uniform(high=1e6)
    assert(isclose(f1(x), arctan_call_r(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), arctan_call_r(-x), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x), arctan_call_r(x))

def test_arctan_phrase(language):
    def arctan_phrase(x : 'float', y : 'float'):
        from numpy import arctan
        a = arctan(x)+arctan(y)
        return a

    f2 = epyccel(arctan_phrase, language = language)
    x = uniform(high=1e6)
    y = uniform(high=1e6)
    assert(isclose(f2(x,y), arctan_phrase(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), arctan_phrase(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), arctan_phrase(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), arctan_phrase(x,-y), rtol=RTOL, atol=ATOL))

#------------------------------- sinh function -------------------------------#
def test_sinh_call_i(language):
    def sinh_call_i(x : 'int'):
        from numpy import sinh
        return sinh(x)

    f1 = epyccel(sinh_call_i, language = language)
    x = randint(100)
    assert(isclose(f1(x), sinh_call_i(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), sinh_call_i(-x), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x), sinh_call_i(x))

def test_sinh_call_r(language):
    def sinh_call_r(x : 'float'):
        from numpy import sinh
        return sinh(x)

    f1 = epyccel(sinh_call_r, language = language)
    x = uniform(high=1e2)
    assert(isclose(f1(x), sinh_call_r(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), sinh_call_r(-x), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x), sinh_call_r(x))

def test_sinh_phrase(language):
    def sinh_phrase(x : 'float', y : 'float'):
        from numpy import sinh
        a = sinh(x)+sinh(y)
        return a

    f2 = epyccel(sinh_phrase, language = language)
    x = uniform(high=1e2)
    y = uniform(high=1e2)
    assert(isclose(f2(x,y), sinh_phrase(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), sinh_phrase(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), sinh_phrase(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), sinh_phrase(x,-y), rtol=RTOL, atol=ATOL))

#------------------------------- sinh function -------------------------------#
def test_cosh_call_i(language):
    def cosh_call_i(x : 'int'):
        from numpy import cosh
        return cosh(x)

    f1 = epyccel(cosh_call_i, language = language)
    x = randint(100)
    assert(isclose(f1(x), cosh_call_i(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), cosh_call_i(-x), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x), cosh_call_i(x))

def test_cosh_call_r(language):
    def cosh_call_r(x : 'float'):
        from numpy import cosh
        return cosh(x)

    f1 = epyccel(cosh_call_r, language = language)
    x = uniform(high=1e2)
    assert(isclose(f1(x), cosh_call_r(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), cosh_call_r(-x), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x), cosh_call_r(x))

def test_cosh_phrase(language):
    def cosh_phrase(x : 'float', y : 'float'):
        from numpy import cosh
        a = cosh(x)+cosh(y)
        return a

    f2 = epyccel(cosh_phrase, language = language)
    x = uniform(high=1e2)
    y = uniform(high=1e2)
    assert(isclose(f2(x,y), cosh_phrase(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), cosh_phrase(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), cosh_phrase(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), cosh_phrase(x,-y), rtol=RTOL, atol=ATOL))

#------------------------------- sinh function -------------------------------#
def test_tanh_call_i(language):
    def tanh_call_i(x : 'int'):
        from numpy import tanh
        return tanh(x)

    f1 = epyccel(tanh_call_i, language = language)
    x = randint(100)
    assert(isclose(f1(x), tanh_call_i(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), tanh_call_i(-x), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x), tanh_call_i(x))

def test_tanh_call_r(language):
    def tanh_call_r(x : 'float'):
        from numpy import tanh
        return tanh(x)

    f1 = epyccel(tanh_call_r, language = language)
    x = uniform(high=1e2)
    assert(isclose(f1(x), tanh_call_r(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), tanh_call_r(-x), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x), tanh_call_r(x))

def test_tanh_phrase(language):
    def tanh_phrase(x : 'float', y : 'float'):
        from numpy import tanh
        a = tanh(x)+tanh(y)
        return a

    f2 = epyccel(tanh_phrase, language = language)
    x = uniform(high=1e2)
    y = uniform(high=1e2)
    assert(isclose(f2(x,y), tanh_phrase(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), tanh_phrase(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), tanh_phrase(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), tanh_phrase(x,-y), rtol=RTOL, atol=ATOL))

#------------------------------ arctan2 function -----------------------------#
def test_arctan2_call_i_i(language):
    def arctan2_call(x : 'int', y : 'int'):
        from numpy import arctan2
        return arctan2(x,y)

    f1 = epyccel(arctan2_call, language = language)
    x = randint(100)
    y = randint(100)
    assert(isclose(f1(x,y), arctan2_call(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x,-y), arctan2_call(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x,y), arctan2_call(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(x,-y), arctan2_call(x,-y), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x, y), arctan2_call(x, y))

def test_arctan2_call_i_r(language):
    def arctan2_call(x : 'int', y : 'float'):
        from numpy import arctan2
        return arctan2(x,y)

    f1 = epyccel(arctan2_call, language = language)
    x = randint(100)
    y = uniform(high=1e2)
    assert(isclose(f1(x,y), arctan2_call(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x,-y), arctan2_call(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x,y), arctan2_call(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(x,-y), arctan2_call(x,-y), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x, y), arctan2_call(x, y))

def test_arctan2_call_r_i(language):
    def arctan2_call(x : 'float', y : 'int'):
        from numpy import arctan2
        return arctan2(x,y)

    f1 = epyccel(arctan2_call, language = language)
    x = uniform(high=1e2)
    y = randint(100)
    assert(isclose(f1(x,y), arctan2_call(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x,-y), arctan2_call(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x,y), arctan2_call(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(x,-y), arctan2_call(x,-y), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x, y), arctan2_call(x, y))

def test_arctan2_call_r_r(language):
    def arctan2_call(x : 'float', y : 'float'):
        from numpy import arctan2
        return arctan2(x,y)

    f1 = epyccel(arctan2_call, language = language)
    x = uniform(high=1e2)
    y = uniform(high=1e2)
    assert(isclose(f1(x,y), arctan2_call(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x,-y), arctan2_call(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x,y), arctan2_call(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(x,-y), arctan2_call(x,-y), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x, y), arctan2_call(x, y))

def test_arctan2_phrase(language):
    def arctan2_phrase(x : 'float', y : 'float', z : 'float'):
        from numpy import arctan2
        a = arctan2(x,y)+arctan2(x,z)
        return a

    f2 = epyccel(arctan2_phrase, language = language)
    x = uniform(high=1e2)
    y = uniform(high=1e2)
    z = uniform(high=1e2)
    assert(isclose(f2(x,y,z), arctan2_phrase(x,y,z), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y,z), arctan2_phrase(-x,y,z), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y,z), arctan2_phrase(-x,-y,z), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y,-z), arctan2_phrase(-x,y,-z), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y,z), arctan2_phrase(x,-y,z), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y,-z), arctan2_phrase(x,-y,-z), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,y,-z), arctan2_phrase(x,y,-z), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y,-z), arctan2_phrase(-x,-y,-z), rtol=RTOL, atol=ATOL))

#-------------------------------- sqrt function ------------------------------#
def test_sqrt_call(language):
    def sqrt_call(x : 'float'):
        from numpy import sqrt
        return sqrt(x)

    f1 = epyccel(sqrt_call, language = language)
    x = rand()
    assert(isclose(f1(x), sqrt_call(x), rtol=RTOL, atol=ATOL))

def test_sqrt_phrase(language):
    def sqrt_phrase(x : 'float', y : 'float'):
        from numpy import sqrt
        a = sqrt(x)*sqrt(y)
        return a

    f2 = epyccel(sqrt_phrase, language = language)
    x = rand()
    y = rand()
    assert(isclose(f2(x,y), sqrt_phrase(x,y), rtol=RTOL, atol=ATOL))

def test_sqrt_return_type_r(language):
    def sqrt_return_type_real(x : 'float'):
        from numpy import sqrt
        a = sqrt(x)
        return a

    f1 = epyccel(sqrt_return_type_real, language = language)
    x = rand()
    assert(isclose(f1(x), sqrt_return_type_real(x), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x), sqrt_return_type_real(x))

def test_sqrt_return_type_c(language):
    def sqrt_return_type_comp(x : 'complex'):
        from numpy import sqrt
        a = sqrt(x)
        return a

    f1 = epyccel(sqrt_return_type_comp, language = language)
    x = rand() + 1j * rand()
    assert(isclose(f1(x), sqrt_return_type_comp(x), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x), sqrt_return_type_comp(x))

#-------------------------------- floor function -----------------------------#
def test_floor_call_i(language):
    def floor_call(x : 'int'):
        from numpy import floor
        return floor(x)

    f1 = epyccel(floor_call, language = language)
    x = randint(1e6)
    assert(isclose(f1(x), floor_call(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), floor_call(-x), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x), floor_call(x))

def test_floor_call_r(language):
    def floor_call(x : 'float'):
        from numpy import floor
        return floor(x)

    f1 = epyccel(floor_call, language = language)
    x = uniform(high=1e6)
    assert(isclose(f1(x), floor_call(x), rtol=RTOL, atol=ATOL))
    assert(isclose(f1(-x), floor_call(-x), rtol=RTOL, atol=ATOL))
    assert matching_types(f1(x), floor_call(x))

def test_floor_phrase(language):
    def floor_phrase(x : 'float', y : 'float'):
        from numpy import floor
        a = floor(x)*floor(y)
        return a

    f2 = epyccel(floor_phrase, language = language)
    x = uniform(high=1e6)
    y = uniform(high=1e6)
    assert(isclose(f2(x,y), floor_phrase(x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,-y), floor_phrase(-x,-y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(-x,y), floor_phrase(-x,y), rtol=RTOL, atol=ATOL))
    assert(isclose(f2(x,-y), floor_phrase(x,-y), rtol=RTOL, atol=ATOL))

def test_shape_indexed(language):
    def test_shape_1d(f : 'int[:]'):
        from numpy import shape
        return shape(f)[0]

    def test_shape_2d(f : 'int[:,:]'):
        from numpy import shape
        a = shape(f)
        return a[0], a[1]

    from numpy import empty
    f1 = epyccel(test_shape_1d, language = language)
    f2 = epyccel(test_shape_2d, language = language)
    n1 = randint(20)
    n2 = randint(20)
    n3 = randint(20)
    x1 = empty(n1,dtype = int)
    x2 = empty((n2,n3), dtype = int)
    assert(f1(x1) == test_shape_1d(x1))
    assert(f2(x2) == test_shape_2d(x2))

def test_shape_property(language):
    def test_shape_1d(f : 'int[:]'):
        return f.shape[0]

    def test_shape_2d(f : 'int[:,:]'):
        a = f.shape
        return a[0], a[1]

    from numpy import empty
    f1 = epyccel(test_shape_1d, language = language)
    f2 = epyccel(test_shape_2d, language = language)
    n1 = randint(20)
    n2 = randint(20)
    n3 = randint(20)
    x1 = empty(n1,dtype = int)
    x2 = empty((n2,n3), dtype = int)
    assert(f1(x1) == test_shape_1d(x1))
    assert(all(isclose(f2(x2), test_shape_2d(x2))))

def test_shape_tuple_output(language):
    def test_shape_1d(f : 'int[:]'):
        from numpy import shape
        s = shape(f)
        return s[0]

    def test_shape_1d_tuple(f : 'int[:]'):
        from numpy import shape
        s, = shape(f)
        return s

    def test_shape_2d(f : 'int[:,:]'):
        from numpy import shape
        a, b = shape(f)
        return a, b

    from numpy import empty
    n1 = randint(20)
    n2 = randint(20)
    n3 = randint(20)
    x1 = empty(n1,dtype = int)
    x2 = empty((n2,n3), dtype = int)
    f1 = epyccel(test_shape_1d, language = language)
    assert(f1(x1)   == test_shape_1d(x1))
    f1_t = epyccel(test_shape_1d_tuple, language = language)
    assert(f1_t(x1) == test_shape_1d_tuple(x1))
    f2 = epyccel(test_shape_2d, language = language)
    assert(f2(x2)   == test_shape_2d(x2))

def test_shape_real(language):
    def test_shape_1d(f : 'float[:]'):
        from numpy import shape
        b = shape(f)
        return b[0]

    def test_shape_2d(f : 'float[:,:]'):
        from numpy import shape
        a = shape(f)
        return a[0], a[1]

    from numpy import empty
    f1 = epyccel(test_shape_1d, language = language)
    f2 = epyccel(test_shape_2d, language = language)
    n1 = randint(20)
    n2 = randint(20)
    n3 = randint(20)
    x1 = empty(n1,dtype = float)
    x2 = empty((n2,n3), dtype = float)
    assert(f1(x1) == test_shape_1d(x1))
    assert(f2(x2) == test_shape_2d(x2))

def test_shape_int(language):
    def test_shape_1d(f : 'int[:]'):
        from numpy import shape
        b = shape(f)
        return b[0]

    def test_shape_2d(f : 'int[:,:]'):
        from numpy import shape
        a = shape(f)
        return a[0], a[1]

    f1 = epyccel(test_shape_1d, language = language)
    f2 = epyccel(test_shape_2d, language = language)

    from numpy import empty
    n1 = randint(20)
    n2 = randint(20)
    n3 = randint(20)
    x1 = empty(n1,dtype = int)
    x2 = empty((n2,n3), dtype = int)
    assert(f1(x1) == test_shape_1d(x1))
    assert(f2(x2) == test_shape_2d(x2))

def test_shape_bool(language):
    def test_shape_1d(f : 'bool[:]'):
        from numpy import shape
        b = shape(f)
        return b[0]

    def test_shape_2d(f : 'bool[:,:]'):
        from numpy import shape
        a = shape(f)
        return a[0], a[1]

    from numpy import empty
    f1 = epyccel(test_shape_1d, language = language)
    f2 = epyccel(test_shape_2d, language = language)
    n1 = randint(20)
    n2 = randint(20)
    n3 = randint(20)
    x1 = empty(n1,dtype = bool)
    x2 = empty((n2,n3), dtype = bool)
    assert(f1(x1) == test_shape_1d(x1))
    assert(f2(x2) == test_shape_2d(x2))

def test_full_basic_int(language):
    def create_full_shape_1d(n : 'int'):
        from numpy import full, shape
        a = full(n,4)
        s = shape(a)
        return len(s),s[0]
    def create_full_shape_2d(n : 'int'):
        from numpy import full, shape
        a = full((n,n),4)
        s = shape(a)
        return len(s),s[0], s[1]
    def create_full_val(val : 'int'):
        from numpy import full
        a = full(3,val)
        return a[0],a[1],a[2]
    def create_full_arg_names(val : 'int'):
        from numpy import full
        a = full(fill_value = val, shape = (2,3))
        return a[0,0],a[0,1],a[0,2],a[1,0],a[1,1],a[1,2]

    size = randint(10)

    f_shape_1d = epyccel(create_full_shape_1d, language = language)
    assert f_shape_1d(size) == create_full_shape_1d(size)

    f_shape_2d = epyccel(create_full_shape_2d, language = language)
    assert f_shape_2d(size) == create_full_shape_2d(size)

    f_val = epyccel(create_full_val, language = language)
    assert f_val(size) == create_full_val(size)
    assert matching_types(f_val(size)[0], create_full_val(size)[0])

    f_arg_names = epyccel(create_full_arg_names, language = language)
    assert f_arg_names(size) == create_full_arg_names(size)
    assert matching_types(f_arg_names(size)[0], create_full_arg_names(size)[0])

def test_size(language):
    def test_size_1d(f: 'int[:]'):
        from numpy import size
        return size(f)

    def test_size_2d(f: 'int[:,:]'):
        from numpy import size
        return size(f)

    def test_size_axis_variable_2d(f: 'int[:,:]', axis :'int'):
        from numpy import size
        return size(f, axis)

    def test_size_axis_literal_3d(f: 'int[:,:,:]'):
        from numpy import size
        return size(f, 2)

    from numpy import empty
    f1 = epyccel(test_size_1d, language = language)
    f2 = epyccel(test_size_2d, language = language)
    f3 = epyccel(test_size_axis_variable_2d, language = language)
    f4 = epyccel(test_size_axis_literal_3d, language = language)
    n1 = randint(20)
    n2 = randint(20)
    n3 = randint(20)
    axis = randint(2)
    x1 = empty(n1, dtype = int)
    x2 = empty((n1, n2), dtype = int)
    x3 = empty((n1, n3), dtype = int)
    x4 = empty((n1, n2, n3), dtype = int)
    assert f1(x1) == test_size_1d(x1)
    assert f2(x2) == test_size_2d(x2)
    assert f3(x3, axis) == test_size_axis_variable_2d(x3, axis)
    assert f4(x4) == test_size_axis_literal_3d(x4)


def test_size_property(language):
    def test_size_1d(f: 'int[:]'):
        return f.size

    def test_size_2d(f: 'int[:,:]'):
        return f.size

    def test_size_3d(f: 'int[:,:,:]'):
        return f.size

    from numpy import empty
    f1 = epyccel(test_size_1d, language = language)
    f2 = epyccel(test_size_2d, language = language)
    f3 = epyccel(test_size_3d, language = language)
    n1 = randint(20)
    n2 = randint(20)
    n3 = randint(20)
    x1 = empty(n1, dtype = int)
    x2 = empty((n1, n2), dtype = int)
    x3 = empty((n1, n2, n3), dtype = int)
    assert f1(x1) == test_size_1d(x1)
    assert f2(x2) == test_size_2d(x2)
    assert f3(x3) == test_size_3d(x3)


def test_full_basic_real(language):
    def create_full_shape_1d(n : 'int'):
        from numpy import full, shape
        a = full(n,4)
        s = shape(a)
        return len(s),s[0]
    def create_full_shape_2d(n : 'int'):
        from numpy import full, shape
        a = full((n,n),4)
        s = shape(a)
        return len(s),s[0], s[1]
    def create_full_val(val : 'float'):
        from numpy import full
        a = full(3,val)
        return a[0],a[1],a[2]
    def create_full_arg_names(val : 'float'):
        from numpy import full
        a = full(fill_value = val, shape = (2,3))
        return a[0,0],a[0,1],a[0,2],a[1,0],a[1,1],a[1,2]

    size = randint(10)
    val  = rand()*5

    f_shape_1d  = epyccel(create_full_shape_1d, language = language)
    assert(f_shape_1d(size)     == create_full_shape_1d(size))

    f_shape_2d  = epyccel(create_full_shape_2d, language = language)
    assert(f_shape_2d(size)     == create_full_shape_2d(size))

    f_val       = epyccel(create_full_val, language = language)
    assert(f_val(val)           == create_full_val(val))
    assert matching_types(f_val(val)[0], create_full_val(val)[0])

    f_arg_names = epyccel(create_full_arg_names, language = language)
    assert(f_arg_names(val)     == create_full_arg_names(val))
    assert matching_types(f_arg_names(val)[0], create_full_arg_names(val)[0])

def test_full_basic_bool(language):
    def create_full_shape_1d(n : 'int'):
        from numpy import full, shape
        a = full(n,4)
        s = shape(a)
        return len(s),s[0]
    def create_full_shape_2d(n : 'int'):
        from numpy import full, shape
        a = full((n,n),4)
        s = shape(a)
        return len(s),s[0], s[1]
    def create_full_val(val : 'bool'):
        from numpy import full
        a = full(3,val)
        return a[0],a[1],a[2]
    def create_full_arg_names(val : 'bool'):
        from numpy import full
        a = full(fill_value = val, shape = (2,3))
        return a[0,0],a[0,1],a[0,2],a[1,0],a[1,1],a[1,2]

    size = randint(10)
    val  = bool(randint(2))

    f_shape_1d  = epyccel(create_full_shape_1d, language = language)
    assert(f_shape_1d(size)     == create_full_shape_1d(size))

    f_shape_2d  = epyccel(create_full_shape_2d, language = language)
    assert(f_shape_2d(size)     == create_full_shape_2d(size))

    f_val       = epyccel(create_full_val, language = language)
    assert(f_val(val)           == create_full_val(val))
    assert matching_types(f_val(val)[0], create_full_val(val)[0])

    f_arg_names = epyccel(create_full_arg_names, language = language)
    assert(f_arg_names(val)     == create_full_arg_names(val))
    assert matching_types(f_arg_names(val)[0], create_full_arg_names(val)[0])

def test_full_order(language):
    def create_full_shape_C(n : 'int', m : 'int'):
        from numpy import full, shape
        a = full((n,m),4, order = 'C')
        s = shape(a)
        return len(s),s[0], s[1]
    def create_full_shape_F(n : 'int', m : 'int'):
        from numpy import full, shape
        a = full((n,m),4, order = 'F')
        s = shape(a)
        return len(s),s[0], s[1]

    size_1 = randint(10)
    size_2 = randint(10)

    f_shape_C  = epyccel(create_full_shape_C, language = language)
    assert(f_shape_C(size_1,size_2) == create_full_shape_C(size_1,size_2))

    f_shape_F  = epyccel(create_full_shape_F, language = language)
    assert(f_shape_F(size_1,size_2) == create_full_shape_F(size_1,size_2))

def test_full_dtype(language):
    def create_full_val_int_int(val : 'int'):
        from numpy import full
        a = full(3,val,int)
        return a[0]
    def create_full_val_int_float(val : 'int'):
        from numpy import full
        a = full(3,val,float)
        return a[0]
    def create_full_val_int_complex(val : 'int'):
        from numpy import full
        a = full(3,val,complex)
        return a[0]
    def create_full_val_real_int32(val : 'float'):
        from numpy import full, int32
        a = full(3,val,int32)
        return a[0]
    def create_full_val_real_float32(val : 'float'):
        from numpy import full, float32
        a = full(3,val,float32)
        return a[0]
    def create_full_val_real_float64(val : 'float'):
        from numpy import full, float64
        a = full(3,val,float64)
        return a[0]
    def create_full_val_real_complex64(val : 'float'):
        from numpy import full, complex64
        a = full(3,val,complex64)
        return a[0]
    def create_full_val_real_complex128(val : 'float'):
        from numpy import full, complex128
        a = full(3,val,complex128)
        return a[0]

    val_int   = randint(100)
    val_float = rand()*100

    f_int_int   = epyccel(create_full_val_int_int, language = language)
    assert(     f_int_int(val_int)        ==      create_full_val_int_int(val_int))
    assert matching_types(f_int_int(val_int), create_full_val_int_int(val_int))

    f_int_float = epyccel(create_full_val_int_float, language = language)
    assert(isclose(     f_int_float(val_int)     ,      create_full_val_int_float(val_int), rtol=RTOL, atol=ATOL))
    assert matching_types(f_int_float(val_int), create_full_val_int_float(val_int))

    f_int_complex = epyccel(create_full_val_int_complex, language = language)
    assert(isclose(     f_int_complex(val_int)     ,      create_full_val_int_complex(val_int), rtol=RTOL, atol=ATOL))
    assert matching_types(f_int_complex(val_int), create_full_val_int_complex(val_int))

    f_real_int32   = epyccel(create_full_val_real_int32, language = language)
    assert(     f_real_int32(val_float)        ==      create_full_val_real_int32(val_float))
    assert matching_types(f_real_int32(val_float), create_full_val_real_int32(val_float))

    f_real_float32   = epyccel(create_full_val_real_float32, language = language)
    assert(isclose(     f_real_float32(val_float)       ,      create_full_val_real_float32(val_float), rtol=RTOL, atol=ATOL))
    assert matching_types(f_real_float32(val_float), create_full_val_real_float32(val_float))

    f_real_float64   = epyccel(create_full_val_real_float64, language = language)
    assert(isclose(     f_real_float64(val_float)       ,      create_full_val_real_float64(val_float), rtol=RTOL, atol=ATOL))
    assert matching_types(f_real_float64(val_float), create_full_val_real_float64(val_float))

    f_real_complex64   = epyccel(create_full_val_real_complex64, language = language)
    assert(isclose(     f_real_complex64(val_float)       ,      create_full_val_real_complex64(val_float), rtol=RTOL, atol=ATOL))
    assert matching_types(f_real_complex64(val_float), create_full_val_real_complex64(val_float))

    f_real_complex128   = epyccel(create_full_val_real_complex128, language = language)
    assert(isclose(     f_real_complex128(val_float)       ,      create_full_val_real_complex128(val_float), rtol=RTOL, atol=ATOL))
    assert matching_types(f_real_complex128(val_float), create_full_val_real_complex128(val_float))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = [
            pytest.mark.skip("full handles types in __new__ so it "
                "cannot be used in a translated interface in python"),
            pytest.mark.python]
        ),
    )
)

def test_full_dtype_auto(language):
    @template(name='T', types=['int','float', 'complex', 'int32',
                               'float32', 'float64', 'complex64', 'complex128'])
    def create_full_val_auto(val : 'T'):
        from numpy import full
        a = full(3,val)
        return a[0]

    integer   = randint(low = min_int,   high = max_int,   dtype=int)
    integer32 = randint(low = min_int32, high = max_int32, dtype=np.int32)

    fl = float(integer)
    fl32 = np.float32(fl)
    fl64 = np.float64(fl)

    cmplx = complex(integer)
    cmplx64 = np.complex64(fl32)
    cmplx128 = np.complex128(fl64)

    f_int = epyccel(create_full_val_auto, language = language)
    assert(f_int(integer) == create_full_val_auto(integer))
    assert matching_types(f_int(integer), create_full_val_auto(integer))

    f_float = epyccel(create_full_val_auto, language = language)
    assert(isclose(f_float(fl), create_full_val_auto(fl), rtol=RTOL, atol=ATOL))
    assert matching_types(f_float(fl), create_full_val_auto(fl))

    f_complex = epyccel(create_full_val_auto, language = language)
    assert(isclose(f_complex(cmplx), create_full_val_auto(cmplx), rtol=RTOL, atol=ATOL))
    assert matching_types(f_complex(cmplx), create_full_val_auto(cmplx))

    f_int32 = epyccel(create_full_val_auto, language = language)
    assert(f_int32(integer32) == create_full_val_auto(integer32))
    assert matching_types(f_int32(integer32), create_full_val_auto(integer32))

    f_float32 = epyccel(create_full_val_auto, language = language)
    assert(isclose(f_float32(fl32)  , create_full_val_auto(fl32), rtol=RTOL, atol=ATOL))
    assert matching_types(f_float32(fl32), create_full_val_auto(fl32))

    f_float64 = epyccel(create_full_val_auto, language = language)
    assert(isclose(f_float64(fl64)  , create_full_val_auto(fl64), rtol=RTOL, atol=ATOL))
    assert matching_types(f_float64(fl64), create_full_val_auto(fl64))

    f_complex64 = epyccel(create_full_val_auto, language = language)
    assert(isclose(f_complex64(cmplx64)  , create_full_val_auto(cmplx64), rtol=RTOL, atol=ATOL))
    assert matching_types(f_complex64(cmplx64), create_full_val_auto(cmplx64))

    f_complex128 = epyccel(create_full_val_auto, language = language)
    assert(isclose(f_complex128(cmplx128)  , create_full_val_auto(cmplx128), rtol=RTOL, atol=ATOL))
    assert matching_types(f_complex128(cmplx128), create_full_val_auto(cmplx128))

def test_full_combined_args(language):
    def create_full_1_shape():
        from numpy import full, shape
        a = full((2,1),4.0,int,'F')
        s = shape(a)
        return len(s),s[0],s[1]
    def create_full_1_val():
        from numpy import full
        a = full((2,1),4.0,int,'F')
        return a[0,0]
    def create_full_2_shape():
        from numpy import full, shape
        a = full((4,2),dtype=float,fill_value=1)
        s = shape(a)
        return len(s),s[0],s[1]
    def create_full_2_val():
        from numpy import full
        a = full((4,2),dtype=float,fill_value=1)
        return a[0,0]
    def create_full_3_shape():
        from numpy import full, shape
        a = full(order = 'F', shape = (4,2),dtype=complex,fill_value=1)
        s = shape(a)
        return len(s),s[0],s[1]
    def create_full_3_val():
        from numpy import full
        a = full(order = 'F', shape = (4,2),dtype=complex,fill_value=1)
        return a[0,0]

    f1_shape = epyccel(create_full_1_shape, language = language)
    f1_val   = epyccel(create_full_1_val, language = language)
    assert(f1_shape() == create_full_1_shape())
    assert(f1_val()   == create_full_1_val()  )
    assert matching_types(f1_val(), create_full_1_val())

    f2_shape = epyccel(create_full_2_shape, language = language)
    f2_val   = epyccel(create_full_2_val, language = language)
    assert(f2_shape() == create_full_2_shape()    )
    assert(isclose(f2_val()  , create_full_2_val()      , rtol=RTOL, atol=ATOL))
    assert matching_types(f2_val(), create_full_2_val())

    f3_shape = epyccel(create_full_3_shape, language = language)
    f3_val   = epyccel(create_full_3_val, language = language)
    assert(             f3_shape() ==    create_full_3_shape()      )
    assert(isclose(     f3_val()  ,      create_full_3_val()        , rtol=RTOL, atol=ATOL))
    assert matching_types(f3_val(), create_full_3_val())

def test_empty_basic(language):
    def create_empty_shape_1d(n : 'int'):
        from numpy import empty, shape
        a = empty(n)
        s = shape(a)
        return len(s),s[0]
    def create_empty_shape_2d(n : 'int'):
        from numpy import empty, shape
        a = empty((n,n))
        s = shape(a)
        return len(s),s[0], s[1]

    size = randint(10)

    f_shape_1d  = epyccel(create_empty_shape_1d, language = language)
    assert(     f_shape_1d(size)      ==      create_empty_shape_1d(size))

    f_shape_2d  = epyccel(create_empty_shape_2d, language = language)
    assert(     f_shape_2d(size)      ==      create_empty_shape_2d(size))

def test_empty_order(language):
    def create_empty_shape_C(n : 'int', m : 'int'):
        from numpy import empty, shape
        a = empty((n,m), order = 'C')
        s = shape(a)
        return len(s),s[0], s[1]
    def create_empty_shape_F(n : 'int', m : 'int'):
        from numpy import empty, shape
        a = empty((n,m), order = 'F')
        s = shape(a)
        return len(s),s[0], s[1]

    size_1 = randint(10)
    size_2 = randint(10)

    f_shape_C  = epyccel(create_empty_shape_C, language = language)
    assert(     f_shape_C(size_1,size_2) == create_empty_shape_C(size_1,size_2))

    f_shape_F  = epyccel(create_empty_shape_F, language = language)
    assert(     f_shape_F(size_1,size_2) == create_empty_shape_F(size_1,size_2))

def test_empty_dtype(language):
    def create_empty_val_int():
        from numpy import empty
        a = empty(3,int)
        return a[0]
    def create_empty_val_float():
        from numpy import empty
        a = empty(3,float)
        return a[0]
    def create_empty_val_complex():
        from numpy import empty
        a = empty(3,complex)
        return a[0]
    def create_empty_val_int32():
        from numpy import empty, int32
        a = empty(3,int32)
        return a[0]
    def create_empty_val_float32():
        from numpy import empty, float32
        a = empty(3,float32)
        return a[0]
    def create_empty_val_float64():
        from numpy import empty, float64
        a = empty(3,float64)
        return a[0]
    def create_empty_val_complex64():
        from numpy import empty, complex64
        a = empty(3,complex64)
        return a[0]
    def create_empty_val_complex128():
        from numpy import empty, complex128
        a = empty(3,complex128)
        return a[0]

    f_int_int   = epyccel(create_empty_val_int, language = language)
    assert matching_types(f_int_int(), create_empty_val_int())

    f_int_float = epyccel(create_empty_val_float, language = language)
    assert matching_types(f_int_float(), create_empty_val_float())

    f_int_complex = epyccel(create_empty_val_complex, language = language)
    assert matching_types(f_int_complex(), create_empty_val_complex())

    f_real_int32   = epyccel(create_empty_val_int32, language = language)
    assert matching_types(f_real_int32(), create_empty_val_int32())

    f_real_float32   = epyccel(create_empty_val_float32, language = language)
    assert matching_types(f_real_float32(), create_empty_val_float32())

    f_real_float64   = epyccel(create_empty_val_float64, language = language)
    assert matching_types(f_real_float64(), create_empty_val_float64())

    f_real_complex64   = epyccel(create_empty_val_complex64, language = language)
    assert matching_types(f_real_complex64(), create_empty_val_complex64())

    f_real_complex128   = epyccel(create_empty_val_complex128, language = language)
    assert matching_types(f_real_complex128(), create_empty_val_complex128())

def test_empty_combined_args(language):
    def create_empty_1_shape():
        from numpy import empty, shape
        a = empty((2,1),int,'F')
        s = shape(a)
        return len(s),s[0],s[1]
    def create_empty_1_val():
        from numpy import empty
        a = empty((2,1),int,'F')
        return a[0,0]
    def create_empty_2_shape():
        from numpy import empty, shape
        a = empty((4,2),dtype=float)
        s = shape(a)
        return len(s),s[0],s[1]
    def create_empty_2_val():
        from numpy import empty
        a = empty((4,2),dtype=float)
        return a[0,0]
    def create_empty_3_shape():
        from numpy import empty, shape
        a = empty(order = 'F', shape = (4,2),dtype=complex)
        s = shape(a)
        return len(s),s[0],s[1]
    def create_empty_3_val():
        from numpy import empty
        a = empty(order = 'F', shape = (4,2),dtype=complex)
        return a[0,0]

    f1_shape = epyccel(create_empty_1_shape, language = language)
    f1_val   = epyccel(create_empty_1_val, language = language)
    assert(     f1_shape() ==      create_empty_1_shape()      )
    assert matching_types(f1_val(), create_empty_1_val())

    f2_shape = epyccel(create_empty_2_shape, language = language)
    f2_val   = epyccel(create_empty_2_val, language = language)
    assert(all(isclose(     f2_shape(),      create_empty_2_shape()      )))
    assert matching_types(f2_val(), create_empty_2_val())

    f3_shape = epyccel(create_empty_3_shape, language = language)
    f3_val   = epyccel(create_empty_3_val, language = language)
    assert(all(isclose(     f3_shape(),      create_empty_3_shape()      )))
    assert matching_types(f3_val(), create_empty_3_val())

def test_ones_basic(language):
    def create_ones_shape_1d(n : 'int'):
        from numpy import ones, shape
        a = ones(n)
        s = shape(a)
        return len(s),s[0]
    def create_ones_shape_2d(n : 'int'):
        from numpy import ones, shape
        a = ones((n,n))
        s = shape(a)
        return len(s),s[0], s[1]

    size = randint(10)

    f_shape_1d  = epyccel(create_ones_shape_1d, language = language)
    assert(     f_shape_1d(size)      ==      create_ones_shape_1d(size))

    f_shape_2d  = epyccel(create_ones_shape_2d, language = language)
    assert(     f_shape_2d(size)      ==      create_ones_shape_2d(size))

def test_ones_order(language):
    def create_ones_shape_C(n : 'int', m : 'int'):
        from numpy import ones, shape
        a = ones((n,m), order = 'C')
        s = shape(a)
        return len(s),s[0], s[1]
    def create_ones_shape_F(n : 'int', m : 'int'):
        from numpy import ones, shape
        a = ones((n,m), order = 'F')
        s = shape(a)
        return len(s),s[0], s[1]

    size_1 = randint(10)
    size_2 = randint(10)

    f_shape_C  = epyccel(create_ones_shape_C, language = language)
    assert(     f_shape_C(size_1,size_2) == create_ones_shape_C(size_1,size_2))

    f_shape_F  = epyccel(create_ones_shape_F, language = language)
    assert(     f_shape_F(size_1,size_2) == create_ones_shape_F(size_1,size_2))

def test_ones_dtype(language):
    def create_ones_val_int():
        from numpy import ones
        a = ones(3,int)
        return a[0]
    def create_ones_val_float():
        from numpy import ones
        a = ones(3,float)
        return a[0]
    def create_ones_val_complex():
        from numpy import ones
        a = ones(3,complex)
        return a[0]
    def create_ones_val_int32():
        from numpy import ones, int32
        a = ones(3,int32)
        return a[0]
    def create_ones_val_float32():
        from numpy import ones, float32
        a = ones(3,float32)
        return a[0]
    def create_ones_val_float64():
        from numpy import ones, float64
        a = ones(3,float64)
        return a[0]
    def create_ones_val_complex64():
        from numpy import ones, complex64
        a = ones(3,complex64)
        return a[0]
    def create_ones_val_complex128():
        from numpy import ones, complex128
        a = ones(3,complex128)
        return a[0]

    f_int_int   = epyccel(create_ones_val_int, language = language)
    assert(     f_int_int()          ==      create_ones_val_int())
    assert matching_types(f_int_int(), create_ones_val_int())

    f_int_float = epyccel(create_ones_val_float, language = language)
    assert(isclose(     f_int_float()       ,      create_ones_val_float(), rtol=RTOL, atol=ATOL))
    assert matching_types(f_int_float(), create_ones_val_float())

    f_int_complex = epyccel(create_ones_val_complex, language = language)
    assert(isclose(     f_int_complex()     ,      create_ones_val_complex(), rtol=RTOL, atol=ATOL))
    assert matching_types(f_int_complex(), create_ones_val_complex())

    f_real_int32   = epyccel(create_ones_val_int32, language = language)
    assert(     f_real_int32()       ==      create_ones_val_int32())
    assert matching_types(f_real_int32(), create_ones_val_int32())

    f_real_float32   = epyccel(create_ones_val_float32, language = language)
    assert(isclose(     f_real_float32()    ,      create_ones_val_float32(), rtol=RTOL, atol=ATOL))
    assert matching_types(f_real_float32(), create_ones_val_float32())

    f_real_float64   = epyccel(create_ones_val_float64, language = language)
    assert(isclose(     f_real_float64()    ,      create_ones_val_float64(), rtol=RTOL, atol=ATOL))
    assert matching_types(f_real_float64(), create_ones_val_float64())

    f_real_complex64   = epyccel(create_ones_val_complex64, language = language)
    assert(isclose(     f_real_complex64()  ,      create_ones_val_complex64(), rtol=RTOL, atol=ATOL))
    assert matching_types(f_real_complex64(), create_ones_val_complex64())

    f_real_complex128   = epyccel(create_ones_val_complex128, language = language)
    assert(isclose(     f_real_complex128() ,      create_ones_val_complex128(), rtol=RTOL, atol=ATOL))
    assert matching_types(f_real_complex128(), create_ones_val_complex128())

def test_ones_combined_args(language):
    def create_ones_1_shape():
        from numpy import ones, shape
        a = ones((2,1),int,'F')
        s = shape(a)
        return len(s),s[0],s[1]
    def create_ones_1_val():
        from numpy import ones
        a = ones((2,1),int,'F')
        return a[0,0]
    def create_ones_2_shape():
        from numpy import ones, shape
        a = ones((4,2),dtype=float)
        s = shape(a)
        return len(s),s[0],s[1]
    def create_ones_2_val():
        from numpy import ones
        a = ones((4,2),dtype=float)
        return a[0,0]
    def create_ones_3_shape():
        from numpy import ones, shape
        a = ones(order = 'F', shape = (4,2),dtype=complex)
        s = shape(a)
        return len(s),s[0],s[1]
    def create_ones_3_val():
        from numpy import ones
        a = ones(order = 'F', shape = (4,2),dtype=complex)
        return a[0,0]

    f1_shape = epyccel(create_ones_1_shape, language = language)
    f1_val   = epyccel(create_ones_1_val, language = language)
    assert(     f1_shape() ==      create_ones_1_shape()      )
    assert(     f1_val()   ==      create_ones_1_val()        )
    assert matching_types(f1_val(), create_ones_1_val())

    f2_shape = epyccel(create_ones_2_shape, language = language)
    f2_val   = epyccel(create_ones_2_val, language = language)
    assert(     f2_shape() ==      create_ones_2_shape()      )
    assert(isclose(     f2_val()  ,      create_ones_2_val()        , rtol=RTOL, atol=ATOL))
    assert matching_types(f2_val(), create_ones_2_val())

    f3_shape = epyccel(create_ones_3_shape, language = language)
    f3_val   = epyccel(create_ones_3_val, language = language)
    assert(     f3_shape() ==      create_ones_3_shape()      )
    assert(isclose(     f3_val()  ,      create_ones_3_val()        , rtol=RTOL, atol=ATOL))
    assert matching_types(f3_val(), create_ones_3_val())

def test_zeros_basic(language):
    def create_zeros_shape_1d(n : 'int'):
        from numpy import zeros, shape
        a = zeros(n)
        s = shape(a)
        return len(s),s[0]
    def create_zeros_shape_2d(n : 'int'):
        from numpy import zeros, shape
        a = zeros((n,n))
        s = shape(a)
        return len(s),s[0], s[1]

    size = randint(10)

    f_shape_1d  = epyccel(create_zeros_shape_1d, language = language)
    assert(     f_shape_1d(size)      ==      create_zeros_shape_1d(size))

    f_shape_2d  = epyccel(create_zeros_shape_2d, language = language)
    assert(     f_shape_2d(size)      ==      create_zeros_shape_2d(size))

def test_zeros_order(language):
    def create_zeros_shape_C(n : 'int', m : 'int'):
        from numpy import zeros, shape
        a = zeros((n,m), order = 'C')
        s = shape(a)
        return len(s),s[0], s[1]
    def create_zeros_shape_F(n : 'int', m : 'int'):
        from numpy import zeros, shape
        a = zeros((n,m), order = 'F')
        s = shape(a)
        return len(s),s[0], s[1]

    size_1 = randint(10)
    size_2 = randint(10)

    f_shape_C  = epyccel(create_zeros_shape_C, language = language)
    assert(     f_shape_C(size_1,size_2) == create_zeros_shape_C(size_1,size_2))

    f_shape_F  = epyccel(create_zeros_shape_F, language = language)
    assert(     f_shape_F(size_1,size_2) == create_zeros_shape_F(size_1,size_2))

def test_zeros_dtype(language):
    def create_zeros_val_int():
        from numpy import zeros
        a = zeros(3,int)
        return a[0]
    def create_zeros_val_float():
        from numpy import zeros
        a = zeros(3,float)
        return a[0]
    def create_zeros_val_complex():
        from numpy import zeros
        a = zeros(3,complex)
        return a[0]
    def create_zeros_val_int32():
        from numpy import zeros, int32
        a = zeros(3,int32)
        return a[0]
    def create_zeros_val_float32():
        from numpy import zeros, float32
        a = zeros(3,float32)
        return a[0]
    def create_zeros_val_float64():
        from numpy import zeros, float64
        a = zeros(3,float64)
        return a[0]
    def create_zeros_val_complex64():
        from numpy import zeros, complex64
        a = zeros(3,complex64)
        return a[0]
    def create_zeros_val_complex128():
        from numpy import zeros, complex128
        a = zeros(3,complex128)
        return a[0]

    f_int_int   = epyccel(create_zeros_val_int, language = language)
    assert(     f_int_int()          ==      create_zeros_val_int())
    assert matching_types(f_int_int(), create_zeros_val_int())

    f_int_float = epyccel(create_zeros_val_float, language = language)
    assert(isclose(     f_int_float()       ,      create_zeros_val_float(), rtol=RTOL, atol=ATOL))
    assert matching_types(f_int_float(), create_zeros_val_float())

    f_int_complex = epyccel(create_zeros_val_complex, language = language)
    assert(isclose(     f_int_complex()     ,      create_zeros_val_complex(), rtol=RTOL, atol=ATOL))
    assert matching_types(f_int_complex(), create_zeros_val_complex())

    f_real_int32   = epyccel(create_zeros_val_int32, language = language)
    assert(     f_real_int32()       ==      create_zeros_val_int32())
    assert matching_types(f_real_int32(), create_zeros_val_int32())

    f_real_float32   = epyccel(create_zeros_val_float32, language = language)
    assert(isclose(     f_real_float32()    ,      create_zeros_val_float32(), rtol=RTOL, atol=ATOL))
    assert matching_types(f_real_float32(), create_zeros_val_float32())

    f_real_float64   = epyccel(create_zeros_val_float64, language = language)
    assert(isclose(     f_real_float64()    ,      create_zeros_val_float64(), rtol=RTOL, atol=ATOL))
    assert matching_types(f_real_float64(), create_zeros_val_float64())

    f_real_complex64   = epyccel(create_zeros_val_complex64, language = language)
    assert(isclose(     f_real_complex64()  ,      create_zeros_val_complex64(), rtol=RTOL, atol=ATOL))
    assert matching_types(f_real_complex64(), create_zeros_val_complex64())

    f_real_complex128   = epyccel(create_zeros_val_complex128, language = language)
    assert(isclose(     f_real_complex128() ,      create_zeros_val_complex128(), rtol=RTOL, atol=ATOL))
    assert matching_types(f_real_complex128(), create_zeros_val_complex128())

def test_zeros_combined_args(language):
    def create_zeros_1_shape():
        from numpy import zeros, shape
        a = zeros((2,1),int,'F')
        s = shape(a)
        return len(s),s[0],s[1]
    def create_zeros_1_val():
        from numpy import zeros
        a = zeros((2,1),int,'F')
        return a[0,0]
    def create_zeros_2_shape():
        from numpy import zeros, shape
        a = zeros((4,2),dtype=float)
        s = shape(a)
        return len(s),s[0],s[1]
    def create_zeros_2_val():
        from numpy import zeros
        a = zeros((4,2),dtype=float)
        return a[0,0]
    def create_zeros_3_shape():
        from numpy import zeros, shape
        a = zeros(order = 'F', shape = (4,2),dtype=complex)
        s = shape(a)
        return len(s),s[0],s[1]
    def create_zeros_3_val():
        from numpy import zeros
        a = zeros(order = 'F', shape = (4,2),dtype=complex)
        return a[0,0]

    f1_shape = epyccel(create_zeros_1_shape, language = language)
    f1_val   = epyccel(create_zeros_1_val, language = language)
    assert(     f1_shape() ==      create_zeros_1_shape()      )
    assert(     f1_val()   ==      create_zeros_1_val()        )
    assert matching_types(f1_val(), create_zeros_1_val())

    f2_shape = epyccel(create_zeros_2_shape, language = language)
    f2_val   = epyccel(create_zeros_2_val, language = language)
    assert(     f2_shape() ==      create_zeros_2_shape()      )
    assert(isclose(     f2_val()  ,      create_zeros_2_val()        , rtol=RTOL, atol=ATOL))
    assert matching_types(f2_val(), create_zeros_2_val())

    f3_shape = epyccel(create_zeros_3_shape, language = language)
    f3_val   = epyccel(create_zeros_3_val, language = language)
    assert(     f3_shape() ==      create_zeros_3_shape()      )
    assert(isclose(     f3_val()  ,      create_zeros_3_val()        , rtol=RTOL, atol=ATOL))
    assert matching_types(f3_val(), create_zeros_3_val())

def test_array(language):
    def create_array_list_val():
        from numpy import array
        a = array([[1,2,3],[4,5,6]])
        return a[0,0]
    def create_array_list_shape():
        from numpy import array, shape
        a = array([[1,2,3],[4,5,6]])
        s = shape(a)
        return len(s), s[0], s[1]
    def create_array_tuple_val():
        from numpy import array
        a = array(((1,2,3),(4,5,6)))
        return a[0,0]
    def create_array_tuple_shape():
        from numpy import array, shape
        a = array(((1,2,3),(4,5,6)))
        s = shape(a)
        return len(s), s[0], s[1]
    f1_shape = epyccel(create_array_list_shape, language = language)
    f1_val   = epyccel(create_array_list_val, language = language)
    assert(f1_shape() == create_array_list_shape())
    assert(f1_val()   == create_array_list_val())
    assert matching_types(f1_val(), create_array_list_val())
    f2_shape = epyccel(create_array_tuple_shape, language = language)
    f2_val   = epyccel(create_array_tuple_val, language = language)
    assert(f2_shape() == create_array_tuple_shape())
    assert(f2_val()   == create_array_tuple_val())
    assert matching_types(f2_val(), create_array_tuple_val())

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="rand not implemented"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_rand_basic(language):
    def create_val():
        from numpy.random import rand # pylint: disable=reimported
        return rand()

    f1 = epyccel(create_val, language = language)
    y = [f1() for i in range(10)]
    assert(all([yi <  1 for yi in y]))
    assert(all([yi >= 0 for yi in y]))
    assert(all([isinstance(yi,float) for yi in y]))
    assert(len(set(y))>1)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="rand not implemented"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_rand_args(language):
    def create_array_size_1d(n : 'int'):
        from numpy.random import rand # pylint: disable=reimported
        from numpy import shape
        a = rand(n)
        return shape(a)[0]

    def create_array_size_2d(n : 'int', m : 'int'):
        from numpy.random import rand # pylint: disable=reimported
        from numpy import shape
        a = rand(n,m)
        return shape(a)[0], shape(a)[1]

    def create_array_size_3d(n : 'int', m : 'int', p : 'int'):
        from numpy.random import rand # pylint: disable=reimported
        from numpy import shape
        a = rand(n,m,p)
        return shape(a)[0], shape(a)[1], shape(a)[2]

    def create_array_vals_1d():
        from numpy.random import rand # pylint: disable=reimported
        a = rand(4)
        return a[0], a[1], a[2], a[3]

    def create_array_vals_2d():
        from numpy.random import rand # pylint: disable=reimported
        a = rand(2,2)
        return a[0,0], a[0,1], a[1,0], a[1,1]

    n = randint(10)
    m = randint(10)
    p = randint(5)
    f_1d = epyccel(create_array_size_1d, language = language)
    assert( f_1d(n)       == create_array_size_1d(n)      )

    f_2d = epyccel(create_array_size_2d, language = language)
    assert( f_2d(n, m)    == create_array_size_2d(n, m)   )

    f_3d = epyccel(create_array_size_3d, language = language)
    assert( f_3d(n, m, p) == create_array_size_3d(n, m, p))

    g_1d = epyccel(create_array_vals_1d, language = language)
    y = g_1d()
    assert(all([yi <  1 for yi in y]))
    assert(all([yi >= 0 for yi in y]))
    assert(all([isinstance(yi,float) for yi in y]))
    assert(len(set(y))>1)

    g_2d = epyccel(create_array_vals_2d, language = language)
    y = g_2d()
    assert(all([yi <  1 for yi in y]))
    assert(all([yi >= 0 for yi in y]))
    assert(all([isinstance(yi,float) for yi in y]))
    assert(len(set(y))>1)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="rand not implemented"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_rand_expr(language):
    def create_val():
        from numpy.random import rand # pylint: disable=reimported
        x = 2*rand()
        return x

    f1 = epyccel(create_val, language = language)
    y = [f1() for i in range(10)]
    assert(all([yi <  2 for yi in y]))
    assert(all([yi >= 0 for yi in y]))
    assert(all([isinstance(yi,float) for yi in y]))
    assert(len(set(y))>1)

@pytest.mark.xfail(reason="a is not allocated")
def test_rand_expr_array(language):
    def create_array_vals_2d():
        from numpy.random import rand # pylint: disable=reimported
        a = rand(2,2)*0.5 + 3
        return a[0,0], a[0,1], a[1,0], a[1,1]

    f2 = epyccel(create_array_vals_2d, language = language)
    y = f2()
    assert(all([yi <  3.5 for yi in y]))
    assert(all([yi >= 3   for yi in y]))
    assert(all([isinstance(yi,float) for yi in y]))
    assert(len(set(y))>1)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="randint not implemented"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_randint_basic(language):
    def create_rand():
        from numpy.random import randint # pylint: disable=reimported
        return randint(-10, 10)

    def create_val(high : 'int'):
        from numpy.random import randint # pylint: disable=reimported
        return randint(high)

    def create_val_low(low : 'int', high : 'int'):
        from numpy.random import randint # pylint: disable=reimported
        return randint(low, high)

    f0 = epyccel(create_rand, language = language)
    y = [f0() for i in range(10)]
    assert(all([yi <  10 for yi in y]))
    assert(all([yi >= -10 for yi in y]))
    assert(all([isinstance(yi,int) for yi in y]))
    assert(len(set(y))>1)

    f1 = epyccel(create_val, language = language)
    y = [f1(100) for i in range(10)]
    assert(all([yi <  100 for yi in y]))
    assert(all([yi >= 0 for yi in y]))
    assert(all([isinstance(yi,int) for yi in y]))
    assert(len(set(y))>1)

    f2 = epyccel(create_val_low, language = language)
    y = [f2(25, 100) for i in range(10)]
    assert(all([yi <  100 for yi in y]))
    assert(all([yi >= 25 for yi in y]))
    assert(all([isinstance(yi,int) for yi in y]))
    assert(len(set(y))>1)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="randint not implemented"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_randint_expr(language):
    def create_val(high : 'int'):
        from numpy.random import randint # pylint: disable=reimported
        x = 2*randint(high)
        return x

    def create_val_low(low : 'int', high : 'int'):
        from numpy.random import randint # pylint: disable=reimported
        x = 2*randint(low, high)
        return x

    f1 = epyccel(create_val, language = language)
    y = [f1(27) for i in range(10)]
    assert(all([yi <  54 for yi in y]))
    assert(all([yi >= 0  for yi in y]))
    assert(all([isinstance(yi,int) for yi in y]))
    assert(len(set(y))>1)

    f2 = epyccel(create_val_low, language = language)
    y = [f2(21,46) for i in range(10)]
    assert(all([yi <  92 for yi in y]))
    assert(all([yi >= 42 for yi in y]))
    assert(all([isinstance(yi,int) for yi in y]))
    assert(len(set(y))>1)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="sum not implemented"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_sum_int(language):
    def sum_call(x : 'int[:]'):
        from numpy import sum as np_sum
        return np_sum(x)

    f1 = epyccel(sum_call, language = language)
    x = randint(99,size=10)
    assert(f1(x) == sum_call(x))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="sum not implemented"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_sum_real(language):
    def sum_call(x : 'float[:]'):
        from numpy import sum as np_sum
        return np_sum(x)

    f1 = epyccel(sum_call, language = language)
    x = rand(10)
    assert(isclose(f1(x), sum_call(x), rtol=RTOL, atol=ATOL))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="sum not implemented"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_sum_phrase(language):
    def sum_phrase(x : 'float[:]', y : 'float[:]'):
        from numpy import sum as np_sum
        a = np_sum(x)*np_sum(y)
        return a

    f2 = epyccel(sum_phrase, language = language)
    x = rand(10)
    y = rand(15)
    assert(isclose(f2(x,y), sum_phrase(x,y), rtol=RTOL, atol=ATOL))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="sum not implemented"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_sum_property(language):
    def sum_call(x : 'int[:]'):
        return x.sum()

    f1 = epyccel(sum_call, language = language)
    x = randint(99,size=10)
    assert(f1(x) == sum_call(x))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="amin not implemented"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_min_int(language):
    def min_call(x : 'int[:]'):
        from numpy import amin
        return amin(x)

    f1 = epyccel(min_call, language = language)
    x = randint(99,size=10)
    assert(f1(x) == min_call(x))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="amin not implemented"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_min_real(language):
    def min_call(x : 'float[:]'):
        from numpy import amin
        return amin(x)

    f1 = epyccel(min_call, language = language)
    x = rand(10)
    assert(isclose(f1(x), min_call(x), rtol=RTOL, atol=ATOL))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="amin not implemented"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_min_phrase(language):
    def min_phrase(x : 'float[:]', y : 'float[:]'):
        from numpy import amin
        a = amin(x)*amin(y)
        return a

    f2 = epyccel(min_phrase, language = language)
    x = rand(10)
    y = rand(15)
    assert(isclose(f2(x,y), min_phrase(x,y), rtol=RTOL, atol=ATOL))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="amin not implemented"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_min_property(language):
    def min_call(x : 'int[:]'):
        return x.min()

    f1 = epyccel(min_call, language = language)
    x = randint(99,size=10)
    assert(f1(x) == min_call(x))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="amax not implemented"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_max_int(language):
    def max_call(x : 'int[:]'):
        from numpy import amax
        return amax(x)

    f1 = epyccel(max_call, language = language)
    x = randint(99,size=10)
    assert(f1(x) == max_call(x))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="amax not implemented"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_max_real(language):
    def max_call(x : 'float[:]'):
        from numpy import amax
        return amax(x)

    f1 = epyccel(max_call, language = language)
    x = rand(10)
    assert(isclose(f1(x), max_call(x), rtol=RTOL, atol=ATOL))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="amax not implemented"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_max_phrase(language):
    def max_phrase(x : 'float[:]', y : 'float[:]'):
        from numpy import amax
        a = amax(x)*amax(y)
        return a

    f2 = epyccel(max_phrase, language = language)
    x = rand(10)
    y = rand(15)
    assert(isclose(f2(x,y), max_phrase(x,y), rtol=RTOL, atol=ATOL))


@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="amax not implemented"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_max_property(language):
    def max_call(x : 'int[:]'):
        return x.max()

    f1 = epyccel(max_call, language = language)
    x = randint(99,size=10)
    assert(f1(x) == max_call(x))


def test_full_like_basic_int(language):
    def create_full_like_shape_1d(n : 'int'):
        from numpy import full_like, shape, array
        arr = array([5, 1, 8, 0, 9])
        a = full_like(arr, n, int, 'F')
        s = shape(a)
        return len(s),s[0]
    def create_full_like_shape_2d(n : 'int'):
        from numpy import full_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = full_like(arr, n, int , 'F')
        s = shape(a)
        return len(s),s[0], s[1]
    def create_full_like_val(val : 'int'):
        from numpy import full_like, array
        arr = array([5, 1, 8, 0, 9])
        a = full_like(arr, val, int, 'F')
        return a[0],a[1],a[2]
    def create_full_like_arg_names(val : 'int'):
        from numpy import full_like, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = full_like(arr, val, int, 'F', shape = (2,3))
        return a[0,0],a[0,1],a[0,2],a[1,0],a[1,1],a[1,2]

    size = randint(10)

    f_shape_1d  = epyccel(create_full_like_shape_1d, language = language)
    assert(f_shape_1d(size) == create_full_like_shape_1d(size))

    f_shape_2d  = epyccel(create_full_like_shape_2d, language = language)
    assert(f_shape_2d(size) == create_full_like_shape_2d(size))

    f_val       = epyccel(create_full_like_val, language = language)
    assert(f_val(size)      == create_full_like_val(size))
    assert matching_types(f_val(size)[0], create_full_like_val(size)[0])

    f_arg_names = epyccel(create_full_like_arg_names, language = language)
    assert(f_arg_names(size) == create_full_like_arg_names(size))
    assert matching_types(f_arg_names(size)[0], create_full_like_arg_names(size)[0])

def test_full_like_basic_real(language):
    def create_full_like_shape_1d(n : 'float'):
        from numpy import full_like, shape, array
        arr = array([5, 1, 8, 0, 9])
        a = full_like(arr, n, float, 'F')
        s = shape(a)
        return len(s),s[0]
    def create_full_like_shape_2d(n : 'float'):
        from numpy import full_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = full_like(arr, n, float, 'F')
        s = shape(a)
        return len(s),s[0], s[1]
    def create_full_like_val(val : 'float'):
        from numpy import full_like, array
        arr = array([5, 1, 8, 0, 9])
        a = full_like(arr, val, float, 'F')
        return a[0],a[1],a[2]
    def create_full_like_arg_names(val : 'float'):
        from numpy import full_like, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = full_like(arr, val, float, 'F', shape = (2,3))
        return a[0,0],a[0,1],a[0,2],a[1,0],a[1,1],a[1,2]

    size = uniform(10)
    val  = rand()*5

    f_shape_1d  = epyccel(create_full_like_shape_1d, language = language)
    assert(f_shape_1d(size)     == create_full_like_shape_1d(size))

    f_shape_2d  = epyccel(create_full_like_shape_2d, language = language)
    assert(f_shape_2d(size)     == create_full_like_shape_2d(size))

    f_val       = epyccel(create_full_like_val, language = language)
    assert(f_val(val)           == create_full_like_val(val))
    assert matching_types(f_val(val)[0], create_full_like_val(val)[0])

    f_arg_names = epyccel(create_full_like_arg_names, language = language)
    assert(f_arg_names(val)     == create_full_like_arg_names(val))
    assert matching_types(f_arg_names(val)[0], create_full_like_arg_names(val)[0])

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Tuples not implemented"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_full_like_basic_bool(language):
    def create_full_like_shape_1d(n : 'int'):
        from numpy import full_like, shape, array
        arr = array([5, 1, 8, 0, 9])
        a = full_like(arr, n, int, 'F')
        s = shape(a)
        return len(s),s[0]
    def create_full_like_shape_2d(n : 'int'):
        from numpy import full_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = full_like(arr, n, int, 'F')
        s = shape(a)
        return len(s),s[0], s[1]
    def create_full_like_val(val : 'bool'):
        from numpy import full_like, array
        arr = array([5, 1, 8, 0, 9])
        a = full_like(arr , 3, bool, 'F')
        return a[0],a[1],a[2]
    def create_full_like_arg_names(val : 'bool'):
        from numpy import full_like, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = full_like(arr ,fill_value = val, dtype=bool, shape = (2,3))
        return a[0,0],a[0,1],a[0,2],a[1,0],a[1,1],a[1,2]

    size = randint(10)
    val  = bool(randint(2))

    f_shape_1d  = epyccel(create_full_like_shape_1d, language = language)
    assert(f_shape_1d(size)     == create_full_like_shape_1d(size))

    f_shape_2d  = epyccel(create_full_like_shape_2d, language = language)
    assert(f_shape_2d(size)     == create_full_like_shape_2d(size))

    f_val       = epyccel(create_full_like_val, language = language)
    assert(f_val(val)           == create_full_like_val(val))
    assert matching_types(f_val(val)[0], create_full_like_val(val)[0])

    f_arg_names = epyccel(create_full_like_arg_names, language = language)
    assert(f_arg_names(val)     == create_full_like_arg_names(val))
    assert matching_types(f_arg_names(val)[0], create_full_like_arg_names(val)[0])

def test_full_like_order(language):
    def create_full_like_shape_C(n : 'int'):
        from numpy import full_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = full_like(arr,4, order = 'C')
        s = shape(a)
        return len(s),s[0], s[1]
    def create_full_like_shape_F(n : 'int'):
        from numpy import full_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = full_like(arr,4, order = 'F')
        s = shape(a)
        return len(s),s[0], s[1]

    size = randint(10)

    f_shape_C  = epyccel(create_full_like_shape_C, language = language)
    assert(f_shape_C(size) == create_full_like_shape_C(size))

    f_shape_F  = epyccel(create_full_like_shape_F, language = language)
    assert(f_shape_F(size) == create_full_like_shape_F(size))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_full_like_dtype(language):
    def create_full_like_val_int_int_auto(val : 'int'):
        from numpy import full_like, array
        arr = array([5, 1, 8, 0, 9], int)
        a = full_like(arr,val)
        return a[0]
    def create_full_like_val_int_int(val : 'int'):
        from numpy import full_like, array
        arr = array([5, 1, 8, 0, 9])
        a = full_like(arr,val,int)
        return a[0]

    def create_full_like_val_int_float_auto(val : 'int'):
        from numpy import full_like, array
        arr = array([5, 1, 8, 0, 9], float)
        a = full_like(arr,val)
        return a[0]
    def create_full_like_val_int_float(val : 'int'):
        from numpy import full_like, array
        arr = array([5, 1, 8, 0, 9])
        a = full_like(arr,val,float)
        return a[0]

    def create_full_like_val_int_complex_auto(val : 'int'):
        from numpy import full_like, array
        arr = array([5, 1, 8, 0, 9], complex)
        a = full_like(arr,val)
        return a[0]
    def create_full_like_val_int_complex(val : 'int'):
        from numpy import full_like, array
        arr = array([5, 1, 8, 0, 9])
        a = full_like(arr,val,complex)
        return a[0]

    def create_full_like_val_real_int32_auto(val : 'float'):
        from numpy import full_like, int32, array
        arr = array([5, 1, 8, 0, 9], int32)
        a = full_like(arr,val)
        return a[0]
    def create_full_like_val_real_int32(val : 'float'):
        from numpy import full_like, int32, array
        arr = array([5, 1, 8, 0, 9])
        a = full_like(arr,val,int32)
        return a[0]

    def create_full_like_val_real_float32_auto(val : 'float'):
        from numpy import full_like, float32, array
        arr = array([5, 1, 8, 0, 9], float32)
        a = full_like(arr,val)
        return a[0]
    def create_full_like_val_real_float32(val : 'float'):
        from numpy import full_like, float32, array
        arr = array([5, 1, 8, 0, 9])
        a = full_like(arr,val,float32)
        return a[0]

    def create_full_like_val_real_float64_auto(val : 'float'):
        from numpy import full_like, float64, array
        arr = array([5, 1, 8, 0, 9], float64)
        a = full_like(arr,val)
        return a[0]
    def create_full_like_val_real_float64(val : 'float'):
        from numpy import full_like, float64, array
        arr = array([5, 1, 8, 0, 9])
        a = full_like(arr,val,float64)
        return a[0]

    def create_full_like_val_real_complex64_auto(val : 'float'):
        from numpy import full_like, complex64, array
        arr = array([5, 1, 8, 0, 9], complex64)
        a = full_like(arr,val)
        return a[0]
    def create_full_like_val_real_complex64(val : 'float'):
        from numpy import full_like, complex64, array
        arr = array([5, 1, 8, 0, 9])
        a = full_like(arr,val,complex64)
        return a[0]

    def create_full_like_val_real_complex128_auto(val : 'float'):
        from numpy import full_like, complex128, array
        arr = array([5, 1, 8, 0, 9], complex128)
        a = full_like(arr,val)
        return a[0]
    def create_full_like_val_real_complex128(val : 'float'):
        from numpy import full_like, complex128, array
        arr = array([5, 1, 8, 0, 9])
        a = full_like(arr,val,complex128)
        return a[0]

    val_int   = randint(100)
    val_float = rand()*100

    f_int_int   = epyccel(create_full_like_val_int_int, language = language)
    assert(     f_int_int(val_int)        ==      create_full_like_val_int_int(val_int))
    assert matching_types(f_int_int(val_int), create_full_like_val_int_int(val_int))

    f_int_float = epyccel(create_full_like_val_int_float, language = language)
    assert(isclose(     f_int_float(val_int)     ,      create_full_like_val_int_float(val_int), rtol=RTOL, atol=ATOL))
    assert matching_types(f_int_float(val_int), create_full_like_val_int_float(val_int))

    f_int_complex = epyccel(create_full_like_val_int_complex, language = language)
    assert(isclose(     f_int_complex(val_int)     ,      create_full_like_val_int_complex(val_int), rtol=RTOL, atol=ATOL))
    assert matching_types(f_int_complex(val_int), create_full_like_val_int_complex(val_int))

    f_real_int32   = epyccel(create_full_like_val_real_int32, language = language)
    assert(     f_real_int32(val_float)        ==      create_full_like_val_real_int32(val_float))
    assert matching_types(f_real_int32(val_float), create_full_like_val_real_int32(val_float))

    f_real_float32   = epyccel(create_full_like_val_real_float32, language = language)
    assert(isclose(     f_real_float32(val_float)       ,      create_full_like_val_real_float32(val_float), rtol=RTOL, atol=ATOL))
    assert matching_types(f_real_float32(val_float), create_full_like_val_real_float32(val_float))

    f_real_float64   = epyccel(create_full_like_val_real_float64, language = language)
    assert(isclose(     f_real_float64(val_float)       ,      create_full_like_val_real_float64(val_float), rtol=RTOL, atol=ATOL))
    assert matching_types(f_real_float64(val_float), create_full_like_val_real_float64(val_float))

    f_real_complex64   = epyccel(create_full_like_val_real_complex64, language = language)
    assert(isclose(     f_real_complex64(val_float)       ,      create_full_like_val_real_complex64(val_float), rtol=RTOL, atol=ATOL))
    assert matching_types(f_real_complex64(val_float), create_full_like_val_real_complex64(val_float))

    f_real_complex128   = epyccel(create_full_like_val_real_complex128, language = language)
    assert(isclose(     f_real_complex128(val_float)       ,      create_full_like_val_real_complex128(val_float), rtol=RTOL, atol=ATOL))
    assert matching_types(f_real_complex128(val_float), create_full_like_val_real_complex128(val_float))

    f_int_int_auto   = epyccel(create_full_like_val_int_int_auto, language = language)
    assert(     f_int_int_auto(val_int)        ==      create_full_like_val_int_int_auto(val_int))
    assert matching_types(f_int_int(val_int), create_full_like_val_int_int_auto(val_int))

    f_int_float_auto = epyccel(create_full_like_val_int_float_auto, language = language)
    assert(isclose(     f_int_float_auto(val_int)     ,      create_full_like_val_int_float_auto(val_int), rtol=RTOL, atol=ATOL))
    assert matching_types(f_int_float_auto(val_int), create_full_like_val_int_float_auto(val_int))

    f_int_complex_auto = epyccel(create_full_like_val_int_complex_auto, language = language)
    assert(isclose(     f_int_complex_auto(val_int)     ,      create_full_like_val_int_complex_auto(val_int), rtol=RTOL, atol=ATOL))
    assert matching_types(f_int_complex_auto(val_int), create_full_like_val_int_complex_auto(val_int))

    f_real_int32_auto   = epyccel(create_full_like_val_real_int32_auto, language = language)
    assert(     f_real_int32_auto(val_float)        ==      create_full_like_val_real_int32_auto(val_float))
    assert matching_types(f_real_int32_auto(val_float), create_full_like_val_real_int32_auto(val_float))

    f_real_float32_auto   = epyccel(create_full_like_val_real_float32_auto, language = language)
    assert(isclose(     f_real_float32_auto(val_float)       ,      create_full_like_val_real_float32_auto(val_float), rtol=RTOL, atol=ATOL))
    assert matching_types(f_real_float32_auto(val_float), create_full_like_val_real_float32_auto(val_float))

    f_real_float64_auto   = epyccel(create_full_like_val_real_float64_auto, language = language)
    assert(isclose(     f_real_float64_auto(val_float)       ,      create_full_like_val_real_float64_auto(val_float), rtol=RTOL, atol=ATOL))
    assert matching_types(f_real_float64_auto(val_float), create_full_like_val_real_float64_auto(val_float))

    f_real_complex64_auto   = epyccel(create_full_like_val_real_complex64_auto, language = language)
    assert(isclose(     f_real_complex64_auto(val_float)       ,      create_full_like_val_real_complex64_auto(val_float), rtol=RTOL, atol=ATOL))
    assert matching_types(f_real_complex64_auto(val_float), create_full_like_val_real_complex64_auto(val_float))

    f_real_complex128_auto   = epyccel(create_full_like_val_real_complex128_auto, language = language)
    assert(isclose(     f_real_complex128_auto(val_float)       ,      create_full_like_val_real_complex128_auto(val_float), rtol=RTOL, atol=ATOL))
    assert matching_types(f_real_complex128_auto(val_float), create_full_like_val_real_complex128_auto(val_float))

def test_full_like_combined_args(language):
    def create_full_like_1_shape():
        from numpy import full_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = full_like(arr,5,int,'F')
        s = shape(a)
        return len(s),s[0],s[1]
    def create_full_like_1_val():
        from numpy import full_like, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = full_like(arr, 4.0, int,'F')
        return a[0,0]
    def create_full_like_2_shape():
        from numpy import full_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = full_like(arr,dtype=float,fill_value=1)
        s = shape(a)
        return len(s),s[0],s[1]
    def create_full_like_2_val():
        from numpy import full_like, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = full_like(arr,dtype=float,fill_value=1)
        return a[0,0]
    def create_full_like_3_shape():
        from numpy import full_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = full_like(arr,order = 'F', shape = (4,2),dtype=complex,fill_value=1)
        s = shape(a)
        return len(s),s[0],s[1]
    def create_full_like_3_val():
        from numpy import full_like, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = full_like(arr,order = 'F', shape = (4,2),dtype=complex,fill_value=1)
        return a[0,0]


    f1_shape = epyccel(create_full_like_1_shape, language = language)
    f1_val   = epyccel(create_full_like_1_val, language = language)
    assert(f1_shape() == create_full_like_1_shape())
    assert(f1_val()   == create_full_like_1_val()  )
    assert matching_types(f1_val(), create_full_like_1_val())

    f2_shape = epyccel(create_full_like_2_shape, language = language)
    f2_val   = epyccel(create_full_like_2_val, language = language)
    assert(f2_shape() == create_full_like_2_shape()    )
    assert(isclose(f2_val()  , create_full_like_2_val()      , rtol=RTOL, atol=ATOL))
    assert matching_types(f2_val(), create_full_like_2_val())

    f3_shape = epyccel(create_full_like_3_shape, language = language)
    f3_val   = epyccel(create_full_like_3_val, language = language)
    assert(             f3_shape() ==    create_full_like_3_shape()      )
    assert(isclose(     f3_val()  ,      create_full_like_3_val()        , rtol=RTOL, atol=ATOL))
    assert matching_types(f3_val(), create_full_like_3_val())

def test_empty_like_basic(language):
    def create_empty_like_shape_1d(n : 'int'):
        from numpy import empty_like, shape, array
        arr = array([5, 1, 8, 0, 9])
        a = empty_like(arr,int)
        s = shape(a)
        return len(s),s[0]
    def create_empty_like_shape_2d(n : 'int'):
        from numpy import empty_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = empty_like(arr,int)
        s = shape(a)
        return len(s),s[0], s[1]

    size = randint(10)

    f_shape_1d  = epyccel(create_empty_like_shape_1d, language = language)
    assert(     f_shape_1d(size)      ==      create_empty_like_shape_1d(size))

    f_shape_2d  = epyccel(create_empty_like_shape_2d, language = language)
    assert(     f_shape_2d(size)      ==      create_empty_like_shape_2d(size))

def test_empty_like_order(language):
    def create_empty_like_shape_C(n : 'int', m : 'int'):
        from numpy import empty_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = empty_like(arr, int, order = 'C')
        s = shape(a)
        return len(s),s[0], s[1]
    def create_empty_like_shape_F(n : 'int', m : 'int'):
        from numpy import empty_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = empty_like(arr, int, order = 'F')
        s = shape(a)
        return len(s),s[0], s[1]

    size_1 = randint(10)
    size_2 = randint(10)

    f_shape_C  = epyccel(create_empty_like_shape_C, language = language)
    assert(     f_shape_C(size_1,size_2) == create_empty_like_shape_C(size_1,size_2))

    f_shape_F  = epyccel(create_empty_like_shape_F, language = language)
    assert(     f_shape_F(size_1,size_2) == create_empty_like_shape_F(size_1,size_2))

def test_empty_like_dtype(language):

    def create_empty_like_val_int_auto():
        from numpy import empty_like, array
        arr = array([5, 1, 8, 0, 9], dtype=int)
        a = empty_like(arr)
        return a[0]

    def create_empty_like_val_int():
        from numpy import empty_like, array
        arr = array([5, 1, 8, 0, 9])
        a = empty_like(arr, int)
        return a[0]

    def create_empty_like_val_float_auto():
        from numpy import empty_like, array
        arr = array([5, 1, 8, 0, 9], dtype=float)
        a = empty_like(arr)
        return a[0]

    def create_empty_like_val_float():
        from numpy import empty_like, array
        arr = array([5, 1, 8, 0, 9])
        a = empty_like(arr, dtype=float)
        return a[0]

    def create_empty_like_val_complex_auto():
        from numpy import empty_like, array
        arr = array([5, 1, 8, 0, 9], dtype=complex)
        a = empty_like(arr)
        return a[0]

    def create_empty_like_val_complex():
        from numpy import empty_like, array
        arr = array([5, 1, 8, 0, 9])
        a = empty_like(arr, dtype=complex)
        return a[0]

    def create_empty_like_val_int32_auto():
        from numpy import empty_like, array, int32
        arr = array([5, 1, 8, 0, 9], dtype=int32)
        a = empty_like(arr)
        return a[0]

    def create_empty_like_val_int32():
        from numpy import empty_like, int32, array
        arr = array([5, 1, 8, 0, 9])
        a = empty_like(arr, dtype=int32)
        return a[0]

    def create_empty_like_val_float32_auto():
        from numpy import empty_like, array, float32
        arr = array([5, 1, 8, 0, 9], dtype='float32')
        a = empty_like(arr)
        return a[0]

    def create_empty_like_val_float32():
        from numpy import empty_like, float32, array
        arr = array([5, 1, 8, 0, 9])
        a = empty_like(arr, dtype=float32)
        return a[0]

    def create_empty_like_val_float64_auto():
        from numpy import empty_like, array, float64
        arr = array([5, 1, 8, 0, 9], dtype=float64)
        a = empty_like(arr)
        return a[0]

    def create_empty_like_val_float64():
        from numpy import empty_like, float64, array
        arr = array([5, 1, 8, 0, 9])
        a = empty_like(arr,dtype=float64)
        return a[0]

    def create_empty_like_val_complex64_auto():
        from numpy import empty_like, array, complex64
        arr = array([5, 1, 8, 0, 9], dtype=complex64)
        a = empty_like(arr)
        return a[0]

    def create_empty_like_val_complex64():
        from numpy import empty_like, complex64, array
        arr = array([5, 1, 8, 0, 9])
        a = empty_like(arr,dtype=complex64)
        return a[0]

    def create_empty_like_val_complex128_auto():
        from numpy import empty_like, array, complex128
        arr = array([5, 1, 8, 0, 9], dtype=complex128)
        a = empty_like(arr)
        return a[0]

    def create_empty_like_val_complex128():
        from numpy import empty_like, complex128, array
        arr = array([5, 1, 8, 0, 9])
        a = empty_like(arr,dtype=complex128)
        return a[0]


    f_int_auto   = epyccel(create_empty_like_val_int_auto, language = language)
    assert matching_types(f_int_auto(), create_empty_like_val_int_auto())

    f_int_int   = epyccel(create_empty_like_val_int, language = language)
    assert matching_types(f_int_int(), create_empty_like_val_int())

    f_float_auto = epyccel(create_empty_like_val_float_auto, language = language)
    assert matching_types(f_float_auto(), create_empty_like_val_float_auto())

    f_int_float = epyccel(create_empty_like_val_float, language = language)
    assert matching_types(f_int_float(), create_empty_like_val_float())

    f_complex_auto = epyccel(create_empty_like_val_complex_auto, language = language)
    assert matching_types(f_complex_auto(), create_empty_like_val_complex_auto())

    f_int_complex = epyccel(create_empty_like_val_complex, language = language)
    assert matching_types(f_int_complex(), create_empty_like_val_complex())

    f_int32_auto   = epyccel(create_empty_like_val_int32_auto, language = language)
    assert matching_types(f_int32_auto(), create_empty_like_val_int32_auto())

    f_real_int32   = epyccel(create_empty_like_val_int32, language = language)
    assert matching_types(f_real_int32(), create_empty_like_val_int32())

    f_float32_auto   = epyccel(create_empty_like_val_float32_auto, language = language)
    assert matching_types(f_float32_auto(), create_empty_like_val_float32_auto())

    f_real_float32   = epyccel(create_empty_like_val_float32, language = language)
    assert matching_types(f_real_float32(), create_empty_like_val_float32())

    f_float64_auto   = epyccel(create_empty_like_val_float64_auto, language = language)
    assert matching_types(f_float64_auto(), create_empty_like_val_float64_auto())

    f_real_float64   = epyccel(create_empty_like_val_float64, language = language)
    assert matching_types(f_real_float64(), create_empty_like_val_float64())

    f_complex64_auto   = epyccel(create_empty_like_val_complex64_auto, language = language)

    assert matching_types(f_complex64_auto(), create_empty_like_val_complex64_auto())

    f_real_complex64   = epyccel(create_empty_like_val_complex64, language = language)
    assert matching_types(f_real_complex64(), create_empty_like_val_complex64())

    f_complex128_auto   = epyccel(create_empty_like_val_complex128_auto, language = language)
    assert matching_types(f_complex128_auto(), create_empty_like_val_complex128_auto())

    f_real_complex128   = epyccel(create_empty_like_val_complex128, language = language)
    assert matching_types(f_real_complex128(), create_empty_like_val_complex128())

def test_empty_like_combined_args(language):

    def create_empty_like_1_shape():
        from numpy import empty_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = empty_like(arr,dtype=int,order='F')
        s = shape(a)
        return len(s),s[0],s[1]

    def create_empty_like_1_val():
        from numpy import empty_like, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = empty_like(arr, dtype=int,order='F')
        return a[0,0]

    def create_empty_like_2_shape():
        from numpy import empty_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = empty_like(arr, dtype=float)
        s = shape(a)
        return len(s),s[0],s[1]

    def create_empty_like_2_val():
        from numpy import empty_like, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = empty_like(arr, dtype=float)
        return a[0,0]

    def create_empty_like_3_shape():
        from numpy import empty_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = empty_like(arr,shape = (4,2), order = 'F',dtype=complex)
        s = shape(a)
        return len(s),s[0],s[1]

    def create_empty_like_3_val():
        from numpy import empty_like, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = empty_like(arr, shape = (4,2),order = 'F',dtype=complex)
        return a[0,0]

    f1_shape = epyccel(create_empty_like_1_shape, language = language)
    f1_val   = epyccel(create_empty_like_1_val, language = language)
    assert(     f1_shape() ==      create_empty_like_1_shape()      )
    assert matching_types(f1_val(), create_empty_like_1_val())

    f2_shape = epyccel(create_empty_like_2_shape, language = language)
    f2_val   = epyccel(create_empty_like_2_val, language = language)
    assert(all(isclose(     f2_shape(),      create_empty_like_2_shape()      )))
    assert matching_types(f2_val(), create_empty_like_2_val())

    f3_shape = epyccel(create_empty_like_3_shape, language = language)
    f3_val   = epyccel(create_empty_like_3_val, language = language)
    assert(all(isclose(     f3_shape(),      create_empty_like_3_shape()      )))
    assert matching_types(f3_val(), create_empty_like_3_val())

def test_ones_like_basic(language):
    def create_ones_like_shape_1d(n : 'int'):
        from numpy import ones_like, shape, array
        arr = array([5, 1, 8, 0, 9])
        a = ones_like(arr)
        s = shape(a)
        return len(s),s[0]
    def create_ones_like_shape_2d(n : 'int'):
        from numpy import ones_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = ones_like(arr)
        s = shape(a)
        return len(s),s[0], s[1]

    size = randint(10)

    f_shape_1d  = epyccel(create_ones_like_shape_1d, language = language)
    assert(     f_shape_1d(size)      ==      create_ones_like_shape_1d(size))

    f_shape_2d  = epyccel(create_ones_like_shape_2d, language = language)
    assert(     f_shape_2d(size)      ==      create_ones_like_shape_2d(size))

def test_ones_like_order(language):
    def create_ones_like_shape_C(n : 'int', m : 'int'):
        from numpy import ones_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = ones_like(arr, order = 'C')
        s = shape(a)
        return len(s),s[0], s[1]
    def create_ones_like_shape_F(n : 'int', m : 'int'):
        from numpy import ones_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = ones_like(arr, order = 'F')
        s = shape(a)
        return len(s),s[0], s[1]

    size_1 = randint(10)
    size_2 = randint(10)

    f_shape_C  = epyccel(create_ones_like_shape_C, language = language)
    assert(     f_shape_C(size_1,size_2) == create_ones_like_shape_C(size_1,size_2))

    f_shape_F  = epyccel(create_ones_like_shape_F, language = language)
    assert(     f_shape_F(size_1,size_2) == create_ones_like_shape_F(size_1,size_2))

def test_ones_like_dtype(language):

    def create_ones_like_val_int_auto():
        from numpy import ones_like, array
        arr = array([5, 1, 8, 0, 9], int)
        a = ones_like(arr)
        return a[0]

    def create_ones_like_val_int():
        from numpy import ones_like, array
        arr = array([5, 1, 8, 0, 9])
        a = ones_like(arr, int)
        return a[0]

    def create_ones_like_val_float_auto():
        from numpy import ones_like, array
        arr = array([5, 1, 8, 0, 9], float)
        a = ones_like(arr)
        return a[0]

    def create_ones_like_val_float():
        from numpy import ones_like, array
        arr = array([5, 1, 8, 0, 9])
        a = ones_like(arr,float)
        return a[0]

    def create_ones_like_val_complex_auto():
        from numpy import ones_like, array
        arr = array([5, 1, 8, 0, 9], complex)
        a = ones_like(arr)
        return a[0]

    def create_ones_like_val_complex():
        from numpy import ones_like, array
        arr = array([5, 1, 8, 0, 9])
        a = ones_like(arr, complex)
        return a[0]

    def create_ones_like_val_int32_auto():
        from numpy import ones_like, int32, array
        arr = array([5, 1, 8, 0, 9], int32)
        a = ones_like(arr)
        return a[0]

    def create_ones_like_val_int32():
        from numpy import ones_like, int32, array
        arr = array([5, 1, 8, 0, 9])
        a = ones_like(arr,int32)
        return a[0]

    def create_ones_like_val_float32_auto():
        from numpy import ones_like, float32, array
        arr = array([5, 1, 8, 0, 9], float32)
        a = ones_like(arr)
        return a[0]

    def create_ones_like_val_float32():
        from numpy import ones_like, float32, array
        arr = array([5, 1, 8, 0, 9])
        a = ones_like(arr, float32)
        return a[0]

    def create_ones_like_val_float64_auto():
        from numpy import ones_like, float64, array
        arr = array([5, 1, 8, 0, 9], float64)
        a = ones_like(arr)
        return a[0]

    def create_ones_like_val_float64():
        from numpy import ones_like, float64, array
        arr = array([5, 1, 8, 0, 9])
        a = ones_like(arr, float64)
        return a[0]

    def create_ones_like_val_complex64_auto():
        from numpy import ones_like, complex64, array
        arr = array([5, 1, 8, 0, 9], complex64)
        a = ones_like(arr)
        return a[0]

    def create_ones_like_val_complex64():
        from numpy import ones_like, complex64, array
        arr = array([5, 1, 8, 0, 9])
        a = ones_like(arr, complex64)
        return a[0]

    def create_ones_like_val_complex128_auto():
        from numpy import ones_like, complex128, array
        arr = array([5, 1, 8, 0, 9], complex128)
        a = ones_like(arr)
        return a[0]

    def create_ones_like_val_complex128():
        from numpy import ones_like, complex128, array
        arr = array([5, 1, 8, 0, 9])
        a = ones_like(arr, complex128)
        return a[0]


    f_int_int   = epyccel(create_ones_like_val_int, language = language)
    assert(     f_int_int()          ==      create_ones_like_val_int())
    assert matching_types(f_int_int(), create_ones_like_val_int())

    f_int_float = epyccel(create_ones_like_val_float, language = language)
    assert(isclose(     f_int_float()       ,      create_ones_like_val_float(), rtol=RTOL, atol=ATOL))
    assert matching_types(f_int_float(), create_ones_like_val_float())

    f_int_complex = epyccel(create_ones_like_val_complex, language = language)
    assert(isclose(     f_int_complex()     ,      create_ones_like_val_complex(), rtol=RTOL, atol=ATOL))
    assert matching_types(f_int_complex(), create_ones_like_val_complex())

    f_real_int32   = epyccel(create_ones_like_val_int32, language = language)
    assert(     f_real_int32()       ==      create_ones_like_val_int32())
    assert matching_types(f_real_int32(), create_ones_like_val_int32())

    f_real_float32   = epyccel(create_ones_like_val_float32, language = language)
    assert(isclose(     f_real_float32()    ,      create_ones_like_val_float32(), rtol=RTOL, atol=ATOL))
    assert matching_types(f_real_float32(), create_ones_like_val_float32())

    f_real_float64   = epyccel(create_ones_like_val_float64, language = language)
    assert(isclose(     f_real_float64()    ,      create_ones_like_val_float64(), rtol=RTOL, atol=ATOL))
    assert matching_types(f_real_float64(), create_ones_like_val_float64())

    f_real_complex64   = epyccel(create_ones_like_val_complex64, language = language)
    assert(isclose(     f_real_complex64()  ,      create_ones_like_val_complex64(), rtol=RTOL, atol=ATOL))
    assert matching_types(f_real_complex64(), create_ones_like_val_complex64())

    f_real_complex128   = epyccel(create_ones_like_val_complex128, language = language)
    assert(isclose(     f_real_complex128() ,      create_ones_like_val_complex128(), rtol=RTOL, atol=ATOL))
    assert matching_types(f_real_complex128(), create_ones_like_val_complex128())

    f_int_int_auto   = epyccel(create_ones_like_val_int_auto, language = language)
    assert(     f_int_int_auto()          ==      create_ones_like_val_int_auto())
    assert matching_types(f_int_int_auto(), create_ones_like_val_int_auto())

    f_int_float_auto = epyccel(create_ones_like_val_float_auto, language = language)
    assert(isclose(     f_int_float_auto()       ,      create_ones_like_val_float_auto(), rtol=RTOL, atol=ATOL))
    assert matching_types(f_int_float_auto(), create_ones_like_val_float_auto())

    f_int_complex_auto = epyccel(create_ones_like_val_complex_auto, language = language)
    assert(isclose(     f_int_complex_auto()     ,      create_ones_like_val_complex_auto(), rtol=RTOL, atol=ATOL))
    assert matching_types(f_int_complex_auto(), create_ones_like_val_complex_auto())

    f_real_int32_auto   = epyccel(create_ones_like_val_int32_auto, language = language)
    assert(     f_real_int32_auto()       ==      create_ones_like_val_int32_auto())
    assert matching_types(f_real_int32_auto(), create_ones_like_val_int32_auto())

    f_real_float32_auto   = epyccel(create_ones_like_val_float32_auto, language = language)
    assert(isclose(     f_real_float32_auto()    ,      create_ones_like_val_float32_auto(), rtol=RTOL, atol=ATOL))
    assert matching_types(f_real_float32_auto(), create_ones_like_val_float32_auto())

    f_real_float64_auto   = epyccel(create_ones_like_val_float64_auto, language = language)
    assert(isclose(     f_real_float64_auto()    ,      create_ones_like_val_float64_auto(), rtol=RTOL, atol=ATOL))
    assert matching_types(f_real_float64_auto(), create_ones_like_val_float64_auto())

    f_real_complex64_auto   = epyccel(create_ones_like_val_complex64_auto, language = language)
    assert(isclose(     f_real_complex64_auto()  ,      create_ones_like_val_complex64_auto(), rtol=RTOL, atol=ATOL))
    assert matching_types(f_real_complex64_auto(), create_ones_like_val_complex64_auto())

    f_real_complex128_auto   = epyccel(create_ones_like_val_complex128_auto, language = language)
    assert(isclose(     f_real_complex128_auto() ,      create_ones_like_val_complex128_auto(), rtol=RTOL, atol=ATOL))
    assert matching_types(f_real_complex128_auto(), create_ones_like_val_complex128_auto())

def test_ones_like_combined_args(language):

    def create_ones_like_1_shape():
        from numpy import ones_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = ones_like(arr,int,'F')
        s = shape(a)
        return len(s),s[0],s[1]

    def create_ones_like_1_val():
        from numpy import ones_like, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = ones_like(arr,int,'F')
        return a[0,0]

    def create_ones_like_2_shape():
        from numpy import ones_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = ones_like(arr,dtype=float)
        s = shape(a)
        return len(s),s[0],s[1]

    def create_ones_like_2_val():
        from numpy import ones_like, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = ones_like(arr,dtype=float)
        return a[0,0]

    def create_ones_like_3_shape():
        from numpy import ones_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = ones_like(arr,shape = (4,2),order = 'F',dtype=complex)
        s = shape(a)
        return len(s),s[0],s[1]

    def create_ones_like_3_val():
        from numpy import ones_like, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = ones_like(arr,shape = (4,2),order = 'F',dtype=complex)
        return a[0,0]

    f1_shape = epyccel(create_ones_like_1_shape, language = language)
    f1_val   = epyccel(create_ones_like_1_val, language = language)
    assert(     f1_shape() ==      create_ones_like_1_shape()      )
    assert(     f1_val()   ==      create_ones_like_1_val()        )
    assert matching_types(f1_val(), create_ones_like_1_val())

    f2_shape = epyccel(create_ones_like_2_shape, language = language)
    f2_val   = epyccel(create_ones_like_2_val, language = language)
    assert(     f2_shape() ==      create_ones_like_2_shape()      )
    assert(isclose(     f2_val()  ,      create_ones_like_2_val()        , rtol=RTOL, atol=ATOL))
    assert matching_types(f2_val(), create_ones_like_2_val())

    f3_shape = epyccel(create_ones_like_3_shape, language = language)
    f3_val   = epyccel(create_ones_like_3_val, language = language)
    assert(     f3_shape() ==      create_ones_like_3_shape()      )
    assert(isclose(     f3_val()  ,      create_ones_like_3_val()        , rtol=RTOL, atol=ATOL))
    assert matching_types(f3_val(), create_ones_like_3_val())

def test_zeros_like_basic(language):
    def create_zeros_like_shape_1d(n : 'int'):
        from numpy import zeros_like, shape, array
        arr = array([5, 1, 8, 0, 9])
        a = zeros_like(arr, int)
        s = shape(a)
        return len(s),s[0]
    def create_zeros_like_shape_2d(n : 'int'):
        from numpy import zeros_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = zeros_like(arr,int)
        s = shape(a)
        return len(s),s[0], s[1]

    size = randint(10)

    f_shape_1d  = epyccel(create_zeros_like_shape_1d, language = language)
    assert(     f_shape_1d(size)      ==      create_zeros_like_shape_1d(size))

    f_shape_2d  = epyccel(create_zeros_like_shape_2d, language = language)
    assert(     f_shape_2d(size)      ==      create_zeros_like_shape_2d(size))

def test_zeros_like_order(language):
    def create_zeros_like_shape_C(n : 'int', m : 'int'):
        from numpy import zeros_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = zeros_like(arr, order = 'C')
        s = shape(a)
        return len(s),s[0], s[1]
    def create_zeros_like_shape_F(n : 'int', m : 'int'):
        from numpy import zeros_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = zeros_like(arr, order = 'F')
        s = shape(a)
        return len(s),s[0], s[1]

    size_1 = randint(10)
    size_2 = randint(10)

    f_shape_C  = epyccel(create_zeros_like_shape_C, language = language)
    assert(     f_shape_C(size_1,size_2) == create_zeros_like_shape_C(size_1,size_2))

    f_shape_F  = epyccel(create_zeros_like_shape_F, language = language)
    assert(     f_shape_F(size_1,size_2) == create_zeros_like_shape_F(size_1,size_2))

def test_zeros_like_dtype(language):

    def create_zeros_like_val_int():
        from numpy import zeros_like, array
        arr = array([5, 1, 8, 0, 9])
        a = zeros_like(arr,int)
        return a[0]

    def create_zeros_like_val_float():
        from numpy import zeros_like, array
        arr = array([5, 1, 8, 0, 9])
        a = zeros_like(arr,float)
        return a[0]

    def create_zeros_like_val_complex():
        from numpy import zeros_like, array
        arr = array([5, 1, 8, 0, 9])
        a = zeros_like(arr,complex)
        return a[0]

    def create_zeros_like_val_int32():
        from numpy import zeros_like, int32, array
        arr = array([5, 1, 8, 0, 9])
        a = zeros_like(arr,int32)
        return a[0]

    def create_zeros_like_val_float32():
        from numpy import zeros_like, float32, array
        arr = array([5, 1, 8, 0, 9])
        a = zeros_like(arr,float32)
        return a[0]

    def create_zeros_like_val_float64():
        from numpy import zeros_like, float64, array
        arr = array([5, 1, 8, 0, 9])
        a = zeros_like(arr,float64)
        return a[0]

    def create_zeros_like_val_complex64():
        from numpy import zeros_like, complex64, array
        arr = array([5, 1, 8, 0, 9])
        a = zeros_like(arr,complex64)
        return a[0]

    def create_zeros_like_val_complex128():
        from numpy import zeros_like, complex128, array
        arr = array([5, 1, 8, 0, 9])
        a = zeros_like(arr,complex128)
        return a[0]

    f_int_int   = epyccel(create_zeros_like_val_int, language = language)
    assert(     f_int_int()          ==      create_zeros_like_val_int())
    assert matching_types(f_int_int(), create_zeros_like_val_int())

    f_int_float = epyccel(create_zeros_like_val_float, language = language)
    assert(isclose(     f_int_float()       ,      create_zeros_like_val_float(), rtol=RTOL, atol=ATOL))
    assert matching_types(f_int_float(), create_zeros_like_val_float())

    f_int_complex = epyccel(create_zeros_like_val_complex, language = language)
    assert(isclose(     f_int_complex()     ,      create_zeros_like_val_complex(), rtol=RTOL, atol=ATOL))
    assert matching_types(f_int_complex(), create_zeros_like_val_complex())

    f_real_int32   = epyccel(create_zeros_like_val_int32, language = language)
    assert(     f_real_int32()       ==      create_zeros_like_val_int32())
    assert matching_types(f_real_int32(), create_zeros_like_val_int32())

    f_real_float32   = epyccel(create_zeros_like_val_float32, language = language)
    assert(isclose(     f_real_float32()    ,      create_zeros_like_val_float32(), rtol=RTOL, atol=ATOL))
    assert matching_types(f_real_float32(), create_zeros_like_val_float32())

    f_real_float64   = epyccel(create_zeros_like_val_float64, language = language)
    assert(isclose(     f_real_float64()    ,      create_zeros_like_val_float64(), rtol=RTOL, atol=ATOL))
    assert matching_types(f_real_float64(), create_zeros_like_val_float64())

    f_real_complex64   = epyccel(create_zeros_like_val_complex64, language = language)
    assert(isclose(     f_real_complex64()  ,      create_zeros_like_val_complex64(), rtol=RTOL, atol=ATOL))
    assert matching_types(f_real_complex64(), create_zeros_like_val_complex64())

    f_real_complex128   = epyccel(create_zeros_like_val_complex128, language = language)
    assert(isclose(     f_real_complex128() ,      create_zeros_like_val_complex128(), rtol=RTOL, atol=ATOL))
    assert matching_types(f_real_complex128(), create_zeros_like_val_complex128())

def test_zeros_like_dtype_auto(language):

    def create_zeros_like_val_int_auto():
        from numpy import zeros_like, array
        arr = array([5, 1, 8, 0, 9], dtype=int)
        a = zeros_like(arr)
        return a[0]

    def create_zeros_like_val_float_auto():
        from numpy import zeros_like, array
        arr = array([5, 1, 8, 0, 9], dtype=float)
        a = zeros_like(arr)
        return a[0]

    def create_zeros_like_val_complex_auto():
        from numpy import zeros_like, array
        arr = array([5, 1, 8, 0, 9], dtype=complex)
        a = zeros_like(arr)
        return a[0]

    def create_zeros_like_val_int32_auto():
        from numpy import zeros_like, array, int32
        arr = array([5, 1, 8, 0, 9], dtype=int32)
        a = zeros_like(arr)
        return a[0]

    def create_zeros_like_val_float32_auto():
        from numpy import zeros_like, array, float32
        arr = array([5, 1, 8, 0, 9], dtype='float32')
        a = zeros_like(arr)
        return a[0]

    def create_zeros_like_val_float64_auto():
        from numpy import zeros_like, array, float64
        arr = array([5, 1, 8, 0, 9], dtype=float64)
        a = zeros_like(arr)
        return a[0]

    def create_zeros_like_val_complex64_auto():
        from numpy import zeros_like, array, complex64
        arr = array([5, 1, 8, 0, 9], dtype=complex64)
        a = zeros_like(arr)
        return a[0]

    def create_zeros_like_val_complex128_auto():
        from numpy import zeros_like, array, complex128
        arr = array([5, 1, 8, 0, 9], dtype=complex128)
        a = zeros_like(arr)
        return a[0]

    f_int_auto   = epyccel(create_zeros_like_val_int_auto, language = language)
    assert matching_types(f_int_auto(), create_zeros_like_val_int_auto())

    f_float_auto = epyccel(create_zeros_like_val_float_auto, language = language)
    assert matching_types(f_float_auto(), create_zeros_like_val_float_auto())

    f_complex_auto = epyccel(create_zeros_like_val_complex_auto, language = language)
    assert matching_types(f_complex_auto(), create_zeros_like_val_complex_auto())

    f_int32_auto   = epyccel(create_zeros_like_val_int32_auto, language = language)
    assert matching_types(f_int32_auto(), create_zeros_like_val_int32_auto())

    f_float32_auto   = epyccel(create_zeros_like_val_float32_auto, language = language)
    assert matching_types(f_float32_auto(), create_zeros_like_val_float32_auto())

    f_float64_auto   = epyccel(create_zeros_like_val_float64_auto, language = language)
    assert matching_types(f_float64_auto(), create_zeros_like_val_float64_auto())

    f_complex64_auto   = epyccel(create_zeros_like_val_complex64_auto, language = language)
    assert matching_types(f_complex64_auto(), create_zeros_like_val_complex64_auto())

    f_complex128_auto   = epyccel(create_zeros_like_val_complex128_auto, language = language)
    assert matching_types(f_complex128_auto(), create_zeros_like_val_complex128_auto())


def test_zeros_like_combined_args(language):

    def create_zeros_like_1_shape():
        from numpy import zeros_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = zeros_like(arr,int,'F')
        s = shape(a)
        return len(s),s[0],s[1]

    def create_zeros_like_1_val():
        from numpy import zeros_like, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = zeros_like(arr, int,'F')
        return a[0,0]

    def create_zeros_like_2_shape():
        from numpy import zeros_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = zeros_like(arr, dtype=float)
        s = shape(a)
        return len(s),s[0],s[1]

    def create_zeros_like_2_val():
        from numpy import zeros_like, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = zeros_like(arr, dtype=float)
        return a[0,0]

    def create_zeros_like_3_shape():
        from numpy import zeros_like, shape, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = zeros_like(arr, shape = (4,2), order = 'F',dtype=complex)
        s = shape(a)
        return len(s),s[0],s[1]

    def create_zeros_like_3_val():
        from numpy import zeros_like, array
        arr = array([[5, 1, 8, 0, 9], [5, 1, 8, 0, 9]])
        a = zeros_like(arr, shape = (4,2), order = 'F',dtype=complex)
        return a[0,0]

    f1_shape = epyccel(create_zeros_like_1_shape, language = language)
    f1_val   = epyccel(create_zeros_like_1_val, language = language)
    assert(     f1_shape() ==      create_zeros_like_1_shape()      )
    assert(     f1_val()   ==      create_zeros_like_1_val()        )
    assert matching_types(f1_val(), create_zeros_like_1_val())

    f2_shape = epyccel(create_zeros_like_2_shape, language = language)
    f2_val   = epyccel(create_zeros_like_2_val, language = language)
    assert(     f2_shape() ==      create_zeros_like_2_shape()      )
    assert(isclose(     f2_val()  ,      create_zeros_like_2_val()        , rtol=RTOL, atol=ATOL))
    assert matching_types(f2_val(), create_zeros_like_2_val())

    f3_shape = epyccel(create_zeros_like_3_shape, language = language)
    f3_val   = epyccel(create_zeros_like_3_val, language = language)
    assert(     f3_shape() ==      create_zeros_like_3_shape()      )
    assert(isclose(     f3_val()  ,      create_zeros_like_3_val()        , rtol=RTOL, atol=ATOL))
    assert matching_types(f3_val(), create_zeros_like_3_val())

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = [
            pytest.mark.skip(reason=("real handles types in __new__ so it "
                "cannot be used in a translated interface in python")),
            pytest.mark.python]
        )
    )
)
def test_numpy_real_scalar(language):

    @template('T', ['bool', 'int', 'int8', 'int16', 'int32', 'int64', 'float', 'float32', 'float64', 'complex64', 'complex128'])
    def get_real(a : 'T'):
        from numpy import real
        b = real(a)
        return b

    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=int)
    integer32 = randint(min_int32, max_int32, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2)
    fl32 = uniform(min_float32 / 2, max_float32 / 2)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2)

    cmplx_from_float32 = uniform(low=min_float32 / 2, high=max_float32 / 2) + \
            uniform(low=min_float32 / 2, high=max_float32 / 2) * 1j
    cmplx_from_float64 = uniform(low=min_float64 / 2, high=max_float64 / 2) + \
            uniform(low=min_float64 / 2, high=max_float64 / 2) * 1j
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx_from_float32)
    cmplx128 = np.complex128(cmplx_from_float64)

    epyccel_func = epyccel(get_real, language=language)

    f_bl_true_output = epyccel_func(True)
    test_bool_true_output = get_real(True)

    f_bl_false_output = epyccel_func(False)
    test_bool_false_output = get_real(False)

    assert f_bl_true_output == test_bool_true_output
    assert f_bl_false_output == test_bool_false_output

    assert matching_types(f_bl_true_output, test_bool_true_output)
    assert matching_types(f_bl_false_output, test_bool_false_output)

    f_integer_output = epyccel_func(integer)
    test_int_output  = get_real(integer)

    assert f_integer_output == test_int_output
    assert matching_types(f_integer_output, test_int_output)

    f_integer8_output = epyccel_func(integer8)
    test_int8_output = get_real(integer8)

    assert f_integer8_output == test_int8_output
    assert matching_types(f_integer8_output, test_int8_output)

    f_integer16_output = epyccel_func(integer16)
    test_int16_output = get_real(integer16)

    assert f_integer16_output == test_int16_output
    assert matching_types(f_integer16_output, test_int16_output)

    f_integer32_output = epyccel_func(integer32)
    test_int32_output = get_real(integer32)

    assert f_integer32_output == test_int32_output
    assert matching_types(f_integer32_output, test_int32_output)

    f_integer64_output = epyccel_func(integer64)
    test_int64_output = get_real(integer64)

    assert f_integer64_output == test_int64_output
    assert matching_types(f_integer64_output, test_int64_output)

    f_fl_output = epyccel_func(fl)
    test_float_output = get_real(fl)

    assert f_fl_output == test_float_output
    assert matching_types(f_fl_output, test_float_output)

    f_fl32_output = epyccel_func(fl32)
    test_float32_output = get_real(fl32)

    assert f_fl32_output == test_float32_output
    assert matching_types(f_fl32_output, test_float32_output)

    f_fl64_output = epyccel_func(fl64)
    test_float64_output = get_real(fl64)

    assert f_fl64_output == test_float64_output
    assert matching_types(f_fl64_output, test_float64_output)

    f_complex64_output = epyccel_func(cmplx64)
    test_complex64_output = get_real(cmplx64)

    assert f_complex64_output == test_complex64_output
    assert matching_types(f_complex64_output, test_complex64_output)

    f_complex128_output = epyccel_func(cmplx128)
    test_complex128_output = get_real(cmplx128)

    assert f_complex128_output == test_complex128_output
    assert matching_types(f_complex64_output, test_complex64_output)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="See https://github.com/pyccel/pyccel/issues/794."),
            pytest.mark.c]
        ),
        pytest.param("python", marks = [
            pytest.mark.skip(reason=("real handles types in __new__ so it "
                "cannot be used in a translated interface in python")),
            pytest.mark.python]
        )
    )
)

def test_numpy_real_array_like_1d(language):

    @template('T', ['bool[:]', 'int[:]', 'int8[:]', 'int16[:]', 'int32[:]', 'int64[:]', 'float[:]', 'float32[:]', 'float64[:]', 'complex64[:]', 'complex128[:]'])
    def get_real(arr : 'T'):
        from numpy import real, shape
        a = real(arr)
        s = shape(a)
        return len(s), s[0], a[0]

    size = 5

    bl = randint(0, 2, size = size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size = size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size = size, dtype=np.int16)
    integer = randint(min_int, max_int, size = size, dtype=int)
    integer32 = randint(min_int32, max_int32, size = size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size = size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size = size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size = size)

    cmplx128_from_float32 = uniform(low=min_float32 / 2, high=max_float32 / 2, size = size) + uniform(low=min_float32 / 2, high=max_float32 / 2, size = size) * 1j
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = uniform(low=min_float64 / 2, high=max_float64 / 2, size = size) + uniform(low=min_float64 / 2, high=max_float64 / 2, size = size) * 1j

    epyccel_func = epyccel(get_real, language=language)

    assert epyccel_func(bl) == get_real(bl)
    assert epyccel_func(integer8) == get_real(integer8)
    assert epyccel_func(integer16) == get_real(integer16)
    assert epyccel_func(integer) == get_real(integer)
    assert epyccel_func(integer32) == get_real(integer32)
    assert epyccel_func(integer64) == get_real(integer64)
    assert epyccel_func(fl) == get_real(fl)
    assert epyccel_func(fl32) == get_real(fl32)
    assert epyccel_func(fl64) == get_real(fl64)
    assert epyccel_func(cmplx64) == get_real(cmplx64)
    assert epyccel_func(cmplx128) == get_real(cmplx128)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="See https://github.com/pyccel/pyccel/issues/794."),
            pytest.mark.c]
        ),
        pytest.param("python", marks = [
            pytest.mark.skip(reason=("real handles types in __new__ so it "
                "cannot be used in a translated interface in python")),
            pytest.mark.python]
        )
    )
)

def test_numpy_real_array_like_2d(language):

    @template('T', ['bool[:,:]', 'int[:,:]', 'int8[:,:]', 'int16[:,:]', 'int32[:,:]', 'int64[:,:]', 'float[:,:]', 'float32[:,:]', 'float64[:,:]', 'complex64[:,:]', 'complex128[:,:]'])
    def get_real(arr : 'T'):
        from numpy import real, shape
        a = real(arr)
        s = shape(a)
        return len(s), s[0], s[1], a[0,1], a[1,0]

    size = (2, 5)

    bl = randint(0, 2, size = size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size = size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size = size, dtype=np.int16)
    integer = randint(min_int, max_int, size = size, dtype=int)
    integer32 = randint(min_int32, max_int32, size = size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size = size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size = size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size=size)

    cmplx128_from_float32 = uniform(low=min_float32 / 2, high=max_float32 / 2, size = size) + uniform(low=min_float32 / 2, high=max_float32 / 2, size = size) * 1j
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = uniform(low=min_float64 / 2, high=max_float64 / 2, size = size) + uniform(low=min_float64 / 2, high=max_float64 / 2, size = size) * 1j

    epyccel_func = epyccel(get_real, language=language)

    assert epyccel_func(bl) == get_real(bl)
    assert epyccel_func(integer8) == get_real(integer8)
    assert epyccel_func(integer16) == get_real(integer16)
    assert epyccel_func(integer) == get_real(integer)
    assert epyccel_func(integer32) == get_real(integer32)
    assert epyccel_func(integer64) == get_real(integer64)
    assert epyccel_func(fl) == get_real(fl)
    assert epyccel_func(fl32) == get_real(fl32)
    assert epyccel_func(fl64) == get_real(fl64)
    assert epyccel_func(cmplx64) == get_real(cmplx64)
    assert epyccel_func(cmplx128) == get_real(cmplx128)


@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = [
            pytest.mark.skip(reason=("imag handles types in __new__ so it "
                "cannot be used in a translated interface in python")),
            pytest.mark.python]
        )
    )
)
def test_numpy_imag_scalar(language):

    @template('T', ['bool', 'int', 'int8', 'int16', 'int32', 'int64', 'float', 'float32', 'float64', 'complex64', 'complex128'])
    def get_imag(a : 'T'):
        from numpy import imag
        b = imag(a)
        return b

    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=int)
    integer32 = randint(min_int32, max_int32, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2)
    fl32 = uniform(min_float32 / 2, max_float32 / 2)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2)

    cmplx_from_float32 = uniform(low=min_float32 / 2, high=max_float32 / 2) + \
            uniform(low=min_float32 / 2, high=max_float32 / 2) * 1j
    cmplx_from_float64 = uniform(low=min_float64 / 2, high=max_float64 / 2) + \
            uniform(low=min_float64 / 2, high=max_float64 / 2) * 1j
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx_from_float32)
    cmplx128 = np.complex128(cmplx_from_float64)

    epyccel_func = epyccel(get_imag, language=language)

    f_bl_true_output = epyccel_func(True)
    test_bool_true_output = get_imag(True)

    f_bl_false_output = epyccel_func(False)
    test_bool_false_output = get_imag(False)

    assert f_bl_true_output == test_bool_true_output
    assert f_bl_false_output == test_bool_false_output

    assert matching_types(f_bl_true_output, test_bool_true_output)
    assert matching_types(f_bl_false_output, test_bool_false_output)

    f_integer_output = epyccel_func(integer)
    test_int_output  = get_imag(integer)

    assert f_integer_output == test_int_output
    assert matching_types(f_integer_output, test_int_output)

    f_integer8_output = epyccel_func(integer8)
    test_int8_output = get_imag(integer8)

    assert f_integer8_output == test_int8_output
    assert matching_types(f_integer8_output, test_int8_output)

    f_integer16_output = epyccel_func(integer16)
    test_int16_output = get_imag(integer16)

    assert f_integer16_output == test_int16_output
    assert matching_types(f_integer16_output, test_int16_output)

    f_integer32_output = epyccel_func(integer32)
    test_int32_output = get_imag(integer32)

    assert f_integer32_output == test_int32_output
    assert matching_types(f_integer32_output, test_int32_output)

    f_integer64_output = epyccel_func(integer64)
    test_int64_output = get_imag(integer64)

    assert f_integer64_output == test_int64_output
    assert matching_types(f_integer64_output, test_int64_output)

    f_fl_output = epyccel_func(fl)
    test_float_output = get_imag(fl)

    assert f_fl_output == test_float_output
    assert matching_types(f_fl_output, test_float_output)

    f_fl32_output = epyccel_func(fl32)
    test_float32_output = get_imag(fl32)

    assert f_fl32_output == test_float32_output
    assert matching_types(f_fl32_output, test_float32_output)

    f_fl64_output = epyccel_func(fl64)
    test_float64_output = get_imag(fl64)

    assert f_fl64_output == test_float64_output
    assert matching_types(f_fl64_output, test_float64_output)

    f_complex64_output = epyccel_func(cmplx64)
    test_complex64_output = get_imag(cmplx64)

    assert f_complex64_output == test_complex64_output
    assert matching_types(f_complex64_output, test_complex64_output)

    f_complex128_output = epyccel_func(cmplx128)
    test_complex128_output = get_imag(cmplx128)

    assert f_complex128_output == test_complex128_output
    assert matching_types(f_complex64_output, test_complex64_output)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = [
            pytest.mark.skip(reason=("imag handles types in __new__ so it "
                "cannot be used in a translated interface in python")),
            pytest.mark.python]
        )
    )
)

def test_numpy_imag_array_like_1d(language):

    @template('T', ['bool[:]', 'int[:]', 'int8[:]', 'int16[:]', 'int32[:]', 'int64[:]', 'float[:]', 'float32[:]', 'float64[:]', 'complex64[:]', 'complex128[:]'])
    def get_imag(arr : 'T'):
        from numpy import imag, shape
        a = imag(arr)
        s = shape(a)
        return len(s), s[0], a[0]

    size = 5

    bl = randint(0, 2, size = size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size = size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size = size, dtype=np.int16)
    integer = randint(min_int, max_int, size = size, dtype=int)
    integer32 = randint(min_int32, max_int32, size = size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size = size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size = size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size=size)

    cmplx128_from_float32 = uniform(low=min_float32 / 2, high=max_float32 / 2, size = size) + uniform(low=min_float32 / 2, high=max_float32 / 2, size = size) * 1j
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = uniform(low=min_float64 / 2, high=max_float64 / 2, size = size) + uniform(low=min_float64 / 2, high=max_float64 / 2, size = size) * 1j

    epyccel_func = epyccel(get_imag, language=language)

    assert epyccel_func(bl) == get_imag(bl)
    assert epyccel_func(integer8) == get_imag(integer8)
    assert epyccel_func(integer16) == get_imag(integer16)
    assert epyccel_func(integer) == get_imag(integer)
    assert epyccel_func(integer32) == get_imag(integer32)
    assert epyccel_func(integer64) == get_imag(integer64)
    assert epyccel_func(fl) == get_imag(fl)
    assert epyccel_func(fl32) == get_imag(fl32)
    assert epyccel_func(fl64) == get_imag(fl64)
    assert epyccel_func(cmplx64) == get_imag(cmplx64)
    assert epyccel_func(cmplx128) == get_imag(cmplx128)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = [
            pytest.mark.skip(reason=("imag handles types in __new__ so it "
                "cannot be used in a translated interface in python")),
            pytest.mark.python]
        )
    )
)

def test_numpy_imag_array_like_2d(language):

    @template('T', ['bool[:,:]', 'int[:,:]', 'int8[:,:]', 'int16[:,:]', 'int32[:,:]', 'int64[:,:]', 'float[:,:]', 'float32[:,:]', 'float64[:,:]', 'complex64[:,:]', 'complex128[:,:]'])
    def get_imag(arr : 'T'):
        from numpy import imag, shape
        a = imag(arr)
        s = shape(a)
        return len(s), s[0], s[1], a[0,1], a[1,0]

    size = (2, 5)

    bl = randint(0, 2, size = size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size = size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size = size, dtype=np.int16)
    integer = randint(min_int, max_int, size = size, dtype=int)
    integer32 = randint(min_int32, max_int32, size = size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size = size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size = size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size=size)

    cmplx128_from_float32 = uniform(low=min_float32 / 2, high=max_float32 / 2, size = size) + uniform(low=min_float32 / 2, high=max_float32 / 2, size = size) * 1j
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = uniform(low=min_float64 / 2, high=max_float64 / 2, size = size) + uniform(low=min_float64 / 2, high=max_float64 / 2, size = size) * 1j

    epyccel_func = epyccel(get_imag, language=language)

    assert epyccel_func(bl) == get_imag(bl)
    assert epyccel_func(integer8) == get_imag(integer8)
    assert epyccel_func(integer16) == get_imag(integer16)
    assert epyccel_func(integer) == get_imag(integer)
    assert epyccel_func(integer32) == get_imag(integer32)
    assert epyccel_func(integer64) == get_imag(integer64)
    assert epyccel_func(fl) == get_imag(fl)
    assert epyccel_func(fl32) == get_imag(fl32)
    assert epyccel_func(fl64) == get_imag(fl64)
    assert epyccel_func(cmplx64) == get_imag(cmplx64)
    assert epyccel_func(cmplx128) == get_imag(cmplx128)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = [
            pytest.mark.skip(reason=("mod has special treatment for bool so it "
                "cannot be used in a translated interface in python")),
            pytest.mark.python]
        )
    )
)

# Not all the arguments supported

def test_numpy_mod_scalar(language):

    @template('T', ['bool', 'int', 'int8', 'int16', 'int32', 'int64', 'float', 'float32', 'float64'])
    def get_mod(a : 'T'):
        from numpy import mod
        b = mod(a, a)
        return b

    epyccel_func = epyccel(get_mod, language=language)

    f_bl_true_output = epyccel_func(True)
    test_bool_true_output = get_mod(True)

    assert f_bl_true_output == test_bool_true_output
    assert matching_types(f_bl_true_output, test_bool_true_output)

    def test_int(min_int, max_int, dtype):
        integer = dtype(randint(min_int, max_int, dtype=dtype) or 1)

        f_integer_output = epyccel_func(integer)
        test_int_output  = get_mod(integer)

        assert f_integer_output == test_int_output
        assert matching_types(f_integer_output, test_int_output)

    test_int(min_int8 , max_int8 , np.int8)
    test_int(min_int16, max_int16, np.int16)
    test_int(min_int  , max_int  , int)
    test_int(min_int32, max_int32, np.int32)
    test_int(min_int64, max_int64, np.int64)

    fl = uniform(min_float / 2, max_float / 2)
    fl32 = uniform(min_float32 / 2, max_float32 / 2)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2)

    f_fl_output = epyccel_func(fl)
    test_float_output = get_mod(fl)

    assert f_fl_output == test_float_output
    assert matching_types(f_fl_output, test_float_output)

    f_fl32_output = epyccel_func(fl32)
    test_float32_output = get_mod(fl32)

    assert f_fl32_output == test_float32_output
    assert matching_types(f_fl32_output, test_float32_output)

    f_fl64_output = epyccel_func(fl64)
    test_float64_output = get_mod(fl64)

    assert f_fl64_output == test_float64_output
    assert matching_types(f_fl64_output, test_float64_output)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = [
            pytest.mark.skip(reason=("mod has special treatment for bool so it "
                "cannot be used in a translated interface in python")),
            pytest.mark.python]
        )
    )
)

def test_numpy_mod_array_like_1d(language):

    @template('T', ['bool[:]', 'int[:]', 'int8[:]', 'int16[:]', 'int32[:]', 'int64[:]', 'float[:]', 'float32[:]', 'float64[:]'])
    def get_mod(arr : 'T'):
        from numpy import mod, shape
        a = mod(arr, arr)
        s = shape(a)
        return len(s), s[0], a[0]

    size = 5

    epyccel_func = epyccel(get_mod, language=language)

    bl = np.full(size, True, dtype= bool)
    assert epyccel_func(bl) == get_mod(bl)

    def test_int(min_int, max_int, dtype):
        integer = randint(min_int, max_int-1, size=size, dtype=dtype)
        integer = np.where(integer==0, 1, integer)
        assert epyccel_func(integer) == get_mod(integer)

    test_int(min_int8 , max_int8 , np.int8)
    test_int(min_int16, max_int16, np.int16)
    test_int(min_int  , max_int  , int)
    test_int(min_int32, max_int32, np.int32)
    test_int(min_int64, max_int64, np.int64)

    fl = uniform(min_float / 2, max_float / 2, size = size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size = size)

    assert epyccel_func(fl) == get_mod(fl)
    assert epyccel_func(fl32) == get_mod(fl32)
    assert epyccel_func(fl64) == get_mod(fl64)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = [
            pytest.mark.skip(reason=("mod has special treatment for bool so it "
                "cannot be used in a translated interface in python")),
            pytest.mark.python]
        )
    )
)

def test_numpy_mod_array_like_2d(language):

    @template('T', ['bool[:,:]', 'int[:,:]', 'int8[:,:]', 'int16[:,:]', 'int32[:,:]', 'int64[:,:]', 'float[:,:]', 'float32[:,:]', 'float64[:,:]'])
    def get_mod(arr : 'T'):
        from numpy import mod, shape
        a = mod(arr, arr)
        s = shape(a)
        return len(s), s[0], s[1], a[0,1], a[1,0]

    size = (2, 5)

    epyccel_func = epyccel(get_mod, language=language)

    bl = np.full(size, True, dtype= bool)
    assert epyccel_func(bl) == get_mod(bl)

    def test_int(min_int, max_int, dtype):
        integer = randint(min_int, max_int-1, size=size, dtype=dtype)
        integer = np.where(integer==0, 1, integer)
        assert epyccel_func(integer) == get_mod(integer)

    test_int(min_int8 , max_int8 , np.int8)
    test_int(min_int16, max_int16, np.int16)
    test_int(min_int  , max_int  , int)
    test_int(min_int32, max_int32, np.int32)
    test_int(min_int64, max_int64, np.int64)

    fl = uniform(min_float / 2, max_float / 2, size = size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size = size)

    assert epyccel_func(fl) == get_mod(fl)
    assert epyccel_func(fl32) == get_mod(fl32)
    assert epyccel_func(fl64) == get_mod(fl64)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Needs a C printer see https://github.com/pyccel/pyccel/issues/791"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = [
            pytest.mark.skip(reason=("prod handles types in __new__ so it "
                "cannot be used in a translated interface in python")),
            pytest.mark.python]
        )
    )
)

# Not all arguments are supported

def test_numpy_prod_scalar(language):

    @template('T', ['bool', 'int', 'int8', 'int16', 'int32', 'int64', 'float', 'float32', 'float64', 'complex64', 'complex128'])
    def get_prod(a : 'T'):
        from numpy import prod
        b = prod(a)
        return b

    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=int)
    integer32 = randint(min_int32, max_int32, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2)
    fl32 = uniform(min_float32 / 2, max_float32 / 2)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2)

    cmplx128_from_float32 = uniform(low=min_float32 / 2, high=max_float32 / 2) + uniform(low=min_float32 / 2, high=max_float32 / 2) * 1j
    cmplx128_from_float64 = uniform(low=min_float64 / 2, high=max_float64 / 2) + uniform(low=min_float64 / 2, high=max_float64 / 2) * 1j
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = np.complex128(cmplx128_from_float64)

    epyccel_func = epyccel(get_prod, language=language)

    f_bl_true_output = epyccel_func(True)
    test_bool_true_output = get_prod(True)

    f_bl_false_output = epyccel_func(False)
    test_bool_false_output = get_prod(False)

    assert f_bl_true_output == test_bool_true_output
    assert f_bl_false_output == test_bool_false_output

    assert matching_types(f_bl_true_output, test_bool_true_output)
    assert matching_types(f_bl_false_output, test_bool_false_output)

    f_integer_output = epyccel_func(integer)
    test_int_output  = get_prod(integer)

    assert f_integer_output == test_int_output
    assert matching_types(f_integer_output, test_int_output)

    f_integer8_output = epyccel_func(integer8)
    test_int8_output = get_prod(integer8)

    assert f_integer8_output == test_int8_output
    assert matching_types(f_integer8_output, test_int8_output)

    f_integer16_output = epyccel_func(integer16)
    test_int16_output = get_prod(integer16)

    assert f_integer16_output == test_int16_output
    assert matching_types(f_integer16_output, test_int16_output)

    f_integer32_output = epyccel_func(integer32)
    test_int32_output = get_prod(integer32)

    assert f_integer32_output == test_int32_output
    assert matching_types(f_integer32_output, test_int32_output)

    f_integer64_output = epyccel_func(integer64)
    test_int64_output = get_prod(integer64)

    assert f_integer64_output == test_int64_output
    assert matching_types(f_integer64_output, test_int64_output)

    f_fl_output = epyccel_func(fl)
    test_float_output = get_prod(fl)

    assert f_fl_output == test_float_output
    assert matching_types(f_fl_output, test_float_output)

    f_fl32_output = epyccel_func(fl32)
    test_float32_output = get_prod(fl32)

    assert f_fl32_output == test_float32_output
    assert matching_types(f_fl32_output, test_float32_output)

    f_fl64_output = epyccel_func(fl64)
    test_float64_output = get_prod(fl64)

    assert f_fl64_output == test_float64_output
    assert matching_types(f_fl64_output, test_float64_output)

    f_complex64_output = get_prod(cmplx64)
    test_complex64_output = get_prod(cmplx64)

    assert f_complex64_output == test_complex64_output
    assert matching_types(f_complex64_output, test_complex64_output)

    f_complex128_output = get_prod(cmplx128)
    test_complex128_output = get_prod(cmplx128)

    assert f_complex128_output == test_complex128_output
    assert matching_types(f_complex64_output, test_complex64_output)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Needs a C printer see https://github.com/pyccel/pyccel/issues/791"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = [
            pytest.mark.skip(reason=("prod handles types in __new__ so it "
                "cannot be used in a translated interface in python")),
            pytest.mark.python]
        )
    )
)

def test_numpy_prod_array_like_1d(language):

    @template('T', ['bool[:]', 'int[:]', 'int8[:]', 'int16[:]', 'int32[:]', 'int64[:]', 'float[:]', 'float32[:]', 'float64[:]', 'complex64[:]', 'complex128[:]'])
    def get_prod(arr : 'T'):
        from numpy import prod
        a = prod(arr)
        return a

    size = 5

    bl = randint(0, 2, size = size, dtype= bool)

    max_ok_int = int(max_int64 ** (1/5))

    integer8  = randint(max(min_int8, -max_ok_int), min(max_ok_int, max_int8), size = size, dtype=np.int8)
    integer16 = randint(max(min_int16, -max_ok_int), min(max_ok_int, max_int16), size = size, dtype=np.int16)
    integer   = randint(max(min_int, -max_ok_int), min(max_ok_int, max_int), size = size, dtype=int)
    integer32 = randint(max(min_int32, -max_ok_int), min(max_ok_int, max_int32), size = size, dtype=np.int32)
    integer64 = randint(-max_ok_int, max_ok_int, size = size, dtype=np.int64)

    fl = uniform(-((-min_float) ** (1/5)), max_float ** (1/5), size = size)

    min_ok_float32 = -((-min_float32) ** (1/5))
    min_ok_float64 = -((-min_float64) ** (1/5))
    max_ok_float32 = max_float32 ** (1/5)
    max_ok_float64 = max_float64 ** (1/5)

    fl32 = uniform(min_ok_float32, max_ok_float32, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_ok_float64, max_ok_float64, size=size)

    cmplx128_from_float32 = uniform(low=min_ok_float32/2,
                                    high=max_ok_float32/2, size = size) + \
                            uniform(low=min_ok_float32/2,
                                    high=max_ok_float32/2, size = size) * 1j
    cmplx128_from_float64 = uniform(low=min_ok_float64/2,
                                    high=max_ok_float64/2, size = size) + \
                            uniform(low=min_ok_float64/2,
                                    high=max_ok_float64/2, size = size) * 1j
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = np.complex128(cmplx128_from_float64)

    epyccel_func = epyccel(get_prod, language=language)

    assert epyccel_func(bl) == get_prod(bl)
    assert epyccel_func(integer8) == get_prod(integer8)
    assert epyccel_func(integer16) == get_prod(integer16)
    assert epyccel_func(integer) == get_prod(integer)
    assert epyccel_func(integer32) == get_prod(integer32)
    assert epyccel_func(integer64) == get_prod(integer64)
    assert np.isclose(epyccel_func(fl), get_prod(fl), rtol=RTOL, atol=ATOL)
    assert np.isclose(epyccel_func(fl32), get_prod(fl32), rtol=RTOL32, atol=ATOL32)
    assert np.isclose(epyccel_func(fl64), get_prod(fl64), rtol=RTOL, atol=ATOL)
    assert np.isclose(epyccel_func(cmplx64), get_prod(cmplx64), rtol=RTOL32, atol=ATOL32)
    assert np.isclose(epyccel_func(cmplx128), get_prod(cmplx128), rtol=RTOL, atol=ATOL)
    assert matching_types(epyccel_func(bl), get_prod(bl))
    assert matching_types(epyccel_func(integer8), get_prod(integer8))
    assert matching_types(epyccel_func(integer16), get_prod(integer16))
    assert matching_types(epyccel_func(integer), get_prod(integer))
    assert matching_types(epyccel_func(integer32), get_prod(integer32))
    assert matching_types(epyccel_func(integer64), get_prod(integer64))
    assert matching_types(epyccel_func(fl), get_prod(fl))
    assert matching_types(epyccel_func(fl32), get_prod(fl32))
    assert matching_types(epyccel_func(fl64), get_prod(fl64))
    assert matching_types(epyccel_func(cmplx64), get_prod(cmplx64))
    assert matching_types(epyccel_func(cmplx128), get_prod(cmplx128))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Needs a C printer see https://github.com/pyccel/pyccel/issues/791"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = [
            pytest.mark.skip(reason=("prod handles types in __new__ so it "
                "cannot be used in a translated interface in python")),
            pytest.mark.python]
        )
    )
)

def test_numpy_prod_array_like_2d(language):

    @template('T', ['bool[:,:]', 'int[:,:]', 'int8[:,:]', 'int16[:,:]', 'int32[:,:]', 'int64[:,:]', 'float[:,:]', 'float32[:,:]', 'float64[:,:]', 'complex64[:,:]', 'complex128[:,:]'])
    def get_prod(arr : 'T'):
        from numpy import prod
        a = prod(arr)
        return a

    size = (2, 5)

    bl = randint(0, 2, size = size, dtype= bool)

    max_ok_int = int(max_int64 ** (1/10))

    integer8  = randint(max(min_int8, -max_ok_int), min(max_ok_int, max_int8), size = size, dtype=np.int8)
    integer16 = randint(max(min_int16, -max_ok_int), min(max_ok_int, max_int16), size = size, dtype=np.int16)
    integer   = randint(max(min_int, -max_ok_int), min(max_ok_int, max_int), size = size, dtype=int)
    integer32 = randint(max(min_int32, -max_ok_int), min(max_ok_int, max_int32), size = size, dtype=np.int32)
    integer64 = randint(-max_ok_int, max_ok_int, size = size, dtype=np.int64)

    fl = uniform(-((-min_float) ** (1/10)), max_float ** (1/10), size = size)

    min_ok_float32 = -((-min_float32) ** (1/10))
    min_ok_float64 = -((-min_float64) ** (1/10))
    max_ok_float32 = max_float32 ** (1/10)
    max_ok_float64 = max_float64 ** (1/10)

    fl32 = uniform(min_ok_float32, max_ok_float32, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_ok_float64, max_ok_float64, size=size)

    cmplx128_from_float32 = uniform(low=min_ok_float32/2,
                                    high=max_ok_float32/2, size = size) + \
                            uniform(low=min_ok_float32/2,
                                    high=max_ok_float32/2, size = size) * 1j
    cmplx128_from_float64 = uniform(low=min_ok_float64/2,
                                    high=max_ok_float64/2, size = size) + \
                            uniform(low=min_ok_float64/2,
                                    high=max_ok_float64/2, size = size) * 1j
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = np.complex128(cmplx128_from_float64)

    epyccel_func = epyccel(get_prod, language=language)

    assert epyccel_func(bl) == get_prod(bl)
    assert epyccel_func(integer8) == get_prod(integer8)
    assert epyccel_func(integer16) == get_prod(integer16)
    assert epyccel_func(integer) == get_prod(integer)
    assert epyccel_func(integer32) == get_prod(integer32)
    assert epyccel_func(integer64) == get_prod(integer64)
    assert np.isclose(epyccel_func(fl), get_prod(fl), rtol=RTOL, atol=ATOL)
    assert np.isclose(epyccel_func(fl32), get_prod(fl32), rtol=RTOL32, atol=ATOL32)
    assert np.isclose(epyccel_func(fl64), get_prod(fl64), rtol=RTOL, atol=ATOL)
    assert np.isclose(epyccel_func(cmplx64), get_prod(cmplx64), rtol=RTOL32, atol=ATOL32)
    assert np.isclose(epyccel_func(cmplx128), get_prod(cmplx128), rtol=RTOL, atol=ATOL)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Still under maintenance, See #769"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = [pytest.mark.python])
    )
)

def test_numpy_norm_scalar(language):

    @template('T', ['bool', 'int', 'int8', 'int16', 'int32', 'int64', 'float', 'float32', 'float64', 'complex64', 'complex128'])
    def get_norm(a : 'T'):
        from numpy.linalg import norm
        b = norm(a)
        return b

    integer8 = randint(min_int8, max_int8, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=int)
    integer32 = randint(min_int32, max_int32, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, dtype=np.int64)

    fl = uniform(low=-(abs(min_float)**(1/2)), high=abs(max_float)**(1/2))
    fl32 = uniform(low=-(abs(min_float32)**(1/2)), high=abs(max_float32)**(1/2))
    fl32 = np.float32(fl32)
    fl64 = uniform(low=-(abs(min_float64)**(1/2)), high=abs(max_float64)**(1/2))

    cmplx128_from_float32 = uniform(low=-((abs(min_float32) / 2)**(1/2)), high=((abs(max_float32) / 2)**(1/2))) + \
                            uniform(low=-((abs(max_float32) / 2)**(1/2)), high=((abs(max_float32) / 2)**(1/2))) * 1j
    cmplx128_from_float64 = uniform(low=-((abs(min_float64) / 2)**(1/2)), high=((abs(max_float64) / 2)**(1/2))) + \
                            uniform(low=-((abs(max_float64) / 2)**(1/2)), high=((abs(max_float64) / 2)**(1/2))) * 1j
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = np.complex128(cmplx128_from_float64)

    epyccel_func = epyccel(get_norm, language=language)

    f_bl_true_output = epyccel_func(True)
    test_bool_true_output = get_norm(True)

    f_bl_false_output = epyccel_func(False)
    test_bool_false_output = get_norm(False)

    assert f_bl_true_output == test_bool_true_output
    assert f_bl_false_output == test_bool_false_output

    assert matching_types(f_bl_false_output, test_bool_false_output)
    assert matching_types(f_bl_true_output, test_bool_true_output)

    f_integer_output = epyccel_func(integer)
    test_int_output  = get_norm(integer)

    assert np.isclose(f_integer_output, test_int_output, rtol=RTOL, atol=ATOL)
    assert matching_types(f_integer_output, test_int_output)

    f_integer8_output = epyccel_func(integer8)
    test_int8_output = get_norm(integer8)

    assert np.isclose(f_integer8_output, test_int8_output, rtol=RTOL, atol=ATOL)
    assert matching_types(f_integer8_output, test_int8_output)

    f_integer16_output = epyccel_func(integer16)
    test_int16_output = get_norm(integer16)

    assert np.isclose(f_integer16_output, test_int16_output, rtol=RTOL, atol=ATOL)
    assert matching_types(f_integer16_output, test_int16_output)

    f_integer32_output = epyccel_func(integer32)
    test_int32_output = get_norm(integer32)

    assert np.isclose(f_integer32_output, test_int32_output, rtol=RTOL, atol=ATOL)
    assert matching_types(f_integer32_output, test_int32_output)

    f_integer64_output = epyccel_func(integer64)
    test_int64_output = get_norm(integer64)

    assert np.isclose(f_integer64_output, test_int64_output, rtol=RTOL, atol=ATOL)
    assert matching_types(f_integer64_output, test_int64_output)

    f_fl_output = epyccel_func(fl)
    test_float_output = get_norm(fl)

    assert np.isclose(f_fl_output, test_float_output, rtol=RTOL, atol=ATOL)
    assert matching_types(f_fl_output, test_float_output)

    f_fl32_output = epyccel_func(fl32)
    test_float32_output = get_norm(fl32)

    assert np.isclose(f_fl32_output, test_float32_output, rtol=RTOL32, atol=ATOL32)
    assert matching_types(f_fl32_output, test_float32_output)

    f_fl64_output = epyccel_func(fl64)
    test_float64_output = get_norm(fl64)

    assert np.isclose(f_fl64_output, test_float64_output, rtol=RTOL, atol=ATOL)
    assert matching_types(f_fl64_output, test_float64_output)

    f_complex64_output = epyccel_func(cmplx64)
    test_complex64_output = get_norm(cmplx64)

    assert np.isclose(f_complex64_output, test_complex64_output, rtol=RTOL32, atol=ATOL32)
    assert matching_types(f_complex64_output, test_complex64_output)

    f_complex128_output = epyccel_func(cmplx128)
    test_complex128_output = get_norm(cmplx128)

    assert np.isclose(f_complex128_output, test_complex128_output, rtol=RTOL, atol=ATOL)
    assert matching_types(f_complex128_output, test_complex128_output)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Still under maintenance, See #769"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = [pytest.mark.python])
    )
)

def test_numpy_norm_array_like_1d(language):

    @template('T', ['bool[:]', 'int[:]', 'int8[:]', 'int16[:]', 'int32[:]', 'int64[:]', 'float[:]', 'float32[:]', 'float64[:]', 'complex64[:]', 'complex128[:]'])
    def get_norm(arr : 'T'):
        from numpy.linalg import norm
        a = norm(arr)
        return a

    size = 5

    bl = randint(0, 2, size=size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=int)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(low=-((abs(min_float) / size)**(1/2)), high=(abs(max_float) / size)**(1/2), size=size)
    fl32 = uniform(low=-((abs(min_float32) / size)**(1/2)), high=(abs(max_float32) / size)**(1/2), size=size)
    fl32 = np.float32(fl32)
    fl64 = uniform(low=-((abs(min_float64) / size)**(1/2)), high=(abs(max_float64) / size)**(1/2), size=size)

    cmplx128_from_float32 = uniform(low=-((abs(min_float32) / (size * 2))**(1/2)),
                                    high=(abs(max_float32) / (size * 2))**(1/2), size=size) + \
                            uniform(low=-((abs(min_float32) / (size * 2))**(1/2)),
                                    high=(abs(max_float32) / (size * 2))**(1/2), size=size) * 1j
    cmplx128_from_float64 = uniform(low=-((abs(min_float64) / (size * 2))**(1/2)),
                                    high=(abs(max_float64) / (size * 2))**(1/2), size=size) + \
                            uniform(low=-((abs(min_float64) / (size * 2))**(1/2)),
                                    high=(abs(max_float64) / (size * 2))**(1/2), size=size) * 1j
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = np.complex128(cmplx128_from_float64)

    epyccel_func = epyccel(get_norm, language=language)

    assert np.isclose(epyccel_func(bl), get_norm(bl), rtol=RTOL, atol=ATOL)
    assert np.isclose(epyccel_func(integer8), get_norm(integer8), rtol=RTOL, atol=ATOL)
    assert np.isclose(epyccel_func(integer16), get_norm(integer16), rtol=RTOL, atol=ATOL)
    assert np.isclose(epyccel_func(integer), get_norm(integer), rtol=RTOL, atol=ATOL)
    assert np.isclose(epyccel_func(integer32), get_norm(integer32), rtol=RTOL, atol=ATOL)
    assert np.isclose(epyccel_func(integer64), get_norm(integer64), rtol=RTOL, atol=ATOL)
    assert np.isclose(epyccel_func(fl), get_norm(fl), rtol=RTOL, atol=ATOL)
    assert np.isclose(epyccel_func(fl32), get_norm(fl32), rtol=RTOL32, atol=ATOL32)
    assert np.isclose(epyccel_func(fl64), get_norm(fl64), rtol=RTOL, atol=ATOL)
    assert np.isclose(epyccel_func(cmplx64), get_norm(cmplx64), rtol=RTOL32, atol=ATOL32)
    assert np.isclose(epyccel_func(cmplx128), get_norm(cmplx128), rtol=RTOL, atol=ATOL)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Still under maintenance, See #769"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = [pytest.mark.python])
    )
)

def test_numpy_norm_array_like_2d(language):

    @template('T', ['bool[:,:]', 'int[:,:]', 'int8[:,:]', 'int16[:,:]', 'int32[:,:]', 'int64[:,:]', 'float[:,:]', 'float32[:,:]', 'float64[:,:]', 'complex64[:,:]', 'complex128[:,:]'])
    def get_norm(arr : 'T'):
        from numpy.linalg import norm
        from numpy import shape
        a = norm(arr)
        return a

    size = (2, 5)

    bl = randint(0, 2, size=size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=int)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(low=-((abs(min_float) / (size[0] * size[1]))**(1/2)), high=(abs(max_float) / (size[0] * size[1]))**(1/2), size=size)
    fl32 = uniform(low=-((abs(min_float32) / (size[0] * size[1]))**(1/2)), high=(abs(max_float32) / (size[0] * size[1]))**(1/2), size=size)
    fl32 = np.float32(fl32)
    fl64 = uniform(low=-((abs(min_float64) / (size[0] * size[1]))**(1/2)), high=(abs(max_float64) / (size[0] * size[1]))**(1/2), size=size)

    cmplx128_from_float32 = uniform(low=-((abs(min_float32) / (size[0] * size[1] * 2))**(1/2)),
                                    high=(abs(max_float32) / (size[0] * size[1] * 2))**(1/2), size=size) + \
                            uniform(low=-((abs(min_float32) / (size[0] * size[1] * 2))**(1/2)),
                                    high=(abs(max_float32) / (size[0] * size[1] * 2))**(1/2), size=size) * 1j
    cmplx128_from_float64 = uniform(low=-((abs(min_float64) / (size[0] * size[1] * 2))**(1/2)),
                                    high=(abs(max_float64) / (size[0] * size[1] * 2))**(1/2), size=size) + \
                            uniform(low=-((abs(min_float64) / (size[0] * size[1] * 2))**(1/2)),
                                    high=(abs(max_float64) / (size[0] * size[1] * 2))**(1/2), size=size) * 1j
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = np.complex128(cmplx128_from_float64)

    epyccel_func = epyccel(get_norm, language=language)

    assert np.allclose(epyccel_func(bl), get_norm(bl), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(integer8), get_norm(integer8), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(integer16), get_norm(integer16), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(integer), get_norm(integer), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(integer32), get_norm(integer32), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(integer64), get_norm(integer64), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(fl), get_norm(fl), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(fl32), get_norm(fl32), rtol=RTOL32, atol=ATOL32)
    assert np.allclose(epyccel_func(fl64), get_norm(fl64), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(cmplx64), get_norm(cmplx64), rtol=RTOL32, atol=ATOL32)
    assert np.allclose(epyccel_func(cmplx128), get_norm(cmplx128), rtol=RTOL, atol=ATOL)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Still under maintenance, See #769"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = [pytest.mark.python])
    )
)

def test_numpy_norm_array_like_2d_fortran_order(language):

    @template('T', ['bool[:,:](order=F)',
                    'int[:,:](order=F)',
                    'int8[:,:](order=F)',
                    'int16[:,:](order=F)',
                    'int32[:,:](order=F)',
                    'int64[:,:](order=F)',
                    'float[:,:](order=F)',
                    'float32[:,:](order=F)',
                    'float64[:,:](order=F)',
                    'complex64[:,:](order=F)',
                    'complex128[:,:](order=F)'])
    def get_norm(arr : 'T'):
        from numpy.linalg import norm
        from numpy import shape
        a = norm(arr, axis=0)
        b = norm(arr, axis=1)
        sa = shape(a)
        sb = shape(b)
        return len(sb), sb[0],len(sa), sa[0], a[0], b[0]

    size = (2, 5)

    bl = randint(0, 2, size=size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=int)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(low=-((abs(min_float) / (size[0] * size[1]))**(1/2)), high=(abs(max_float) / (size[0] * size[1]))**(1/2), size=size)
    fl32 = uniform(low=-((abs(min_float32) / (size[0] * size[1]))**(1/2)), high=(abs(max_float32) / (size[0] * size[1]))**(1/2), size=size)
    fl32 = np.float32(fl32)
    fl64 = uniform(low=-((abs(min_float64) / (size[0] * size[1]))**(1/2)), high=(abs(max_float64) / (size[0] * size[1]))**(1/2), size=size)

    cmplx128_from_float32 = uniform(low=-((abs(min_float32) / (size[0] * size[1] * 2))**(1/2)),
                                    high=(abs(max_float32) / (size[0] * size[1] * 2))**(1/2), size=size) + \
                            uniform(low=-((abs(min_float32) / (size[0] * size[1] * 2))**(1/2)),
                                    high=(abs(max_float32) / (size[0] * size[1] * 2))**(1/2), size=size) * 1j
    cmplx128_from_float64 = uniform(low=-((abs(min_float64) / (size[0] * size[1] * 2))**(1/2)),
                                    high=(abs(max_float64) / (size[0] * size[1] * 2))**(1/2), size=size) + \
                            uniform(low=-((abs(min_float64) / (size[0] * size[1] * 2))**(1/2)),
                                    high=(abs(max_float64) / (size[0] * size[1] * 2))**(1/2), size=size) * 1j
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = np.complex128(cmplx128_from_float64)

    epyccel_func = epyccel(get_norm, language=language)

    #re-ordering to Fortran order
    bl = np.ndarray(size, buffer=bl, order='F', dtype=bool)
    integer8 = np.ndarray(size, buffer=integer8, order='F', dtype=np.int8)
    integer16 = np.ndarray(size, buffer=integer16, order='F', dtype=np.int16)
    integer = np.ndarray(size, buffer=integer, order='F', dtype=int)
    integer32 = np.ndarray(size, buffer=integer32, order='F', dtype=np.int32)
    integer64 = np.ndarray(size, buffer=integer64, order='F', dtype=np.int64)
    fl = np.ndarray(size, buffer=fl, order='F', dtype=float)
    fl32 = np.ndarray(size, buffer=fl32, order='F', dtype=np.float32)
    fl64 = np.ndarray(size, buffer=fl64, order='F', dtype=np.float64)
    cmplx64 = np.ndarray(size, buffer=cmplx64, order='F', dtype=np.complex64)
    cmplx128 = np.ndarray(size, buffer=cmplx128, order='F', dtype=np.complex128)

    assert np.allclose(epyccel_func(bl), get_norm(bl), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(integer8), get_norm(integer8), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(integer16), get_norm(integer16), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(integer), get_norm(integer), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(integer32), get_norm(integer32), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(integer64), get_norm(integer64), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(fl), get_norm(fl), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(fl32), get_norm(fl32), rtol=RTOL32, atol=ATOL32)
    assert np.allclose(epyccel_func(fl64), get_norm(fl64), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(cmplx64), get_norm(cmplx64), rtol=RTOL32, atol=ATOL32)
    assert np.allclose(epyccel_func(cmplx128), get_norm(cmplx128), rtol=RTOL, atol=ATOL)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Still under maintenance, See #769"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = [pytest.mark.python])
    )
)

def test_numpy_norm_array_like_3d(language):

    @template('T', ['bool[:,:,:]',
                    'int[:,:,:]',
                    'int8[:,:,:]',
                    'int16[:,:,:]',
                    'int32[:,:,:]',
                    'int64[:,:,:]',
                    'float[:,:,:]',
                    'float32[:,:,:]',
                    'float64[:,:,:]',
                    'complex64[:,:,:]',
                    'complex128[:,:,:]'])
    def get_norm(arr : 'T'):
        from numpy.linalg import norm
        a = norm(arr)
        return a

    size = (2, 5, 5)

    bl = randint(0, 2, size=size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=int)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(low=-((abs(min_float) / (size[0] * size[1] * size[2]))**(1/2)), high=(abs(max_float) / (size[0] * size[1] * size[2]))**(1/2), size=size)
    fl32 = uniform(low=-((abs(min_float32) / (size[0] * size[1] * size[2]))**(1/2)), high=(abs(max_float32) / (size[0] * size[1] * size[2]))**(1/2), size=size)
    fl32 = np.float32(fl32)
    fl64 = uniform(low=-((abs(min_float64) / (size[0] * size[1] * size[2]))**(1/2)), high=(abs(max_float64) / (size[0] * size[1] * size[2]))**(1/2), size=size)

    cmplx128_from_float32 = uniform(low=-((abs(min_float32) / (size[0] * size[1] * size[2] * 2))**(1/2)),
                                    high=(abs(max_float32) / (size[0] * size[1] * size[2] * 2))**(1/2), size=size) + \
                            uniform(low=-((abs(min_float32) / (size[0] * size[1] * size[2] * 2))**(1/2)),
                                    high=(abs(max_float32) / (size[0] * size[1] * size[2] * 2))**(1/2), size=size) * 1j
    cmplx128_from_float64 = uniform(low=-((abs(min_float64) / (size[0] * size[1] * size[2] * 2))**(1/2)),
                                    high=(abs(max_float64) / (size[0] * size[1] * size[2] * 2))**(1/2), size=size) + \
                            uniform(low=-((abs(min_float64) / (size[0] * size[1] * size[2] * 2))**(1/2)),
                                    high=(abs(max_float64) / (size[0] * size[1] * size[2] * 2))**(1/2), size=size) * 1j
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = np.complex128(cmplx128_from_float64)

    epyccel_func = epyccel(get_norm, language=language)

    assert np.allclose(epyccel_func(bl), get_norm(bl), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(integer8), get_norm(integer8), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(integer16), get_norm(integer16), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(integer), get_norm(integer), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(integer32), get_norm(integer32), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(integer64), get_norm(integer64), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(fl), get_norm(fl), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(fl32), get_norm(fl32), rtol=RTOL32, atol=ATOL32)
    assert np.allclose(epyccel_func(fl64), get_norm(fl64), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(cmplx64), get_norm(cmplx64), rtol=RTOL32, atol=ATOL32)
    assert np.allclose(epyccel_func(cmplx128), get_norm(cmplx128), rtol=RTOL, atol=ATOL)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Still under maintenance, See #769"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = [pytest.mark.python])
    )
)

def test_numpy_norm_array_like_3d_fortran_order(language):

    @template('T', ['bool[:,:,:](order=F)', 'int[:,:,:](order=F)', 'int8[:,:,:](order=F)',
                    'int16[:,:,:](order=F)', 'int32[:,:,:](order=F)', 'int64[:,:,:](order=F)',
                    'float[:,:,:](order=F)', 'float32[:,:,:](order=F)', 'float64[:,:,:](order=F)',
                    'complex64[:,:,:](order=F)', 'complex128[:,:,:](order=F)'])
    def get_norm(arr : 'T'):
        from numpy.linalg import norm
        from numpy import shape
        a = norm(arr, axis=0)
        b = norm(arr, axis=1)
        c = norm(arr, axis=2)
        sa = shape(a)
        sb = shape(b)
        sc = shape(c)
        return len(sc), sc[0],len(sb), sb[0],len(sa), sa[0], a[0][0], b[0][0], c[0][0]

    size = (2, 5, 5)

    bl = randint(0, 2, size=size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=int)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(low=-((abs(min_float) / (size[0] * size[1] * size[2]))**(1/2)), high=(abs(max_float) / (size[0] * size[1] * size[2]))**(1/2), size=size)
    fl32 = uniform(low=-((abs(min_float32) / (size[0] * size[1] * size[2]))**(1/2)), high=(abs(max_float32) / (size[0] * size[1] * size[2]))**(1/2), size=size)
    fl32 = np.float32(fl32)
    fl64 = uniform(low=-((abs(min_float64) / (size[0] * size[1] * size[2]))**(1/2)), high=(abs(max_float64) / (size[0] * size[1] * size[2]))**(1/2), size=size)

    cmplx128_from_float32 = uniform(low=-((abs(min_float32) / (size[0] * size[1] * size[2] * 2))**(1/2)),
                                    high=(abs(max_float32) / (size[0] * size[1] * size[2] * 2))**(1/2), size=size) + \
                            uniform(low=-((abs(min_float32) / (size[0] * size[1] * size[2] * 2))**(1/2)),
                                    high=(abs(max_float32) / (size[0] * size[1] * size[2] * 2))**(1/2), size=size) * 1j
    cmplx128_from_float64 = uniform(low=-((abs(min_float64) / (size[0] * size[1] * size[2] * 2))**(1/2)),
                                    high=(abs(max_float64) / (size[0] * size[1] * size[2] * 2))**(1/2), size=size) + \
                            uniform(low=-((abs(min_float64) / (size[0] * size[1] * size[2] * 2))**(1/2)),
                                    high=(abs(max_float64) / (size[0] * size[1] * size[2] * 2))**(1/2), size=size) * 1j
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = np.complex128(cmplx128_from_float64)

    epyccel_func = epyccel(get_norm, language=language)

    #re-ordering to Fortran order
    bl = np.ndarray(size, buffer=bl, order='F', dtype=bool)
    integer8 = np.ndarray(size, buffer=integer8, order='F', dtype=np.int8)
    integer16 = np.ndarray(size, buffer=integer16, order='F', dtype=np.int16)
    integer = np.ndarray(size, buffer=integer, order='F', dtype=int)
    integer32 = np.ndarray(size, buffer=integer32, order='F', dtype=np.int32)
    integer64 = np.ndarray(size, buffer=integer64, order='F', dtype=np.int64)
    fl = np.ndarray(size, buffer=fl, order='F', dtype=float)
    fl32 = np.ndarray(size, buffer=fl32, order='F', dtype=np.float32)
    fl64 = np.ndarray(size, buffer=fl64, order='F', dtype=np.float64)
    cmplx64 = np.ndarray(size, buffer=cmplx64, order='F', dtype=np.complex64)
    cmplx128 = np.ndarray(size, buffer=cmplx128, order='F', dtype=np.complex128)

    assert np.allclose(epyccel_func(bl), get_norm(bl), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(integer8), get_norm(integer8), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(integer16), get_norm(integer16), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(integer), get_norm(integer), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(integer32), get_norm(integer32), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(integer64), get_norm(integer64), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(fl), get_norm(fl), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(fl32), get_norm(fl32), rtol=RTOL32, atol=ATOL32)
    assert np.allclose(epyccel_func(fl64), get_norm(fl64), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(cmplx64), get_norm(cmplx64), rtol=RTOL32, atol=ATOL32)
    assert np.allclose(epyccel_func(cmplx128), get_norm(cmplx128), rtol=RTOL, atol=ATOL)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Needs a C printer see https://github.com/pyccel/pyccel/issues/791"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = [
            pytest.mark.python])
    )
)

def test_numpy_matmul_array_like_1d(language):

    @template('T', ['bool[:]', 'int[:]', 'int8[:]', 'int16[:]', 'int32[:]', 'int64[:]', 'float[:]', 'float32[:]', 'float64[:]', 'complex64[:]', 'complex128[:]'])
    def get_matmul(arr : 'T'):
        from numpy import matmul
        a = matmul(arr, arr)
        return a

    size = 5

    bl = randint(0, 2, size=size, dtype= bool)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=int)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl = uniform(-((max_float / size)**(1/2)), (max_float / size)**(1/2), size = size)
    fl32 = uniform(-((max_float32 / size)**(1/2)), (max_float32 / size)**(1/2), size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(-((max_float64 / size)**(1/2)), (max_float64 / size)**(1/2), size = size)

    cmplx128_from_float32 = uniform(low=-((max_float32 / (size * 2))**(1/2)),
                                    high=(max_float32 / (size * 2))**(1/2), size=size) + \
                            uniform(low=-((max_float32 / (size * 2))**(1/2)),
                                    high=(max_float32 / (size * 2))**(1/2), size=size) * 1j
    cmplx128_from_float64 = uniform(low=-((max_float64 / (size * 2))**(1/2)),
                                    high=(max_float64 / (size * 2))**(1/2), size=size) + \
                            uniform(low=-((max_float64 / (size * 2))**(1/2)),
                                    high=(max_float64 / (size * 2))**(1/2), size=size) * 1j
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = np.complex128(cmplx128_from_float64)

    epyccel_func = epyccel(get_matmul, language=language)

    assert epyccel_func(bl) == get_matmul(bl)
    assert epyccel_func(integer8) == get_matmul(integer8)
    assert epyccel_func(integer16) == get_matmul(integer16)
    assert epyccel_func(integer) == get_matmul(integer)
    assert epyccel_func(integer32) == get_matmul(integer32)
    assert epyccel_func(integer64) == get_matmul(integer64)
    assert isclose(epyccel_func(fl),get_matmul(fl), rtol=RTOL, atol=ATOL)
    assert isclose(epyccel_func(fl32),get_matmul(fl32), rtol=RTOL32, atol=ATOL32)
    assert isclose(epyccel_func(fl64),get_matmul(fl64), rtol=RTOL, atol=ATOL)
    assert isclose(epyccel_func(cmplx64),get_matmul(cmplx64), rtol=RTOL32, atol=ATOL32)
    assert isclose(epyccel_func(cmplx128),get_matmul(cmplx128), rtol=RTOL, atol=ATOL)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [pytest.mark.fortran]),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Needs a C printer see https://github.com/pyccel/pyccel/issues/791"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = [
            pytest.mark.python],
        )
    )
)

def test_numpy_matmul_array_like_2x2d(language):

    @template('T', ['bool[:,:]', 'int[:,:]', 'int8[:,:]', 'int16[:,:]', 'int32[:,:]', 'int64[:,:]', 'float[:,:]', 'float32[:,:]', 'float64[:,:]', 'complex64[:,:]', 'complex128[:,:]'])
    def get_matmul(arr : 'T'):
        from numpy import matmul, shape
        a = matmul(arr, arr)
        s = shape(a)
        return len(s) , s[0] , s[1] , a[0,1] , a[1,0]

    size = (2, 2)

    bl = randint(0, 2, size=size, dtype= bool)

    def calculate_max_values(min_for_type, max_for_type):
        cast = type(min_for_type)
        min_test = -np.sqrt(abs(min_for_type) / size[0])
        max_test = np.sqrt(abs(max_for_type) / size[0])
        print(min_test, max_test, cast(min_test), cast(max_test))
        return cast(min_test), cast(max_test)

    integer8 = randint(*calculate_max_values(min_int8, max_int8), size=size, dtype=np.int8)
    integer16 = randint(*calculate_max_values(min_int16, max_int16), size=size, dtype=np.int16)
    integer = randint(*calculate_max_values(min_int, max_int), size=size, dtype=int)
    integer32 = randint(*calculate_max_values(min_int32, max_int32), size=size, dtype=np.int32)
    integer64 = randint(*calculate_max_values(min_int64, max_int64), size=size, dtype=np.int64)

    fl = uniform(*calculate_max_values(min_float, max_float), size = size)
    fl32 = uniform(*calculate_max_values(min_float32, max_float32), size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(*calculate_max_values(min_float64, max_float64), size = size)

    cmplx128_from_float32 = uniform(*calculate_max_values(min_int, max_int), size=size) + uniform(*calculate_max_values(min_int, max_int), size=size) * 1j
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)
    cmplx128 = uniform(*calculate_max_values(min_int, max_int), size=size) + uniform(*calculate_max_values(min_int, max_int), size=size) * 1j

    integer8  = np.full(size, calculate_max_values(min_int8, max_int8)[1])
    integer16 = np.full(size, calculate_max_values(min_int16, max_int16)[1])
    integer   = np.full(size, calculate_max_values(min_int, max_int)[1])
    integer32 = np.full(size, calculate_max_values(min_int32, max_int32)[1])
    integer64 = np.full(size, calculate_max_values(min_int64, max_int64)[1])

    fl   = np.full(size, calculate_max_values(min_float, max_float)[1])
    fl32 = np.full(size, calculate_max_values(min_float32, max_float32)[1])
    fl64 = np.full(size, calculate_max_values(min_float64, max_float64)[1])

    cmplx64  = np.full(size, np.complex64(integer + integer * 1j))
    cmplx128 = np.full(size, integer + integer * 1j)

    epyccel_func = epyccel(get_matmul, language=language)

    assert np.allclose(epyccel_func(bl), get_matmul(bl), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(integer8), get_matmul(integer8), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(integer16), get_matmul(integer16), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(integer), get_matmul(integer), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(integer32), get_matmul(integer32), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(integer64), get_matmul(integer64), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(fl), get_matmul(fl), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(fl32), get_matmul(fl32), rtol=RTOL32, atol=ATOL32)
    assert np.allclose(epyccel_func(fl64), get_matmul(fl64), rtol=RTOL, atol=ATOL)
    assert np.allclose(epyccel_func(cmplx64), get_matmul(cmplx64), rtol=RTOL32, atol=ATOL32)
    assert np.allclose(epyccel_func(cmplx128), get_matmul(cmplx128), rtol=RTOL, atol=ATOL)

def test_numpy_where_array_like_1d_with_condition(language):

    @template('T', ['bool[:]', 'int[:]', 'int8[:]', 'int16[:]', 'int32[:]', 'int64[:]', 'float[:]', 'float32[:]', 'float64[:]'])
    def get_chosen_elements(arr : 'T'):
        from numpy import where, shape
        a = where(arr > 5, arr, arr * 2)
        s = shape(a)
        return len(s), s[0], a[1], a[0]

    size = 5

    bl = randint(0, 2, size=size, dtype= bool)

    integer8  = randint(min_int8//2,  max_int8//2, size=size, dtype=np.int8)
    integer16 = randint(min_int16//2, max_int16//2, size=size, dtype=np.int16)
    integer   = randint(min_int//2,   max_int//2, size=size, dtype=int)
    integer32 = randint(min_int32//2, max_int32//2, size=size, dtype=np.int32)
    integer64 = randint(min_int64//2, max_int64//2, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size = size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size = size)

    epyccel_func = epyccel(get_chosen_elements, language=language)

    assert epyccel_func(bl) == get_chosen_elements(bl)
    assert epyccel_func(integer8) == get_chosen_elements(integer8)
    assert epyccel_func(integer16) == get_chosen_elements(integer16)
    assert epyccel_func(integer) == get_chosen_elements(integer)
    assert epyccel_func(integer32) == get_chosen_elements(integer32)
    assert epyccel_func(integer64) == get_chosen_elements(integer64)
    assert epyccel_func(fl) == get_chosen_elements(fl)
    assert epyccel_func(fl32) == get_chosen_elements(fl32)
    assert epyccel_func(fl64) == get_chosen_elements(fl64)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="nonzero not implemented"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_numpy_where_array_like_1d_1_arg(language):

    @template('T', ['int[:]', 'int8[:]', 'int16[:]', 'int32[:]', 'int64[:]', 'float[:]', 'float32[:]', 'float64[:]'])
    def get_chosen_elements(arr : 'T'):
        from numpy import where, shape
        a = where(arr > 5)
        s = shape(a)
        return len(s), s[1], a[0][1], a[0][0]

    size = 5

    # Arrays must have at least 2 elements larger than 5 to avoid IndexError
    integer8  = np.array([6,1,8,2,3], dtype = np.int8)
    integer16 = np.array([6,1,8,2,3], dtype = np.int16)
    integer   = np.array([6,1,8,2,3], dtype = int)
    integer32 = np.array([6,1,8,2,3], dtype = np.int32)
    integer64 = np.array([6,1,8,2,3], dtype = np.int64)

    fl   = np.array([6,22,1,8,2,3], dtype = float)
    fl32 = np.array([6,22,1,8,2,3], dtype = np.float32)
    fl64 = np.array([6,22,1,8,2,3], dtype = np.float64)

    epyccel_func = epyccel(get_chosen_elements, language=language)

    assert epyccel_func(integer8) == get_chosen_elements(integer8)
    assert epyccel_func(integer16) == get_chosen_elements(integer16)
    assert epyccel_func(integer) == get_chosen_elements(integer)
    assert epyccel_func(integer32) == get_chosen_elements(integer32)
    assert epyccel_func(integer64) == get_chosen_elements(integer64)
    assert epyccel_func(fl) == get_chosen_elements(fl)
    assert epyccel_func(fl32) == get_chosen_elements(fl32)
    assert epyccel_func(fl64) == get_chosen_elements(fl64)

def test_numpy_where_array_like_2d_with_condition(language):

    @template('T', ['bool[:,:]', 'int[:,:]', 'int8[:,:]', 'int16[:,:]', 'int32[:,:]', 'int64[:,:]', 'float[:,:]', 'float32[:,:]', 'float64[:,:]'])
    def get_chosen_elements(arr : 'T'):
        from numpy import where, shape
        a = where(arr < 0, arr, arr + 1)
        s = shape(a)
        return len(s), s[0], a[0,0], a[0,1], a[1,0], a[1,1]

    size = (2, 5)

    bl = randint(0, 2, size=size, dtype= bool)

    integer8 = randint(min_int8, max_int8-1, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16-1, size=size, dtype=np.int16)
    integer = randint(min_int, max_int-1, size=size, dtype=int)
    integer32 = randint(min_int32, max_int32-1, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64-1, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size = size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size = size)

    epyccel_func = epyccel(get_chosen_elements, language=language)

    assert epyccel_func(bl) == get_chosen_elements(bl)
    assert epyccel_func(integer8) == get_chosen_elements(integer8)
    assert epyccel_func(integer16) == get_chosen_elements(integer16)
    assert epyccel_func(integer) == get_chosen_elements(integer)
    assert epyccel_func(integer32) == get_chosen_elements(integer32)
    assert epyccel_func(integer64) == get_chosen_elements(integer64)
    assert epyccel_func(fl) == get_chosen_elements(fl)
    assert epyccel_func(fl32) == get_chosen_elements(fl32)
    assert epyccel_func(fl64) == get_chosen_elements(fl64)

def test_numpy_where_complex(language):
    @types('complex64[:]', 'complex64[:]', 'bool[:]')
    @types('complex128[:]', 'complex128[:]', 'bool[:]')
    def where_wrapper(arr1, arr2, cond):
        from numpy import where, shape
        a = where(cond, arr1, arr2)
        s = shape(a)
        return len(s), s[0], a[1], a[0]

    size = 7

    cond = randint(0, 1, size=size, dtype= bool)

    cmplx128_from_float32_1 = uniform(low=min_float32 / 2, high=max_float32 / 2, size=size) + uniform(low=min_float32 / 2, high=max_float32 / 2, size=size) * 1j
    cmplx128_from_float32_2 = uniform(low=min_float32 / 2, high=max_float32 / 2, size=size) + uniform(low=min_float32 / 2, high=max_float32 / 2, size=size) * 1j
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64_1 = np.complex64(cmplx128_from_float32_1)
    cmplx64_2 = np.complex64(cmplx128_from_float32_2)
    cmplx128_1 = uniform(low=min_float64 / 2, high=max_float64 / 2, size=size) + uniform(low=min_float64 / 2, high=max_float64 / 2, size=size) * 1j
    cmplx128_2 = uniform(low=min_float64 / 2, high=max_float64 / 2, size=size) + uniform(low=min_float64 / 2, high=max_float64 / 2, size=size) * 1j

    epyccel_func = epyccel(where_wrapper, language=language)

    assert epyccel_func(cmplx64_1, cmplx64_2, cond)  == where_wrapper(cmplx64_1, cmplx64_2, cond)
    assert epyccel_func(cmplx128_1, cmplx128_2, cond) == where_wrapper(cmplx128_1, cmplx128_2, cond)

def test_where_combined_types(language):
    @types('bool[:]','int32[:]','int64[:]')
    @types('bool[:]','int32[:]','float32[:]')
    @types('bool[:]','float64[:]','int64[:]')
    @types('bool[:]','complex128[:]','int64[:]')
    def where_wrapper(cond, arr1, arr2):
        from numpy import where, shape
        a = where(cond, arr1, arr2)
        s = shape(a)
        return len(s), s[0], a[1], a[0]

    size = 6

    cond = randint(0, 1, size=size, dtype= bool)

    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    float32 = uniform(min_float32, max_float32, size = size)
    float32 = np.float32(float32)
    float64 = uniform(min_float64/2, max_float64/2, size = size)

    complex128 = uniform(low=min_float64 / 2, high=max_float64 / 2, size=size) + uniform(low=min_float64 / 2, high=max_float64 / 2, size=size) * 1j

    epyccel_func = epyccel(where_wrapper, language=language)

    res_pyc = epyccel_func (cond, integer32, integer64)
    res_pyt = where_wrapper(cond, integer32, integer64)
    assert res_pyc == res_pyt
    assert matching_types(res_pyc, res_pyt)
    res_pyc = epyccel_func (cond, integer32, float32)
    res_pyt = where_wrapper(cond, integer32, float32)
    assert res_pyc == res_pyt
    assert matching_types(res_pyc, res_pyt)
    res_pyc = epyccel_func (cond, float64, integer64)
    res_pyt = where_wrapper(cond, float64, integer64)
    assert res_pyc == res_pyt
    assert matching_types(res_pyc, res_pyt)
    res_pyc = epyccel_func (cond, complex128, integer64)
    res_pyt = where_wrapper(cond, complex128, integer64)
    assert res_pyc == res_pyt
    assert matching_types(res_pyc, res_pyt)

def test_numpy_linspace_scalar(language):
    from numpy import linspace

    @types('int', 'int', 'int')
    @types('int8', 'int', 'int')
    @types('int16', 'int', 'int')
    @types('int32', 'int', 'int')
    @types('int64', 'int', 'int')
    @types('float', 'int', 'int')
    @types('float32', 'int', 'int')
    @types('float64', 'int', 'int')
    def get_linspace(start, steps, num):
        from numpy import linspace
        stop = start + steps
        b = linspace(start, stop, num)
        x = 0.0
        for bi in b:
            x += bi
        return x

    def test_linspace(start : 'complex64', end : 'complex64'):
        from numpy import linspace
        x = linspace(start, end, 5)
        return x[0], x[1], x[2], x[3], x[4]

    def test_linspace2(start : 'complex128', end : 'complex128'):
        from numpy import linspace
        x = linspace(start, end, 5)
        return x[0], x[1], x[2], x[3], x[4]

    def test_linspace_type(start : 'int', end : 'int', result : 'int64[:]'):
        from numpy import linspace
        import numpy as np
        x = linspace(start + 4, end, 15, dtype=np.int64)
        ret = 1
        for i in range(len(x)):
            if x[i] != result[i]:
                ret = 0
        return ret, x[int(len(x) / 2)]

    def test_linspace_type2(start : 'int', end : 'int', result : 'complex128[:]'):
        from numpy import linspace
        x = linspace(start, end * 2, 15, dtype='complex128')
        for i in range(len(x)):
            result[i] = x[i]

    integer8 = randint(min_int8, max_int8 // 2, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, dtype=np.int16)
    integer = randint(min_int, max_int, dtype=int)
    integer32 = randint(min_int32, max_int32, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, dtype=np.int64)

    fl = uniform(min_float / 200, max_float / 200)
    fl32 = uniform(min_float32 / 200, max_float32 / 200)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 200, max_float64 / 200)

    epyccel_func = epyccel(get_linspace, language=language)
    epyccel_func_type = epyccel(test_linspace_type, language=language)
    epyccel_func_type2 = epyccel(test_linspace_type2, language=language)

    x = linspace(0 + 4, 10, 15, dtype=np.int64)
    ret, ele = epyccel_func_type(0, 10, x)
    assert (ret == 1)
    assert (ele.dtype == np.int64)
    x = linspace(0, 10 * 2, 15, dtype='complex128')
    out = np.empty_like(x)
    epyccel_func_type2(0, 10, out)
    assert (np.allclose(x, out))
    arr = np.zeros
    x = randint(1, 60)
    assert np.isclose(epyccel_func(integer8, x, 30), get_linspace(integer8, x, 30), rtol=RTOL, atol=ATOL)
    assert matching_types(epyccel_func(integer8, x, 100), get_linspace(integer8, x, 100))
    x = randint(100, 200)
    assert np.isclose(epyccel_func(integer, x, 30), get_linspace(integer, x, 30), rtol=RTOL, atol=ATOL)
    assert matching_types(epyccel_func(integer, x, 100), get_linspace(integer, x, 100))
    x = randint(100, 200)
    assert np.isclose(epyccel_func(integer16, x, 30), get_linspace(integer16, x, 30), rtol=RTOL, atol=ATOL)
    assert matching_types(epyccel_func(integer16, x, 100), get_linspace(integer16, x, 100))
    x = randint(100, 200)
    assert np.isclose(epyccel_func(integer32, x, 30), get_linspace(integer32, x, 30), rtol=RTOL, atol=ATOL)
    assert matching_types(epyccel_func(integer32, x, 100), get_linspace(integer32, x, 100))
    x = randint(100, 200)
    assert np.isclose(epyccel_func(integer64, x, 200), get_linspace(integer64, x, 200), rtol=RTOL, atol=ATOL)
    assert matching_types(epyccel_func(integer64, x, 100), get_linspace(integer64, x, 100))
    x = randint(100, 200)
    assert np.isclose(epyccel_func(fl, x, 100), get_linspace(fl, x, 100), rtol=RTOL, atol=ATOL)
    assert matching_types(epyccel_func(fl, x, 100), get_linspace(fl, x, 100))
    x = randint(100, 200)
    assert np.isclose(epyccel_func(fl32, x, 200), get_linspace(fl32, x, 200), rtol=RTOL, atol=ATOL)
    assert matching_types(epyccel_func(fl32, x, 100), get_linspace(fl32, x, 100))
    x = randint(100, 200)
    assert np.isclose(epyccel_func(fl64, x, 200), get_linspace(fl64, x, 200), rtol=RTOL, atol=ATOL)
    assert matching_types(epyccel_func(fl64, x, 100), get_linspace(fl64, x, 100))

    epyccel_func1 = epyccel(test_linspace, language=language)
    epyccel_func2 = epyccel(test_linspace2, language=language)
    assert (epyccel_func1(np.complex64(3+6j), np.complex64(5+1j)) == test_linspace(np.complex64(3+6j), np.complex64(5+1j)))
    assert (epyccel_func1(np.complex64(-3+6j), np.complex64(5-1j)) == test_linspace(np.complex64(-3+6j), np.complex64(5-1j)))
    assert (epyccel_func2(np.complex128(3+6j), np.complex128(5+1j)) == test_linspace(np.complex128(3+6j), np.complex128(5+1j)))
    assert (epyccel_func2(np.complex128(-3+6j), np.complex128(5-1j)) == test_linspace(np.complex128(-3+6j), np.complex128(5-1j)))

    res_pyc = epyccel_func2(np.complex128(3+6j), np.complex128(5+1j))
    res_pyt = test_linspace(np.complex128(3+6j), np.complex128(5+1j))
    for pyc, pyt in zip(res_pyc,res_pyt):
        assert matching_types(pyc, pyt)

def test_numpy_linspace_array_like_1d(language):
    from numpy import linspace

    @types('int[:]', 'int', 'float[:,:]', 'bool')
    @types('int8[:]', 'int', 'float[:,:]', 'bool')
    @types('int16[:]', 'int', 'float[:,:]', 'bool')
    @types('int32[:]', 'int', 'float[:,:]', 'bool')
    @types('float[:]', 'int', 'float[:,:]', 'bool')
    @types('float32[:]', 'int', 'float32[:,:]', 'bool')
    @types('float64[:]', 'int', 'float64[:,:]', 'bool')
    def test_linspace(start, stop, out, endpoint):
        from numpy import linspace
        numberOfSamplesToGenerate = 7
        a = linspace(start, stop, numberOfSamplesToGenerate, endpoint=endpoint)
        for i in range(len(out)):
            for j in range(len(out[i])):
                out[i][j] = a[i][j]

    def test_linspace2(start : 'complex128[:]', stop : 'int', out : 'complex128[:,:]', endpoint : 'bool'):
        from numpy import linspace
        numberOfSamplesToGenerate = 7
        a = linspace(start, stop, numberOfSamplesToGenerate, endpoint=endpoint)
        for i in range(len(out)):
            for j in range(len(out[i])):
                out[i][j] = a[i][j]

    @types('int[:]', 'int', 'int32[:,:]', 'bool')
    @types('float64[:]', 'int', 'int32[:,:]', 'bool')
    def test_linspace_dtype(start, stop, out, endpoint):
        from numpy import linspace
        import numpy as np
        numberOfSamplesToGenerate = 7
        a = linspace(start, stop, numberOfSamplesToGenerate, endpoint=(endpoint == True), dtype=np.int32)
        for i in range(len(out)):
            for j in range(len(out[i])):
                out[i][j] = a[i][j]

    size = 5
    integer8 = randint(min_int8 / 2, max_int8 / 2, size=size, dtype=np.int8)
    integer16 = randint(min_int16 / 2, max_int16 / 2, size=size, dtype=np.int16)
    integer = randint(-10000, 10000, size=size, dtype=int)
    integer32 = randint(min_int32 / 2, max_int32 / 2, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)

    fl32 = np.array([1.5, 2.2, 3.3, 4.4, 5.5], dtype=np.float32)

    epyccel_func = epyccel(test_linspace, language=language)
    epyccel_func2 = epyccel(test_linspace2, language=language)

    epyccel_func_dtype = epyccel(test_linspace_dtype, language=language)

    arr = linspace(integer, 5, 7)
    out = np.empty_like(arr)
    epyccel_func(integer, 5, out, True)
    assert np.allclose(arr, out)
    arr = linspace(integer, 5, 7, endpoint=False)
    out = np.empty_like(arr)
    epyccel_func(integer, 5, out, False)
    assert np.allclose(arr, out)

    arr = linspace(integer8, 5, 7)
    out = np.empty_like(arr)
    epyccel_func(integer8, 5, out, True)
    assert np.allclose(arr, out)
    arr = linspace(integer8, 5, 7, endpoint=False)
    out = np.empty_like(arr)
    epyccel_func(integer8, 5, out, False)
    assert np.allclose(arr, out)

    arr = linspace(integer16, 5, 7)
    out = np.empty_like(arr)
    epyccel_func(integer16, 5, out, True)
    assert np.allclose(arr, out)
    arr = linspace(integer16, 5, 7, endpoint=False)
    out = np.empty_like(arr)
    epyccel_func(integer16, 5, out, False)
    assert np.allclose(arr, out)

    arr = linspace(integer32, 5, 7)
    out = np.empty_like(arr)
    epyccel_func(integer32, 5, out, True)
    assert np.allclose(arr, out)
    arr = linspace(integer32, 5, 7, endpoint=False)
    out = np.empty_like(arr)
    epyccel_func(integer32, 5, out, False)
    assert np.allclose(arr, out)

    if sys.platform != 'win32':
        arr = linspace(integer64, 5, 7)
        out = np.empty_like(arr)
        epyccel_func(integer64, 5, out, True)
        assert np.allclose(arr, out)
        arr = linspace(integer64, 5, 7, endpoint=False)
        out = np.empty_like(arr)
        epyccel_func(integer64, 5, out, False)
        assert np.allclose(arr, out)


    arr = linspace(fl32, 5, 7)
    out = np.empty_like(arr)
    epyccel_func(fl32, 5, out, True)
    assert np.allclose(arr, out)
    arr = linspace(fl32, 5, 7, endpoint=False)
    out = np.empty_like(arr)
    epyccel_func(fl32, 5, out, False)
    assert np.allclose(arr, out)

    rng = np.random.default_rng()
    fl64 = rng.random((5,), dtype=np.float64)
    arr = linspace(fl64, 2, 7)
    out = np.empty_like(arr)
    epyccel_func(fl64, 2, out, True)
    assert np.allclose(arr, out)
    arr = linspace(fl64, 5, 7, endpoint=False)
    out = np.empty_like(arr)
    epyccel_func(fl64, 5, out, False)
    assert np.allclose(arr, out)

    cmplx = (np.random.random(5)*75) + (np.random.random(5)*50) * 1j
    arr = linspace(cmplx, 0, 7)
    out = np.empty_like(arr)
    epyccel_func2(cmplx, 0, out, True)
    assert np.allclose(arr, out)
    arr = linspace(cmplx, 0, 7, endpoint=False)
    out = np.empty_like(arr)
    epyccel_func2(cmplx, 0, out, False)
    assert np.allclose(arr, out)

def test_numpy_linspace_array_like_2d(language):
    from numpy import linspace

    @types('int[:,:]', 'int', 'float[:,:,:]', 'bool')
    @types('int8[:,:]', 'int', 'float[:,:,:]', 'bool')
    @types('int16[:,:]', 'int', 'float[:,:,:]', 'bool')
    @types('int32[:,:]', 'int', 'float[:,:,:]', 'bool')
    @types('float[:,:]', 'int', 'float[:,:,:]', 'bool')
    @types('float32[:,:]', 'int', 'float32[:,:,:]', 'bool')
    @types('float64[:,:]', 'int', 'float64[:,:,:]', 'bool')
    def test_linspace(start, stop, out, endpoint):
        from numpy import linspace
        numberOfSamplesToGenerate = 7
        a = linspace(start, stop, numberOfSamplesToGenerate, endpoint=endpoint)
        for i in range(len(out)):
            for j in range(len(out[i])):
                for k in range(len(out[i][j])):
                    out[i][j][k] = a[i][j][k]

    def test_linspace3(start : 'complex128[:,:]', stop : 'int', out : 'complex128[:,:,:]', endpoint : 'bool'):
        from numpy import linspace
        numberOfSamplesToGenerate = 7
        a = linspace(start, stop, numberOfSamplesToGenerate, endpoint=endpoint)
        for i in range(len(out)):
            for j in range(len(out[i])):
                for k in range(len(out[i][j])):
                    out[i][j][k] = a[i][j][k]

    def test_linspace2(start : 'int[:,:]', stop : 'int[:,:]', out : 'float[:,:,:]', endpoint : 'bool'):
        from numpy import linspace
        numberOfSamplesToGenerate = 7
        a = linspace(start, stop, numberOfSamplesToGenerate, endpoint=endpoint)
        for i in range(len(out)):
            for j in range(len(out[i])):
                for k in range(len(out[i][j])):
                    out[i][j][k] = a[i][j][k]

    def test_linspace4(start : 'complex128[:,:]', stop : 'complex128[:,:]', out : 'complex128[:,:,:]', endpoint : 'bool'):
        from numpy import linspace
        numberOfSamplesToGenerate = 7
        a = linspace(start, stop, numberOfSamplesToGenerate, endpoint=endpoint)
        for i in range(len(out)):
            for j in range(len(out[i])):
                for k in range(len(out[i][j])):
                    out[i][j][k] = a[i][j][k]

    size = (2, 5)

    integer8 = randint(min_int8, max_int8, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16, size=size, dtype=np.int16)
    integer = randint(min_int, max_int, size=size, dtype=int)
    integer32 = randint(min_int32, max_int32, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64, size=size, dtype=np.int64)
    fl32 = np.array([[1.5, 2.2, 3.3, 4.4, 5.5],[5.4,2.1,7.1,10.46,11.0]], dtype=np.float32)
    cmplx = (np.random.random((2,5))*75) + (np.random.random((2,5))*50) * 1j

    epyccel_func = epyccel(test_linspace, language=language)
    epyccel_func3 = epyccel(test_linspace3, language=language)
    epyccel_func2 = epyccel(test_linspace2, language=language)
    epyccel_func4 = epyccel(test_linspace4, language=language)

    arr = linspace(integer, 5, 7)
    out = np.empty_like(arr)
    epyccel_func(integer, 5, out, True)
    assert np.allclose(arr, out)
    arr = linspace(integer, 5, 7, endpoint=False)
    out = np.empty_like(arr)
    epyccel_func(integer, 5, out, False)
    assert np.allclose(arr, out)
    arr = linspace(integer8, 5, 7)
    out = np.empty_like(arr)
    epyccel_func(integer8, 5, out, True)
    assert np.allclose(arr, out)
    arr = linspace(integer8, 5, 7, endpoint=False)
    out = np.empty_like(arr)
    epyccel_func(integer8, 5, out, False)
    assert np.allclose(arr, out)
    arr = linspace(integer16, 5, 7)
    out = np.empty_like(arr)
    epyccel_func(integer16, 5, out, True)
    assert np.allclose(arr, out)
    arr = linspace(integer16, 5, 7, endpoint=False)
    out = np.empty_like(arr)
    epyccel_func(integer16, 5, out, False)
    assert np.allclose(arr, out)
    arr = linspace(integer32, 5, 7)
    out = np.empty_like(arr)
    epyccel_func(integer32, 5, out, True)
    assert np.allclose(arr, out)
    arr = linspace(integer32, 5, 7, endpoint=False)
    out = np.empty_like(arr)
    epyccel_func(integer32, 5, out, False)
    assert np.allclose(arr, out)
    integer   = randint(min_int / 2, max_int / 2, size=size, dtype=int)
    integer_2 = np.array([[1, 2, 3, 4, 5],[5,2,7,10,11]], dtype=int)
    arr = linspace(integer, integer_2, 7, endpoint=False)
    out = np.empty_like(arr)
    epyccel_func2(integer, integer_2, out, False)
    assert np.allclose(arr, out)
    if sys.platform != 'win32':
        arr = linspace(integer64, 5, 7)
        out = np.empty_like(arr)
        epyccel_func(integer64, 5, out, True)
        assert np.allclose(arr, out)
        arr = linspace(integer64, 5, 7, endpoint=False)
        out = np.empty_like(arr)
        epyccel_func(integer64, 5, out, False)
        assert np.allclose(arr, out)

    arr = linspace(fl32, 5, 7)
    out = np.empty_like(arr)
    epyccel_func(fl32, 5, out, True)
    assert np.allclose(arr, out)
    arr = linspace(fl32, 5, 7, endpoint=False)
    out = np.empty_like(arr)
    epyccel_func(fl32, 5, out, False)
    assert np.allclose(arr, out)
    rng = np.random.default_rng()
    fl64 = rng.random((2,5), dtype=np.float64)
    arr = linspace(fl64, 5, 7)
    out = np.empty_like(arr)
    epyccel_func(fl64, 5, out, True)
    assert np.allclose(arr, out)
    arr = linspace(fl64, 5, 7, endpoint=False)
    out = np.empty_like(arr)
    epyccel_func(fl64, 5, out, False)
    assert np.allclose(arr, out)

    arr = linspace(cmplx, 5, 7)
    out = np.empty_like(arr)
    epyccel_func3(cmplx, 5, out, True)
    assert np.allclose(arr, out)
    arr = linspace(cmplx, 5, 7, endpoint=False)
    out = np.empty_like(arr)
    epyccel_func3(cmplx, 5, out, False)
    assert np.allclose(arr, out)
    cmplx  = (np.random.random((2,5))*55) + (np.random.random((2,5))*50) * 1j
    cmplx2 = (np.random.random((2,5))*14) + (np.random.random((2,5))*15) * 1j
    arr = linspace(cmplx, cmplx2, 7)
    out = np.empty_like(arr)
    epyccel_func4(cmplx, cmplx2, out, True)
    assert np.allclose(arr, out)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="count_nonzero not implemented"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_numpy_count_non_zero_1d(language):
    @template('T', ['bool[:]', 'int[:]', 'int8[:]', 'int16[:]', 'int32[:]', 'int64[:]', 'float[:]', 'float32[:]', 'float64[:]'])
    def count(arr : 'T'):
        from numpy import count_nonzero
        return count_nonzero(arr)

    size = 5

    bl = randint(0, 2, size=size, dtype= bool)

    integer8  = randint(min_int8//2,  max_int8//2, size=size, dtype=np.int8)
    integer16 = randint(min_int16//2, max_int16//2, size=size, dtype=np.int16)
    integer   = randint(min_int//2,   max_int//2, size=size, dtype=int)
    integer32 = randint(min_int32//2, max_int32//2, size=size, dtype=np.int32)
    integer64 = randint(min_int64//2, max_int64//2, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size = size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size = size)

    epyccel_func = epyccel(count, language=language)

    assert epyccel_func(bl) == count(bl)
    assert epyccel_func(integer8) == count(integer8)
    assert epyccel_func(integer16) == count(integer16)
    assert epyccel_func(integer) == count(integer)
    assert epyccel_func(integer32) == count(integer32)
    assert epyccel_func(integer64) == count(integer64)
    assert epyccel_func(fl) == count(fl)
    assert epyccel_func(fl32) == count(fl32)
    assert epyccel_func(fl64) == count(fl64)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="count_nonzero not implemented"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_numpy_count_non_zero_2d(language):
    @template('T', ['bool[:,:]', 'int[:,:]', 'int8[:,:]', 'int16[:,:]', 'int32[:,:]', 'int64[:,:]', 'float[:,:]', 'float32[:,:]', 'float64[:,:]'])
    def count(arr : 'T'):
        from numpy import count_nonzero
        return count_nonzero(arr)

    size = (2, 5)

    bl = randint(0, 2, size=size, dtype= bool)

    integer8  = randint(min_int8, max_int8-1, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16-1, size=size, dtype=np.int16)
    integer   = randint(min_int, max_int-1, size=size, dtype=int)
    integer32 = randint(min_int32, max_int32-1, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64-1, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size = size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size = size)

    epyccel_func = epyccel(count, language=language)

    assert epyccel_func(bl) == count(bl)
    assert epyccel_func(integer8) == count(integer8)
    assert epyccel_func(integer16) == count(integer16)
    assert epyccel_func(integer) == count(integer)
    assert epyccel_func(integer32) == count(integer32)
    assert epyccel_func(integer64) == count(integer64)
    assert epyccel_func(fl) == count(fl)
    assert epyccel_func(fl32) == count(fl32)
    assert epyccel_func(fl64) == count(fl64)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="count_nonzero not implemented"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_numpy_count_non_zero_1d_keep_dims(language):
    @template('T', ['bool[:]', 'int[:]', 'int8[:]', 'int16[:]', 'int32[:]', 'int64[:]', 'float[:]', 'float32[:]', 'float64[:]'])
    def count(arr : 'T'):
        from numpy import count_nonzero
        a = count_nonzero(arr, keepdims=True)
        s = a.shape
        return s[0], a[0]

    size = 5

    bl = randint(0, 2, size=size, dtype= bool)

    integer8  = randint(min_int8//2,  max_int8//2, size=size, dtype=np.int8)
    integer16 = randint(min_int16//2, max_int16//2, size=size, dtype=np.int16)
    integer   = randint(min_int//2,   max_int//2, size=size, dtype=int)
    integer32 = randint(min_int32//2, max_int32//2, size=size, dtype=np.int32)
    integer64 = randint(min_int64//2, max_int64//2, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size = size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size = size)

    epyccel_func = epyccel(count, language=language)

    assert epyccel_func(bl) == count(bl)
    assert epyccel_func(integer8) == count(integer8)
    assert epyccel_func(integer16) == count(integer16)
    assert epyccel_func(integer) == count(integer)
    assert epyccel_func(integer32) == count(integer32)
    assert epyccel_func(integer64) == count(integer64)
    assert epyccel_func(fl) == count(fl)
    assert epyccel_func(fl32) == count(fl32)
    assert epyccel_func(fl64) == count(fl64)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="count_nonzero not implemented"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_numpy_count_non_zero_2d_keep_dims(language):
    @template('T', ['bool[:,:]', 'int[:,:]', 'int8[:,:]', 'int16[:,:]', 'int32[:,:]', 'int64[:,:]', 'float[:,:]', 'float32[:,:]', 'float64[:,:]'])
    def count(arr : 'T'):
        from numpy import count_nonzero
        a = count_nonzero(arr, keepdims=True)
        s = a.shape
        return s[0], s[1], a[0,0]

    size = (2, 5)

    bl = randint(0, 2, size=size, dtype= bool)

    integer8  = randint(min_int8, max_int8-1, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16-1, size=size, dtype=np.int16)
    integer   = randint(min_int, max_int-1, size=size, dtype=int)
    integer32 = randint(min_int32, max_int32-1, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64-1, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size = size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size = size)

    epyccel_func = epyccel(count, language=language)

    assert epyccel_func(bl) == count(bl)
    assert epyccel_func(integer8) == count(integer8)
    assert epyccel_func(integer16) == count(integer16)
    assert epyccel_func(integer) == count(integer)
    assert epyccel_func(integer32) == count(integer32)
    assert epyccel_func(integer64) == count(integer64)
    assert epyccel_func(fl) == count(fl)
    assert epyccel_func(fl32) == count(fl32)
    assert epyccel_func(fl64) == count(fl64)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="count_nonzero not implemented"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_numpy_count_non_zero_axis(language):
    @template('T', ['bool[:,:,:]', 'int[:,:,:]', 'int8[:,:,:]', 'int16[:,:,:]', 'int32[:,:,:]', 'int64[:,:,:]', 'float[:,:,:]', 'float32[:,:,:]', 'float64[:,:,:]'])
    def count(arr : 'T'):
        from numpy import count_nonzero
        a = count_nonzero(arr, axis = 1)
        s = a.shape
        return len(s), s[0], s[1], a[0,0], a[0,-1]

    size = (2, 5, 3)

    bl = randint(0, 2, size=size, dtype= bool)

    integer8  = randint(min_int8, max_int8-1, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16-1, size=size, dtype=np.int16)
    integer   = randint(min_int, max_int-1, size=size, dtype=int)
    integer32 = randint(min_int32, max_int32-1, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64-1, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size = size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size = size)

    epyccel_func = epyccel(count, language=language)

    assert epyccel_func(bl) == count(bl)
    assert epyccel_func(integer8) == count(integer8)
    assert epyccel_func(integer16) == count(integer16)
    assert epyccel_func(integer) == count(integer)
    assert epyccel_func(integer32) == count(integer32)
    assert epyccel_func(integer64) == count(integer64)
    assert epyccel_func(fl) == count(fl)
    assert epyccel_func(fl32) == count(fl32)
    assert epyccel_func(fl64) == count(fl64)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="count_nonzero not implemented"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_numpy_count_non_zero_axis_keep_dims(language):
    @template('T', ['bool[:,:,:]', 'int[:,:,:]', 'int8[:,:,:]', 'int16[:,:,:]', 'int32[:,:,:]', 'int64[:,:,:]', 'float[:,:,:]', 'float32[:,:,:]', 'float64[:,:,:]'])
    def count(arr : 'T'):
        from numpy import count_nonzero, empty
        a = count_nonzero(arr, axis = 0, keepdims=True)
        s = a.shape
        return len(s), s[0], s[1], s[2], a[0,0,0], a[0,0,-1]

    size = (5, 2, 3)

    bl = randint(0, 2, size=size, dtype= bool)

    integer8  = randint(min_int8, max_int8-1, size=size, dtype=np.int8)
    integer16 = randint(min_int16, max_int16-1, size=size, dtype=np.int16)
    integer   = randint(min_int, max_int-1, size=size, dtype=int)
    integer32 = randint(min_int32, max_int32-1, size=size, dtype=np.int32)
    integer64 = randint(min_int64, max_int64-1, size=size, dtype=np.int64)

    fl = uniform(min_float / 2, max_float / 2, size = size)
    fl32 = uniform(min_float32 / 2, max_float32 / 2, size = size)
    fl32 = np.float32(fl32)
    fl64 = uniform(min_float64 / 2, max_float64 / 2, size = size)

    epyccel_func = epyccel(count, language=language)

    assert epyccel_func(bl) == count(bl)
    assert epyccel_func(integer8) == count(integer8)
    assert epyccel_func(integer16) == count(integer16)
    assert epyccel_func(integer) == count(integer)
    assert epyccel_func(integer32) == count(integer32)
    assert epyccel_func(integer64) == count(integer64)
    assert epyccel_func(fl) == count(fl)
    assert epyccel_func(fl32) == count(fl32)
    assert epyccel_func(fl64) == count(fl64)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="count_nonzero not implemented"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_numpy_count_non_zero_axis_keep_dims_F(language):
    @template('T', ['bool[:,:,:](order=F)',
                    'int[:,:,:](order=F)',
                    'int8[:,:,:](order=F)',
                    'int16[:,:,:](order=F)',
                    'int32[:,:,:](order=F)',
                    'int64[:,:,:](order=F)',
                    'float[:,:,:](order=F)',
                    'float32[:,:,:](order=F)',
                    'float64[:,:,:](order=F)'])
    def count(arr : 'T'):
        from numpy import count_nonzero
        a = count_nonzero(arr, axis = 1, keepdims=True)
        s = a.shape
        return len(s), s[0], s[1], s[2], a[0,0,0], a[0,0,-1]

    size = (2, 5, 3)

    bl = np.array(randint(0, 2, size=size), dtype= bool, order='F')

    integer8  = np.array(randint(min_int8,  max_int8-1,  size=size, dtype=np.int8), order='F')
    integer16 = np.array(randint(min_int16, max_int16-1, size=size, dtype=np.int16), order='F')
    integer   = np.array(randint(min_int,   max_int-1,   size=size, dtype=int), order='F')
    integer32 = np.array(randint(min_int32, max_int32-1, size=size, dtype=np.int32), order='F')
    integer64 = np.array(randint(min_int64, max_int64-1, size=size, dtype=np.int64), order='F')

    fl   = np.array(uniform(min_float / 2, max_float / 2, size = size), dtype=float, order='F')
    fl32 = np.array(uniform(min_float32 / 2, max_float32 / 2, size = size), dtype=np.float32, order='F')
    fl64 = np.array(uniform(min_float64 / 2, max_float64 / 2, size = size), dtype=np.float64, order='F')

    epyccel_func = epyccel(count, language=language)

    assert epyccel_func(bl) == count(bl)
    assert epyccel_func(integer8) == count(integer8)
    assert epyccel_func(integer16) == count(integer16)
    assert epyccel_func(integer) == count(integer)
    assert epyccel_func(integer32) == count(integer32)
    assert epyccel_func(integer64) == count(integer64)
    assert epyccel_func(fl) == count(fl)
    assert epyccel_func(fl32) == count(fl32)
    assert epyccel_func(fl64) == count(fl64)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="nonzero not implemented"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_nonzero(language):

    @template('T', ['bool[:]', 'int[:]', 'int8[:]', 'int16[:]', 'int32[:]', 'int64[:]', 'float[:]', 'float32[:]', 'float64[:]'])
    def nonzero_func(a : 'T'):
        from numpy import nonzero
        b = nonzero(a)
        return len(b), b[0][0], b[0][1]

    # Arrays must have at least 2 non-zero elements to avoid IndexError
    bl = np.array([True, False, True, False, True])
    integer8  = np.array([6,1,8,2,3], dtype = np.int8)
    integer16 = np.array([6,1,8,2,3], dtype = np.int16)
    integer   = np.array([6,1,8,2,3], dtype = int)
    integer32 = np.array([6,1,8,2,3], dtype = np.int32)
    integer64 = np.array([6,1,8,2,3], dtype = np.int64)

    fl   = np.array([6,22,1,8,2,3], dtype = float)
    fl32 = np.array([6,22,1,8,2,3], dtype = np.float32)
    fl64 = np.array([6,22,1,8,2,3], dtype = np.float64)

    epyccel_func = epyccel(nonzero_func, language=language)

    assert epyccel_func(bl) == nonzero_func(bl)
    assert epyccel_func(integer8) == nonzero_func(integer8)
    assert epyccel_func(integer16) == nonzero_func(integer16)
    assert epyccel_func(integer) == nonzero_func(integer)
    assert epyccel_func(integer32) == nonzero_func(integer32)
    assert epyccel_func(integer64) == nonzero_func(integer64)
    assert epyccel_func(fl) == nonzero_func(fl)
    assert epyccel_func(fl32) == nonzero_func(fl32)
    assert epyccel_func(fl64) == nonzero_func(fl64)

def test_dtype(language):

    @template('T', ['bool[:]', 'int[:]', 'int8[:]', 'int16[:]', 'int32[:]', 'int64[:]',
                    'float[:]', 'float32[:]', 'float64[:]'])
    def func(a : 'T'):
        from numpy import zeros
        b = zeros(5, dtype=a.dtype)
        return b[0]

    bl = np.array([True, False, True, False, True])
    integer8  = np.array([6,1,8,2,3], dtype = np.int8)
    integer16 = np.array([6,1,8,2,3], dtype = np.int16)
    integer   = np.array([6,1,8,2,3], dtype = int)
    integer32 = np.array([6,1,8,2,3], dtype = np.int32)
    integer64 = np.array([6,1,8,2,3], dtype = np.int64)

    fl   = np.array([6,22,1,8,2,3], dtype = float)
    fl32 = np.array([6,22,1,8,2,3], dtype = np.float32)
    fl64 = np.array([6,22,1,8,2,3], dtype = np.float64)

    epyccel_func = epyccel(func, language=language)

    assert matching_types(epyccel_func(bl), func(bl))
    assert matching_types(epyccel_func(integer8), func(integer8))
    assert matching_types(epyccel_func(integer16), func(integer16))
    assert matching_types(epyccel_func(integer), func(integer))
    assert matching_types(epyccel_func(integer32), func(integer32))
    assert matching_types(epyccel_func(integer64), func(integer64))
    assert matching_types(epyccel_func(fl), func(fl))
    assert matching_types(epyccel_func(fl32), func(fl32))
    assert matching_types(epyccel_func(fl64), func(fl64))

def test_result_type(language):
    def int_vs_int_array():
        import numpy as np
        b = np.zeros(5, dtype=np.result_type(3, np.arange(7, dtype=np.int32)))
        return b[0]

    def type_comparison():
        import numpy as np
        b = np.zeros(5, dtype=np.result_type(np.int32, np.int16))
        return b[0]

    def type_comparison2():
        import numpy as np
        b = np.zeros(5, dtype=np.result_type(np.int32, np.complex64))
        return b[0]

    def value_types():
        import numpy as np
        b = np.zeros(5, dtype=np.result_type(3.0, -2))
        return b[0]

    def pass_through_type():
        import numpy as np
        b = np.zeros(5, dtype=np.result_type(np.float64))
        return b[0]

    def expression_type():
        import numpy as np
        a = np.array([6,1,8,2,3], dtype = np.int64)
        b = np.array([6,22,1,8,2], dtype = np.float32)
        c = np.zeros(5, dtype=np.result_type(a+b))
        return c[0]

    epyccel_int_vs_int_array = epyccel(int_vs_int_array, language=language)
    epyccel_type_comparison = epyccel(type_comparison, language=language)
    epyccel_type_comparison2 = epyccel(type_comparison2, language=language)
    epyccel_value_types = epyccel(value_types, language=language)

    assert matching_types(epyccel_int_vs_int_array(), int_vs_int_array())
    assert matching_types(epyccel_type_comparison(), type_comparison())
    assert matching_types(epyccel_type_comparison2(), type_comparison2())
    assert matching_types(epyccel_value_types(), value_types())

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = [
            pytest.mark.skip("Template causes problems with order"),
            pytest.mark.python]
        ),
    )
)
def test_copy(language):
    @template('T', ['int[:]', 'float[:,:]', 'complex[:,:,:](order=F)'])
    def copy_array(a : 'T'):
        b = a.copy()
        return b

    @template('T', ['float[:,:]', 'complex[:,:,:](order=F)'])
    def copy_array_to_F(a : 'T'):
        b = a.copy(order='F')
        return b

    @template('T', ['float[:,:]', 'complex[:,:,:](order=F)'])
    def copy_array_to_C(a : 'T'):
        b = a.copy(order='C')
        return b

    arr_1d = randint(min_int, max_int, size=5)
    arr_2d = uniform(min_float64 / 2, max_float64 / 2, size=(3,4))
    arr_3d = (uniform(min_float64 / 2, max_float64 / 2, size=(3,4,5)) \
            + uniform(min_float64 / 2, max_float64 / 2, size=(3,4,5))*1j).T

    funcs = [(f.__name__, f, epyccel(f, language=language)) for f in (copy_array, copy_array_to_F, copy_array_to_C)]

    _, f, epyc_f = funcs[0]
    res_1d_pyt = f(arr_1d)
    res_1d_pyc = epyc_f(arr_1d)
    assert np.array_equal(res_1d_pyt, res_1d_pyc)
    assert res_1d_pyt.dtype is res_1d_pyc.dtype

    for _, f, epyc_f in funcs:
        res_2d_pyt = f(arr_2d)
        res_2d_pyc = epyc_f(arr_2d)
        assert np.array_equal(res_2d_pyt, res_2d_pyc)
        assert res_2d_pyt.dtype is res_2d_pyc.dtype
        assert res_2d_pyt.flags.c_contiguous == res_2d_pyc.flags.c_contiguous
        assert res_2d_pyt.flags.f_contiguous == res_2d_pyc.flags.f_contiguous

        res_3d_pyt = f(arr_3d)
        res_3d_pyc = epyc_f(arr_3d)
        assert np.array_equal(res_3d_pyt, res_3d_pyc)
        assert res_3d_pyt.dtype is res_3d_pyc.dtype
        assert res_3d_pyt.flags.c_contiguous == res_3d_pyc.flags.c_contiguous
        assert res_3d_pyt.flags.f_contiguous == res_3d_pyc.flags.f_contiguous
