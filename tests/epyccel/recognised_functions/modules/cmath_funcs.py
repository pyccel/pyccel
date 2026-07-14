# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import TypeVar

T = TypeVar("T", float, complex)


def sqrt_call(x: complex):
    from cmath import sqrt

    return sqrt(x)


def sqrt_mod_call(x: complex):
    import cmath

    return cmath.sqrt(x)


def sqrt_phrase(x: complex, y: complex):
    from cmath import sqrt

    a = sqrt(x) * sqrt(y)
    return a


def sqrt_return_type_real(x: complex):
    from cmath import sqrt

    a = sqrt(x)
    return a


def sqrt_complex_abs(x: "complex"):
    from cmath import sqrt

    a = sqrt(x * x.conjugate()) + sqrt(x.conjugate() * x)
    return a


def sin_call(x: complex):
    from cmath import sin

    return sin(x)


def sin_phrase(x: complex, y: complex):
    from cmath import sin

    a = sin(x) + sin(y)
    return a


def cos_call(x: complex):
    from cmath import cos

    return cos(x)


def cos_phrase(x: complex, y: complex):
    from cmath import cos

    a = cos(x) + cos(y)
    return a


def tan_call(x: complex):
    from cmath import tan

    return tan(x)


def tan_phrase(x: complex, y: complex):
    from cmath import tan

    a = tan(x) + tan(y)
    return a


def exp_call(x: complex):
    from cmath import exp

    return exp(x)


def exp_phrase(x: complex, y: complex):
    from cmath import exp

    a = exp(x) + exp(y)
    return a


def asin_call(x: complex):
    from cmath import asin

    return asin(x)


def asin_phrase(x: complex, y: complex):
    from cmath import asin

    a = asin(x) + asin(y)
    return a


def acos_call(x: complex):
    from cmath import acos

    return acos(x)


def acos_phrase(x: complex, y: complex):
    from cmath import acos

    a = acos(x) + acos(y)
    return a


def atan_call(x: complex):
    from cmath import atan

    return atan(x)


def atan_phrase(x: complex, y: complex):
    from cmath import atan

    a = atan(x) + atan(y)
    return a


def sinh_call(x: complex):
    from cmath import sinh

    return sinh(x)


def sinh_phrase(x: complex, y: complex):
    from cmath import sinh

    a = sinh(x) + sinh(y)
    return a


def cosh_call(x: complex):
    from cmath import cosh

    return cosh(x)


def cosh_phrase(x: complex, y: complex):
    from cmath import cosh

    a = cosh(x) + cosh(y)
    return a


def tanh_call(x: complex):
    from cmath import tanh

    return tanh(x)


def tanh_phrase(x: complex, y: complex):
    from cmath import tanh

    a = tanh(x) + tanh(y)
    return a


# def isfinite_call(x: T):
#    from cmath import isfinite
#
#    return isfinite(x)
#
# def isinf_call(x: T):
#    from cmath import isinf
#
#    return isinf(x)
#
# def isnan_call(x: T):
#    from cmath import isnan
#
#    return isnan(x)


def acosh_call(x: complex):
    from cmath import acosh

    return acosh(x)


def acosh_phrase(x: complex, y: complex):
    from cmath import acosh

    a = acosh(x) + acosh(y)
    return a


def asinh_call(x: complex):
    from cmath import asinh

    return asinh(x)


def asinh_phrase(x: complex, y: complex):
    from cmath import asinh

    a = asinh(x) + asinh(y)
    return a


def atanh_call(x: complex):
    from cmath import atanh

    return atanh(x)


def atanh_phrase(x: complex, y: complex):
    from cmath import atanh

    a = atanh(x) + atanh(y)
    return a


def phase_call(x: T):
    from cmath import phase

    return phase(x)


def polar_call(x: complex):
    from cmath import polar

    r, t = polar(x)
    return r, t


def rect_call(r: float, phi: float):
    from cmath import rect

    return rect(r, phi)
