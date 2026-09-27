# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import TypeVar


def fabs_call(x: "float"):
    from math import fabs

    return fabs(x)


def fabs_phrase(x: "float", y: "float"):
    from math import fabs

    a = fabs(x) * fabs(y)
    return a


def fabs_return_type(x: "int"):
    from math import fabs

    a = fabs(x)
    return a


def sqrt_call(x: "float"):
    from math import sqrt

    return sqrt(x)


def sqrt_module_call(x: "float"):
    import math

    return math.sqrt(x)


def sqrt_phrase(x: "float", y: "float"):
    from math import sqrt

    a = sqrt(x) * sqrt(y)
    return a


def sqrt_return_type_real(x: "float"):
    from math import sqrt

    a = sqrt(x)
    return a


def sin_call(x: "float"):
    from math import sin

    return sin(x)


def sin_phrase(x: "float", y: "float"):
    from math import sin

    a = sin(x) + sin(y)
    return a


def cos_call(x: "float"):
    from math import cos

    return cos(x)


def cos_phrase(x: "float", y: "float"):
    from math import cos

    a = cos(x) + cos(y)
    return a


def tan_call(x: "float"):
    from math import tan

    return tan(x)


def tan_phrase(x: "float", y: "float"):
    from math import tan

    a = tan(x) + tan(y)
    return a


def exp_call(x: "float"):
    from math import exp

    return exp(x)


def exp_phrase(x: "float", y: "float"):
    from math import exp

    a = exp(x) + exp(y)
    return a


def log_call(x: "float"):
    from math import log

    return log(x)


def log_phrase(x: "float", y: "float"):
    from math import log

    a = log(x) + log(y)
    return a


def asin_call(x: "float"):
    from math import asin

    return asin(x)


def asin_phrase(x: "float", y: "float"):
    from math import asin

    a = asin(x) + asin(y)
    return a


def acos_call(x: "float"):
    from math import acos

    return acos(x)


def acos_phrase(x: "float", y: "float"):
    from math import acos

    a = acos(x) + acos(y)
    return a


def atan_call(x: "float"):
    from math import atan

    return atan(x)


def atan_phrase(x: "float", y: "float"):
    from math import atan

    a = atan(x) + atan(y)
    return a


def sinh_call(x: "float"):
    from math import sinh

    return sinh(x)


def sinh_phrase(x: "float", y: "float"):
    from math import sinh

    a = sinh(x) + sinh(y)
    return a


def cosh_call(x: "float"):
    from math import cosh

    return cosh(x)


def cosh_phrase(x: "float", y: "float"):
    from math import cosh

    a = cosh(x) + cosh(y)
    return a


def tanh_call(x: "float"):
    from math import tanh

    return tanh(x)


def tanh_phrase(x: "float", y: "float"):
    from math import tanh

    a = tanh(x) + tanh(y)
    return a


def atan2_call(x: "float", y: "float"):
    from math import atan2

    return atan2(x, y)


def atan2_phrase(x: "float", y: "float", z: "float"):
    from math import atan2

    a = atan2(x, y) + atan2(y, z)
    return a


def copysign_call(x: "float", y: "float"):
    from math import copysign

    return copysign(x, y)


def copysign_zero_case(x: "int", y: "int"):
    from math import copysign

    return copysign(x, y)


def copysign_return_type(x: "float", y: "float"):
    from math import copysign

    a = copysign(x, y)
    return a


def copysign_return_type_2(x: "int", y: "int"):
    from math import copysign

    a = copysign(x, y)
    return a


def copysign_return_type_3(x: "int", y: "float"):
    from math import copysign

    a = copysign(x, y)
    return a


def copysign_return_type_4(x: "float", y: "int"):
    from math import copysign

    a = copysign(x, y)
    return a


def isfinite_call(x: "float"):
    from math import isfinite

    return isfinite(x)


def isinf_call(x: "float"):
    from math import isinf

    return isinf(x)


def isnan_call(x: "float"):
    from math import isnan

    return isnan(x)


def ldexp_call(x: "float", exp: "int"):
    from math import ldexp

    return ldexp(x, exp)


def ldexp_type(x: "float", exp: "int"):
    from math import ldexp

    return ldexp(x, exp)


def remainder_call(x: "float", y: "float"):
    from math import remainder

    return remainder(x, y)


def remainder_type(x: "float", y: "float"):
    from math import remainder

    return remainder(x, y)


def trunc_call(x: "float"):
    from math import trunc

    return trunc(x)


def trunc_call_int(x: "int"):
    from math import trunc

    return trunc((x))


def trunc_type(x: "float"):
    from math import trunc

    return trunc(x)


def expm1_call(x: "float"):
    from math import expm1

    return expm1(x)


def expm1_call_special_case(x: "float"):
    from math import expm1

    return expm1(x)


def expm1_phrase(x: "float", y: "float"):
    from math import expm1

    a = expm1(x) + expm1(y)
    return a


def expm1_type(x: "float"):
    from math import expm1

    return expm1(x)


def log1p_call(x: "float"):
    from math import log1p

    return log1p(x)


def log1p_phrase(x: "float", y: "float"):
    from math import log1p

    a = log1p(x) + log1p(y)
    return a


def log2_call(x: "float"):
    from math import log2

    return log2(x)


def log2_phrase(x: "float", y: "float"):
    from math import log2

    a = log2(x) + log2(y)
    return a


def log10_call(x: "float"):
    from math import log10

    return log10(x)


def log10_phrase(x: "float", y: "float"):
    from math import log10

    a = log10(x) + log10(y)
    return a


T_pow_call = TypeVar("T_pow_call", int, float)


def pow_call(x: float, y: T_pow_call):
    from math import pow as my_pow

    return my_pow(x, y)


def hypot_call(x: "float", y: "float"):
    from math import hypot

    return hypot(x, y)


def acosh_call(x: "float"):
    from math import acosh

    return acosh(x)


def acosh_phrase(x: "float", y: "float"):
    from math import acosh

    a = acosh(x) + acosh(y)
    return a


def asinh_call(x: "float"):
    from math import asinh

    return asinh(x)


def asinh_phrase(x: "float", y: "float"):
    from math import asinh

    a = asinh(x) + asinh(y)
    return a


def atanh_call(x: "float"):
    from math import atanh

    return atanh(x)


def atanh_phrase(x: "float", y: "float"):
    from math import atanh

    a = atanh(x) + atanh(y)
    return a


def erf_call(x: "float"):
    from math import erf

    return erf(x)


def erf_phrase(x: "float", y: "float"):
    from math import erf

    a = erf(x) + erf(y)
    return a


def erfc_call(x: "float"):
    from math import erfc

    return erfc(x)


def erfc_phrase(x: "float", y: "float"):
    from math import erfc

    a = erfc(x) + erfc(y)
    return a


def gamma_call(x: "float"):
    from math import gamma

    return gamma(x)


def gamma_phrase(x: "float", y: "float"):
    from math import gamma

    a = gamma(x) + gamma(y)
    return a


def lgamma_call(x: "float"):
    from math import lgamma

    return lgamma(x)


def lgamma_phrase(x: "float", y: "float"):
    from math import lgamma

    a = lgamma(x) + lgamma(y)
    return a
