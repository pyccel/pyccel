# pylint: disable=missing-function-docstring, missing-module-docstring
import os
import sys
from cmath import infj, nanj, pi
from typing import TypeVar

import pytest
from numpy import isclose
from numpy.random import rand, uniform

from pyccel import epyccel
from modules import cmath_funcs
from utilities import epyccel_module_with_fallback


@pytest.fixture(scope="module")
def epyc_cmath_funcs_mod(language):
    return epyccel_module_with_fallback(cmath_funcs, language)



RTOL = sys.float_info.epsilon * 1000
ATOL = sys.float_info.epsilon * 100

T = TypeVar("T", float, complex)

max_float = 3.40282e5  # maximum positive float
min_float = sys.float_info.min  # Minimum positive float


def test_sqrt_call(epyc_cmath_funcs_mod):
    sqrt_call = cmath_funcs.sqrt_call
    f1 = epyc_cmath_funcs_mod.sqrt_call
    x = rand() + rand() * 1j
    assert isclose(f1(x), sqrt_call(x), rtol=RTOL, atol=ATOL)


def test_sqrt_mod_call(epyc_cmath_funcs_mod):
    sqrt_call = cmath_funcs.sqrt_mod_call
    f1 = epyc_cmath_funcs_mod.sqrt_mod_call
    x = rand() + rand() * 1j
    assert isclose(f1(x), sqrt_call(x), rtol=RTOL, atol=ATOL)


def test_sqrt_phrase(epyc_cmath_funcs_mod):
    sqrt_phrase = cmath_funcs.sqrt_phrase
    f2 = epyc_cmath_funcs_mod.sqrt_phrase
    x = rand() + rand() * 1j
    y = rand() + rand() * 1j
    assert isclose(f2(x, y), sqrt_phrase(x, y), rtol=RTOL, atol=ATOL)


def test_sqrt_return_type(epyc_cmath_funcs_mod):
    sqrt_return_type_real = cmath_funcs.sqrt_return_type_real
    f1 = epyc_cmath_funcs_mod.sqrt_return_type_real
    x = rand() + rand() * 1j
    assert isclose(f1(x), sqrt_return_type_real(x), rtol=RTOL, atol=ATOL)
    assert type(f1(x)) == type(
        sqrt_return_type_real(x)
    )  # pylint: disable=unidiomatic-typecheck


def test_sqrt_complex_abs(epyc_cmath_funcs_mod):
    sqrt_complex_abs = cmath_funcs.sqrt_complex_abs
    f1 = epyc_cmath_funcs_mod.sqrt_complex_abs
    x = rand() + 1j * rand()
    assert isclose(f1(x), sqrt_complex_abs(x), rtol=RTOL, atol=ATOL)
    assert type(f1(x)) == type(
        sqrt_complex_abs(x)
    )  # pylint: disable=unidiomatic-typecheck


def test_sin_call(epyc_cmath_funcs_mod):
    sin_call = cmath_funcs.sin_call
    f1 = epyc_cmath_funcs_mod.sin_call
    x = rand() + rand() * 1j
    assert isclose(f1(x), sin_call(x), rtol=RTOL, atol=ATOL)


def test_sin_phrase(epyc_cmath_funcs_mod):
    sin_phrase = cmath_funcs.sin_phrase
    f2 = epyc_cmath_funcs_mod.sin_phrase
    x = rand() + rand() * 1j
    y = rand() + rand() * 1j
    assert isclose(f2(x, y), sin_phrase(x, y), rtol=RTOL, atol=ATOL)


def test_cos_call(epyc_cmath_funcs_mod):
    cos_call = cmath_funcs.cos_call
    f1 = epyc_cmath_funcs_mod.cos_call
    x = rand() + rand() * 1j
    assert isclose(f1(x), cos_call(x), rtol=RTOL, atol=ATOL)


def test_cos_phrase(epyc_cmath_funcs_mod):
    cos_phrase = cmath_funcs.cos_phrase
    f2 = epyc_cmath_funcs_mod.cos_phrase
    x = rand() + rand() * 1j
    y = rand() + rand() * 1j
    assert isclose(f2(x, y), cos_phrase(x, y), rtol=RTOL, atol=ATOL)


def test_tan_call(epyc_cmath_funcs_mod):
    tan_call = cmath_funcs.tan_call
    f1 = epyc_cmath_funcs_mod.tan_call
    x = rand() + rand() * 1j
    assert isclose(f1(x), tan_call(x), rtol=RTOL, atol=ATOL)


def test_tan_phrase(epyc_cmath_funcs_mod):
    tan_phrase = cmath_funcs.tan_phrase
    f2 = epyc_cmath_funcs_mod.tan_phrase
    x = rand() + rand() * 1j
    y = rand() + rand() * 1j
    assert isclose(f2(x, y), tan_phrase(x, y), rtol=RTOL, atol=ATOL)


def test_exp_call(epyc_cmath_funcs_mod):
    exp_call = cmath_funcs.exp_call
    f1 = epyc_cmath_funcs_mod.exp_call
    x = rand() + rand() * 1j
    assert isclose(f1(x), exp_call(x), rtol=RTOL, atol=ATOL)


def test_exp_phrase(epyc_cmath_funcs_mod):
    exp_phrase = cmath_funcs.exp_phrase
    f2 = epyc_cmath_funcs_mod.exp_phrase
    x = rand() + rand() * 1j
    y = rand() + rand() * 1j
    assert isclose(f2(x, y), exp_phrase(x, y), rtol=RTOL, atol=ATOL)


def test_asin_call(epyc_cmath_funcs_mod):
    asin_call = cmath_funcs.asin_call
    f1 = epyc_cmath_funcs_mod.asin_call
    x = rand() + rand() * 1j
    assert isclose(f1(x), asin_call(x), rtol=RTOL, atol=ATOL)


def test_asin_phrase(epyc_cmath_funcs_mod):
    asin_phrase = cmath_funcs.asin_phrase
    f2 = epyc_cmath_funcs_mod.asin_phrase
    x = rand() + rand() * 1j
    y = rand() + rand() * 1j
    assert isclose(f2(x, y), asin_phrase(x, y), rtol=RTOL, atol=ATOL)


def test_acos_call(epyc_cmath_funcs_mod):
    acos_call = cmath_funcs.acos_call
    f1 = epyc_cmath_funcs_mod.acos_call
    x = rand() + rand() * 1j
    assert isclose(f1(x), acos_call(x), rtol=RTOL, atol=ATOL)


def test_acos_phrase(epyc_cmath_funcs_mod):
    acos_phrase = cmath_funcs.acos_phrase
    f2 = epyc_cmath_funcs_mod.acos_phrase
    x = rand() + rand() * 1j
    y = rand() + rand() * 1j
    assert isclose(f2(x, y), acos_phrase(x, y), rtol=RTOL, atol=ATOL)


def test_atan_call(epyc_cmath_funcs_mod):
    atan_call = cmath_funcs.atan_call
    f1 = epyc_cmath_funcs_mod.atan_call
    x = rand() + rand() * 1j
    assert isclose(f1(x), atan_call(x), rtol=RTOL, atol=ATOL)


def test_atan_phrase(epyc_cmath_funcs_mod):
    atan_phrase = cmath_funcs.atan_phrase
    f2 = epyc_cmath_funcs_mod.atan_phrase
    x = rand() + rand() * 1j
    y = rand() + rand() * 1j
    assert isclose(f2(x, y), atan_phrase(x, y), rtol=RTOL, atol=ATOL)


def test_sinh_call(epyc_cmath_funcs_mod):
    sinh_call = cmath_funcs.sinh_call
    f1 = epyc_cmath_funcs_mod.sinh_call
    x = rand() + rand() * 1j
    assert isclose(f1(x), sinh_call(x), rtol=RTOL, atol=ATOL)


def test_sinh_phrase(epyc_cmath_funcs_mod):
    sinh_phrase = cmath_funcs.sinh_phrase
    f2 = epyc_cmath_funcs_mod.sinh_phrase
    x = rand() + rand() * 1j
    y = rand() + rand() * 1j
    assert isclose(f2(x, y), sinh_phrase(x, y), rtol=RTOL, atol=ATOL)


def test_cosh_call(epyc_cmath_funcs_mod):
    cosh_call = cmath_funcs.cosh_call
    f1 = epyc_cmath_funcs_mod.cosh_call
    x = rand() + rand() * 1j
    assert isclose(f1(x), cosh_call(x), rtol=RTOL, atol=ATOL)


def test_cosh_phrase(epyc_cmath_funcs_mod):
    cosh_phrase = cmath_funcs.cosh_phrase
    f2 = epyc_cmath_funcs_mod.cosh_phrase
    x = rand() + rand() * 1j
    y = rand() + rand() * 1j
    assert isclose(f2(x, y), cosh_phrase(x, y), rtol=RTOL, atol=ATOL)


def test_tanh_call(epyc_cmath_funcs_mod):
    tanh_call = cmath_funcs.tanh_call
    f1 = epyc_cmath_funcs_mod.tanh_call
    x = rand() + rand() * 1j
    assert isclose(f1(x), tanh_call(x), rtol=RTOL, atol=ATOL)


def test_tanh_phrase(epyc_cmath_funcs_mod):
    tanh_phrase = cmath_funcs.tanh_phrase
    f2 = epyc_cmath_funcs_mod.tanh_phrase
    x = rand() + rand() * 1j
    y = rand() + rand() * 1j
    assert isclose(f2(x, y), tanh_phrase(x, y), rtol=RTOL, atol=ATOL)


# ----------------------------- isfinite function -----------------------------#
@pytest.mark.parametrize(
    "language",
    (
        pytest.param("c", marks=pytest.mark.c),
        pytest.param(
            "fortran",
            marks=[
                pytest.mark.xfail(reason="isfinite not implemented"),
                pytest.mark.fortran,
            ],
        ),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
@pytest.mark.skipif(
    os.environ.get("PYCCEL_DEFAULT_COMPILER", None) == "intel",
    reason="Nan not correctly passed to intel function",
)
def test_isfinite_call(language):  # isfinite
    def isfinite_call(x: T):
        from cmath import isfinite
        
        return isfinite(x)

    f1 = epyccel(isfinite_call, language=language)
    x = rand()
    y = rand() + rand() * 1j

    assert isfinite_call(x) == f1(x)
    assert isfinite_call(y) == f1(y)

    # Test not a number
    assert isfinite_call(nanj) == f1(nanj)
    # Test infinite number
    assert isfinite_call(infj) == f1(infj)
    # Test negative infinite number
    assert isfinite_call(-infj) == f1(-infj)


# ------------------------------- isinf function ------------------------------#
@pytest.mark.parametrize(
    "language",
    (
        pytest.param("c", marks=pytest.mark.c),
        pytest.param(
            "fortran",
            marks=[
                pytest.mark.xfail(reason="isinf not implemented"),
                pytest.mark.fortran,
            ],
        ),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_isinf_call(language):  # isinf
    def isinf_call(x: T):
        from cmath import isinf

        return isinf(x)

    f1 = epyccel(isinf_call, language=language)
    x = rand()
    y = rand() + rand() * 1j

    assert isinf_call(x) == f1(x)
    assert isinf_call(y) == f1(y)

    # Test not a number
    assert isinf_call(nanj) == f1(nanj)
    # Test infinite number
    assert isinf_call(infj) == f1(infj)
    # Test negative infinite number
    assert isinf_call(-infj) == f1(-infj)


# ------------------------------- isnan function ------------------------------#


@pytest.mark.parametrize(
    "language",
    (
        pytest.param(
            "c",
            marks=[
                pytest.mark.xfail(reason="infj not properly passed through"),
                pytest.mark.c,
            ],
        ),
        pytest.param(
            "fortran",
            marks=[
                pytest.mark.xfail(reason="infj not properly passed through"),
                pytest.mark.fortran,
            ],
        ),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_isnan_call(language):  # isnan
    def isnan_call(x: T):
        from cmath import isnan

        return isnan(x)

    f1 = epyccel(isnan_call, language=language)
    x = rand()
    y = rand() + rand() * 1j

    assert isnan_call(x) == f1(x)
    assert isnan_call(y) == f1(y)

    # Test not a number
    assert isnan_call(nanj) == f1(nanj)
    # Test infinite number
    assert isnan_call(infj) == f1(infj)
    # Test negative infinite number
    assert isnan_call(-infj) == f1(-infj)


# ------------------------------- Acosh function ------------------------------#


def test_acosh_call(epyc_cmath_funcs_mod):
    acosh_call = cmath_funcs.acosh_call
    f1 = epyc_cmath_funcs_mod.acosh_call

    x = uniform(low=1, high=max_float) + uniform(low=1, high=max_float) * 1j
    assert isclose(f1(x), acosh_call(x), rtol=RTOL, atol=ATOL)
    assert isinstance(f1(x), type(acosh_call(x)))


def test_acosh_phrase(epyc_cmath_funcs_mod):
    acosh_phrase = cmath_funcs.acosh_phrase
    f2 = epyc_cmath_funcs_mod.acosh_phrase

    x = uniform(low=1, high=max_float) + uniform(low=1, high=max_float) * 1j
    y = uniform(low=1, high=max_float) + uniform(low=1, high=max_float) * 1j
    assert isclose(f2(x, y), acosh_phrase(x, y), rtol=RTOL, atol=ATOL)


# ------------------------------- Asinh function ------------------------------#


@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows asinh gives different results to Python"
)
def test_asinh_call(epyc_cmath_funcs_mod):
    asinh_call = cmath_funcs.asinh_call
    f1 = epyc_cmath_funcs_mod.asinh_call

    x = uniform(high=max_float) + uniform(high=max_float) * 1j
    assert isclose(f1(x), asinh_call(x), rtol=RTOL, atol=ATOL)
    assert isinstance(f1(x), type(asinh_call(x)))

    # Negative value
    assert isclose(f1(-x), asinh_call(-x), rtol=RTOL, atol=ATOL)


@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows asinh gives different results to Python"
)
def test_asinh_phrase(epyc_cmath_funcs_mod):
    asinh_phrase = cmath_funcs.asinh_phrase
    f2 = epyc_cmath_funcs_mod.asinh_phrase
    x = uniform(high=max_float) + uniform(high=max_float) * 1j
    y = uniform(high=max_float) + uniform(high=max_float) * 1j
    assert isclose(f2(x, y), asinh_phrase(x, y), rtol=RTOL, atol=ATOL)
    # Negative value
    assert isclose(f2(-x, -y), asinh_phrase(-x, -y), rtol=RTOL, atol=ATOL)


# ------------------------------- Atanh function ------------------------------#


def test_atanh_call(epyc_cmath_funcs_mod):
    atanh_call = cmath_funcs.atanh_call
    f1 = epyc_cmath_funcs_mod.atanh_call
    low = -1 + min_float
    high = 1 - min_float
    x = uniform(low=low, high=high) + uniform(low=low, high=high) * 1j
    assert isclose(f1(x), atanh_call(x), rtol=RTOL, atol=ATOL)
    assert isinstance(f1(x), type(atanh_call(x)))


def test_atanh_phrase(epyc_cmath_funcs_mod):
    atanh_phrase = cmath_funcs.atanh_phrase
    f2 = epyc_cmath_funcs_mod.atanh_phrase

    # Domain ]-1, 1[
    low = -1 + min_float
    high = 1 - min_float
    x = uniform(low=low, high=high) + uniform(low=low, high=high) * 1j
    y = uniform(low=low, high=high) + uniform(low=low, high=high) * 1j
    assert isclose(f2(x, y), atanh_phrase(x, y), rtol=RTOL, atol=ATOL)


# ------------------------------- Polar functions ------------------------------#


@pytest.mark.skipif_by_language(True,
    language="python",
                    reason="Printed code differs between types. See #1334"
)
def test_phase_call(epyc_cmath_funcs_mod):
    phase_call = cmath_funcs.phase_call
    f1 = epyc_cmath_funcs_mod.phase_call
    low = 1 + min_float
    high = max_float
    x = uniform(low=low, high=high) + uniform(low=low, high=high) * 1j
    assert isclose(f1(x), phase_call(x), rtol=RTOL, atol=ATOL)
    assert isinstance(f1(x), type(phase_call(x)))

    y = uniform(low=low, high=high)
    assert isclose(f1(y), phase_call(y), rtol=RTOL, atol=ATOL)
    assert isinstance(f1(y), type(phase_call(y)))


def test_polar_call(epyc_cmath_funcs_mod):
    polar_call = cmath_funcs.polar_call
    f1 = epyc_cmath_funcs_mod.polar_call
    low = -1 + min_float
    high = 1 - min_float
    x = uniform(low=low, high=high) + uniform(low=low, high=high) * 1j
    assert isclose(f1(x), polar_call(x), rtol=RTOL, atol=ATOL).all()
    assert isinstance(f1(x), type(polar_call(x)))


def test_rect_call(epyc_cmath_funcs_mod):
    rect_call = cmath_funcs.rect_call
    f1 = epyc_cmath_funcs_mod.rect_call
    r = uniform(low=0, high=max_float)
    phi = uniform(low=-2 * pi, high=2 * pi)
    assert isclose(f1(r, phi), rect_call(r, phi), rtol=RTOL, atol=ATOL)
    assert isinstance(f1(r, phi), type(rect_call(r, phi)))
