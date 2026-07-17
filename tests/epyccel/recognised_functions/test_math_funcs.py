# pylint: disable=missing-function-docstring, missing-module-docstring
import os
import sys
from math import inf, modf, nan
from typing import TypeVar

import pytest
from tolerances import ATOL, RTOL
from modules import math_funcs
from numpy import isclose
from numpy.random import rand, randint, uniform
from epyccel_utilities import epyccel_module_with_fallback

from pyccel import epyccel

from tolerances import ATOL, RTOL

@pytest.fixture(scope="module")
def epyc_math_funcs_mod(language):
    return epyccel_module_with_fallback(math_funcs, language)


max_float = 3.40282e5  # maximum positive float
min_float = sys.float_info.min  # Minimum positive float


def test_fabs_call(epyc_math_funcs_mod):
    fabs_call = math_funcs.fabs_call
    f1 = epyc_math_funcs_mod.fabs_call
    x = rand()
    assert isclose(f1(x), fabs_call(x), rtol=RTOL, atol=ATOL)


def test_fabs_phrase(epyc_math_funcs_mod):
    fabs_phrase = math_funcs.fabs_phrase
    f2 = epyc_math_funcs_mod.fabs_phrase
    x = rand()
    y = rand()
    assert isclose(f2(x, y), fabs_phrase(x, y), rtol=RTOL, atol=ATOL)


def test_fabs_return_type(epyc_math_funcs_mod):
    fabs_return_type = math_funcs.fabs_return_type
    f1 = epyc_math_funcs_mod.fabs_return_type
    x = randint(100)
    assert isclose(f1(x), fabs_return_type(x), rtol=RTOL, atol=ATOL)
    assert type(f1(x)) == type(
        fabs_return_type(x)
    )  # pylint: disable=unidiomatic-typecheck


def test_sqrt_call(epyc_math_funcs_mod):
    sqrt_call = math_funcs.sqrt_call
    f1 = epyc_math_funcs_mod.sqrt_call
    x = rand()
    assert isclose(f1(x), sqrt_call(x), rtol=RTOL, atol=ATOL)


def test_sqrt_module_call(epyc_math_funcs_mod):
    sqrt_call = math_funcs.sqrt_module_call
    f1 = epyc_math_funcs_mod.sqrt_module_call
    x = rand()
    assert isclose(f1(x), sqrt_call(x), rtol=RTOL, atol=ATOL)


def test_sqrt_phrase(epyc_math_funcs_mod):
    sqrt_phrase = math_funcs.sqrt_phrase
    f2 = epyc_math_funcs_mod.sqrt_phrase
    x = rand()
    y = rand()
    assert isclose(f2(x, y), sqrt_phrase(x, y), rtol=RTOL, atol=ATOL)


def test_sqrt_return_type(epyc_math_funcs_mod):
    sqrt_return_type_real = math_funcs.sqrt_return_type_real
    f1 = epyc_math_funcs_mod.sqrt_return_type_real
    x = rand()
    assert isclose(f1(x), sqrt_return_type_real(x), rtol=RTOL, atol=ATOL)
    assert type(f1(x)) == type(
        sqrt_return_type_real(x)
    )  # pylint: disable=unidiomatic-typecheck


def test_sin_call(epyc_math_funcs_mod):
    sin_call = math_funcs.sin_call
    f1 = epyc_math_funcs_mod.sin_call
    x = rand()
    assert isclose(f1(x), sin_call(x), rtol=RTOL, atol=ATOL)


def test_sin_phrase(epyc_math_funcs_mod):
    sin_phrase = math_funcs.sin_phrase
    f2 = epyc_math_funcs_mod.sin_phrase
    x = rand()
    y = rand()
    assert isclose(f2(x, y), sin_phrase(x, y), rtol=RTOL, atol=ATOL)


def test_cos_call(epyc_math_funcs_mod):
    cos_call = math_funcs.cos_call
    f1 = epyc_math_funcs_mod.cos_call
    x = rand()
    assert isclose(f1(x), cos_call(x), rtol=RTOL, atol=ATOL)


def test_cos_phrase(epyc_math_funcs_mod):
    cos_phrase = math_funcs.cos_phrase
    f2 = epyc_math_funcs_mod.cos_phrase
    x = rand()
    y = rand()
    assert isclose(f2(x, y), cos_phrase(x, y), rtol=RTOL, atol=ATOL)


def test_tan_call(epyc_math_funcs_mod):
    tan_call = math_funcs.tan_call
    f1 = epyc_math_funcs_mod.tan_call
    x = rand()
    assert isclose(f1(x), tan_call(x), rtol=RTOL, atol=ATOL)


def test_tan_phrase(epyc_math_funcs_mod):
    tan_phrase = math_funcs.tan_phrase
    f2 = epyc_math_funcs_mod.tan_phrase
    x = rand()
    y = rand()
    assert isclose(f2(x, y), tan_phrase(x, y), rtol=RTOL, atol=ATOL)


def test_exp_call(epyc_math_funcs_mod):
    exp_call = math_funcs.exp_call
    f1 = epyc_math_funcs_mod.exp_call
    x = rand()
    assert isclose(f1(x), exp_call(x), rtol=RTOL, atol=ATOL)


def test_exp_phrase(epyc_math_funcs_mod):
    exp_phrase = math_funcs.exp_phrase
    f2 = epyc_math_funcs_mod.exp_phrase
    x = rand()
    y = rand()
    assert isclose(f2(x, y), exp_phrase(x, y), rtol=RTOL, atol=ATOL)


def test_log_call(epyc_math_funcs_mod):
    log_call = math_funcs.log_call
    f1 = epyc_math_funcs_mod.log_call
    x = rand()
    assert isclose(f1(x), log_call(x), rtol=RTOL, atol=ATOL)


def test_log_phrase(epyc_math_funcs_mod):
    log_phrase = math_funcs.log_phrase
    f2 = epyc_math_funcs_mod.log_phrase
    x = rand()
    y = rand()
    assert isclose(f2(x, y), log_phrase(x, y), rtol=RTOL, atol=ATOL)


def test_asin_call(epyc_math_funcs_mod):
    asin_call = math_funcs.asin_call
    f1 = epyc_math_funcs_mod.asin_call
    x = rand()
    assert isclose(f1(x), asin_call(x), rtol=RTOL, atol=ATOL)


def test_asin_phrase(epyc_math_funcs_mod):
    asin_phrase = math_funcs.asin_phrase
    f2 = epyc_math_funcs_mod.asin_phrase
    x = rand()
    y = rand()
    assert isclose(f2(x, y), asin_phrase(x, y), rtol=RTOL, atol=ATOL)


def test_acos_call(epyc_math_funcs_mod):
    acos_call = math_funcs.acos_call
    f1 = epyc_math_funcs_mod.acos_call
    x = rand()
    assert isclose(f1(x), acos_call(x), rtol=RTOL, atol=ATOL)


def test_acos_phrase(epyc_math_funcs_mod):
    acos_phrase = math_funcs.acos_phrase
    f2 = epyc_math_funcs_mod.acos_phrase
    x = rand()
    y = rand()
    assert isclose(f2(x, y), acos_phrase(x, y), rtol=RTOL, atol=ATOL)


def test_atan_call(epyc_math_funcs_mod):
    atan_call = math_funcs.atan_call
    f1 = epyc_math_funcs_mod.atan_call
    x = rand()
    assert isclose(f1(x), atan_call(x), rtol=RTOL, atol=ATOL)


def test_atan_phrase(epyc_math_funcs_mod):
    atan_phrase = math_funcs.atan_phrase
    f2 = epyc_math_funcs_mod.atan_phrase
    x = rand()
    y = rand()
    assert isclose(f2(x, y), atan_phrase(x, y), rtol=RTOL, atol=ATOL)


def test_sinh_call(epyc_math_funcs_mod):
    sinh_call = math_funcs.sinh_call
    f1 = epyc_math_funcs_mod.sinh_call
    x = rand()
    assert isclose(f1(x), sinh_call(x), rtol=RTOL, atol=ATOL)


def test_sinh_phrase(epyc_math_funcs_mod):
    sinh_phrase = math_funcs.sinh_phrase
    f2 = epyc_math_funcs_mod.sinh_phrase
    x = rand()
    y = rand()
    assert isclose(f2(x, y), sinh_phrase(x, y), rtol=RTOL, atol=ATOL)


def test_cosh_call(epyc_math_funcs_mod):
    cosh_call = math_funcs.cosh_call
    f1 = epyc_math_funcs_mod.cosh_call
    x = rand()
    assert isclose(f1(x), cosh_call(x), rtol=RTOL, atol=ATOL)


def test_cosh_phrase(epyc_math_funcs_mod):
    cosh_phrase = math_funcs.cosh_phrase
    f2 = epyc_math_funcs_mod.cosh_phrase
    x = rand()
    y = rand()
    assert isclose(f2(x, y), cosh_phrase(x, y), rtol=RTOL, atol=ATOL)


def test_tanh_call(epyc_math_funcs_mod):
    tanh_call = math_funcs.tanh_call
    f1 = epyc_math_funcs_mod.tanh_call
    x = rand()
    assert isclose(f1(x), tanh_call(x), rtol=RTOL, atol=ATOL)


def test_tanh_phrase(epyc_math_funcs_mod):
    tanh_phrase = math_funcs.tanh_phrase
    f2 = epyc_math_funcs_mod.tanh_phrase
    x = rand()
    y = rand()
    assert isclose(f2(x, y), tanh_phrase(x, y), rtol=RTOL, atol=ATOL)


def test_atan2_call(epyc_math_funcs_mod):
    atan2_call = math_funcs.atan2_call
    f1 = epyc_math_funcs_mod.atan2_call
    x = rand()
    y = rand()
    assert isclose(f1(x, y), atan2_call(x, y), rtol=RTOL, atol=ATOL)


def test_atan2_phrase(epyc_math_funcs_mod):
    atan2_phrase = math_funcs.atan2_phrase
    f2 = epyc_math_funcs_mod.atan2_phrase
    x = rand()
    y = rand()
    z = rand()
    assert isclose(f2(x, y, z), atan2_phrase(x, y, z), rtol=RTOL, atol=ATOL)


# ------------------------------- Floor function ------------------------------#
def test_floor_call(language):
    def floor_call(x: "float"):
        from math import floor

        return floor(x)

    flags = "-Werror -Wconversion"
    f1 = epyccel(floor_call, language=language, flags=flags)
    x = rand()
    assert isclose(f1(x), floor_call(x), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x), floor_call(-x), rtol=RTOL, atol=ATOL)


def test_floor_phrase(language):
    def floor_phrase(x: "float", y: "float"):
        from math import floor

        a = floor(x) * floor(y)
        return a

    flags = "-Werror -Wconversion"
    f2 = epyccel(floor_phrase, language=language, flags=flags)
    x = rand()
    y = rand()
    assert isclose(f2(x, y), floor_phrase(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, y), floor_phrase(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(x, -y), floor_phrase(x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), floor_phrase(-x, -y), rtol=RTOL, atol=ATOL)


def test_floor_return_type(language):
    def floor_return_type_int(x: "int"):
        from math import floor

        a = floor(x)
        return a

    def floor_return_type_real(x: "float"):
        from math import floor

        a = floor(x)
        return a

    flags = "-Werror -Wconversion"
    f1 = epyccel(floor_return_type_int, language=language, flags=flags)

    x = randint(100)
    assert isclose(f1(x), floor_return_type_int(x), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x), floor_return_type_int(-x), rtol=RTOL, atol=ATOL)
    assert type(f1(x)) == type(
        floor_return_type_int(x)
    )  # pylint: disable=unidiomatic-typecheck

    flags = "-Werror -Wconversion"
    f1 = epyccel(floor_return_type_real, language=language, flags=flags)

    x = uniform(100)
    assert isclose(f1(x), floor_return_type_real(x), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x), floor_return_type_real(-x), rtol=RTOL, atol=ATOL)
    assert type(f1(x)) == type(
        floor_return_type_real(x)
    )  # pylint: disable=unidiomatic-typecheck


# ------------------------------- Ceil function -------------------------------#
def test_ceil_call_r(language):
    def ceil_call(x: "float"):
        from math import ceil

        return ceil(x)

    flags = "-Werror -Wconversion"
    f1 = epyccel(ceil_call, language=language, flags=flags)

    x = rand()
    assert ceil_call(x) == f1(x)
    assert ceil_call(-x) == f1(-x)

    assert isinstance(ceil_call(x), type(f1(x)))


def test_ceil_call_i(language):
    def ceil_call(x: "int"):
        from math import ceil

        return ceil(x)

    flags = "-Werror -Wconversion"
    f1 = epyccel(ceil_call, language=language, flags=flags)

    x = randint(10)
    assert ceil_call(x) == f1(x)
    assert ceil_call(-x) == f1(-x)

    assert isinstance(ceil_call(x), type(f1(x)))


def test_ceil_phrase(language):
    def ceil_phrase(x: "float", y: "float"):
        from math import ceil

        a = ceil(x) * ceil(y)
        return a

    flags = "-Werror -Wconversion"
    f2 = epyccel(ceil_phrase, language=language, flags=flags)

    x = rand()
    y = rand()
    assert isclose(ceil_phrase(x, y), f2(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(ceil_phrase(-x, y), f2(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(ceil_phrase(x, -y), f2(x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(ceil_phrase(-x, -y), f2(-x, -y), rtol=RTOL, atol=ATOL)


# ------------------------------- copysign function -------------------------------#


def test_copysign_call(epyc_math_funcs_mod):
    copysign_call = math_funcs.copysign_call
    f1 = epyc_math_funcs_mod.copysign_call
    x = rand()
    y = rand()
    # Same sign
    assert isclose(copysign_call(x, y), f1(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(copysign_call(-x, -y), f1(-x, -y), rtol=RTOL, atol=ATOL)
    # Different sign
    assert isclose(copysign_call(-x, y), f1(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(copysign_call(x, -y), f1(x, -y), rtol=RTOL, atol=ATOL)
    # x =/= 0, y = 0 and x = 0, y =/= 0
    assert isclose(copysign_call(x, 0.0), f1(x, 0.0), rtol=RTOL, atol=ATOL)
    assert isclose(copysign_call(0.0, y), f1(0.0, y), rtol=RTOL, atol=ATOL)


def test_copysign_call_zero_case(epyc_math_funcs_mod):
    copysign_zero_case = math_funcs.copysign_zero_case
    f1 = epyc_math_funcs_mod.copysign_zero_case
    x = 0
    y = 0
    # Same sign
    assert isclose(copysign_zero_case(x, y), f1(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(copysign_zero_case(-x, -y), f1(-x, -y), rtol=RTOL, atol=ATOL)
    # Different sign
    assert isclose(copysign_zero_case(-x, y), f1(-x, y), rtol=RTOL, atol=ATOL)
    assert isclose(copysign_zero_case(x, -y), f1(x, -y), rtol=RTOL, atol=ATOL)


def test_copysign_return_type_1(epyc_math_funcs_mod):  # copysign
    """test type copysign(real, real) => should return real number"""

    copysign_return_type = math_funcs.copysign_return_type
    f1 = epyc_math_funcs_mod.copysign_return_type
    x = rand()  # real
    y = rand()  # real

    # Same sign
    assert isinstance(f1(x, y), type(copysign_return_type(x, y)))
    assert isinstance(f1(-x, -y), type(copysign_return_type(-x, -y)))
    # Different sign
    assert isinstance(f1(-x, y), type(copysign_return_type(-x, y)))
    assert isinstance(f1(x, -y), type(copysign_return_type(x, -y)))


def test_copysign_return_type_2(epyc_math_funcs_mod):  # copysign
    """test type copysign(int, int) => should return real type"""

    copysign_return_type = math_funcs.copysign_return_type_2
    f1 = epyc_math_funcs_mod.copysign_return_type_2
    high = 10000000
    x = randint(high)  # int
    y = randint(high)  # int

    # Same sign
    assert isinstance(f1(x, y), type(copysign_return_type(x, y)))
    assert isinstance(f1(-x, -y), type(copysign_return_type(-x, -y)))
    # Different sign
    assert isinstance(f1(-x, y), type(copysign_return_type(-x, y)))
    assert isinstance(f1(x, -y), type(copysign_return_type(x, -y)))


def test_copysign_return_type_3(epyc_math_funcs_mod):  # copysign
    """test type copysign(int, real) => should return real type"""

    copysign_return_type = math_funcs.copysign_return_type_3
    f1 = epyc_math_funcs_mod.copysign_return_type_3
    high = 10000000
    x = randint(high)  # int
    y = rand()  # real

    # Same sign
    assert isinstance(f1(x, y), type(copysign_return_type(x, y)))
    assert isinstance(f1(-x, -y), type(copysign_return_type(-x, -y)))
    # Different sign
    assert isinstance(f1(-x, y), type(copysign_return_type(-x, y)))
    assert isinstance(f1(x, -y), type(copysign_return_type(x, -y)))


def test_copysign_return_type_4(epyc_math_funcs_mod):  # copysign
    """test type copysign(real, int) => should return real type"""

    copysign_return_type = math_funcs.copysign_return_type_4
    f1 = epyc_math_funcs_mod.copysign_return_type_4
    high = 10000000
    x = rand()  # real
    y = randint(high)  # int

    # Same sign
    assert isinstance(f1(x, y), type(copysign_return_type(x, y)))
    assert isinstance(f1(-x, -y), type(copysign_return_type(-x, -y)))
    # Different sign
    assert isinstance(f1(-x, y), type(copysign_return_type(-x, y)))
    assert isinstance(f1(x, -y), type(copysign_return_type(x, -y)))


# ----------------------------- isfinite function -----------------------------#
@pytest.mark.skipif_by_language(
    True, language="fortran", reason="isfinite not implemented"
)
@pytest.mark.skipif(
    os.environ.get("PYCCEL_DEFAULT_COMPILER", None) == "intel",
    reason="Nan not correctly passed to intel function",
)
def test_isfinite_call(epyc_math_funcs_mod):  # isfinite
    isfinite_call = math_funcs.isfinite_call
    f1 = epyc_math_funcs_mod.isfinite_call
    x = rand()

    assert isfinite_call(x) == f1(x)

    # Test not a number
    assert isfinite_call(nan) == f1(nan)
    # Test infinite number
    assert isfinite_call(inf) == f1(inf)
    # Test negative infinite number
    assert isfinite_call(-inf) == f1(-inf)


# ------------------------------- isinf function ------------------------------#
@pytest.mark.skipif_by_language(
    True, language="fortran", reason="isinf not implemented"
)
def test_isinf_call(epyc_math_funcs_mod):  # isinf
    isinf_call = math_funcs.isinf_call
    f1 = epyc_math_funcs_mod.isinf_call
    x = rand()

    assert isinf_call(x) == f1(x)

    # Test not a number
    assert isinf_call(nan) == f1(nan)
    # Test infinite number
    assert isinf_call(inf) == f1(inf)
    # Test negative infinite number
    assert isinf_call(-inf) == f1(-inf)


# ------------------------------- isnan function ------------------------------#


def test_isnan_call(epyc_math_funcs_mod):  # isnan
    isnan_call = math_funcs.isnan_call
    f1 = epyc_math_funcs_mod.isnan_call
    x = rand()

    assert isnan_call(x) == f1(x)

    # Test not a number
    assert isnan_call(nan) == f1(nan)
    # Test infinite number
    assert isnan_call(inf) == f1(inf)
    # Test negative infinite number
    assert isnan_call(-inf) == f1(-inf)


# ------------------------------- ldexp function ------------------------------#
@pytest.mark.skipif_by_language(
    True, language="fortran", reason="ldexp not implemented"
)
def test_ldexp_call(epyc_math_funcs_mod):  # ldexp
    ldexp_call = math_funcs.ldexp_call
    f1 = epyc_math_funcs_mod.ldexp_call
    high = 100
    x = rand()
    exp = randint(high)

    assert isclose(ldexp_call(x, exp), f1(x, exp), rtol=RTOL, atol=ATOL)
    # Negative exponent
    assert isclose(ldexp_call(x, -exp), f1(x, -exp), rtol=RTOL, atol=ATOL)
    # Negative value
    assert isclose(ldexp_call(-x, exp), f1(-x, exp), rtol=RTOL, atol=ATOL)
    # Negative value and negative exponent
    assert isclose(ldexp_call(-x, -exp), f1(-x, -exp), rtol=RTOL, atol=ATOL)


@pytest.mark.skipif_by_language(
    True, language="fortran", reason="ldexp not implemented"
)
def test_ldexp_return_type(epyc_math_funcs_mod):  # ldexp
    ldexp_type = math_funcs.ldexp_type
    f1 = epyc_math_funcs_mod.ldexp_type
    high = 100
    x = rand()
    exp = randint(high)

    assert isinstance(ldexp_type(x, exp), type(f1(x, exp)))
    # Negative exponent
    assert isinstance(ldexp_type(x, -exp), type(f1(x, -exp)))
    # Negative value
    assert isinstance(ldexp_type(-x, exp), type(f1(-x, exp)))
    # Negative value and negative exponent
    assert isinstance(ldexp_type(-x, -exp), type(f1(-x, -exp)))


# --------------------------- remainder function ------------------------------#


@pytest.mark.skipif_by_language(
    True, language="fortran", reason="remainder not implemented"
)
def test_remainder_call(epyc_math_funcs_mod):  # remainder
    remainder_call = math_funcs.remainder_call
    f1 = epyc_math_funcs_mod.remainder_call
    x = rand()
    y = rand() + 1
    # Same sign
    assert isclose(remainder_call(x, y), f1(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(remainder_call(-x, -y), f1(-x, -y), rtol=RTOL, atol=ATOL)

    # Different sign
    assert isclose(remainder_call(x, -y), f1(x, -y), rtol=RTOL, atol=ATOL)
    assert isclose(remainder_call(-x, y), f1(-x, y), rtol=RTOL, atol=ATOL)


@pytest.mark.skipif_by_language(
    True, language="fortran", reason="remainder not implemented"
)
def test_remainder_return_type(epyc_math_funcs_mod):  # remainder
    remainder_type = math_funcs.remainder_type
    f1 = epyc_math_funcs_mod.remainder_type
    x = rand()
    y = rand()

    # Same sign
    assert isinstance(remainder_type(x, y), type(f1(x, y)))
    assert isinstance(remainder_type(-x, -y), type(f1(-x, -y)))

    # Different sign
    assert isinstance(remainder_type(x, -y), type(f1(x, -y)))
    assert isinstance(remainder_type(-x, y), type(f1(-x, y)))


# ----------------------------- trunc function --------------------------------#


def test_trunc_call(epyc_math_funcs_mod):  # trunc
    trunc_call = math_funcs.trunc_call
    f1 = epyc_math_funcs_mod.trunc_call
    x = uniform(high=10000.0)

    # positive number
    assert trunc_call(x) == f1(x)
    # Negative number
    assert trunc_call(-x) == f1(-x)


def test_trunc_call_int(epyc_math_funcs_mod):  # trunc
    trunc_call = math_funcs.trunc_call_int
    f1 = epyc_math_funcs_mod.trunc_call_int
    high = 10000
    x = randint(high)

    # positive number
    assert trunc_call(x) == f1(x)
    # Negative number
    assert trunc_call(-x) == f1(-x)


def test_trunc_return_type(epyc_math_funcs_mod):  # trunc
    trunc_type = math_funcs.trunc_type
    f1 = epyc_math_funcs_mod.trunc_type
    x = uniform(high=10000.0)

    assert isinstance(trunc_type((x)), type(f1((x))))
    assert isinstance(trunc_type(-x), type(f1(-x)))


# --------------------------- expm1 function ------------------------------#
@pytest.mark.skipif_by_language(
    True, language="fortran", reason="expm1 not implemented"
)
def test_expm1_call(epyc_math_funcs_mod):  # expm1
    expm1_call = math_funcs.expm1_call
    f1 = epyc_math_funcs_mod.expm1_call
    x = rand()
    assert isclose(f1(x), expm1_call(x), rtol=RTOL, atol=ATOL)


@pytest.mark.skipif_by_language(
    True, language="fortran", reason="expm1 not implemented"
)
def test_expm1_call_special_case(epyc_math_funcs_mod):  # expm1
    # should give result accurate to full precision better than exp()
    expm1_call = math_funcs.expm1_call_special_case
    x = 1e-5
    f1 = epyc_math_funcs_mod.expm1_call_special_case
    assert isclose(f1(x), expm1_call(x), rtol=RTOL, atol=ATOL)


@pytest.mark.skipif_by_language(
    True, language="fortran", reason="expm1 not implemented"
)
def test_expm1_phrase(epyc_math_funcs_mod):  # expm1
    expm1_phrase = math_funcs.expm1_phrase
    f2 = epyc_math_funcs_mod.expm1_phrase
    x = rand()
    y = rand()
    assert isclose(f2(x, y), expm1_phrase(x, y), rtol=RTOL, atol=ATOL)


@pytest.mark.skipif_by_language(
    True, language="fortran", reason="expm1 not implemented"
)
def test_expm1_return_type(epyc_math_funcs_mod):  # expm1 # expm1
    expm1_type = math_funcs.expm1_type
    f1 = epyc_math_funcs_mod.expm1_type
    x = uniform(high=700.0)

    assert isinstance(expm1_type(x), type(f1(x)))
    assert isinstance(expm1_type(-x), type(f1(-x)))


# --------------------------- log1p function ------------------------------#


@pytest.mark.skipif_by_language(
    True, language="fortran", reason="log1p not implemented"
)
def test_log1p_call(epyc_math_funcs_mod):
    log1p_call = math_funcs.log1p_call
    f1 = epyc_math_funcs_mod.log1p_call
    x = rand()
    assert isclose(f1(x), log1p_call(x), rtol=RTOL, atol=ATOL)
    assert isinstance(f1(x), type(log1p_call(x)))


@pytest.mark.skipif_by_language(
    True, language="fortran", reason="log1p not implemented"
)
def test_log1p_phrase(epyc_math_funcs_mod):
    log1p_phrase = math_funcs.log1p_phrase
    f2 = epyc_math_funcs_mod.log1p_phrase
    x = rand()
    y = rand()
    assert isclose(f2(x, y), log1p_phrase(x, y), rtol=RTOL, atol=ATOL)


# --------------------------- log2 function ------------------------------#
@pytest.mark.skipif_by_language(True, language="fortran", reason="log2 not implemented")
def test_log2_call(epyc_math_funcs_mod):
    log2_call = math_funcs.log2_call
    f1 = epyc_math_funcs_mod.log2_call
    low = min_float
    high = max_float
    x = uniform(low=low, high=high)
    assert isclose(f1(x), log2_call(x), rtol=RTOL, atol=ATOL)
    assert isinstance(f1(x), type(log2_call(x)))


@pytest.mark.skipif_by_language(True, language="fortran", reason="log2 not implemented")
def test_log2_phrase(epyc_math_funcs_mod):
    log2_phrase = math_funcs.log2_phrase
    f2 = epyc_math_funcs_mod.log2_phrase
    low = min_float
    high = max_float
    x = uniform(low=low, high=high)
    y = uniform(low=low, high=high)
    assert isclose(f2(x, y), log2_phrase(x, y), rtol=RTOL, atol=ATOL)


# --------------------------- log10 function ------------------------------#


def test_log10_call(epyc_math_funcs_mod):
    log10_call = math_funcs.log10_call
    f1 = epyc_math_funcs_mod.log10_call
    low = min_float
    high = max_float
    x = uniform(low=low, high=high)
    assert isclose(f1(x), log10_call(x), rtol=RTOL, atol=ATOL)
    assert isinstance(f1(x), type(log10_call(x)))


def test_log10_phrase(epyc_math_funcs_mod):
    log10_phrase = math_funcs.log10_phrase
    f2 = epyc_math_funcs_mod.log10_phrase
    low = min_float
    high = max_float
    x = uniform(low=low, high=high)
    y = uniform(low=low, high=high)
    assert isclose(f2(x, y), log10_phrase(x, y), rtol=RTOL, atol=ATOL)


# --------------------------------- Pow function ------------------------------#


def test_pow_call(epyc_math_funcs_mod):
    pow_call = math_funcs.pow_call
    f1 = epyc_math_funcs_mod.pow_call
    high = 10
    # case 1: x > 0
    x = uniform(low=min_float)
    y = uniform(low=-high, high=high)
    assert isclose(f1(x, y), pow_call(x, y), rtol=RTOL, atol=ATOL)
    assert isinstance(f1(x, y), type(pow_call(x, y)))

    # case 2: x = 0 and y > 0
    x = 0.0
    y = uniform(high=high)
    assert isclose(f1(x, y), pow_call(x, y), rtol=RTOL, atol=ATOL)

    # case 3: x < 0 and y is integer
    x = uniform(low=-high, high=0)
    y = randint(high)
    assert isclose(f1(x, y), pow_call(x, y), rtol=RTOL, atol=ATOL)


# ------------------------------- Hypot function ------------------------------#


def test_hypot_call(epyc_math_funcs_mod):
    hypot_call = math_funcs.hypot_call
    f1 = epyc_math_funcs_mod.hypot_call
    high = 10
    x = uniform(low=-high, high=high)
    y = uniform(low=-high, high=high)
    assert isclose(f1(x, y), hypot_call(x, y), rtol=RTOL, atol=ATOL)
    assert isinstance(f1(x, y), type(hypot_call(x, y)))


# ------------------------------- Acosh function ------------------------------#


def test_acosh_call(epyc_math_funcs_mod):
    acosh_call = math_funcs.acosh_call
    f1 = epyc_math_funcs_mod.acosh_call

    x = uniform(low=1, high=max_float)
    assert isclose(f1(x), acosh_call(x), rtol=RTOL, atol=ATOL)
    assert isinstance(f1(x), type(acosh_call(x)))


def test_acosh_phrase(epyc_math_funcs_mod):
    acosh_phrase = math_funcs.acosh_phrase
    f2 = epyc_math_funcs_mod.acosh_phrase

    x = uniform(low=1, high=max_float)
    y = uniform(low=1, high=max_float)
    assert isclose(f2(x, y), acosh_phrase(x, y), rtol=RTOL, atol=ATOL)


# ------------------------------- Asinh function ------------------------------#


def test_asinh_call(epyc_math_funcs_mod):
    asinh_call = math_funcs.asinh_call
    f1 = epyc_math_funcs_mod.asinh_call

    x = uniform(high=max_float)
    assert isclose(f1(x), asinh_call(x), rtol=RTOL, atol=ATOL)
    assert isinstance(f1(x), type(asinh_call(x)))

    # Negative value
    assert isclose(f1(-x), asinh_call(-x), rtol=RTOL, atol=ATOL)


def test_asinh_phrase(epyc_math_funcs_mod):
    asinh_phrase = math_funcs.asinh_phrase
    f2 = epyc_math_funcs_mod.asinh_phrase
    x = uniform(high=max_float)
    y = uniform(high=max_float)
    assert isclose(f2(x, y), asinh_phrase(x, y), rtol=RTOL, atol=ATOL)
    # Negative value
    assert isclose(f2(-x, -y), asinh_phrase(-x, -y), rtol=RTOL, atol=ATOL)


# ------------------------------- Atanh function ------------------------------#


def test_atanh_call(epyc_math_funcs_mod):
    atanh_call = math_funcs.atanh_call
    f1 = epyc_math_funcs_mod.atanh_call
    low = -1 + min_float
    high = 1 - min_float
    x = uniform(low=low, high=high)
    assert isclose(f1(x), atanh_call(x), rtol=RTOL, atol=ATOL)
    assert isinstance(f1(x), type(atanh_call(x)))


def test_atanh_phrase(epyc_math_funcs_mod):
    atanh_phrase = math_funcs.atanh_phrase
    f2 = epyc_math_funcs_mod.atanh_phrase

    # Domain ]-1, 1[
    low = -1 + min_float
    high = 1 - min_float
    x = uniform(low=low, high=high)
    y = uniform(low=low, high=high)
    assert isclose(f2(x, y), atanh_phrase(x, y), rtol=RTOL, atol=ATOL)


# --------------------------------- Erf function ------------------------------#


def test_erf_call(epyc_math_funcs_mod):
    erf_call = math_funcs.erf_call
    f1 = epyc_math_funcs_mod.erf_call

    # Domain ]-inf, +inf[
    x = uniform(high=max_float)
    assert isclose(f1(x), erf_call(x), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x), erf_call(-x), rtol=RTOL, atol=ATOL)
    assert isinstance(f1(x), type(erf_call(x)))


def test_erf_phrase(epyc_math_funcs_mod):
    erf_phrase = math_funcs.erf_phrase
    f2 = epyc_math_funcs_mod.erf_phrase

    # Domain ]-inf, +inf[
    x = uniform(high=max_float)
    y = uniform(high=max_float)
    assert isclose(f2(x, y), erf_phrase(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), erf_phrase(-x, -y), rtol=RTOL, atol=ATOL)


# -------------------------------- Erfc function ------------------------------#


def test_erfc_call(epyc_math_funcs_mod):
    erfc_call = math_funcs.erfc_call
    f1 = epyc_math_funcs_mod.erfc_call

    # Domain ]-inf, +inf[
    x = uniform(high=max_float)
    assert isclose(f1(x), erfc_call(x), rtol=RTOL, atol=ATOL)
    assert isclose(f1(-x), erfc_call(-x), rtol=RTOL, atol=ATOL)
    assert isinstance(f1(x), type(erfc_call(x)))


def test_erfc_phrase(epyc_math_funcs_mod):
    erfc_phrase = math_funcs.erfc_phrase
    f2 = epyc_math_funcs_mod.erfc_phrase

    # Domain ]-inf, +inf[
    x = uniform(high=max_float)
    y = uniform(high=max_float)
    assert isclose(f2(x, y), erfc_phrase(x, y), rtol=RTOL, atol=ATOL)
    assert isclose(f2(-x, -y), erfc_phrase(-x, -y), rtol=RTOL, atol=ATOL)


# -------------------------------- gamma function -----------------------------#


def test_gamma_call(epyc_math_funcs_mod):
    gamma_call = math_funcs.gamma_call
    f1 = epyc_math_funcs_mod.gamma_call

    # Domain ]0, +inf[ || (x < 0 and x.fraction not null)
    x = uniform(low=min_float)
    assert isclose(f1(x), gamma_call(x), rtol=RTOL, atol=ATOL)
    # make fractional part different from zero to test negative case
    if modf(x)[0] == 0:
        x += -0.1
    assert isclose(f1(-x), gamma_call(-x), rtol=RTOL, atol=ATOL)

    assert isinstance(f1(x), type(gamma_call(x)))


def test_gamma_phrase(epyc_math_funcs_mod):
    gamma_phrase = math_funcs.gamma_phrase
    f2 = epyc_math_funcs_mod.gamma_phrase

    # Domain ]0, +inf[ || (x < 0 and fractional part of x not null)
    x = uniform(low=min_float)
    y = uniform(low=min_float)
    assert isclose(f2(x, y), gamma_phrase(x, y), rtol=RTOL, atol=ATOL)


# ------------------------------- lgamma function -----------------------------#


def test_lgamma_call(epyc_math_funcs_mod):
    lgamma_call = math_funcs.lgamma_call
    f1 = epyc_math_funcs_mod.lgamma_call

    # Domain ]0, +inf[ || (x < 0 and x.fraction not null)
    x = uniform(low=min_float)
    assert isclose(f1(x), lgamma_call(x), rtol=RTOL, atol=ATOL)
    _, f = modf(x)
    # make fractional part different from zero to test negative case
    if f == 0:
        x += -0.1
    assert isclose(f1(-x), lgamma_call(-x), rtol=RTOL, atol=ATOL)
    assert isinstance(f1(x), type(lgamma_call(x)))


def test_lgamma_phrase(epyc_math_funcs_mod):
    lgamma_phrase = math_funcs.lgamma_phrase
    f2 = epyc_math_funcs_mod.lgamma_phrase

    # Domain ]0, +inf[ || (x < 0 and fractional part of x not null)
    x = uniform(low=min_float)
    y = uniform(low=min_float)
    assert isclose(f2(x, y), lgamma_phrase(x, y), rtol=RTOL, atol=ATOL)
