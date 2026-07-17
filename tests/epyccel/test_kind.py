# pylint: disable=missing-function-docstring, missing-module-docstring
import pytest
from modules import kind
from epyccel_utilities import epyccel_module_with_fallback

from pyccel import epyccel


@pytest.fixture(scope="module")
def epyc_kind_mod(language):
    return epyccel_module_with_fallback(kind, language)


def test_or_boolean(epyc_kind_mod):
    or_bool = kind.or_bool
    epyc_or_bool = epyc_kind_mod.or_bool

    assert epyc_or_bool(True, True) == or_bool(True, True)
    assert epyc_or_bool(True, False) == or_bool(True, False)
    assert epyc_or_bool(False, False) == or_bool(False, False)


def test_real_greater_bool(epyc_kind_mod):
    real_greater_bool = kind.real_greater_bool
    epyc_real_greater_bool = epyc_kind_mod.real_greater_bool

    assert real_greater_bool(1.0, 2.0) == epyc_real_greater_bool(1.0, 2.0)
    assert real_greater_bool(1.5, 1.2) == epyc_real_greater_bool(1.5, 1.2)


# Skip test if PYCCEL_DEFAULT_COMPILER=LLVM
@pytest.mark.skip_llvm
def test_input_output_matching_types(language):
    def add_real(a: "float", b: "float"):
        c = a + b
        return c

    flags = "-Werror -Wconversion"
    if language == "fortran":
        flags = flags + "-extra"

    epyc_add_real = epyccel(add_real, flags=flags, language=language)

    assert add_real(1.0, 2.0) == epyc_add_real(1.0, 2.0)


def test_output_types_1(epyc_kind_mod):
    cast_to_int = kind.cast_to_int
    f = epyc_kind_mod.cast_to_int
    assert type(cast_to_int(5.2)) == type(
        f(5.2)
    )  # pylint: disable=unidiomatic-typecheck


def test_output_types_2(epyc_kind_mod):
    cast_to_float = kind.cast_to_float
    f = epyc_kind_mod.cast_to_float
    assert type(cast_to_float(5)) == type(f(5))  # pylint: disable=unidiomatic-typecheck


def test_output_types_3(epyc_kind_mod):
    cast_to_bool = kind.cast_to_bool
    f = epyc_kind_mod.cast_to_bool
    assert cast_to_bool(1) == f(1)
