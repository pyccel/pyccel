# pylint: disable=missing-function-docstring, missing-module-docstring
import pytest
import numpy as np

from modules import base
from utilities import epyccel_module_with_fallback


@pytest.fixture(scope="module")
def epyc_base_mod(language):
    return epyccel_module_with_fallback(base, language)


def test_is_false(epyc_base_mod):
    f = base.is_false
    epyc_f = epyc_base_mod.is_false
    np.equal(f(True), epyc_f(True))
    np.equal(f(False), epyc_f(False))


def test_is_true(epyc_base_mod):
    f = base.is_true
    epyc_f = epyc_base_mod.is_true
    np.equal(f(True), epyc_f(True))
    np.equal(f(False), epyc_f(False))


def test_compare_is(epyc_base_mod):
    f = base.compare_is
    epyc_f = epyc_base_mod.compare_is
    np.equal(f(True, True), epyc_f(True, True))
    np.equal(f(True, False), epyc_f(True, False))
    np.equal(f(False, True), epyc_f(False, True))
    np.equal(f(False, False), epyc_f(False, False))


def test_compare_is_not(epyc_base_mod):
    f = base.compare_is_not
    epyc_f = epyc_base_mod.compare_is_not
    np.equal(f(True, True), epyc_f(True, True))
    np.equal(f(True, False), epyc_f(True, False))
    np.equal(f(False, True), epyc_f(False, True))
    np.equal(f(False, False), epyc_f(False, False))


def test_compare_is_int(epyc_base_mod):
    f = base.compare_is_int
    epyc_f = epyc_base_mod.compare_is_int
    np.equal(f(True, 1), epyc_f(True, 1))
    np.equal(f(True, 0), epyc_f(True, 0))
    np.equal(f(False, 1), epyc_f(False, 1))
    np.equal(f(False, 0), epyc_f(False, 0))


def test_compare_is_not_int(epyc_base_mod):
    f = base.compare_is_not_int
    epyc_f = epyc_base_mod.compare_is_not_int
    np.equal(f(True, 1), epyc_f(True, 1))
    np.equal(f(True, 0), epyc_f(True, 0))
    np.equal(f(False, 1), epyc_f(False, 1))
    np.equal(f(False, 0), epyc_f(False, 0))


def test_not_false(epyc_base_mod):
    f = base.not_false
    epyc_f = epyc_base_mod.not_false
    np.equal(f(True), epyc_f(True))
    np.equal(f(False), epyc_f(False))


def test_not_true(epyc_base_mod):
    f = base.not_true
    epyc_f = epyc_base_mod.not_true
    np.equal(f(True), epyc_f(True))
    np.equal(f(False), epyc_f(False))


def test_eq_false(epyc_base_mod):
    f = base.eq_false
    epyc_f = epyc_base_mod.eq_false
    np.equal(f(True), epyc_f(True))
    np.equal(f(False), epyc_f(False))


def test_eq_true(epyc_base_mod):
    f = base.eq_true
    epyc_f = epyc_base_mod.eq_true
    np.equal(f(True), epyc_f(True))
    np.equal(f(False), epyc_f(False))


def test_neq_false(epyc_base_mod):
    f = base.eq_false
    epyc_f = epyc_base_mod.eq_false
    np.equal(f(True), epyc_f(True))
    np.equal(f(False), epyc_f(False))


def test_neq_true(epyc_base_mod):
    f = base.eq_true
    epyc_f = epyc_base_mod.eq_true
    np.equal(f(True), epyc_f(True))
    np.equal(f(False), epyc_f(False))


def test_not(epyc_base_mod):
    f = base.not_val
    epyc_f = epyc_base_mod.not_val
    np.equal(f(True), epyc_f(True))
    np.equal(f(False), epyc_f(False))


def test_not_int(epyc_base_mod):
    f = base.not_int
    epyc_f = epyc_base_mod.not_int
    np.equal(f(0), epyc_f(0))
    np.equal(f(4), epyc_f(4))


def test_compare_is_nil(epyc_base_mod):
    f = base.is_nil
    epyc_f = epyc_base_mod.is_nil
    np.equal(f(None), epyc_f(None))


def test_compare_is_not_nil(epyc_base_mod):
    f = base.is_not_nil
    epyc_f = epyc_base_mod.is_not_nil
    np.equal(f(None), epyc_f(None))


def test_cast_int(epyc_base_mod):
    f = base.cast_int
    epyc_f = epyc_base_mod.cast_int
    np.equal(f(4), epyc_f(4))
    f = base.cast_float_to_int
    epyc_f = epyc_base_mod.cast_float_to_int
    np.equal(f(4.5), epyc_f(4.5))


def test_cast_bool(epyc_base_mod):
    f = base.cast_bool
    epyc_f = epyc_base_mod.cast_bool
    np.equal(f(True), epyc_f(True))


def test_cast_float(epyc_base_mod):
    f = base.cast_float
    epyc_f = epyc_base_mod.cast_float
    np.equal(f(4.5), epyc_f(4.5))
    f = base.cast_int_to_float
    epyc_f = epyc_base_mod.cast_int_to_float
    np.equal(f(4), epyc_f(4))


def test_if_0_int(epyc_base_mod):
    f = base.if_0_int
    epyc_f = epyc_base_mod.if_0_int
    np.equal(f(22), epyc_f(22))
    np.equal(f(0), epyc_f(0))


def test_if_0_real(epyc_base_mod):
    f = base.if_0_real
    epyc_f = epyc_base_mod.if_0_real
    np.equal(f(22.3), epyc_f(22.3))
    np.equal(f(0.0), epyc_f(0.0))


def test_same_int(epyc_base_mod):
    f = base.is_same_int
    epyc_f = epyc_base_mod.is_same_int
    np.equal(f(22), epyc_f(22))
    f = base.isnot_same_int
    epyc_f = epyc_base_mod.isnot_same_int
    np.equal(f(22), epyc_f(22))


def test_same_float(epyc_base_mod):
    f = base.is_same_float
    epyc_f = epyc_base_mod.is_same_float
    np.equal(f(22.2), epyc_f(22.2))
    f = base.isnot_same_float
    epyc_f = epyc_base_mod.isnot_same_float
    np.equal(f(22.2), epyc_f(22.2))


def test_same_string(epyc_base_mod):
    f = base.is_same_string
    epyc_f = epyc_base_mod.is_same_string
    np.equal(f(), epyc_f())
    f = base.isnot_same_string
    epyc_f = epyc_base_mod.isnot_same_string
    np.equal(f(), epyc_f())


def test_same_complex(epyc_base_mod):
    f = base.is_same_complex
    epyc_f = epyc_base_mod.is_same_complex
    np.equal(f(complex(2, 3)), epyc_f(complex(2, 3)))
    f = base.isnot_same_complex
    epyc_f = epyc_base_mod.isnot_same_complex
    np.equal(f(complex(2, 3)), epyc_f(complex(2, 3)))


def test_is_types(epyc_base_mod):
    f = base.is_types
    epyc_f = epyc_base_mod.is_types
    np.equal(f(1, 1.0), epyc_f(1, 1.0))


def test_isnot_types(epyc_base_mod):
    f = base.isnot_types
    epyc_f = epyc_base_mod.isnot_types
    np.equal(f(1, 1.0), epyc_f(1, 1.0))


def test_none_is_none(epyc_base_mod):
    f = base.none_is_none
    epyc_f = epyc_base_mod.none_is_none
    np.equal(f(), epyc_f())


def test_none_isnot_none(epyc_base_mod):
    f = base.none_isnot_none
    epyc_f = epyc_base_mod.none_isnot_none
    np.equal(f(), epyc_f())


def test_pass_if(epyc_base_mod):
    f = base.pass_if
    epyc_f = epyc_base_mod.pass_if
    np.equal(f(2), epyc_f(2))


def test_pass2_if(epyc_base_mod):
    f = base.pass2_if
    epyc_f = epyc_base_mod.pass2_if
    np.equal(f(0.2), epyc_f(0.2))
    np.equal(f(0.0), epyc_f(0.0))


def test_use_optional(epyc_base_mod):
    f = base.use_optional
    epyc_f = epyc_base_mod.use_optional
    np.equal(f(), epyc_f())
    np.equal(f(6), epyc_f(6))


def test_check_optional(epyc_base_mod):
    f = base.check_optional
    epyc_f = epyc_base_mod.check_optional
    np.equal(f(), epyc_f())
    np.equal(f(6), epyc_f(6))
    np.equal(f(4), epyc_f(4))
    np.equal(f(1), epyc_f(1))


def test_none_equality(epyc_base_mod):
    f = base.none_equality
    epyc_f = epyc_base_mod.none_equality
    np.equal(f(), epyc_f())
    np.equal(f(6), epyc_f(6))


def test_none_none_equality(epyc_base_mod):
    f = base.none_none_equality
    epyc_f = epyc_base_mod.none_none_equality
    np.equal(f(), epyc_f())


def test_none_literal_equality(epyc_base_mod):
    f = base.none_literal_equality
    epyc_f = epyc_base_mod.none_literal_equality
    np.equal(f(), epyc_f())
