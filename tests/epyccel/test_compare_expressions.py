# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np
import pytest
from modules import compare_expressions
from epyccel_utilities import epyccel_module_with_fallback


@pytest.fixture(scope="module")
def epyc_compare_expressions_mod(language):
    return epyccel_module_with_fallback(compare_expressions, language)


# ==============================================================================
def test_mod_eq_pow(epyc_compare_expressions_mod):
    f = compare_expressions.mod_eq_pow
    epyc_f = epyc_compare_expressions_mod.mod_eq_pow
    # True
    np.equal(f(10, 3, 1), epyc_f(10, 3, 1))
    np.equal(f(19, 10, 3), epyc_f(19, 10, 3))
    np.equal(f(21, 3, 0), epyc_f(21, 3, 0))
    # False
    np.equal(f(10, 5, 2), epyc_f(10, 5, 2))
    np.equal(f(19, 10, 1), epyc_f(19, 10, 1))
    np.equal(f(21, 3, 1), epyc_f(21, 3, 1))


def test_mod_neq_pow(epyc_compare_expressions_mod):
    f = compare_expressions.mod_neq_pow
    epyc_f = epyc_compare_expressions_mod.mod_neq_pow
    # True
    np.equal(f(10, 5, 2), epyc_f(10, 5, 2))
    np.equal(f(19, 10, 1), epyc_f(19, 10, 1))
    np.equal(f(21, 3, 1), epyc_f(21, 3, 1))
    # False
    np.equal(f(10, 3, 1), epyc_f(10, 3, 1))
    np.equal(f(19, 10, 3), epyc_f(19, 10, 3))
    np.equal(f(21, 3, 0), epyc_f(21, 3, 0))


def test_idiv_gt_add(epyc_compare_expressions_mod):
    f = compare_expressions.idiv_gt_add
    epyc_f = epyc_compare_expressions_mod.idiv_gt_add
    # True
    np.equal(f(10, 3, 2), epyc_f(10, 3, 2))
    np.equal(f(8, 2, 2), epyc_f(8, 2, 2))
    np.equal(f(16, 3, 4), epyc_f(16, 3, 4))
    # False
    np.equal(f(10, 3, 2), epyc_f(10, 3, 2))
    np.equal(f(8, 2, 3), epyc_f(8, 2, 3))
    np.equal(f(16, 3, 5), epyc_f(16, 3, 5))


def test_in_range(epyc_compare_expressions_mod):
    f = compare_expressions.in_range
    epyc_f = epyc_compare_expressions_mod.in_range
    # True
    np.equal(f(0.0, 1.0, 2.0), epyc_f(0.0, 1.0, 2.0))
    np.equal(f(-2.0, -1.0, 2.0), epyc_f(-2.0, -1.0, 2.0))
    np.equal(f(-2.0, -1.3, -1.0), epyc_f(-2.0, -1.3, -1.0))
    # False
    np.equal(f(0.0, 10.0, 2.0), epyc_f(0.0, 10.0, 2.0))
    np.equal(f(-2.0, -10.0, 2.0), epyc_f(-2.0, -10.0, 2.0))
    np.equal(f(-2.0, -0.3, -1.0), epyc_f(-2.0, -0.3, -1.0))
