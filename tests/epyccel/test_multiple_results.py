# pylint: disable=missing-function-docstring, missing-module-docstring
import pytest
from modules import multiple_results

from epyccel_utilities import compare_epyccel, epyccel_module_with_fallback

# ==============================================================================


@pytest.fixture(scope="module")
def epyc_multiple_results_mod(language):
    return epyccel_module_with_fallback(multiple_results, language)


# ==============================================================================
def test_const_int_float(epyc_multiple_results_mod):
    compare_epyccel(
        multiple_results.const_int_float, epyc_multiple_results_mod.const_int_float
    )


# ...
def test_const_complex_bool_int(epyc_multiple_results_mod):
    compare_epyccel(
        multiple_results.const_complex_bool_int,
        epyc_multiple_results_mod.const_complex_bool_int,
    )


# ...
def test_expr_float_int_bool(epyc_multiple_results_mod):
    compare_epyccel(
        multiple_results.expr_complex_int_bool,
        epyc_multiple_results_mod.expr_complex_int_bool,
        3,
    )
