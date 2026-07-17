# pylint: disable=missing-function-docstring, missing-module-docstring
# coding: utf-8

import numpy as np
import pytest
from modules import epyccel_decorators

from pyccel import epyccel

from epyccel_utilities import epyccel_module_with_fallback


@pytest.fixture(scope="module")
def epyc_epyccel_decorators_mod(language):
    return epyccel_module_with_fallback(epyccel_decorators, language)


@pytest.mark.skipif_by_language(
    True, language="python", reason="Can't hide functions in Python"
)
def test_private(epyc_epyccel_decorators_mod):
    # Attribute error when extracting f from module
    with pytest.raises(AttributeError):
        epyc_epyccel_decorators_mod.hidden


def test_inline_1_out(epyc_epyccel_decorators_mod):
    f = epyccel_decorators.inline_1_out
    g = epyc_epyccel_decorators_mod.inline_1_out

    assert f() == g()


def test_inline_0_out(epyc_epyccel_decorators_mod):
    f = epyccel_decorators.inline_0_out
    g = epyc_epyccel_decorators_mod.inline_0_out

    x = np.ones(4, dtype=int)
    y = np.ones(4, dtype=int)

    f(x)
    g(y)

    assert all(x == y)


def test_inline_local(epyc_epyccel_decorators_mod):
    f = epyccel_decorators.inline_local
    g = epyc_epyccel_decorators_mod.inline_local

    assert f() == g()


def test_inline_local_name_clash(epyc_epyccel_decorators_mod):
    f = epyccel_decorators.inline_local_name_clash
    g = epyc_epyccel_decorators_mod.inline_local_name_clash

    assert f() == g()


def test_inline_optional(epyc_epyccel_decorators_mod):
    f = epyccel_decorators.inline_optional
    g = epyc_epyccel_decorators_mod.inline_optional

    assert f() == g()


def test_inline_array(epyc_epyccel_decorators_mod):
    f = epyccel_decorators.inline_array
    g = epyc_epyccel_decorators_mod.inline_array

    assert f() == g()


def test_nested_inline_call(epyc_epyccel_decorators_mod):
    f = epyccel_decorators.nested_inline_call
    g = epyc_epyccel_decorators_mod.nested_inline_call

    assert f() == g()


def test_inline_return(epyc_epyccel_decorators_mod):
    f = epyccel_decorators.inline_return
    g = epyc_epyccel_decorators_mod.inline_return

    assert f() == g()


def test_inline_multiple_results(epyc_epyccel_decorators_mod):
    f = epyccel_decorators.inline_multiple_results
    g = epyc_epyccel_decorators_mod.inline_multiple_results

    assert f() == g()


def test_inline_literal_return(epyc_epyccel_decorators_mod):
    f = epyccel_decorators.inline_literal_return
    g = epyc_epyccel_decorators_mod.inline_literal_return

    assert f() == g()


def test_inline_array_return(epyc_epyccel_decorators_mod):
    f = epyccel_decorators.inline_array_return
    g = epyc_epyccel_decorators_mod.inline_array_return

    out_pyth = f()
    out_pycc = g()
    assert np.array_equal(out_pyth[0], out_pycc[0])
    assert out_pyth[1] == out_pycc[1]


def test_inline_multiple_return(epyc_epyccel_decorators_mod):
    f = epyccel_decorators.inline_multiple_return
    g = epyc_epyccel_decorators_mod.inline_multiple_return

    assert f() == g()


def test_inline_homogeneous_tuple_result(epyc_epyccel_decorators_mod):
    f = epyccel_decorators.inline_homogeneous_tuple_result
    g = epyc_epyccel_decorators_mod.inline_homogeneous_tuple_result

    assert f() == g()


def test_inline_inhomogeneous_tuple_result(epyc_epyccel_decorators_mod):
    f = epyccel_decorators.inline_inhomogeneous_tuple_result
    g = epyc_epyccel_decorators_mod.inline_inhomogeneous_tuple_result

    assert f() == g()


def test_inhomogeneous_tuple_in_inline(epyc_epyccel_decorators_mod):
    f = epyccel_decorators.inhomogeneous_tuple_in_inline
    g = epyc_epyccel_decorators_mod.inhomogeneous_tuple_in_inline

    assert f() == g()


def test_multi_level_inhomogeneous_tuple_in_inline(epyc_epyccel_decorators_mod):
    f = epyccel_decorators.multi_level
    g = epyc_epyccel_decorators_mod.multi_level

    assert f() == g()


def test_indexed_template(epyc_epyccel_decorators_mod):
    my_sum = epyccel_decorators.my_sum
    pyccel_sum = epyc_epyccel_decorators_mod.my_sum

    x = np.ones(4, dtype=float)

    python_fl = my_sum(x)
    pyccel_fl = pyccel_sum(x)

    assert python_fl == pyccel_fl
    assert isinstance(python_fl, type(pyccel_fl))

    y = np.full(4, 1 + 3j)

    python_cmplx = my_sum(y)
    pyccel_cmplx = pyccel_sum(y)

    assert python_cmplx == pyccel_cmplx
    assert isinstance(python_cmplx, type(pyccel_cmplx))


@pytest.mark.parametrize(
    "language",
    (
        pytest.param(
            "fortran",
            marks=[
                pytest.mark.skip(reason="lists not implemented in fortran"),
                pytest.mark.fortran,
            ],
        ),
        pytest.param("c", marks=pytest.mark.c),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_allow_negative_index_list(language):
    def allow_negative_index_annotation():
        a = [1, 2, 3, 4]
        return a[-1], a[-2], a[-3], a[0]

    epyc_allow_negative_index_annotation = epyccel(
        allow_negative_index_annotation, language=language
    )

    assert epyc_allow_negative_index_annotation() == allow_negative_index_annotation()
    assert isinstance(
        epyc_allow_negative_index_annotation(), type(allow_negative_index_annotation())
    )
