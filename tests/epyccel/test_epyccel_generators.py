# pylint: disable=missing-function-docstring, missing-module-docstring
from typing import TypeVar

import numpy as np
import pytest
from modules import epyccel_generators
from numpy.random import randint
from utilities import epyccel_module_with_fallback

from pyccel import epyccel


@pytest.fixture(scope="module")
def epyc_epyccel_generators_mod(language):
    return epyccel_module_with_fallback(epyccel_generators, language)


def test_sum_range(epyc_epyccel_generators_mod):
    f = epyccel_generators.sum_range
    n = randint(1, 50)
    x = np.array(randint(100, size=n), dtype=int)

    f_epyc = epyc_epyccel_generators_mod.sum_range

    assert f(x) == f_epyc(x)


def test_sum_var(epyc_epyccel_generators_mod):
    f = epyccel_generators.sum_var
    n = randint(1, 50)
    x = np.array(randint(100, size=n), dtype=int)

    f_epyc = epyc_epyccel_generators_mod.sum_var

    assert f(x) == f_epyc(x)


def test_sum_var2(epyc_epyccel_generators_mod):
    f = epyccel_generators.sum_var2
    n1 = randint(1, 10)
    n2 = randint(1, 10)
    x = np.array(randint(10, size=(n1, n2)), dtype=int)

    f_epyc = epyc_epyccel_generators_mod.sum_var2

    assert f(x) == f_epyc(x)


def test_sum_var3(epyc_epyccel_generators_mod):
    f = epyccel_generators.sum_var3
    n1 = randint(1, 10)
    n2 = randint(1, 10)
    n3 = randint(1, 10)
    x = np.array(randint(10, size=(n1, n2, n3)), dtype=int)

    f_epyc = epyc_epyccel_generators_mod.sum_var3

    assert f(x) == f_epyc(x)


def test_sum_var4(epyc_epyccel_generators_mod):
    f = epyccel_generators.sum_var4
    n = randint(1, 50)
    x = np.array(randint(100, size=n), dtype=int)

    f_epyc = epyc_epyccel_generators_mod.sum_var4

    assert f(x) == f_epyc(x)


def test_sum_var5(epyc_epyccel_generators_mod):
    f = epyccel_generators.sum_var5
    n = randint(1, 50)
    x = np.ones(n, dtype=bool)

    f_epyc = epyc_epyccel_generators_mod.sum_var5

    assert f(x) == f_epyc(x)


@pytest.mark.parametrize(
    "language",
    (
        pytest.param("fortran", marks=pytest.mark.fortran),
        pytest.param(
            "c",
            marks=[
                pytest.mark.xfail(reason="Max not implemented in C for integers"),
                pytest.mark.c,
            ],
        ),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_max(language):
    def f():
        return max(i if i > k else k for i in range(5) for k in range(10))

    f_epyc = epyccel(f, language=language)

    assert f() == f_epyc()


@pytest.mark.parametrize(
    "language",
    (
        pytest.param("fortran", marks=pytest.mark.fortran),
        pytest.param(
            "c",
            marks=[
                pytest.mark.xfail(reason="Min not implemented in C for integers"),
                pytest.mark.c,
            ],
        ),
        pytest.param("python", marks=pytest.mark.python),
    ),
)
def test_min(language):
    def f():
        return min(
            k if i > k else 0 if i == k else i for i in range(5) for k in range(10)
        )

    f_epyc = epyccel(f, language=language)

    assert f() == f_epyc()


def test_expression1(epyc_epyccel_generators_mod):
    f = epyccel_generators.expression1
    n = randint(1, 10)
    x = np.array(randint(100, size=n), dtype=float)

    f_epyc = epyc_epyccel_generators_mod.expression1

    assert np.isclose(f(x), f_epyc(x), rtol=1e-14, atol=1e-14)


def test_expression2(epyc_epyccel_generators_mod):
    f = epyccel_generators.expression2
    n = randint(1, 10)
    x = randint(100, size=n).astype(np.int64)

    f_epyc = epyc_epyccel_generators_mod.expression2

    assert f(x) == f_epyc(x)


def test_nested_generators1(epyc_epyccel_generators_mod):
    f = epyccel_generators.nested_generators1
    x = randint(0, 50, size=(5, 5, 5, 5)).astype(float)

    f_epyc = epyc_epyccel_generators_mod.nested_generators1

    assert f(x) == f_epyc(x)


def test_nested_generators2(epyc_epyccel_generators_mod):
    f = epyccel_generators.nested_generators2
    x = randint(0, 50, size=(5, 5, 5, 5)).astype(float)

    f_epyc = epyc_epyccel_generators_mod.nested_generators2

    assert f(x) == f_epyc(x)


def test_nested_generators3(epyc_epyccel_generators_mod):
    f = epyccel_generators.nested_generators3
    x = randint(0, 10, size=(5, 5, 5, 5)).astype(float)

    f_epyc = epyc_epyccel_generators_mod.nested_generators3

    assert f(x) == f_epyc(x)


def test_nested_generators4(epyc_epyccel_generators_mod):
    f = epyccel_generators.nested_generators4
    x = randint(0, 10, size=(5, 5, 5, 5)).astype(float)

    f_epyc = epyc_epyccel_generators_mod.nested_generators4

    assert f(x) == f_epyc(x)


def test_sum_range_overwrite(epyc_epyccel_generators_mod):
    f = epyccel_generators.sum_range_overwrite
    n = randint(1, 50)
    x = np.array(randint(100, size=n), dtype=int)

    f_epyc = epyc_epyccel_generators_mod.sum_range_overwrite

    assert f(x) == f_epyc(x)


def test_sum_with_condition(epyc_epyccel_generators_mod):
    f = epyccel_generators.sum_with_condition
    f_epyc = epyc_epyccel_generators_mod.sum_with_condition
    assert f() == f_epyc()


def test_sum_with_multiple_conditions(epyc_epyccel_generators_mod):
    f = epyccel_generators.sum_with_multiple_conditions
    f_epyc = epyc_epyccel_generators_mod.sum_with_multiple_conditions
    assert f() == f_epyc()


def test_max_with_condition(epyc_epyccel_generators_mod):
    f = epyccel_generators.max_with_condition
    f_epyc = epyc_epyccel_generators_mod.max_with_condition
    assert f() == f_epyc()


def test_max_with_condition_float(epyc_epyccel_generators_mod):
    f = epyccel_generators.max_with_condition_float
    f_epyc = epyc_epyccel_generators_mod.max_with_condition_float
    assert f() == f_epyc()


def test_max_with_multiple_conditions(epyc_epyccel_generators_mod):
    f = epyccel_generators.max_with_multiple_conditions
    f_epyc = epyc_epyccel_generators_mod.max_with_multiple_conditions
    assert f() == f_epyc()


def test_min_with_condition(epyc_epyccel_generators_mod):
    f = epyccel_generators.min_with_condition
    f_epyc = epyc_epyccel_generators_mod.min_with_condition
    assert f() == f_epyc()


def test_min_with_condition_float(epyc_epyccel_generators_mod):
    f = epyccel_generators.min_with_condition_float
    f_epyc = epyc_epyccel_generators_mod.min_with_condition_float
    assert f() == f_epyc()


def test_min_with_multiple_conditions(epyc_epyccel_generators_mod):
    f = epyccel_generators.min_with_multiple_conditions
    f_epyc = epyc_epyccel_generators_mod.min_with_multiple_conditions
    assert f() == f_epyc()


def test_sum_with_two_variables(epyc_epyccel_generators_mod):
    f = epyccel_generators.sum_with_two_variables
    f_epyc = epyc_epyccel_generators_mod.sum_with_two_variables

    assert f() == f_epyc()


def test_min_max_values(epyc_epyccel_generators_mod):
    f = epyccel_generators.min_max_values
    f_epyc = epyc_epyccel_generators_mod.min_max_values

    for dtype in (np.int16, np.int32, np.int64, np.float32, np.float64):
        x = randint(0, 100, size=(5,)).astype(dtype)
        print(f(x))
        print(f_epyc(x))
        assert f(x) == f_epyc(x)
