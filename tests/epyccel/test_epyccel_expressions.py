# pylint: disable=missing-function-docstring, missing-module-docstring
import pytest
import numpy as np
from numpy import iinfo
from numpy.random import rand, randint

from pyccel import epyccel
from modules import epyccel_expressions
from utilities import epyccel_module_with_fallback


@pytest.fixture(scope="module")
def epyc_epyccel_expressions_mod(language):
    return epyccel_module_with_fallback(epyccel_expressions, language)



# Use int32 for Windows compatibility
min_int = iinfo(np.int32).min
max_int = iinfo(np.int32).max


def test_swap_basic(epyc_epyccel_expressions_mod):
    swp = epyccel_expressions.swp
    f = epyc_epyccel_expressions_mod.swp
    assert f(2, 4) == swp(2, 4)
    assert f(-2, 4) == swp(-2, 4)
    assert f(4, 100) == swp(4, 100)
    x = randint(min_int, max_int)
    y = randint(min_int, max_int)
    assert f(x, y) == swp(x, y)


def test_swap_basic_2(epyc_epyccel_expressions_mod):
    swp = epyccel_expressions.swap_basic_2
    f = epyc_epyccel_expressions_mod.swap_basic_2
    assert f(2, 4) == swp(2, 4)
    assert f(-2, 4) == swp(-2, 4)
    assert f(4, 100) == swp(4, 100)
    x = randint(min_int, max_int)
    y = randint(min_int, max_int)
    assert f(x, y) == swp(x, y)


def test_swap_basic_3(epyc_epyccel_expressions_mod):
    swp = epyccel_expressions.swap_basic_3
    f = epyc_epyccel_expressions_mod.swap_basic_3
    assert f(2, 4, 8) == swp(2, 4, 8)
    assert f(-2, 4, -6) == swp(-2, 4, -6)
    assert f(4, 100, 234) == swp(4, 100, 234)
    x = randint(min_int, max_int)
    y = randint(min_int, max_int)
    z = randint(min_int, max_int)
    assert f(x, y, z) == swp(x, y, z)


def test_swap_basic_4(epyc_epyccel_expressions_mod):
    swp = epyccel_expressions.swap_basic_4
    f = epyc_epyccel_expressions_mod.swap_basic_4
    assert f(2, 4, 8) == swp(2, 4, 8)
    assert f(-2, 4, -6) == swp(-2, 4, -6)
    assert f(4, 100, 234) == swp(4, 100, 234)
    x = randint(min_int, max_int)
    y = randint(min_int, max_int)
    z = randint(min_int, max_int)
    assert f(x, y, z) == swp(x, y, z)


def test_swap_index_1(epyc_epyccel_expressions_mod):
    swp = epyccel_expressions.swap_index_1
    f = epyc_epyccel_expressions_mod.swap_index_1
    assert f(2, 4, 8) == swp(2, 4, 8)
    assert f(-2, 4, -6) == swp(-2, 4, -6)
    assert f(4, 100, 234) == swp(4, 100, 234)
    x = randint(min_int, max_int)
    y = randint(min_int, max_int)
    z = randint(min_int, max_int)
    assert f(x, y, z) == swp(x, y, z)


def test_swap_index_2(epyc_epyccel_expressions_mod):
    swp = epyccel_expressions.swap_index_2
    f = epyc_epyccel_expressions_mod.swap_index_2
    assert f(0, 1) == swp(0, 1)
    assert f(1, 0) == swp(1, 0)
    assert f(2, 1) == swp(2, 1)


def test_multi_level_swap(epyc_epyccel_expressions_mod):
    swp = epyccel_expressions.multi_level_swap
    f = epyc_epyccel_expressions_mod.multi_level_swap
    assert f(2, 4, 8) == swp(2, 4, 8)
    assert f(-2, 4, -6) == swp(-2, 4, -6)
    assert f(4, 100, 234) == swp(4, 100, 234)
    x = randint(min_int, max_int)
    y = randint(min_int, max_int)
    z = randint(min_int, max_int)
    assert f(x, y, z) == swp(x, y, z)


def test_multi_type_swap(epyc_epyccel_expressions_mod):
    swp = epyccel_expressions.multi_type_swap
    b = randint(min_int, max_int)
    d = randint(min_int, max_int)
    a = rand() * 100
    c = rand() * 100

    f = epyc_epyccel_expressions_mod.multi_type_swap
    assert f(a, b, c, d) == swp(a, b, c, d)
    assert f(-2.0, 4, -6.0, 10) == swp(-2.0, 4, -6.0, 10)


def test_tuple_assign(epyc_epyccel_expressions_mod):
    tup_assign = epyccel_expressions.tup_assign
    f = epyc_epyccel_expressions_mod.tup_assign
    assert f(2, 4) == tup_assign(2, 4)
    assert f(-2, 4) == tup_assign(-2, 4)
    assert f(4, 100) == tup_assign(4, 100)
    x = randint(min_int // 2, max_int // 2)
    y = randint(min_int // 2, max_int // 2)
    assert f(x, y) == tup_assign(x, y)


def test_tuple_assign2(epyc_epyccel_expressions_mod):
    tup_assign = epyccel_expressions.tuple_assign2
    f = epyc_epyccel_expressions_mod.tuple_assign2
    assert f(2, 4) == tup_assign(2, 4)
    assert f(-2, 4) == tup_assign(-2, 4)
    assert f(4, 100) == tup_assign(4, 100)
    x = randint(min_int // 2, max_int // 2)
    y = randint(min_int // 2, max_int // 2)
    assert f(x, y) == tup_assign(x, y)


def test_tuple_assign3(epyc_epyccel_expressions_mod):
    tup_assign = epyccel_expressions.tuple_assign3
    f = epyc_epyccel_expressions_mod.tuple_assign3
    assert f(2) == tup_assign(2)
    assert f(-2) == tup_assign(-2)
    assert f(40) == tup_assign(40)
    x = randint(min_int, max_int - 5)
    assert f(x) == tup_assign(x)
