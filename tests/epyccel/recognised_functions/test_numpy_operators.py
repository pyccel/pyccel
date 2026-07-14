# pylint: disable=missing-function-docstring, missing-module-docstring
import os

import numpy as np
import pytest
from modules import numpy_operators
from numpy.random import randint
from utilities import epyccel_module_with_fallback

from pyccel import epyccel


@pytest.fixture(scope="module")
def epyc_numpy_operators_mod(language):
    return epyccel_module_with_fallback(numpy_operators, language)


int_types = (bool, np.int8, np.int64)


def test_numpy_bit_and(epyc_numpy_operators_mod):
    f = numpy_operators.numpy_bit_and_1
    g = numpy_operators.numpy_bit_and_2
    h = numpy_operators.numpy_bit_and_3
    epyc_f = epyc_numpy_operators_mod.numpy_bit_and_1
    epyc_g = epyc_numpy_operators_mod.numpy_bit_and_2
    epyc_h = epyc_numpy_operators_mod.numpy_bit_and_3

    for t_x in int_types:
        for t_y in int_types:
            x = randint(2 if t_x is bool else 127, size=(2, 3, 4), dtype=t_x)
            y = randint(2 if t_y is bool else 127, size=(2, 3, 4), dtype=t_y)

            z = f(x, y)
            epyc_z = epyc_f(x, y)
            assert np.array_equal(epyc_z, z)
            assert z.dtype is epyc_z.dtype

            y = randint(2 if t_y is bool else 127, dtype=t_y)

            z2 = g(x, y)
            epyc_z2 = epyc_g(x, y)
            assert np.array_equal(epyc_z2, z2)
            assert z2.dtype is epyc_z2.dtype

            z3 = h(y, x)
            epyc_z3 = epyc_h(y, x)
            assert np.array_equal(epyc_z3, z3)
            assert z3.dtype is epyc_z3.dtype


def test_numpy_bit_or(epyc_numpy_operators_mod):
    f = numpy_operators.numpy_bit_or_1
    g = numpy_operators.numpy_bit_or_2
    h = numpy_operators.numpy_bit_or_3
    epyc_f = epyc_numpy_operators_mod.numpy_bit_or_1
    epyc_g = epyc_numpy_operators_mod.numpy_bit_or_2
    epyc_h = epyc_numpy_operators_mod.numpy_bit_or_3

    for t_x in int_types:
        for t_y in int_types:
            x = randint(2 if t_x is bool else 127, size=(2, 3, 4), dtype=t_x)
            y = randint(2 if t_y is bool else 127, size=(2, 3, 4), dtype=t_y)

            z = f(x, y)
            epyc_z = epyc_f(x, y)
            assert np.array_equal(epyc_z, z)
            assert z.dtype is epyc_z.dtype

            y = randint(2 if t_y is bool else 127, dtype=t_y)

            z2 = g(x, y)
            epyc_z2 = epyc_g(x, y)
            assert np.array_equal(epyc_z2, z2)
            assert z2.dtype is epyc_z2.dtype

            z3 = h(y, x)
            epyc_z3 = epyc_h(y, x)
            assert np.array_equal(epyc_z3, z3)
            assert z3.dtype is epyc_z3.dtype


def test_numpy_bit_xor(epyc_numpy_operators_mod):
    f = numpy_operators.numpy_bit_xor_1
    g = numpy_operators.numpy_bit_xor_2
    h = numpy_operators.numpy_bit_xor_3
    epyc_f = epyc_numpy_operators_mod.numpy_bit_xor_1
    epyc_g = epyc_numpy_operators_mod.numpy_bit_xor_2
    epyc_h = epyc_numpy_operators_mod.numpy_bit_xor_3

    for t_x in int_types:
        for t_y in int_types:
            x = randint(2 if t_x is bool else 127, size=(2, 3, 4), dtype=t_x)
            y = randint(2 if t_y is bool else 127, size=(2, 3, 4), dtype=t_y)

            z = f(x, y)
            epyc_z = epyc_f(x, y)
            assert np.array_equal(epyc_z, z)
            assert z.dtype is epyc_z.dtype

            y = randint(2 if t_y is bool else 127, dtype=t_y)

            z2 = g(x, y)
            epyc_z2 = epyc_g(x, y)
            assert np.array_equal(epyc_z2, z2)
            assert z2.dtype is epyc_z2.dtype

            z3 = h(y, x)
            epyc_z3 = epyc_h(y, x)
            assert np.array_equal(epyc_z3, z3)
            assert z3.dtype is epyc_z3.dtype


def test_numpy_bit_lshift(epyc_numpy_operators_mod):
    f = numpy_operators.numpy_bit_lshift_1
    g = numpy_operators.numpy_bit_lshift_2
    h = numpy_operators.numpy_bit_lshift_3
    epyc_f = epyc_numpy_operators_mod.numpy_bit_lshift_1
    epyc_g = epyc_numpy_operators_mod.numpy_bit_lshift_2
    epyc_h = epyc_numpy_operators_mod.numpy_bit_lshift_3

    for t_x in int_types:
        for t_y in int_types:
            x = randint(2 if t_x is bool else 32, size=(2, 3, 4), dtype=t_x)
            y = randint(2 if t_y is bool else 5, size=(2, 3, 4), dtype=t_y)

            z = f(x, y)
            epyc_z = epyc_f(x, y)
            assert np.array_equal(epyc_z, z)
            assert z.dtype is epyc_z.dtype

            y = randint(2 if t_y is bool else 5, dtype=t_y)

            z2 = g(x, y)
            epyc_z2 = epyc_g(x, y)
            assert np.array_equal(epyc_z2, z2)
            assert z2.dtype is epyc_z2.dtype

            x = randint(2 if t_x is bool else 32, dtype=t_x)
            y = randint(2 if t_y is bool else 5, size=(2, 3, 4), dtype=t_y)

            z3 = h(x, y)
            epyc_z3 = epyc_h(x, y)
            assert np.array_equal(epyc_z3, z3)
            assert z3.dtype is epyc_z3.dtype


def test_numpy_bit_rshift(epyc_numpy_operators_mod):
    f = numpy_operators.numpy_bit_rshift_1
    g = numpy_operators.numpy_bit_rshift_2
    h = numpy_operators.numpy_bit_rshift_3
    epyc_f = epyc_numpy_operators_mod.numpy_bit_rshift_1
    epyc_g = epyc_numpy_operators_mod.numpy_bit_rshift_2
    epyc_h = epyc_numpy_operators_mod.numpy_bit_rshift_3

    for t_x in int_types:
        for t_y in int_types:
            x = randint(2 if t_x is bool else 32, size=(2, 3, 4), dtype=t_x)
            y = randint(2 if t_y is bool else 5, size=(2, 3, 4), dtype=t_y)

            z = f(x, y)
            epyc_z = epyc_f(x, y)
            assert np.array_equal(epyc_z, z)
            assert z.dtype is epyc_z.dtype

            y = randint(2 if t_y is bool else 5, dtype=t_y)

            z2 = g(x, y)
            epyc_z2 = epyc_g(x, y)
            assert np.array_equal(epyc_z2, z2)
            assert z2.dtype is epyc_z2.dtype

            x = randint(2 if t_x is bool else 32, dtype=t_x)
            y = randint(2 if t_y is bool else 5, size=(2, 3, 4), dtype=t_y)

            z3 = h(x, y)
            epyc_z3 = epyc_h(x, y)
            assert np.array_equal(epyc_z3, z3)
            assert z3.dtype is epyc_z3.dtype


@pytest.mark.skipif(
    os.environ.get("PYCCEL_DEFAULT_COMPILER", "GNU") == "intel",
    reason="Intel's invert implementation does not seem to match.",
)
def test_numpy_bit_invert(epyc_numpy_operators_mod):
    f = numpy_operators.numpy_bit_invert
    epyc_f = epyc_numpy_operators_mod.numpy_bit_invert

    for t_x in int_types:
        x = randint(2 if t_x is bool else 127, size=(2, 3, 4), dtype=t_x)

        z = f(x)
        epyc_z = epyc_f(x)
        assert np.array_equal(epyc_z, z)
        assert z.dtype is epyc_z.dtype
