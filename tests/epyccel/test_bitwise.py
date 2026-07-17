# pylint: disable=missing-function-docstring, missing-module-docstring
import sys

import pytest
from modules import bitwise
from epyccel_utilities import epyccel_module_with_fallback

from pyccel import epyccel


@pytest.fixture(scope="module")
def epyc_bitwise_mod(language):
    return epyccel_module_with_fallback(bitwise, language)


@pytest.mark.parametrize("a, b", [(True, False), (True, True)])
def test_right_shift_b_b(epyc_bitwise_mod, a, b):
    f1 = bitwise.right_shift_b_b
    f2 = epyc_bitwise_mod.right_shift_b_b
    r1 = f1(a, b)
    r2 = f2(a, b)
    assert r1 == r2
    assert type(r1) is type(r2)


@pytest.mark.parametrize("a, b", [(1, 1), (1, 2)])
def test_right_shift_i_i(epyc_bitwise_mod, a, b):
    f1 = bitwise.right_shift_i_i
    f2 = epyc_bitwise_mod.right_shift_i_i
    r1 = f1(a, b)
    r2 = f2(a, b)
    assert r1 == r2
    assert type(r1) is type(r2)


@pytest.mark.parametrize("a, b", [(True, 2), (True, 1)])
def test_right_shift_b_i(epyc_bitwise_mod, a, b):
    f1 = bitwise.right_shift_b_i
    f2 = epyc_bitwise_mod.right_shift_b_i
    r1 = f1(a, b)
    r2 = f2(a, b)
    assert r1 == r2
    assert type(r1) is type(r2)


@pytest.mark.parametrize("a, b", [(True, 1), (True, 2)])
def test_left_shift_b_i(epyc_bitwise_mod, a, b):
    f1 = bitwise.right_shift_b_i
    f2 = epyc_bitwise_mod.right_shift_b_i
    r1 = f1(a, b)
    r2 = f2(a, b)
    assert r1 == r2
    assert type(r1) is type(r2)


@pytest.mark.parametrize("a, b", [(1, 1), (1, 2)])
def test_left_shift_i_i(epyc_bitwise_mod, a, b):
    f1 = bitwise.right_shift_i_i
    f2 = epyc_bitwise_mod.right_shift_i_i
    r1 = f1(a, b)
    r2 = f2(a, b)
    assert r1 == r2
    assert type(r1) is type(r2)


@pytest.mark.parametrize("a, b", [(True, False), (True, True)])
def test_left_shift_b_b(epyc_bitwise_mod, a, b):
    f1 = bitwise.right_shift_b_b
    f2 = epyc_bitwise_mod.right_shift_b_b
    r1 = f1(a, b)
    r2 = f2(a, b)
    assert r1 == r2
    assert type(r1) is type(r2)


def test_bit_xor_b_b(epyc_bitwise_mod):
    f1 = bitwise.bit_xor_b_b
    f2 = epyc_bitwise_mod.bit_xor_b_b
    r1 = f1(True, False)
    r2 = f2(True, False)
    assert r1 == r2
    assert type(r1) is type(r2)


@pytest.mark.parametrize("a, b, c", [(True, False, False), (True, True, True)])
def test_bit_xor_b_b_b(epyc_bitwise_mod, a, b, c):
    f1 = bitwise.bit_xor_b_b_b
    f2 = epyc_bitwise_mod.bit_xor_b_b_b
    r1 = f1(a, b, c)
    r2 = f2(a, b, c)
    assert r1 == r2


@pytest.mark.parametrize("a, b", [(1, 1), (1, 2)])
def test_bit_xor_i_i(epyc_bitwise_mod, a, b):
    f1 = bitwise.bit_xor_i_i
    f2 = epyc_bitwise_mod.bit_xor_i_i
    r1 = f1(a, b)
    r2 = f2(a, b)
    assert r1 == r2
    assert type(r1) is type(r2)


@pytest.mark.parametrize("a, b", [(False, 2), (True, 1)])
def test_bit_xor_b_i(epyc_bitwise_mod, a, b):
    f1 = bitwise.bit_xor_b_i
    f2 = epyc_bitwise_mod.bit_xor_b_i
    r1 = f1(a, b)
    r2 = f2(a, b)
    assert r1 == r2
    assert type(r1) is type(r2)


@pytest.mark.parametrize("a, b", [(1, False), (1, True)])
def test_bit_or_i_b(epyc_bitwise_mod, a, b):
    f1 = bitwise.bit_or_i_b
    f2 = epyc_bitwise_mod.bit_or_i_b
    r1 = f1(a, b)
    r2 = f2(a, b)
    assert r1 == r2
    assert type(r1) is type(r2)


@pytest.mark.parametrize("a, b", [(1, 1), (1, 2)])
def test_bit_or_i_i(epyc_bitwise_mod, a, b):
    f1 = bitwise.bit_or_i_i
    f2 = epyc_bitwise_mod.bit_or_i_i
    r1 = f1(a, b)
    r2 = f2(a, b)
    assert r1 == r2
    assert type(r1) is type(r2)


def test_bit_or_b_b(epyc_bitwise_mod):
    f1 = bitwise.bit_or_b_b
    f2 = epyc_bitwise_mod.bit_or_b_b
    r1 = f1(False, True)
    r2 = f2(False, True)
    assert r1 == r2
    assert type(r1) is type(r2)


@pytest.mark.parametrize("a, b", [(1, True), (1, False)])
def test_bit_and_i_b(epyc_bitwise_mod, a, b):
    f1 = bitwise.bit_and_i_b
    f2 = epyc_bitwise_mod.bit_and_i_b
    r1 = f1(a, b)
    r2 = f2(a, b)
    assert r1 == r2
    assert type(r1) is type(r2)


@pytest.mark.parametrize("a, b", [(1, 1), (1, 2)])
def test_bit_and_i_i(epyc_bitwise_mod, a, b):
    f1 = bitwise.bit_and_i_i
    f2 = epyc_bitwise_mod.bit_and_i_i
    r1 = f1(a, b)
    r2 = f2(a, b)
    assert r1 == r2
    assert type(r1) is type(r2)


def test_bit_and_b_b(epyc_bitwise_mod):
    f1 = bitwise.bit_and_b_b
    f2 = epyc_bitwise_mod.bit_and_b_b
    r1 = f1(True, True)
    r2 = f2(True, True)
    assert r1 == r2
    assert type(r1) is type(r2)


@pytest.mark.parametrize("a, b, c", [(1, 0, 4), (1, 0, 4)])
def test_bit_and_i_i_i(epyc_bitwise_mod, a, b, c):
    f1 = bitwise.bit_and_i_i_i
    f2 = epyc_bitwise_mod.bit_and_i_i_i
    r1 = f1(a, b, c)
    r2 = f2(a, b, c)
    assert r1 == r2
    assert type(r1) is type(r2)


@pytest.mark.parametrize("a, b, c", [(True, True, 4), (True, False, 4)])
def test_bit_and_b_b_i(epyc_bitwise_mod, a, b, c):
    f1 = bitwise.bit_and_b_b_i
    f2 = epyc_bitwise_mod.bit_and_b_b_i
    r1 = f1(a, b, c)
    r2 = f2(a, b, c)
    assert r1 == r2
    assert type(r1) is type(r2)


@pytest.mark.skipif(
    sys.version_info >= (3, 16),
    reason="Bitwise inversion of bools was removed in Python 3.16",
)
@pytest.mark.filterwarnings("ignore:.*Bitwise inversion*:DeprecationWarning")
def test_invert_b(epyc_bitwise_mod):
    f1 = bitwise.invert_b
    f2 = epyc_bitwise_mod.invert_b
    for a in [True, False]:
        r1 = f1(a)
        r2 = f2(a)
        assert r1 == r2
        assert type(r1) is type(r2)


def test_invert_i(epyc_bitwise_mod):
    f1 = bitwise.invert_i
    f2 = epyc_bitwise_mod.invert_i
    for a in [0, 1, 60, -45]:
        r1 = f1(a)
        r2 = f2(a)
        assert r1 == r2
        assert type(r1) is type(r2)


def test_or_ints(epyc_bitwise_mod):
    f1 = bitwise.or_ints
    f2 = epyc_bitwise_mod.or_ints
    for a in [0, 1, 60, -45]:
        r1 = f1(a)
        r2 = f2(a)
        assert r1 == r2
        assert type(r1) is type(r2)
