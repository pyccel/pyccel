# pylint: disable=missing-function-docstring, missing-module-docstring
import sys

import pytest

from pyccel import epyccel
from modules import bitwise

@pytest.mark.parametrize("a, b", [(True, False), (True, True)])
def test_right_shift_b_b(language, a, b):
    f1 = bitwise.right_shift_b_b
    f2 = epyccel(f1, language=language)
    r1 = f1(a, b)
    r2 = f2(a, b)
    assert r1 == r2
    assert type(r1) is type(r2)

@pytest.mark.parametrize("a, b", [(1, 1), (1, 2)])
def test_right_shift_i_i(language, a, b):
    f1 = bitwise.right_shift_i_i
    f2 = epyccel(f1, language=language)
    r1 = f1(a, b)
    r2 = f2(a, b)
    assert r1 == r2
    assert type(r1) is type(r2)

@pytest.mark.parametrize("a, b", [(True, 2), (True, 1)])
def test_right_shift_b_i(language, a, b):
    f1 = bitwise.right_shift_b_i
    f2 = epyccel(f1, language=language)
    r1 = f1(a, b)
    r2 = f2(a, b)
    assert r1 == r2
    assert type(r1) is type(r2)

@pytest.mark.parametrize("a, b", [(True, 1), (True, 2)])
def test_left_shift_b_i(language, a, b):
    f1 = bitwise.right_shift_b_i
    f2 = epyccel(f1, language=language)
    r1 = f1(a, b)
    r2 = f2(a, b)
    assert r1 == r2
    assert type(r1) is type(r2)

@pytest.mark.parametrize("a, b", [(1, 1), (1, 2)])
def test_left_shift_i_i(language, a, b):
    f1 = bitwise.right_shift_i_i
    f2 = epyccel(f1, language=language)
    r1 = f1(a, b)
    r2 = f2(a, b)
    assert r1 == r2
    assert type(r1) is type(r2)

@pytest.mark.parametrize("a, b", [(True, False), (True, True)])
def test_left_shift_b_b(language, a, b):
    f1 = bitwise.right_shift_b_b
    f2 = epyccel(f1, language=language)
    r1 = f1(a, b)
    r2 = f2(a, b)
    assert r1 == r2
    assert type(r1) is type(r2)

def test_bit_xor_b_b(language):
    f1 = bitwise.bit_xor_b_b
    f2 = epyccel(f1, language=language)
    r1 = f1(True, False)
    r2 = f2(True, False)
    assert r1 == r2
    assert type(r1) is type(r2)

@pytest.mark.parametrize("a, b, c", [(True, False, False), (True, True, True)])
def test_bit_xor_b_b_b(language, a, b, c):
    f1 = bitwise.bit_xor_b_b_b
    f2 = epyccel(f1, language=language)
    r1 = f1(a, b, c)
    r2 = f2(a, b, c)
    assert r1 == r2

@pytest.mark.parametrize("a, b", [(1, 1), (1, 2)])
def test_bit_xor_i_i(language, a, b):
    f1 = bitwise.bit_xor_i_i
    f2 = epyccel(f1, language=language)
    r1 = f1(a, b)
    r2 = f2(a, b)
    assert r1 == r2
    assert type(r1) is type(r2)

@pytest.mark.parametrize("a, b", [(False, 2), (True, 1)])
def test_bit_xor_b_i(language, a, b):
    f1 = bitwise.bit_xor_b_i
    f2 = epyccel(f1, language=language)
    r1 = f1(a, b)
    r2 = f2(a, b)
    assert r1 == r2
    assert type(r1) is type(r2)

@pytest.mark.parametrize("a, b", [(1, False), (1, True)])
def test_bit_or_i_b(language, a, b):
    f1 = bitwise.bit_or_i_b
    f2 = epyccel(f1, language=language)
    r1 = f1(a, b)
    r2 = f2(a, b)
    assert r1 == r2
    assert type(r1) is type(r2)

@pytest.mark.parametrize("a, b", [(1, 1), (1, 2)])
def test_bit_or_i_i(language, a, b):
    f1 = bitwise.bit_or_i_i
    f2 = epyccel(f1, language=language)
    r1 = f1(a, b)
    r2 = f2(a, b)
    assert r1 == r2
    assert type(r1) is type(r2)

def test_bit_or_b_b(language):
    f1 = bitwise.bit_or_b_b
    f2 = epyccel(f1, language=language)
    r1 = f1(False, True)
    r2 = f2(False, True)
    assert r1 == r2
    assert type(r1) is type(r2)

@pytest.mark.parametrize("a, b", [(1, True), (1, False)])
def test_bit_and_i_b(language, a, b):
    f1 = bitwise.bit_and_i_b
    f2 = epyccel(f1, language=language)
    r1 = f1(a, b)
    r2 = f2(a, b)
    assert r1 == r2
    assert type(r1) is type(r2)

@pytest.mark.parametrize("a, b", [(1, 1), (1, 2)])
def test_bit_and_i_i(language, a, b):
    f1 = bitwise.bit_and_i_i
    f2 = epyccel(f1, language=language)
    r1 = f1(a, b)
    r2 = f2(a, b)
    assert r1 == r2
    assert type(r1) is type(r2)

def test_bit_and_b_b(language):
    f1 = bitwise.bit_and_b_b
    f2 = epyccel(f1, language=language)
    r1 = f1(True, True)
    r2 = f2(True, True)
    assert r1 == r2
    assert type(r1) is type(r2)

@pytest.mark.parametrize("a, b, c", [(1, 0, 4), (1, 0, 4)])
def test_bit_and_i_i_i(language, a, b, c):
    f1 = bitwise.bit_and_i_i_i
    f2 = epyccel(f1, language=language)
    r1 = f1(a, b, c)
    r2 = f2(a, b, c)
    assert r1 == r2
    assert type(r1) is type(r2)

@pytest.mark.parametrize("a, b, c", [(True, True, 4), (True, False, 4)])
def test_bit_and_b_b_i(language, a, b, c):
    f1 = bitwise.bit_and_b_b_i
    f2 = epyccel(f1, language=language)
    r1 = f1(a, b, c)
    r2 = f2(a, b, c)
    assert r1 == r2
    assert type(r1) is type(r2)

@pytest.mark.skipif(
    sys.version_info >= (3, 16),
    reason="Bitwise inversion of bools was removed in Python 3.16",
)
@pytest.mark.filterwarnings("ignore:.*Bitwise inversion*:DeprecationWarning")
def test_invert_b(language):
    f1 = bitwise.invert_b
    f2 = epyccel(f1, language=language)
    for a in [True, False]:
        r1 = f1(a)
        r2 = f2(a)
        assert r1 == r2
        assert type(r1) is type(r2)

def test_invert_i(language):
    f1 = bitwise.invert_i
    f2 = epyccel(f1, language=language)
    for a in [0, 1, 60, -45]:
        r1 = f1(a)
        r2 = f2(a)
        assert r1 == r2
        assert type(r1) is type(r2)

def test_or_ints(language):
    f1 = bitwise.or_ints
    f2 = epyccel(f1, language=language)
    for a in [0, 1, 60, -45]:
        r1 = f1(a)
        r2 = f2(a)
        assert r1 == r2
        assert type(r1) is type(r2)
