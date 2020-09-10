import pytest
import numpy as np

from pyccel.epyccel import epyccel
from modules import bitwise
from conftest import *

@pytest.mark.parametrize("a, b",[(1, 1),(1, 2)])
def test_right_shift(language, a, b):
    f1 = bitwise.right_shift
    f2 = epyccel( f1, language = language, verbose = True )
    assert f1(a, b) == f2(a, b)

@pytest.mark.parametrize("a, b",[(1, 1),(1, 2)])
def test_left_shift(language, a, b):
    f1 = bitwise.right_shift
    f2 = epyccel( f1, language = language, verbose = True )
    assert f1(a, b) == f2(a, b)

@pytest.mark.parametrize("a, b",[(1, 1),(1, 2)])
def test_bit_xor(language, a, b):
    f1 = bitwise.bit_xor
    f2 = epyccel( f1, language = language, verbose = True )
    assert f1(a, b) == f2(a, b)

@pytest.mark.parametrize("a, b",[(1, 1),(1, 2)])
def test_bit_or(language, a, b):
    f1 = bitwise.bit_or
    f2 = epyccel( f1, language = language, verbose = True )
    assert f1(a, b) == f2(a, b)

@pytest.mark.parametrize("a, b",[(1, 1),(1, 2)])
def test_bit_and_f2(language, a, b):
    f1 = bitwise.bit_and_f2
    f2 = epyccel( f1, language = language, verbose = True )
    assert f1(a, b) == f2(a, b)

@pytest.mark.parametrize("a",[1, 0])
def test_invert(language, a):
    f1 = bitwise.invert
    f2 = epyccel( f1, language = language, verbose = True )
    assert f1(a) == f2(a)

@pytest.mark.parametrize("a, b, c",[(1, 0, 4), (1, 0, 4)])
def test_bit_and_f3(language, a, b, c):
    f1 = bitwise.bit_and_f3
    f2 = epyccel( f1, language = language, verbose = True )
    assert f1(a, b, c) == f2(a, b, c)
