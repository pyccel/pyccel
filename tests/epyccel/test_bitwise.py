# pylint: disable=missing-function-docstring, missing-module-docstring/
import pytest

from pyccel.epyccel import epyccel
from modules import bitwise

@pytest.mark.parametrize("a, b",[(True, False),(True, True)])
def test_right_shift_b_b(language, a, b):
    f1 = bitwise.right_shift_b_b
    f2 = epyccel( f1, language = language )
    assert f1(a, b) == f2(a, b)
    assert(type(f1(a, b)) == type(f2(a, b))) # pylint: disable=unidiomatic-typecheck

@pytest.mark.parametrize("a, b",[(1, 1),(1, 2)])
def test_right_shift_i_i(language, a, b):
    f1 = bitwise.right_shift_i_i
    f2 = epyccel( f1, language = language )
    assert f1(a, b) == f2(a, b)
    assert(type(f1(a, b)) == type(f2(a, b))) # pylint: disable=unidiomatic-typecheck

@pytest.mark.parametrize("a, b",[(True, 2),(True, 1)])
def test_right_shift_b_i(language, a, b):
    f1 = bitwise.right_shift_b_i
    f2 = epyccel( f1, language = language )
    assert f1(a, b) == f2(a, b)
    assert(type(f1(a, b)) == type(f2(a, b))) # pylint: disable=unidiomatic-typecheck

@pytest.mark.parametrize("a, b",[(True, 1),(True, 2)])
def test_left_shift_b_i(language, a, b):
    f1 = bitwise.right_shift_b_i
    f2 = epyccel( f1, language = language )
    assert f1(a, b) == f2(a, b)
    assert(type(f1(a, b)) == type(f2(a, b))) # pylint: disable=unidiomatic-typecheck

@pytest.mark.parametrize("a, b",[(1, 1),(1, 2)])
def test_left_shift_i_i(language, a, b):
    f1 = bitwise.right_shift_i_i
    f2 = epyccel( f1, language = language )
    assert f1(a, b) == f2(a, b)
    assert(type(f1(a, b)) == type(f2(a, b))) # pylint: disable=unidiomatic-typecheck

@pytest.mark.parametrize("a, b",[(True, False),(True, True)])
def test_left_shift_b_b(language, a, b):
    f1 = bitwise.right_shift_b_b
    f2 = epyccel( f1, language = language )
    assert f1(a, b) == f2(a, b)
    assert(type(f1(a, b)) == type(f2(a, b))) # pylint: disable=unidiomatic-typecheck

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="problem in wrapping issue #370"),
            pytest.mark.fortran]
        )
    ]
)
def test_bit_xor_b_b(language):
    f1 = bitwise.bit_xor_b_b
    f2 = epyccel( f1, language = language )
    assert f1(True, False) == f2(True, False)
    assert(type(f1(True, False)) == type(f2(True, False))) # pylint: disable=unidiomatic-typecheck

@pytest.mark.parametrize("a, b, c",[(True, False, False),(True, True, True)])
def test_bit_xor_b_b_b(language, a, b, c):
    f1 = bitwise.bit_xor_b_b_b
    f2 = epyccel( f1, language = language )
    assert f1(a, b, c) == f2(a, b, c)

@pytest.mark.parametrize("a, b",[(1, 1),(1, 2)])
def test_bit_xor_i_i(language, a, b):
    f1 = bitwise.bit_xor_i_i
    f2 = epyccel( f1, language = language )
    assert f1(a, b) == f2(a, b)
    assert(type(f1(a, b)) == type(f2(a, b))) # pylint: disable=unidiomatic-typecheck

@pytest.mark.parametrize("a, b",[(False, 2), (True, 1)])
def test_bit_xor_b_i(language, a, b):
    f1 = bitwise.bit_xor_b_i
    f2 = epyccel( f1, language = language )
    assert f1(a, b) == f2(a, b)
    assert(type(f1(a, b)) == type(f2(a, b))) # pylint: disable=unidiomatic-typecheck

@pytest.mark.parametrize("a, b",[(1, False),(1, True)])
def test_bit_or_i_b(language, a, b):
    f1 = bitwise.bit_or_i_b
    f2 = epyccel( f1, language = language )
    assert f1(a, b) == f2(a, b)
    assert(type(f1(a, b)) == type(f2(a, b))) # pylint: disable=unidiomatic-typecheck

@pytest.mark.parametrize("a, b",[(1, 1),(1, 2)])
def test_bit_or_i_i(language, a, b):
    f1 = bitwise.bit_or_i_i
    f2 = epyccel( f1, language = language )
    assert f1(a, b) == f2(a, b)
    assert(type(f1(a, b)) == type(f2(a, b))) # pylint: disable=unidiomatic-typecheck

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="problem in wrapping issue #370"),
            pytest.mark.fortran]
        )
    ]
)
def test_bit_or_b_b(language):
    f1 = bitwise.bit_or_b_b
    f2 = epyccel( f1, language = language )
    assert f1(False, True) == f2(False, True)
    assert(type(f1(False, True)) == type(f2(False, True))) # pylint: disable=unidiomatic-typecheck

@pytest.mark.parametrize("a, b",[(1, True),(1, False)])
def test_bit_and_i_b(language, a, b):
    f1 = bitwise.bit_and_i_b
    f2 = epyccel( f1, language = language )
    assert f1(a, b) == f2(a, b)
    assert(type(f1(a, b)) == type(f2(a, b))) # pylint: disable=unidiomatic-typecheck

@pytest.mark.parametrize("a, b",[(1, 1),(1, 2)])
def test_bit_and_i_i(language, a, b):
    f1 = bitwise.bit_and_i_i
    f2 = epyccel( f1, language = language )
    assert f1(a, b) == f2(a, b)
    assert(type(f1(a, b)) == type(f2(a, b))) # pylint: disable=unidiomatic-typecheck


@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="problem in wrapping issue #370"),
            pytest.mark.fortran]
        )
    ]
)
def test_bit_and_b_b(language):
    f1 = bitwise.bit_and_b_b
    f2 = epyccel( f1, language = language )
    assert f1(True, True) == f2(True, True)
    assert(type(f1(True, True)) == type(f2(True, True))) # pylint: disable=unidiomatic-typecheck

@pytest.mark.parametrize("a, b, c",[(1, 0, 4), (1, 0, 4)])
def test_bit_and_i_i_i(language, a, b, c):
    f1 = bitwise.bit_and_i_i_i
    f2 = epyccel( f1, language = language )
    assert f1(a, b, c) == f2(a, b, c)
    assert(type(f1(a, b, c)) == type(f2(a, b, c))) # pylint: disable=unidiomatic-typecheck

@pytest.mark.parametrize("a, b, c",[(True, True, 4), (True, False, 4)])
def test_bit_and_b_b_i(language, a, b, c):
    f1 = bitwise.bit_and_b_b_i
    f2 = epyccel( f1, language = language )
    assert f1(a, b, c) == f2(a, b, c)
    assert(type(f1(a, b, c)) == type(f2(a, b, c))) # pylint: disable=unidiomatic-typecheck

@pytest.mark.parametrize("a",[True, False])
def test_invert_b(language, a):
    f1 = bitwise.invert_b
    f2 = epyccel( f1, language = language )
    assert f1(a) == f2(a)
    assert(type(f1(a)) == type(f2(a))) # pylint: disable=unidiomatic-typecheck

@pytest.mark.parametrize("a",[1, 0])
def test_invert_i(language, a):
    f1 = bitwise.invert_i
    f2 = epyccel( f1, language = language )
    assert f1(a) == f2(a)
    assert(type(f1(a)) == type(f2(a))) # pylint: disable=unidiomatic-typecheck
