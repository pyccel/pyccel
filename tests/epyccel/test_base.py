# pylint: disable=missing-function-docstring, missing-module-docstring/
import pytest
import numpy as np

from pyccel.epyccel import epyccel
from modules import base


def compare_epyccel(f, *args, language='fortran'):
    f2 = epyccel(f, language=language)
    out1 = f(*args)
    out2 = f2(*args)
    assert np.equal(out1, out2)

def test_is_false(language):
    compare_epyccel(base.is_false, True, language=language)
    compare_epyccel(base.is_false, False, language=language)

def test_is_true(language):
    compare_epyccel(base.is_true, True, language=language)
    compare_epyccel(base.is_true, False, language=language)

def test_compare_is(language):
    compare_epyccel(base.compare_is, True, True, language=language)
    compare_epyccel(base.compare_is, True, False, language=language)
    compare_epyccel(base.compare_is, False, True, language=language)
    compare_epyccel(base.compare_is, False, False, language=language)

def test_compare_is_not(language):
    compare_epyccel(base.compare_is_not, True, True, language=language)
    compare_epyccel(base.compare_is_not, True, False, language=language)
    compare_epyccel(base.compare_is_not, False, True, language=language)
    compare_epyccel(base.compare_is_not, False, False, language=language)

def test_compare_is_int(language):
    compare_epyccel(base.compare_is_int, True, 1, language=language)
    compare_epyccel(base.compare_is_int, True, 0, language=language)
    compare_epyccel(base.compare_is_int, False, 1, language=language)
    compare_epyccel(base.compare_is_int, False, 0, language=language)

def test_compare_is_not_int(language):
    compare_epyccel(base.compare_is_not_int, True, 1, language=language)
    compare_epyccel(base.compare_is_not_int, True, 0, language=language)
    compare_epyccel(base.compare_is_not_int, False, 1, language=language)
    compare_epyccel(base.compare_is_not_int, False, 0, language=language)

def test_not_false(language):
    compare_epyccel(base.not_false, True, language=language)
    compare_epyccel(base.not_false, False, language=language)

def test_not_true(language):
    compare_epyccel(base.not_true, True, language=language)
    compare_epyccel(base.not_true, False, language=language)

def test_eq_false(language):
    compare_epyccel(base.eq_false, True, language=language)
    compare_epyccel(base.eq_false, False, language=language)

def test_eq_true(language):
    compare_epyccel(base.eq_true, True, language=language)
    compare_epyccel(base.eq_true, False, language=language)

def test_neq_false(language):
    compare_epyccel(base.eq_false, True, language=language)
    compare_epyccel(base.eq_false, False, language=language)

def test_neq_true(language):
    compare_epyccel(base.eq_true, True, language=language)
    compare_epyccel(base.eq_true, False, language=language)

def test_not(language):
    compare_epyccel(base.not_val, True, language=language)
    compare_epyccel(base.not_val, False, language=language)

@pytest.mark.parametrize( 'language', [
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="f2py does not support optional arguments"),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c)
    ]
)
def test_compare_is_nil(language):
    compare_epyccel(base.is_nil, None, language=language)

@pytest.mark.parametrize( 'language', [
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="f2py does not support optional arguments"),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c)
    ]
)
def test_compare_is_not_nil(language):
    compare_epyccel(base.is_not_nil, None, language=language)

def test_cast_int(language):
    compare_epyccel(base.cast_int, 4, language=language)
    compare_epyccel(base.cast_float_to_int, 4.5, language=language)

def test_cast_bool(language):
    compare_epyccel(base.cast_bool, True, language=language)

def test_cast_float(language):
    compare_epyccel(base.cast_float, 4.5, language=language)
    compare_epyccel(base.cast_int_to_float, 4, language=language)

def test_if_0_int(language):
    compare_epyccel(base.if_0_int, 22, language=language)
    compare_epyccel(base.if_0_int, 0, language=language)

def test_if_0_real(language):
    compare_epyccel(base.if_0_real, 22.3, language=language)
    compare_epyccel(base.if_0_real, 0.0, language=language)

def test_same_int(language):
    compare_epyccel(base.is_same_int, 22, language=language)
    compare_epyccel(base.isnot_same_int, 22, language=language)

def test_same_float(language):
    compare_epyccel(base.is_same_float, 22.2, language=language)
    compare_epyccel(base.isnot_same_float, 22.2, language=language)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="Strings are not yet implemented for C language"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_same_string(language):
    compare_epyccel(base.is_same_string, language=language)
    compare_epyccel(base.isnot_same_string, language=language)

def test_same_complex(language):
    compare_epyccel(base.is_same_complex, complex(2,3), language=language)
    compare_epyccel(base.isnot_same_complex, complex(2,3), language=language)

def test_is_types(language):
    compare_epyccel(base.is_types, 1, 1.0, language=language)

def test_isnot_types(language):
    compare_epyccel(base.isnot_types, 1, 1.0, language=language)

def test_none_is_none(language):
    compare_epyccel(base.none_is_none, language=language)

def test_none_isnot_none(language):
    compare_epyccel(base.none_isnot_none, language=language)
