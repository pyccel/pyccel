# pylint: disable=missing-function-docstring, missing-module-docstring
import pytest
import numpy as np
from numpy.random import randint

from pyccel import epyccel
from modules import class_expressions
from utilities import epyccel_module_with_fallback


@pytest.fixture(scope="module")
def epyc_class_expressions_mod(language):
    return epyccel_module_with_fallback(class_expressions, language)




def test_complex_imag(epyc_class_expressions_mod):
    f = class_expressions.complex_imag
    epyc_f = epyc_class_expressions_mod.complex_imag

    r = f()
    epyc_r = epyc_f()

    assert r == epyc_r
    assert isinstance(r, type(epyc_r))


def test_complex_imag_expr(epyc_class_expressions_mod):
    f = class_expressions.complex_imag_expr
    epyc_f = epyc_class_expressions_mod.complex_imag_expr

    a = randint(20) + 1j * randint(20)
    b = randint(20) + 1j * randint(20)

    r = f(a, b)
    epyc_r = epyc_f(a, b)

    assert r == epyc_r
    assert isinstance(r, type(epyc_r))


def test_float_imag(epyc_class_expressions_mod):
    f = class_expressions.float_imag
    epyc_f = epyc_class_expressions_mod.float_imag

    r = f()
    epyc_r = epyc_f()

    assert r == epyc_r
    assert isinstance(r, type(epyc_r))


def test_complex_real(epyc_class_expressions_mod):
    f = class_expressions.complex_real
    epyc_f = epyc_class_expressions_mod.complex_real

    r = f()
    epyc_r = epyc_f()

    assert r == epyc_r
    assert isinstance(r, type(epyc_r))


def test_complex_real_expr(epyc_class_expressions_mod):
    f = class_expressions.complex_real_expr
    epyc_f = epyc_class_expressions_mod.complex_real_expr

    a = randint(20) + 1j * randint(20)
    b = randint(20) + 1j * randint(20)

    r = f(a, b)
    epyc_r = epyc_f(a, b)

    assert r == epyc_r
    assert isinstance(r, type(epyc_r))


def test_complex_conjugate(epyc_class_expressions_mod):
    f = class_expressions.complex_conjugate
    epyc_f = epyc_class_expressions_mod.complex_conjugate

    a = randint(20) + 1j * randint(20)
    b = randint(20) + 1j * randint(20)

    r = f(a, b)
    epyc_r = epyc_f(a, b)

    assert r == epyc_r
    assert isinstance(r, type(epyc_r))


def test_complex64_conjugate(epyc_class_expressions_mod):
    f = class_expressions.complex64_conjugate
    epyc_f = epyc_class_expressions_mod.complex64_conjugate

    a = np.complex64(randint(20) + 1j * randint(20))
    b = np.complex64(randint(20) + 1j * randint(20))

    r = f(a, b)
    epyc_r = epyc_f(a, b)

    assert r == epyc_r
    assert isinstance(r, type(epyc_r))


def test_float_conjugate(epyc_class_expressions_mod):
    f = class_expressions.float_conjugate
    epyc_f = epyc_class_expressions_mod.float_conjugate

    a = float(randint(20))
    b = float(randint(20))

    r = f(a, b)
    epyc_r = epyc_f(a, b)

    assert r == epyc_r
    assert isinstance(r, type(epyc_r))


def test_float64_conjugate(epyc_class_expressions_mod):
    f = class_expressions.float64_conjugate
    epyc_f = epyc_class_expressions_mod.float64_conjugate

    a = np.float64(randint(20))
    b = np.float64(randint(20))

    r = f(a, b)
    epyc_r = epyc_f(a, b)

    assert r == epyc_r
    assert isinstance(r, type(epyc_r))


def test_int_conjugate(epyc_class_expressions_mod):
    f = class_expressions.int_conjugate
    epyc_f = epyc_class_expressions_mod.int_conjugate

    a = randint(20)
    b = randint(20)

    r = f(a, b)
    epyc_r = epyc_f(a, b)

    assert r == epyc_r
    assert isinstance(r, type(epyc_r))


def test_int32_conjugate(epyc_class_expressions_mod):
    f = class_expressions.int32_conjugate
    epyc_f = epyc_class_expressions_mod.int32_conjugate

    a = randint(20, dtype=np.int32)
    b = randint(20, dtype=np.int32)

    r = f(a, b)
    epyc_r = epyc_f(a, b)

    assert r == epyc_r
    assert isinstance(r, type(epyc_r))


def test_bool_conjugate(epyc_class_expressions_mod):
    f = class_expressions.bool_conjugate
    epyc_f = epyc_class_expressions_mod.bool_conjugate

    a = bool(randint(2))
    b = bool(randint(2))

    r = f(a, b)
    epyc_r = epyc_f(a, b)

    assert r == epyc_r
    assert isinstance(r, type(epyc_r))


def test_ndarray_var_from_expr(epyc_class_expressions_mod):
    f = class_expressions.ndarray_var_from_expr
    epyc_f = epyc_class_expressions_mod.ndarray_var_from_expr

    a = np.ones(6, dtype=int)
    b = np.ones(6, dtype=int)

    r = f(a, b)
    epyc_r = epyc_f(a, b)

    assert r == epyc_r


def test_ndarray_var_from_slice(epyc_class_expressions_mod):
    f = class_expressions.ndarray_var_from_slice
    a = np.ones(6, dtype=int)

    epyc_f = epyc_class_expressions_mod.ndarray_var_from_slice

    r = f(a)
    epyc_r = epyc_f(a)
    assert r == epyc_r
