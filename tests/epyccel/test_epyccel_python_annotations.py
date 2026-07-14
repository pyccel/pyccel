# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np
import pytest
from modules import python_annotations
from numpy.random import randint
from utilities import epyccel_module_with_fallback

from pyccel import epyccel


@pytest.fixture(scope="module")
def epyc_python_annotations_mod(language):
    return epyccel_module_with_fallback(python_annotations, language)


def test_array_int32_1d_scalar_add(epyc_python_annotations_mod):

    f1 = python_annotations.array_int32_1d_scalar_add
    f2 = epyc_python_annotations_mod.array_int32_1d_scalar_add

    x1 = np.array([1, 2, 3], dtype=np.int32)
    x2 = np.copy(x1)
    a = randint(low=-1e9, high=1e9, dtype=np.int32)

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal(x1, x2)


def test_array_int32_2d_C_scalar_add(epyc_python_annotations_mod):

    f1 = python_annotations.array_int32_2d_C_scalar_add
    f2 = epyc_python_annotations_mod.array_int32_2d_C_scalar_add

    x1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    x2 = np.copy(x1)
    a = randint(low=-1e9, high=1e9, dtype=np.int32)

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal(x1, x2)


def test_array_int32_2d_F_add(epyc_python_annotations_mod):

    f1 = python_annotations.array_int32_2d_F_add
    f2 = epyc_python_annotations_mod.array_int32_2d_F_add

    x1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32, order="F")
    x2 = np.copy(x1)
    a = np.array([[-1, -2, -3], [-4, -5, -6]], dtype=np.int32, order="F")

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal(x1, x2)


def test_array_int_1d_scalar_add(epyc_python_annotations_mod):

    f1 = python_annotations.array_int_1d_scalar_add
    f2 = epyc_python_annotations_mod.array_int_1d_scalar_add

    x1 = np.array([1, 2, 3])
    x2 = np.copy(x1)
    a = 5

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal(x1, x2)


def test_array_float_1d_scalar_add(epyc_python_annotations_mod):

    f1 = python_annotations.array_float_1d_scalar_add
    f2 = epyc_python_annotations_mod.array_float_1d_scalar_add

    x1 = np.array([1.0, 2.0, 3.0])
    x2 = np.copy(x1)
    a = 5.0

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal(x1, x2)


def test_array_float_2d_F_scalar_add(epyc_python_annotations_mod):

    f1 = python_annotations.array_float_2d_F_scalar_add
    f2 = epyc_python_annotations_mod.array_float_2d_F_scalar_add

    x1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], order="F")
    x2 = np.copy(x1)
    a = 5.0

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal(x1, x2)


def test_array_float_2d_F_add(epyc_python_annotations_mod):

    f1 = python_annotations.array_float_2d_F_add
    f2 = epyc_python_annotations_mod.array_float_2d_F_add

    x1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], order="F")
    x2 = np.copy(x1)
    a = np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], order="F")

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal(x1, x2)


def test_array_int32_2d_F_complex_3d_expr(epyc_python_annotations_mod):

    f1 = python_annotations.array_int32_2d_F_complex_3d_expr
    f2 = epyc_python_annotations_mod.array_int32_2d_F_complex_3d_expr

    x1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32, order="F")
    x2 = np.copy(x1)
    a = np.array([[-1, -2, -3], [-4, -5, -6]], dtype=np.int32, order="F")

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal(x1, x2)


def test_array_float_1d_complex_3d_expr(epyc_python_annotations_mod):

    f1 = python_annotations.array_float_1d_complex_3d_expr
    f2 = epyc_python_annotations_mod.array_float_1d_complex_3d_expr

    x1 = np.array([1.0, 2.0, 3.0])
    x2 = np.copy(x1)
    a = np.array([-1.0, -2.0, -3.0])

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal(x1, x2)


def test_fib(epyc_python_annotations_mod):
    f1 = python_annotations.fib
    f2 = epyc_python_annotations_mod.fib
    assert f1(10) == f2(10)
