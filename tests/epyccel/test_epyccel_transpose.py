# pylint: disable=missing-function-docstring, missing-module-docstring
import pytest
from modules import epyccel_transpose
from numpy import array, array_equal, empty
from numpy.random import randint


from epyccel_utilities import epyccel_module_with_fallback


@pytest.fixture(scope="module")
def epyc_epyccel_transpose_mod(language):
    return epyccel_module_with_fallback(epyccel_transpose, language)


def test_transpose_shape(epyc_epyccel_transpose_mod):

    f1 = epyccel_transpose.transpose_shape_1
    f2 = epyc_epyccel_transpose_mod.transpose_shape_2
    x1 = array(randint(50, size=(2, 5)), dtype=int)
    x2 = array(randint(50, size=(2, 3, 7)), dtype=int)

    f1_epyc = epyc_epyccel_transpose_mod.transpose_shape_1
    assert f1(x1) == f1_epyc(x1)

    f2_epyc = epyc_epyccel_transpose_mod.transpose_shape_2
    assert f2(x2) == f2_epyc(x2)


def test_transpose_property(epyc_epyccel_transpose_mod):

    f1 = epyccel_transpose.transpose_property_1
    f2 = epyccel_transpose.transpose_property_2
    x1 = array(randint(50, size=(2, 5)), dtype=int)
    x2 = array(randint(50, size=(2, 3, 7)), dtype=int)

    f1_epyc = epyc_epyccel_transpose_mod.transpose_property_1
    assert f1(x1) == f1_epyc(x1)

    f2_epyc = epyc_epyccel_transpose_mod.transpose_property_2
    assert f2(x2) == f2_epyc(x2)


def test_transpose_in_expression(epyc_epyccel_transpose_mod):

    f1 = epyccel_transpose.transpose_in_expression_1
    f2 = epyccel_transpose.transpose_in_expression_2
    x1 = array(randint(50, size=(2, 5)), dtype=int)
    x2 = array(randint(50, size=(2, 3, 7)), dtype=int)

    f1_epyc = epyc_epyccel_transpose_mod.transpose_in_expression_1
    assert f1(x1) == f1_epyc(x1)

    f2_epyc = epyc_epyccel_transpose_mod.transpose_in_expression_2
    assert f2(x2) == f2_epyc(x2)


def test_mixed_order(epyc_epyccel_transpose_mod):

    f1 = epyccel_transpose.mixed_order_1
    f2 = epyccel_transpose.mixed_order_2
    f3 = epyccel_transpose.mixed_order_3
    x1 = array(randint(50, size=(2, 5)), dtype=int)
    x2 = array(randint(50, size=(2, 3, 7)), dtype=int)

    f1_epyc = epyc_epyccel_transpose_mod.mixed_order_1
    assert f1(x1) == f1_epyc(x1)

    f2_epyc = epyc_epyccel_transpose_mod.mixed_order_2
    assert f2(x1) == f2_epyc(x1)

    f3_epyc = epyc_epyccel_transpose_mod.mixed_order_3
    assert f3(x2) == f3_epyc(x2)


def test_transpose_pointer(epyc_epyccel_transpose_mod):

    f1 = epyccel_transpose.transpose_pointer_1
    f2 = epyccel_transpose.transpose_pointer_2
    x1 = array(randint(50, size=(2, 5)), dtype=int)
    x1_copy = x1.copy()
    x2 = array(randint(50, size=(2, 3, 7)), dtype=int)
    x2_copy = x2.copy()

    f1_epyc = epyc_epyccel_transpose_mod.transpose_pointer_1
    assert f1(x1) == f1_epyc(x1_copy)

    f2_epyc = epyc_epyccel_transpose_mod.transpose_pointer_2
    assert f2(x2) == f2_epyc(x2_copy)


def test_transpose_of_expression(epyc_epyccel_transpose_mod):

    f1 = epyccel_transpose.transpose_of_expression_1
    f2 = epyccel_transpose.transpose_of_expression_2
    x1 = array(randint(50, size=(2, 5)), dtype=int)
    x2 = array(randint(50, size=(2, 3, 7)), dtype=int)

    f1_epyc = epyc_epyccel_transpose_mod.transpose_of_expression_1
    assert f1(x1) == f1_epyc(x1)

    f2_epyc = epyc_epyccel_transpose_mod.transpose_of_expression_2
    assert f2(x2) == f2_epyc(x2)


def test_force_transpose(epyc_epyccel_transpose_mod):

    f1 = epyccel_transpose.force_transpose_1
    f2 = epyccel_transpose.force_transpose_2
    x1 = array(randint(50, size=(2, 5)), dtype=int)
    x2 = array(randint(50, size=(2, 3, 7)), dtype=int)

    f1_epyc = epyc_epyccel_transpose_mod.force_transpose_1
    assert f1(x1) == f1_epyc(x1)

    f2_epyc = epyc_epyccel_transpose_mod.force_transpose_2
    assert f2(x2) == f2_epyc(x2)


def test_transpose_to_inner_indexes(epyc_epyccel_transpose_mod):

    f1 = epyccel_transpose.transpose_to_inner_indexes_1
    f2 = epyccel_transpose.transpose_to_inner_indexes_2
    f3 = epyccel_transpose.transpose_to_inner_indexes_3
    x1 = array(randint(50, size=(2, 5)), dtype=int)
    x2 = array(randint(50, size=(2, 5, 3)), dtype=int)

    y1_pyt = empty((1, 5, 2, 1), dtype=int)
    y2_pyt = empty((1, 5, 1, 2, 1), dtype=int)
    y3_pyt = empty((1, 3, 5, 2, 1), dtype=int)

    y1_pyc = empty((1, 5, 2, 1), dtype=int)
    y2_pyc = empty((1, 5, 1, 2, 1), dtype=int)
    y3_pyc = empty((1, 3, 5, 2, 1), dtype=int)

    f1_epyc = epyc_epyccel_transpose_mod.transpose_to_inner_indexes_1
    f1(x1, y1_pyt)
    f1_epyc(x1, y1_pyc)
    assert array_equal(y1_pyt, y1_pyc)

    f2_epyc = epyc_epyccel_transpose_mod.transpose_to_inner_indexes_2
    f2(x1, y2_pyt)
    f2_epyc(x1, y2_pyc)
    assert array_equal(y2_pyt, y2_pyc)

    f3_epyc = epyc_epyccel_transpose_mod.transpose_to_inner_indexes_3
    f3(x2, y3_pyt)
    f3_epyc(x2, y3_pyc)
    assert array_equal(y3_pyt, y3_pyc)
