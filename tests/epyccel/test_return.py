# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np
import pytest
from modules import return_module
from utilities import epyccel_module_with_fallback

from pyccel import epyccel


@pytest.fixture(scope="module")
def epyc_return_mod(language):
    return epyccel_module_with_fallback(
        return_module, language, flags="-Werror  -Wunused-variable"
    )


# Most of the tests are currently skipped for LLVM because
# flang-new does not support most -W* flags, except -Werror


def test_single_return_var_assign(epyc_return_mod):
    single_return_var_assign = return_module.single_return_var_assign
    epyc_single_return_var_assign = epyc_return_mod.single_return_var_assign
    assert epyc_single_return_var_assign() == single_return_var_assign()


@pytest.mark.skip_llvm
def test_assign_vars_return(epyc_return_mod):
    assign_vars_return = return_module.assign_vars_return
    epyc_assign_vars_return = epyc_return_mod.assign_vars_return
    assert epyc_assign_vars_return(3, 4) == assign_vars_return(3, 4)


@pytest.mark.skip_llvm
def test_sum_in_single_return(epyc_return_mod):
    sum_in_single_return = return_module.sum_in_single_return
    epyc_sum_in_single_return = epyc_return_mod.sum_in_single_return
    assert epyc_sum_in_single_return(7, 2) == sum_in_single_return(7, 2)


@pytest.mark.skip_llvm
def test_return_expr(epyc_return_mod):
    return_expr = return_module.return_expr
    epyc_return_expr = epyc_return_mod.return_expr
    assert epyc_return_expr(7, 2) == return_expr(7, 2)


@pytest.mark.skip_llvm
def test_return_single_var(epyc_return_mod):
    return_single_var = return_module.return_single_var
    epyc_return_single_var = epyc_return_mod.return_single_var
    assert epyc_return_single_var(7) == return_single_var(7)


@pytest.mark.skip_llvm
def test_return_scalare(epyc_return_mod):
    return_scalare = return_module.return_scalare
    epyc_return_scalare = epyc_return_mod.return_scalare
    assert epyc_return_scalare() == return_scalare()


@pytest.mark.skip_llvm
def test_multi_return_scalare(epyc_return_mod):
    multi_return_scalare = return_module.multi_return_scalare
    epyc_multi_return_scalare = epyc_return_mod.multi_return_scalare
    assert epyc_multi_return_scalare() == multi_return_scalare()


@pytest.mark.skip_llvm
def test_multi_return_vars(epyc_return_mod):
    multi_return_vars = return_module.multi_return_vars
    epyc_multi_return_vars = epyc_return_mod.multi_return_vars
    assert epyc_multi_return_vars(7, 2) == multi_return_vars(7, 2)


@pytest.mark.skip_llvm
def test_multi_return_vars_expr(epyc_return_mod):
    multi_return_vars_expr = return_module.multi_return_vars_expr
    epyc_multi_return_vars_expr = epyc_return_mod.multi_return_vars_expr
    assert epyc_multi_return_vars_expr(7, 2) == multi_return_vars_expr(7, 2)


@pytest.mark.skip_llvm
def test_scalare_multi_return_stmts(epyc_return_mod):
    scalare_multi_return_stmts = return_module.scalare_multi_return_stmts
    epyc_scalare_multi_return_stmts = epyc_return_mod.scalare_multi_return_stmts
    assert epyc_scalare_multi_return_stmts(7) == scalare_multi_return_stmts(7)


@pytest.mark.skip_llvm
def test_create_arr(epyc_return_mod):
    create_arr = return_module.create_arr
    epyc_create_arr = epyc_return_mod.create_arr
    assert epyc_create_arr(7) == create_arr(7)


@pytest.mark.skip_llvm
def test_return_arr_element(epyc_return_mod):
    return_arr_element = return_module.return_arr_element
    epyc_return_arr_element = epyc_return_mod.return_arr_element
    assert epyc_return_arr_element(7) == return_arr_element(7)


@pytest.mark.skip_llvm
def test_create_multi_arrs(epyc_return_mod):
    create_multi_arrs = return_module.create_multi_arrs
    epyc_create_multi_arrs = epyc_return_mod.create_multi_arrs
    assert epyc_create_multi_arrs(7) == create_multi_arrs(7)


@pytest.mark.skip_llvm
def test_expr_arrs_elements(epyc_return_mod):
    expr_arrs_elements = return_module.expr_arrs_elements
    epyc_expr_arrs_elements = epyc_return_mod.expr_arrs_elements
    assert epyc_expr_arrs_elements(7) == expr_arrs_elements(7)


@pytest.mark.skip_llvm
def test_complex_expr(epyc_return_mod):
    complex_expr = return_module.complex_expr
    epyc_complex_expr = epyc_return_mod.complex_expr
    assert epyc_complex_expr(7) == complex_expr(7)


@pytest.mark.skip_llvm
def test_multi_allocs(epyc_return_mod):
    multi_allocs = return_module.multi_allocs
    epyc_multi_allocs = epyc_return_mod.multi_allocs
    assert epyc_multi_allocs(7) == multi_allocs(7)


def test_return_nothing(epyc_return_mod):
    divide_by = return_module.divide_by
    epyc_divide_by = epyc_return_mod.divide_by
    x = np.ones(5)
    x_copy = x.copy()
    b = 0.01
    divide_by(x, b)
    epyc_divide_by(x_copy, b)
    assert np.allclose(x, x_copy, rtol=1e-13, atol=1e-14)
    b = 4.0
    divide_by(x, b)
    epyc_divide_by(x_copy, b)
    assert np.allclose(x, x_copy, rtol=1e-13, atol=1e-14)


def test_return_None(epyc_return_mod):
    divide_by = return_module.return_None
    epyc_divide_by = epyc_return_mod.return_None
    x = np.ones(5)
    x_copy = x.copy()
    b = 0.01
    divide_by(x, b)
    epyc_divide_by(x_copy, b)
    assert np.allclose(x, x_copy, rtol=1e-13, atol=1e-14)
    b = 4.0
    divide_by(x, b)
    epyc_divide_by(x_copy, b)
    assert np.allclose(x, x_copy, rtol=1e-13, atol=1e-14)


@pytest.mark.skip_llvm
def test_arg_arr_element_op(epyc_return_mod):
    arr = np.array([1, 2, 3, 4], dtype=float)
    return_mult_arr_arg_element = return_module.return_mult_arr_arg_element
    epyc_return_mult_arr_arg_element = epyc_return_mod.return_mult_arr_arg_element
    assert epyc_return_mult_arr_arg_element(7, arr) == return_mult_arr_arg_element(
        7, arr
    )
    return_add_arr_arg_element = return_module.return_add_arr_arg_element
    epyc_return_add_arr_arg_element = epyc_return_mod.return_add_arr_arg_element
    assert epyc_return_add_arr_arg_element(7, arr) == return_add_arr_arg_element(7, arr)
    return_op_arr_arg_element = return_module.return_op_arr_arg_element
    epyc_return_op_arr_arg_element = epyc_return_mod.return_op_arr_arg_element
    assert epyc_return_op_arr_arg_element(7, arr) == return_op_arr_arg_element(7, arr)
