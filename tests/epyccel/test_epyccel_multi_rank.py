# pylint: disable=missing-function-docstring, missing-module-docstring
import pytest
import numpy as np
from numpy.random import rand, randint

from modules        import multi_rank
from pyccel.epyccel import epyccel

@pytest.mark.parametrize('f1',[multi_rank.add_mixed_order,
    multi_rank.mul_mixed_order,
    multi_rank.sub_mixed_order,
    multi_rank.div_mixed_order,
    multi_rank.augadd_mixed_order,
    multi_rank.augmul_mixed_order,
    multi_rank.augsub_mixed_order,
    multi_rank.augdiv_mixed_order])
def test_add_mixed_order(f1, language):
    f2 = epyccel( f1, language = language )

    x1 = np.array(np.int64(rand(4,5))*100, dtype=float)
    x2 = np.copy(x1)

    y1 = np.array(np.int64(rand(4,5)*100)+1, dtype=float, order = 'F')
    y2 = np.copy(y1)

    f1(x1, y1)
    f2(x2, y2)

    assert np.array_equal( x1, x2 )

def test_mul_by_vector_C(language):
    f1 = multi_rank.mul_by_vector_C
    f2 = epyccel( f1, language = language )

    x1 = np.array(rand(4,5)*10, dtype=int)
    x2 = np.copy(x1)

    y1 = np.array(rand(5)*10, dtype=int)
    y2 = np.copy(y1)

    f1(x1, y1)
    f2(x2, y2)

    assert np.array_equal( x1, x2 )

def test_mul_by_vector_F(language):
    f1 = multi_rank.mul_by_vector_F
    f2 = epyccel( f1, language = language )

    x1 = np.array(rand(4,5)*10, dtype=int, order='F')
    x2 = np.copy(x1)

    y1 = np.array(rand(5)*10, dtype=int)
    y2 = np.copy(y1)

    f1(x1, y1)
    f2(x2, y2)

    assert np.array_equal( x1, x2 )

def test_mul_by_vector_dim_1_C_C(language):
    f1 = multi_rank.mul_by_vector_dim_1_C_C
    f2 = epyccel( f1, language = language )

    x1 = np.array(rand(3,5)*10, dtype=int)
    x2 = np.copy(x1)

    f1(x1)
    f2(x2)

    assert np.array_equal( x1, x2 )

def test_mul_by_vector_dim_1_C_F(language):
    f1 = multi_rank.mul_by_vector_dim_1_C_F
    f2 = epyccel( f1, language = language )

    x1 = np.array(rand(3,5)*10, dtype=int)
    x2 = np.copy(x1)

    f1(x1)
    f2(x2)

    assert np.array_equal( x1, x2 )

def test_mul_by_vector_dim_1_F_C(language):
    f1 = multi_rank.mul_by_vector_dim_1_F_C
    f2 = epyccel( f1, language = language )

    x1 = np.array(rand(3,5)*10, dtype=int, order='F')
    x2 = np.copy(x1)

    f1(x1)
    f2(x2)

    assert np.array_equal( x1, x2 )

def test_mul_by_vector_dim_1_F_F(language):
    f1 = multi_rank.mul_by_vector_dim_1_F_F
    f2 = epyccel( f1, language = language )

    x1 = np.array(rand(3,5)*10, dtype=int, order='F')
    x2 = np.copy(x1)

    f1(x1)
    f2(x2)

    assert np.array_equal( x1, x2 )

def test_multi_dim_sum(language):
    f1 = multi_rank.multi_dim_sum
    f2 = epyccel( f1, language = language )

    dims = [randint(1,10) for _ in range(3)]
    x1 = np.array(rand(*dims)*10, dtype=int)
    y1 = np.copy(x1)
    x2 = np.array(rand(*dims[1:])*10, dtype=int)
    y2 = np.copy(x2)
    x3 = np.array(rand(dims[2])*10, dtype=int)
    y3 = np.copy(x3)
    x4 = int(rand()*10)
    y4 = x4

    pyccel_result = np.empty(dims, dtype=int)
    python_result = np.empty(dims, dtype=int)

    f1(pyccel_result, x1, x2, x3, x4)
    f2(python_result, y1, y2, y3, y4)

    assert np.array_equal( pyccel_result, python_result )

# The remaining tests use np.sum

def test_multi_dim_sum_ones():
    f1 = multi_rank.multi_dim_sum_ones
    f2 = epyccel( f1 )

    dims = [randint(1, 10) for _ in range(3)]
    x1 = np.array(rand(*dims)*10, dtype=int)
    y1 = np.copy(x1)

    pyccel_result = np.empty(dims, dtype=int)
    python_result = np.empty(dims, dtype=int)

    f1(pyccel_result, x1)
    f2(python_result, y1)

    assert np.array_equal( pyccel_result, python_result )

def test_multi_expression_assign():
    f1 = multi_rank.multi_expression_assign
    f2 = epyccel( f1 )

    x1 = np.array(rand(4,5)*10, dtype=int)
    x2 = np.copy(x1)

    y1 = np.array(rand(5)*10, dtype=int)
    y2 = np.copy(y1)

    f1(x1, y1)
    f2(x2, y2)

    assert np.array_equal( x1, x2 )

def test_multi_expression_augassign():
    f1 = multi_rank.multi_expression_augassign
    f2 = epyccel( f1 )

    x1 = np.array(rand(4,5)*10, dtype=int)
    x2 = np.copy(x1)

    y1 = np.array(rand(5)*10, dtype=int)
    y2 = np.copy(y1)

    f1(x1, y1)
    f2(x2, y2)

    assert np.array_equal( x1, x2 )

def test_grouped_expressions():
    f1 = multi_rank.grouped_expressions
    f2 = epyccel( f1 )

    x1 = np.array(rand(4,5)*10, dtype=int, order='F')
    x2 = np.copy(x1)

    y1 = np.array(rand(4,5)*10, dtype=int)
    y2 = np.copy(y1)

    z1 = np.array(rand(5)*10, dtype=int)
    z2 = np.copy(z1)

    f1(x1, y1, z1)
    f2(x2, y2, z2)

    assert np.array_equal( x1, x2 )

def test_grouped_expressions2():
    f1 = multi_rank.grouped_expressions2
    f2 = epyccel( f1 )

    x1 = np.array(rand(3,4,5)*10, dtype=int)
    x2 = np.copy(x1)

    y1 = np.array(rand(4,5)*10, dtype=int)
    y2 = np.copy(y1)

    z1 = np.array(rand(5)*10, dtype=int)
    z2 = np.copy(z1)

    f1(x1, y1, z1)
    f2(x2, y2, z2)

    assert np.array_equal( x1, x2 )

def test_dependencies():
    f1 = multi_rank.dependencies
    f2 = epyccel( f1 )

    x1 = np.array(rand(4,5)*10, dtype=int)
    x2 = np.copy(x1)

    y1 = np.array(rand(5)*10, dtype=int)
    y2 = np.copy(y1)

    f1(x1, y1)
    f2(x2, y2)

    assert np.array_equal( x1, x2 )

def test_auto_dependencies():
    f1 = multi_rank.auto_dependencies
    f2 = epyccel( f1 )

    x1 = np.array(rand(4,5)*10, dtype=int)
    x2 = np.copy(x1)

    y1 = np.array(rand(5)*10, dtype=int)
    y2 = np.copy(y1)

    f1(x1, y1)
    f2(x2, y2)

    assert np.array_equal( x1, x2 )
