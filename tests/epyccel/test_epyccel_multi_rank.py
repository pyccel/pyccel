# pylint: disable=missing-function-docstring, missing-module-docstring/
import numpy as np
from numpy.random import rand, randint

from pyccel.epyccel import epyccel
from modules        import multi_rank

def test_mul_by_vector_C():
    f1 = multi_rank.mul_by_vector_C
    f2 = epyccel( f1 )

    x1 = np.array(rand(4,5)*10, dtype=int)
    x2 = np.copy(x1)

    y1 = np.array(rand(5)*10, dtype=int)
    y2 = np.copy(y1)

    f1(x1, y1)
    f2(x2, y2)

    assert np.array_equal( x1, x2 )

def test_mul_by_vector_F():
    f1 = multi_rank.mul_by_vector_F
    f2 = epyccel( f1 )

    x1 = np.array(rand(4,5)*10, dtype=int, order='F')
    x2 = np.copy(x1)

    y1 = np.array(rand(5)*10, dtype=int)
    y2 = np.copy(y1)

    f1(x1, y1)
    f2(x2, y2)

    assert np.array_equal( x1, x2 )

def test_mul_by_vector_dim_1_C_C():
    f1 = multi_rank.mul_by_vector_dim_1_C_C
    f2 = epyccel( f1 )

    x1 = np.array(rand(3,5)*10, dtype=int)
    x2 = np.copy(x1)

    f1(x1)
    f2(x2)

    assert np.array_equal( x1, x2 )

def test_mul_by_vector_dim_1_C_F():
    f1 = multi_rank.mul_by_vector_dim_1_C_F
    f2 = epyccel( f1 )

    x1 = np.array(rand(3,5)*10, dtype=int)
    x2 = np.copy(x1)

    f1(x1)
    f2(x2)

    assert np.array_equal( x1, x2 )

def test_mul_by_vector_dim_1_F_C():
    f1 = multi_rank.mul_by_vector_dim_1_F_C
    f2 = epyccel( f1 )

    x1 = np.array(rand(3,5)*10, dtype=int, order='F')
    x2 = np.copy(x1)

    f1(x1)
    f2(x2)

    assert np.array_equal( x1, x2 )

def test_mul_by_vector_dim_1_F_F():
    f1 = multi_rank.mul_by_vector_dim_1_F_F
    f2 = epyccel( f1 )

    x1 = np.array(rand(3,5)*10, dtype=int, order='F')
    x2 = np.copy(x1)

    f1(x1)
    f2(x2)

    assert np.array_equal( x1, x2 )

def test_multi_dim_sum():
    f1 = multi_rank.multi_dim_sum
    f2 = epyccel( f1 )

    dims = [randint(10) for _ in range(3)]
    x1 = np.array(rand(*dims)*10, dtype=int)
    y1 = np.copy(x1)
    x2 = np.array(rand(*dims[1:])*10, dtype=int)
    y2 = np.copy(x2)
    x3 = np.array(rand(dims[2])*10, dtype=int)
    y3 = np.copy(x3)
    x4 = int(rand()*10)
    y4 = x4

    pyccel_result = np.empty(dims)
    python_result = np.empty(dims)

    f1(pyccel_result, x1, x2, x3, x4)
    f1(python_result, y1, y2, y3, y4)

    assert np.array_equal( pyccel_result, python_result )

def test_multi_dim_sum_ones():
    f1 = multi_rank.multi_dim_sum_ones
    f2 = epyccel( f1 )

    dims = [randint(10) for _ in range(3)]
    x1 = np.array(rand(*dims)*10, dtype=int)
    y1 = np.copy(x1)

    pyccel_result = np.empty(dims)
    python_result = np.empty(dims)

    f1(pyccel_result, x1)
    f1(python_result, y1)

    assert np.array_equal( pyccel_result, python_result )

def test_mul_by_vector_C():
    f1 = multi_rank.multi_expression
    f2 = epyccel( f1 )

    x1 = np.array(rand(4,5)*10, dtype=int)
    x2 = np.copy(x1)

    y1 = np.array(rand(5)*10, dtype=int)
    y2 = np.copy(y1)

    f1(x1, y1)
    f2(x2, y2)

    assert np.array_equal( x1, x2 )
