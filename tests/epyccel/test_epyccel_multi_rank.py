# pylint: disable=missing-function-docstring, missing-module-docstring/
import numpy as np
from numpy.random import rand

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
