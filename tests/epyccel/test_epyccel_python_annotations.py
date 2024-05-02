# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np
from numpy.random import randint

from pyccel.epyccel import epyccel
from modules import python_annotations

def test_array_int32_1d_scalar_add(language):

    f1 = python_annotations.array_int32_1d_scalar_add
    f2 = epyccel( f1, language = language )

    x1 = np.array( [1,2,3], dtype=np.int32 )
    x2 = np.copy(x1)
    a = randint(low = -1e9, high = 1e9, dtype = np.int32)

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int32_2d_C_scalar_add(language):

    f1 = python_annotations.array_int32_2d_C_scalar_add
    f2 = epyccel( f1, language = language )

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32 )
    x2 = np.copy(x1)
    a = randint(low = -1e9, high = 1e9, dtype = np.int32)

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int32_2d_F_add(language):

    f1 = python_annotations.array_int32_2d_F_add
    f2 = epyccel( f1, language = language )

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32, order='F' )
    x2 = np.copy(x1)
    a  = np.array( [[-1,-2,-3], [-4,-5,-6]], dtype=np.int32, order='F' )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_1d_scalar_add(language):

    f1 = python_annotations.array_int_1d_scalar_add
    f2 = epyccel( f1, language = language )

    x1 = np.array( [1,2,3] )
    x2 = np.copy(x1)
    a = 5

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_real_1d_scalar_add(language):

    f1 = python_annotations.array_real_1d_scalar_add
    f2 = epyccel( f1, language = language )

    x1 = np.array( [1.,2.,3.] )
    x2 = np.copy(x1)
    a = 5.

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )


def test_array_real_2d_F_scalar_add(language):

    f1 = python_annotations.array_real_2d_F_scalar_add
    f2 = epyccel( f1, language = language )

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]], order='F' )
    x2 = np.copy(x1)
    a = 5.

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_real_2d_F_add(language):

    f1 = python_annotations.array_real_2d_F_add
    f2 = epyccel( f1, language = language )

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]], order='F' )
    x2 = np.copy(x1)
    a  = np.array( [[-1.,-2.,-3.], [-4.,-5.,-6.]], order='F' )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int32_2d_F_complex_3d_expr(language):

    f1 = python_annotations.array_int32_2d_F_complex_3d_expr
    f2 = epyccel( f1, language = language )

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32, order='F' )
    x2 = np.copy(x1)
    a  = np.array( [[-1,-2,-3], [-4,-5,-6]], dtype=np.int32, order='F' )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_real_1d_complex_3d_expr(language):

    f1 = python_annotations.array_real_1d_complex_3d_expr
    f2 = epyccel( f1, language = language )

    x1 = np.array( [1.,2.,3.] )
    x2 = np.copy(x1)
    a  = np.array( [-1.,-2.,-3.] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_fib(language):
    f1 = python_annotations.fib
    f2 = epyccel(f1, language=language)
    assert f1(10) == f2(10)

