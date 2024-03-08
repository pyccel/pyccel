# pylint: disable=missing-function-docstring, missing-module-docstring
import pytest
import numpy as np
from numpy import iinfo, finfo
from numpy.random import randint, uniform

from pyccel.epyccel import epyccel
from modules        import arrays

RTOL = 1e-12
ATOL = 1e-16

def check_array_equal(a, b):
    """
    Check that two arrays are equal.

    Check that two arrays are equal. To be equal they must have the same
    shape, dtype, and order.
    """
    assert np.array_equal(a, b)
    assert a.dtype is b.dtype
    assert a.flags.c_contiguous == b.flags.c_contiguous
    assert a.flags.f_contiguous == b.flags.f_contiguous
#==============================================================================
# TEST: VERIFY ARRAY'S DTYPE CORRESPONDENCE TO THE PASSED ELEMENTS
#==============================================================================

def test_array_assigned_dtype(language):
    integer   = randint(low = iinfo('int').min,   high = iinfo('int').max,   dtype=int)
    integer8  = randint(low = iinfo('int8').min,  high = iinfo('int8').max,  dtype=np.int8)
    integer16 = randint(low = iinfo('int16').min, high = iinfo('int16').max, dtype=np.int16)
    integer32 = randint(low = iinfo('int32').min, high = iinfo('int32').max, dtype=np.int32)
    integer64 = randint(low = iinfo('int64').min, high = iinfo('int64').max, dtype=np.int64)

    fl = float(integer)
    fl32 = np.float32(fl)
    fl64 = np.float64(fl)

    cmplx64 = np.complex64(fl32)
    cmplx128 = np.complex128(fl64)

    epyccel_func = epyccel(arrays.array_return_first_element, language=language)

    f_integer_output = epyccel_func(integer, integer)
    test_int_output  = arrays.array_return_first_element(integer, integer)
    assert isinstance(f_integer_output, type(test_int_output))

    f_integer8_output = epyccel_func(integer8, integer8)
    test_int8_output = arrays.array_return_first_element(integer8, integer8)
    assert isinstance(f_integer8_output, type(test_int8_output))

    f_integer16_output = epyccel_func(integer16, integer16)
    test_int16_output = arrays.array_return_first_element(integer16, integer16)
    assert isinstance(f_integer16_output, type(test_int16_output))

    f_integer32_output = epyccel_func(integer32, integer32)
    test_int32_output = arrays.array_return_first_element(integer32, integer32)
    assert isinstance(f_integer32_output, type(test_int32_output))

    f_integer64_output = epyccel_func(integer64, integer64)
    test_int64_output = arrays.array_return_first_element(integer64, integer64)
    assert isinstance(f_integer64_output, type(test_int64_output))

    f_fl_output = epyccel_func(fl, fl)
    test_float_output = arrays.array_return_first_element(fl, fl)
    assert isinstance(f_fl_output, type(test_float_output))

    f_fl32_output = epyccel_func(fl32, fl32)
    test_float32_output = arrays.array_return_first_element(fl32, fl32)
    assert isinstance(f_fl32_output, type(test_float32_output))

    f_fl64_output = epyccel_func(fl64, fl64)
    test_float64_output = arrays.array_return_first_element(fl64, fl64)
    assert isinstance(f_fl64_output, type(test_float64_output))

    f_cmplx64_output = epyccel_func(cmplx64, cmplx64)
    test_cmplx64_output = arrays.array_return_first_element(cmplx64, cmplx64)
    assert isinstance(f_cmplx64_output, type(test_cmplx64_output))

    f_cmplx128_output = epyccel_func(cmplx128, cmplx128)
    test_cmplx128_output = arrays.array_return_first_element(cmplx128, cmplx128)
    assert isinstance(f_cmplx128_output, type(test_cmplx128_output))

#==============================================================================
# TEST: 1D ARRAYS OF INT-32
#==============================================================================

def test_array_int32_1d_scalar_add(language):

    f1 = arrays.array_int32_1d_scalar_add
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1,2,3], dtype=np.int32 )
    x2 = np.copy(x1)
    a = randint(low = -1e9, high = 1e9, dtype = np.int32)

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int32_1d_scalar_add_stride(language):

    f1 = arrays.array_int32_1d_scalar_add
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1,2,3,4,5,6,7,8], dtype=np.int32 )
    x2 = np.copy(x1)
    a = randint(low = -1e9, high = 1e9, dtype = np.int32)

    f1(x1[::3], a)
    f2(x2[::3], a)

    assert np.array_equal( x1, x2 )

def test_array_int32_1d_scalar_sub(language):

    f1 = arrays.array_int32_1d_scalar_sub
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1,2,3], dtype=np.int32 )
    x2 = np.copy(x1)
    a = randint(low = -1e9, high = 1e9, dtype = np.int32)

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int32_1d_scalar_sub_stride(language):

    f1 = arrays.array_int32_1d_scalar_sub
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1,2,3,4,5,6,7,8,9], dtype=np.int32 )
    x2 = np.copy(x1)
    a = randint(low = -1e9, high = 1e9, dtype = np.int32)

    f1(x1[::2], a)
    f2(x2[::2], a)

    assert np.array_equal( x1, x2 )

def test_array_int32_1d_scalar_mul(language):

    f1 = arrays.array_int32_1d_scalar_mul
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1,2,3], dtype=np.int32 )
    x2 = np.copy(x1)
    a = randint(low = -1e9, high = 1e9, dtype = np.int32)

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int32_1d_scalar_mul_stride(language):

    f1 = arrays.array_int32_1d_scalar_mul
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1,2,3,4,5,6,7,8,9], dtype=np.int32 )
    x2 = np.copy(x1)
    a = randint(low = -1e9, high = 1e9, dtype = np.int32)

    f1(x1[3:7:2], a)
    f2(x2[3:7:2], a)

    assert np.array_equal( x1, x2 )

def test_array_int32_1d_scalar_div(language):

    f1 = arrays.array_int32_1d_scalar_div
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1,2,3], dtype=np.int32 )
    x2 = np.copy(x1)
    a = randint(low = 1, high = 1e9, dtype = np.int32)

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int32_1d_scalar_idiv(language):

    f1 = arrays.array_int32_1d_scalar_idiv
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1,2,3], dtype=np.int32 )
    x2 = np.copy(x1)
    a = randint(low = 1, high = 1e9, dtype = np.int32)

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int32_1d_scalar_idiv_stride(language):

    f1 = arrays.array_int32_1d_scalar_idiv
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1,2,3,4,5,6,7,8,9], dtype=np.int32 )
    x2 = np.copy(x1)
    a = randint(low = 1, high = 1e9, dtype = np.int32)

    f1(x1[:3:2], a)
    f2(x2[:3:2], a)

    assert np.array_equal( x1, x2 )

def test_array_int32_1d_add(language):

    f1 = arrays.array_int32_1d_add
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1,2,3], dtype=np.int32 )
    x2 = np.copy(x1)
    a  = np.array( [1,2,3], dtype=np.int32 )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int32_1d_sub(language):

    f1 = arrays.array_int32_1d_sub
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1,2,3], dtype=np.int32 )
    x2 = np.copy(x1)
    a  = np.array( [1,2,3], dtype=np.int32 )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int32_1d_mul(language):

    f1 = arrays.array_int32_1d_mul
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1,2,3], dtype=np.int32 )
    x2 = np.copy(x1)
    a  = np.array( [1,2,3], dtype=np.int32 )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int32_1d_idiv(language):

    f1 = arrays.array_int32_1d_idiv
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1,2,3], dtype=np.int32 )
    x2 = np.copy(x1)
    a  = np.array( [1,2,3], dtype=np.int32 )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int32_1d_add_augassign(language):

    f1 = arrays.array_int32_1d_add_augassign
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1,2,3], dtype=np.int32 )
    x2 = np.copy(x1)
    a  = np.array( [1,2,3], dtype=np.int32 )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int32_1d_sub_augassign(language):

    f1 = arrays.array_int32_1d_sub_augassign
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1,2,3], dtype=np.int32 )
    x2 = np.copy(x1)
    a  = np.array( [1,2,3], dtype=np.int32 )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_1d_initialization_1(language):

    f1 = arrays.array_int_1d_initialization_1
    f2 = epyccel( f1 , language = language)

    assert f1() == f2()

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Lists not yet supported"),
            pytest.mark.c]
        ),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_array_int_1d_initialization_2(language):

    f1 = arrays.array_int_1d_initialization_2
    f2 = epyccel( f1 , language = language)

    assert f1() == f2()

def test_array_int_1d_initialization_3(language):

    f1 = arrays.array_int_1d_initialization_3
    f2 = epyccel( f1 , language = language)

    assert f1() == f2()

#==============================================================================
# TEST: 2D ARRAYS OF INT-32 WITH C ORDERING
#==============================================================================

def test_array_int32_2d_C_scalar_add(language):

    f1 = arrays.array_int32_2d_C_scalar_add
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32 )
    x2 = np.copy(x1)
    a = randint(low = -1e9, high = 1e9, dtype = np.int32)

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int32_2d_C_scalar_add_stride(language):

    f1 = arrays.array_int32_2d_C_scalar_add
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32 )
    x2 = np.copy(x1)
    a = randint(low = -1e9, high = 1e9, dtype = np.int32)

    f1(x1[::2,:], a)
    f2(x2[::2,:], a)

    assert np.array_equal( x1, x2 )

def test_array_int32_2d_C_scalar_sub(language):

    f1 = arrays.array_int32_2d_C_scalar_sub
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32 )
    x2 = np.copy(x1)
    a = randint(low = -1e9, high = 1e9, dtype = np.int32)

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="Ordering is unknown on non-contiguous array"),
            pytest.mark.c]),
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="Ordering is unknown on non-contiguous array"),
            pytest.mark.fortran]),
        pytest.param("python", marks = pytest.mark.python)
    ]
)
def test_array_int32_2d_C_scalar_sub_stride(language):

    f1 = arrays.array_int32_2d_C_scalar_sub
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3,7], [4,5,6,8]], dtype=np.int32 )
    x2 = np.copy(x1)
    a = randint(low = -1e9, high = 1e9, dtype = np.int32)

    f1(x1[:,::2], a)
    f2(x2[:,::2], a)

    assert np.array_equal( x1, x2 )

def test_array_int32_2d_C_scalar_mul(language):

    f1 = arrays.array_int32_2d_C_scalar_mul
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32 )
    x2 = np.copy(x1)
    a = randint(low = -1e9, high = 1e9, dtype = np.int32)

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int32_2d_C_scalar_mul_stride(language):

    f1 = arrays.array_int32_2d_C_scalar_mul
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32 )
    x2 = np.copy(x1)
    a = randint(low = -1e9, high = 1e9, dtype = np.int32)

    f1(x1[1:,:], a)
    f2(x2[1:,:], a)

    assert np.array_equal( x1, x2 )

def test_array_int32_2d_C_scalar_idiv(language):

    f1 = arrays.array_int32_2d_C_scalar_idiv
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32 )
    x2 = np.copy(x1)
    a = randint(low = 1, high = 1e9, dtype = np.int32)

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="Ordering is unknown on non-contiguous array"),
            pytest.mark.c]),
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="Ordering is unknown on non-contiguous array"),
            pytest.mark.fortran]),
        pytest.param("python", marks = pytest.mark.python)
    ]
)
def test_array_int32_2d_C_scalar_idiv_stride(language):

    f1 = arrays.array_int32_2d_C_scalar_idiv
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32 )
    x2 = np.copy(x1)
    a = randint(low = 1, high = 1e9, dtype = np.int32)

    f1(x1[:,1:], a)
    f2(x2[:,1:], a)

    assert np.array_equal( x1, x2 )

def test_array_int32_2d_C_add(language):

    f1 = arrays.array_int32_2d_C_add
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32 )
    x2 = np.copy(x1)
    a  = np.array( [[-1,-2,-3], [-4,-5,-6]], dtype=np.int32 )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int32_2d_C_sub(language):

    f1 = arrays.array_int32_2d_C_sub
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32 )
    x2 = np.copy(x1)
    a  = np.array( [[-1,-2,-3], [-4,-5,-6]], dtype=np.int32 )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int32_2d_C_mul(language):

    f1 = arrays.array_int32_2d_C_mul
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32 )
    x2 = np.copy(x1)
    a  = np.array( [[-1,-2,-3], [-4,-5,-6]], dtype=np.int32 )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int32_2d_C_idiv(language):

    f1 = arrays.array_int32_2d_C_idiv
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32 )
    x2 = np.copy(x1)
    a  = np.array( [[-1,-2,-3], [-4,-5,-6]], dtype=np.int32 )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

#==============================================================================
# TEST: 2D ARRAYS OF INT-32 WITH F ORDERING
#==============================================================================

def test_array_int32_2d_F_scalar_add(language):

    f1 = arrays.array_int32_2d_F_scalar_add
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32, order='F' )
    x2 = np.copy(x1)
    a = randint(low = -1e9, high = 1e9, dtype = np.int32)

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="Ordering is unknown on non-contiguous array"),
            pytest.mark.c]),
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="Ordering is unknown on non-contiguous array"),
            pytest.mark.fortran]),
        pytest.param("python", marks = pytest.mark.python)
    ]
)
def test_array_int32_2d_F_scalar_add_stride(language):

    f1 = arrays.array_int32_2d_F_scalar_add
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6], [7,8,9]], dtype=np.int32, order='F' )
    x2 = np.copy(x1)
    a = randint(low = -1e9, high = 1e9, dtype = np.int32)

    f1(x1[1::2,::2], a)
    f2(x2[1::2,::2], a)

    assert np.array_equal( x1, x2 )

def test_array_int32_2d_F_scalar_sub(language):

    f1 = arrays.array_int32_2d_F_scalar_sub
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32, order='F' )
    x2 = np.copy(x1)
    a = randint(low = -1e9, high = 1e9, dtype = np.int32)

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="Ordering is unknown on non-contiguous array"),
            pytest.mark.c]),
        pytest.param("fortran", marks = [
            pytest.mark.xfail(reason="Ordering is unknown on non-contiguous array"),
            pytest.mark.fortran]),
        pytest.param("python", marks = pytest.mark.python)
    ]
)
def test_array_int32_2d_F_scalar_sub_stride(language):

    f1 = arrays.array_int32_2d_F_scalar_sub
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6], [7,8,9]], dtype=np.int32, order='F' )
    x2 = np.copy(x1)
    a = randint(low = -1e9, high = 1e9, dtype = np.int32)

    f1(x1[::2,1::2], a)
    f2(x2[::2,1::2], a)

    assert np.array_equal( x1, x2 )

def test_array_int32_2d_F_scalar_mul(language):

    f1 = arrays.array_int32_2d_F_scalar_mul
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32, order='F' )
    x2 = np.copy(x1)
    a = randint(low = -1e9, high = 1e9, dtype = np.int32)

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int32_2d_F_scalar_idiv(language):

    f1 = arrays.array_int32_2d_F_scalar_idiv
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32, order='F' )
    x2 = np.copy(x1)
    a = randint(low = 1, high = 1e9, dtype = np.int32)

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int32_2d_F_add(language):

    f1 = arrays.array_int32_2d_F_add
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32, order='F' )
    x2 = np.copy(x1)
    a  = np.array( [[-1,-2,-3], [-4,-5,-6]], dtype=np.int32, order='F' )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int32_2d_F_sub(language):

    f1 = arrays.array_int32_2d_F_sub
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32, order='F' )
    x2 = np.copy(x1)
    a  = np.array( [[-1,-2,-3], [-4,-5,-6]], dtype=np.int32, order='F' )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int32_2d_F_mul(language):

    f1 = arrays.array_int32_2d_F_mul
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32, order='F' )
    x2 = np.copy(x1)
    a  = np.array( [[-1,-2,-3], [-4,-5,-6]], dtype=np.int32, order='F' )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int32_2d_F_idiv(language):

    f1 = arrays.array_int32_2d_F_idiv
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32, order='F' )
    x2 = np.copy(x1)
    a  = np.array( [[-1,-2,-3], [-4,-5,-6]], dtype=np.int32, order='F' )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )


#==============================================================================
# TEST: 1D ARRAYS OF INT-64
#==============================================================================

def test_array_int_1d_scalar_add(language):

    f1 = arrays.array_int_1d_scalar_add
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1,2,3] )
    x2 = np.copy(x1)
    a = 5

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_1d_scalar_sub(language):

    f1 = arrays.array_int_1d_scalar_sub
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1,2,3] )
    x2 = np.copy(x1)
    a = 5

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_1d_scalar_mul(language):

    f1 = arrays.array_int_1d_scalar_mul
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1,2,3] )
    x2 = np.copy(x1)
    a = 5

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_1d_scalar_idiv(language):

    f1 = arrays.array_int_1d_scalar_idiv
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1,2,3] )
    x2 = np.copy(x1)
    a = 5

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_1d_add(language):

    f1 = arrays.array_int_1d_add
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1,2,3] )
    x2 = np.copy(x1)
    a  = np.array( [1,2,3] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_1d_sub(language):

    f1 = arrays.array_int_1d_sub
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1,2,3] )
    x2 = np.copy(x1)
    a  = np.array( [1,2,3] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_1d_mul(language):

    f1 = arrays.array_int_1d_mul
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1,2,3] )
    x2 = np.copy(x1)
    a  = np.array( [1,2,3] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_1d_idiv(language):

    f1 = arrays.array_int_1d_idiv
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1,2,3] )
    x2 = np.copy(x1)
    a  = np.array( [1,2,3] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

#==============================================================================
# TEST: 2D ARRAYS OF INT-64 WITH C ORDERING
#==============================================================================

def test_array_int_2d_C_scalar_add(language):

    f1 = arrays.array_int_2d_C_scalar_add
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]] )
    x2 = np.copy(x1)
    a = 5

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_2d_C_scalar_sub(language):

    f1 = arrays.array_int_2d_C_scalar_sub
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]] )
    x2 = np.copy(x1)
    a = 5

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_2d_C_scalar_mul(language):

    f1 = arrays.array_int_2d_C_scalar_mul
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]] )
    x2 = np.copy(x1)
    a = 5

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_2d_C_scalar_idiv(language):

    f1 = arrays.array_int_2d_C_scalar_idiv
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]] )
    x2 = np.copy(x1)
    a = 5

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_2d_C_add(language):

    f1 = arrays.array_int_2d_C_add
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]] )
    x2 = np.copy(x1)
    a  = np.array( [[-1,-2,-3], [-4,-5,-6]] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_2d_C_sub(language):

    f1 = arrays.array_int_2d_C_sub
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]] )
    x2 = np.copy(x1)
    a  = np.array( [[-1,-2,-3], [-4,-5,-6]] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_2d_C_mul(language):

    f1 = arrays.array_int_2d_C_mul
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]] )
    x2 = np.copy(x1)
    a  = np.array( [[-1,-2,-3], [-4,-5,-6]] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_2d_C_idiv(language):

    f1 = arrays.array_int_2d_C_idiv
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]] )
    x2 = np.copy(x1)
    a  = np.array( [[-1,-2,-3], [-4,-5,-6]] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_2d_C_initialization(language):

    f1 = arrays.array_int_2d_C_initialization
    f2 = epyccel(f1, language = language)

    x1 = np.zeros((2, 3), dtype=int)
    x2 = np.ones_like(x1)

    f1(x1)
    f2(x2)

    assert np.array_equal(x1, x2)

#==============================================================================
# TEST: 2D ARRAYS OF INT-64 WITH F ORDERING
#==============================================================================

def test_array_int_2d_F_scalar_add(language):

    f1 = arrays.array_int_2d_F_scalar_add
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]], order='F' )
    x2 = np.copy(x1)
    a = 5

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_2d_F_scalar_sub(language):

    f1 = arrays.array_int_2d_F_scalar_sub
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]], order='F' )
    x2 = np.copy(x1)
    a = 5

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_2d_F_scalar_mul(language):

    f1 = arrays.array_int_2d_F_scalar_mul
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]], order='F' )
    x2 = np.copy(x1)
    a = 5

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_2d_F_scalar_idiv(language):

    f1 = arrays.array_int_2d_F_scalar_idiv
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]], order='F' )
    x2 = np.copy(x1)
    a = 5

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_2d_F_add(language):

    f1 = arrays.array_int_2d_F_add
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]], order='F' )
    x2 = np.copy(x1)
    a  = np.array( [[-1,-2,-3], [-4,-5,-6]], order='F' )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_2d_F_sub(language):

    f1 = arrays.array_int_2d_F_sub
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]], order='F' )
    x2 = np.copy(x1)
    a  = np.array( [[-1,-2,-3], [-4,-5,-6]], order='F' )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_2d_F_mul(language):

    f1 = arrays.array_int_2d_F_mul
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]], order='F' )
    x2 = np.copy(x1)
    a  = np.array( [[-1,-2,-3], [-4,-5,-6]], order='F' )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_2d_F_idiv(language):

    f1 = arrays.array_int_2d_F_idiv
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]], order='F' )
    x2 = np.copy(x1)
    a  = np.array( [[-1,-2,-3], [-4,-5,-6]], order='F' )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_2d_F_initialization(language):

    f1 = arrays.array_int_2d_F_initialization
    f2 = epyccel(f1, language = language)

    x1 = np.zeros((2, 3), dtype=int, order='F')
    x2 = np.ones_like(x1)

    f1(x1)
    f2(x2)

    assert np.array_equal(x1, x2)

#==============================================================================
# TEST: 1D ARRAYS OF REAL
#==============================================================================

def test_array_float_1d_scalar_add(language):

    f1 = arrays.array_float_1d_scalar_add
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1.,2.,3.] )
    x2 = np.copy(x1)
    a = 5.

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_float_1d_scalar_sub(language):

    f1 = arrays.array_float_1d_scalar_sub
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1.,2.,3.] )
    x2 = np.copy(x1)
    a = 5.

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_float_1d_scalar_mul(language):

    f1 = arrays.array_float_1d_scalar_mul
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1.,2.,3.] )
    x2 = np.copy(x1)
    a = 5.

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_float_1d_scalar_div(language):

    f1 = arrays.array_float_1d_scalar_div
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1.,2.,3.] )
    x2 = np.copy(x1)
    a = 5.

    f1(x1, a)
    f2(x2, a)

    assert np.allclose(x1, x2, rtol=RTOL, atol=ATOL)

def test_array_float_1d_scalar_mod(language):
    f1 = arrays.array_float_1d_scalar_mod
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1.,2.,3.] )
    x2 = np.copy(x1)
    a = 5.

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_float_1d_scalar_idiv(language):

    f1 = arrays.array_float_1d_scalar_idiv
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1.,2.,3.] )
    x2 = np.copy(x1)
    a = 5.

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_float_1d_add(language):

    f1 = arrays.array_float_1d_add
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1.,2.,3.] )
    x2 = np.copy(x1)
    a  = np.array( [1.,2.,3.] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_float_1d_sub(language):

    f1 = arrays.array_float_1d_sub
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1.,2.,3.] )
    x2 = np.copy(x1)
    a  = np.array( [1.,2.,3.] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_float_1d_mul(language):

    f1 = arrays.array_float_1d_mul
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1.,2.,3.] )
    x2 = np.copy(x1)
    a  = np.array( [1.,2.,3.] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_float_1d_div(language):

    f1 = arrays.array_float_1d_div
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1.,2.,3.] )
    x2 = np.copy(x1)
    a  = np.array( [1.,2.,3.] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_float_1d_mod(language):

    f1 = arrays.array_float_1d_mod
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1.,2.,3.] )
    x2 = np.copy(x1)
    a  = np.array( [1.,2.,3.] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2)

def test_array_float_1d_idiv(language):

    f1 = arrays.array_float_1d_idiv
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1.,2.,3.] )
    x2 = np.copy(x1)
    a  = np.array( [1.,2.,3.] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

#==============================================================================
# TEST: 2D ARRAYS OF REAL WITH C ORDERING
#==============================================================================

def test_array_float_2d_C_scalar_add(language):

    f1 = arrays.array_float_2d_C_scalar_add
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]] )
    x2 = np.copy(x1)
    a = 5.

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_float_2d_C_scalar_sub(language):

    f1 = arrays.array_float_2d_C_scalar_sub
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]] )
    x2 = np.copy(x1)
    a = 5.

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_float_2d_C_scalar_mul(language):

    f1 = arrays.array_float_2d_C_scalar_mul
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]] )
    x2 = np.copy(x1)
    a = 5.

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_float_2d_C_scalar_div(language):

    f1 = arrays.array_float_2d_C_scalar_div
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]] )
    x2 = np.copy(x1)
    a = 5.

    f1(x1, a)
    f2(x2, a)

    assert np.allclose(x1, x2, rtol=RTOL, atol=ATOL)

def test_array_float_2d_C_scalar_mod(language):

    f1 = arrays.array_float_2d_C_scalar_mod
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]] )
    x2 = np.copy(x1)
    a = 5.

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_float_2d_C_add(language):

    f1 = arrays.array_float_2d_C_add
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]] )
    x2 = np.copy(x1)
    a  = np.array( [[-1.,-2.,-3.], [-4.,-5.,-6.]] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_float_2d_C_sub(language):

    f1 = arrays.array_float_2d_C_sub
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]] )
    x2 = np.copy(x1)
    a  = np.array( [[-1.,-2.,-3.], [-4.,-5.,-6.]] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_float_2d_C_mul(language):

    f1 = arrays.array_float_2d_C_mul
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]] )
    x2 = np.copy(x1)
    a  = np.array( [[-1.,-2.,-3.], [-4.,-5.,-6.]] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_float_2d_C_div(language):

    f1 = arrays.array_float_2d_C_div
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]] )
    x2 = np.copy(x1)
    a  = np.array( [[-1.,-2.,-3.], [-4.,-5.,-6.]] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_float_2d_C_mod(language):

    f1 = arrays.array_float_2d_C_mod
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]] )
    x2 = np.copy(x1)
    a  = np.array( [[-1.,-2.,-3.], [-4.,-5.,-6.]] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_float_2d_C_array_initialization(language):

    f1 = arrays.array_float_2d_C_array_initialization
    f2 = epyccel(f1, language = language)

    x1 = np.zeros((2, 3), dtype=float )
    x2 = np.ones_like(x1)

    f1(x1)
    f2(x2)

    assert np.array_equal(x1, x2)

def test_array_float_3d_C_array_initialization_1(language):

    f1 = arrays.array_float_3d_C_array_initialization_1
    f2 = epyccel(f1, language = language)

    x  = np.random.random((3,2))
    y  = np.random.random((3,2))
    a  = np.array([x,y])

    x1 = np.zeros_like(a)
    x2 = np.zeros_like(a)

    f1(x, y, x1)
    f2(x, y, x2)

    assert np.array_equal(x1, x2)

def test_array_float_3d_C_array_initialization_2(language):

    f1 = arrays.array_float_3d_C_array_initialization_2
    f2 = epyccel(f1, language = language)

    x1 = np.zeros((2,3,4))
    x2 = np.zeros((2,3,4))

    f1(x1)
    f2(x2)

    assert np.array_equal(x1, x2)

def test_array_float_4d_C_array_initialization(language):

    f1 = arrays.array_float_4d_C_array_initialization
    f2 = epyccel(f1, language = language)

    x  = np.random.random((3,2,4))
    y  = np.random.random((3,2,4))
    a  = np.array([x,y])

    x1 = np.zeros_like(a)
    x2 = np.zeros_like(a)

    f1(x, y, x1)
    f2(x, y, x2)

    assert np.array_equal(x1, x2)
#==============================================================================
# TEST: 2D ARRAYS OF REAL WITH F ORDERING
#==============================================================================

def test_array_float_2d_F_scalar_add(language):

    f1 = arrays.array_float_2d_F_scalar_add
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]], order='F' )
    x2 = np.copy(x1)
    a = 5.

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_float_2d_F_scalar_sub(language):

    f1 = arrays.array_float_2d_F_scalar_sub
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]], order='F' )
    x2 = np.copy(x1)
    a = 5.

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_float_2d_F_scalar_mul(language):

    f1 = arrays.array_float_2d_F_scalar_mul
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]], order='F' )
    x2 = np.copy(x1)
    a = 5.

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_float_2d_F_scalar_div(language):

    f1 = arrays.array_float_2d_F_scalar_div
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]], order='F' )
    x2 = np.copy(x1)
    a = 5.

    f1(x1, a)
    f2(x2, a)

    assert np.allclose(x1, x2, rtol=RTOL, atol=ATOL)

def test_array_float_2d_F_scalar_mod(language):

    f1 = arrays.array_float_2d_F_scalar_mod
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]], order='F' )
    x2 = np.copy(x1)
    a = 5.

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_float_2d_F_add(language):

    f1 = arrays.array_float_2d_F_add
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]], order='F' )
    x2 = np.copy(x1)
    a  = np.array( [[-1.,-2.,-3.], [-4.,-5.,-6.]], order='F' )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_float_2d_F_sub(language):

    f1 = arrays.array_float_2d_F_sub
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]], order='F' )
    x2 = np.copy(x1)
    a  = np.array( [[-1.,-2.,-3.], [-4.,-5.,-6.]], order='F' )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_float_2d_F_mul(language):

    f1 = arrays.array_float_2d_F_mul
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]], order='F' )
    x2 = np.copy(x1)
    a  = np.array( [[-1.,-2.,-3.], [-4.,-5.,-6.]], order='F' )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_float_2d_F_div(language):

    f1 = arrays.array_float_2d_F_div
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]], order='F' )
    x2 = np.copy(x1)
    a  = np.array( [[-1.,-2.,-3.], [-4.,-5.,-6.]], order='F' )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_float_2d_F_mod(language):

    f1 = arrays.array_float_2d_F_mod
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]], order='F' )
    x2 = np.copy(x1)
    a  = np.array( [[-1.,-2.,-3.], [-4.,-5.,-6.]], order='F' )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_float_2d_F_array_initialization(language):

    f1 = arrays.array_float_2d_F_array_initialization
    f2 = epyccel(f1, language = language)

    x1 = np.zeros((2, 3), dtype=float, order='F')
    x2 = np.ones_like(x1)

    f1(x1)
    f2(x2)

    assert np.array_equal(x1, x2)

def test_array_float_3d_F_array_initialization_1(language):

    f1 = arrays.array_float_3d_F_array_initialization_1
    f2 = epyccel(f1, language = language)

    x  = np.random.random((3,2)).copy(order='F')
    y  = np.random.random((3,2)).copy(order='F')
    a  = np.array([x,y], order='F')

    x1 = np.zeros_like(a)
    x2 = np.zeros_like(a)

    f1(x, y, x1)
    f2(x, y, x2)

    assert np.array_equal(x1, x2)

def test_array_float_3d_F_array_initialization_2(language):

    f1 = arrays.array_float_3d_F_array_initialization_2
    f2 = epyccel(f1, language = language)

    x1 = np.zeros((2,3,4), order='F')
    x2 = np.zeros((2,3,4), order='F')

    f1(x1)
    f2(x2)

    assert np.array_equal(x1, x2)

def test_array_float_4d_F_array_initialization(language):

    f1 = arrays.array_float_4d_F_array_initialization
    f2 = epyccel(f1, language = language)

    x  = np.random.random((3,2,4)).copy(order='F')
    y  = np.random.random((3,2,4)).copy(order='F')
    a  = np.array([x,y], order='F')

    x1 = np.zeros_like(a)
    x2 = np.zeros_like(a)

    f1(x, y, x1)
    f2(x, y, x2)

    assert np.array_equal(x1, x2)

@pytest.mark.xfail(reason='Inhomogeneous arguments due to unknown shape')
def test_array_float_4d_F_array_initialization_mixed_ordering(language):

    f1 = arrays.array_float_4d_F_array_initialization_mixed_ordering
    f2 = epyccel(f1, language = language)

    x  = np.array([[16., 17.], [18., 19.]], dtype='float', order='F')
    a  = np.array(([[[0., 1.], [2., 3.]],
                  [[4., 5.], [6., 7.]],
                  [[8., 9.], [10., 11.]]],
                  [[[12., 13.], [14., 15.]],
                  x,
                  [[20., 21.], [22., 23.]]]),
                  dtype='float', order='F')

    x1 = np.zeros_like(a)
    x2 = np.zeros_like(a)

    f1(x, x1)
    f2(x, x2)

    assert np.array_equal(x1, x2)
#==============================================================================
# TEST: COMPLEX EXPRESSIONS IN 3D : TEST CONSTANT AND UNKNOWN SHAPES
#==============================================================================


def test_array_int32_1d_complex_3d_expr(language):

    f1 = arrays.array_int32_1d_complex_3d_expr
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1,2,3], dtype=np.int32 )
    x2 = np.copy(x1)
    a  = np.array( [-1,-2,-3], dtype=np.int32 )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int32_2d_C_complex_3d_expr(language):

    f1 = arrays.array_int32_2d_C_complex_3d_expr
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32 )
    x2 = np.copy(x1)
    a  = np.array( [[-1,-2,-3], [-4,-5,-6]], dtype=np.int32 )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int32_2d_F_complex_3d_expr(language):

    f1 = arrays.array_int32_2d_F_complex_3d_expr
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32, order='F' )
    x2 = np.copy(x1)
    a  = np.array( [[-1,-2,-3], [-4,-5,-6]], dtype=np.int32, order='F' )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int32_in_bool_out_1d_complex_3d_expr(language):

    f1 = arrays.array_int32_in_bool_out_1d_complex_3d_expr
    f2 = epyccel( f1 , language = language)

    x  = np.array( [1,2,3], dtype=np.int32 )
    a  = np.array( [-1,-2,-3], dtype=np.int32 )
    r1 = np.empty( 3 , dtype=bool )
    r2 = np.copy(r1)

    f1(x, a, r1)
    f2(x, a, r2)

    assert np.array_equal( r1, r2 )

def test_array_int32_in_bool_out_2d_C_complex_3d_expr(language):

    f1 = arrays.array_int32_in_bool_out_2d_C_complex_3d_expr
    f2 = epyccel( f1 , language = language)

    x  = np.array( [[1,2,3], [4,5,6]], dtype=np.int32 )
    a  = np.array( [[-1,-2,-3], [-4,-5,-6]], dtype=np.int32 )
    r1 = np.empty( (2,3) , dtype=bool )
    r2 = np.copy(r1)

    f1(x, a, r1)
    f2(x, a, r2)

    assert np.array_equal( r1, r2 )

def test_array_int32_in_bool_out_2d_F_complex_3d_expr(language):

    f1 = arrays.array_int32_in_bool_out_2d_F_complex_3d_expr
    f2 = epyccel( f1 , language = language)

    x  = np.array( [[1,2,3], [4,5,6]], dtype=np.int32, order='F' )
    a  = np.array( [[-1,-2,-3], [-4,-5,-6]], dtype=np.int32, order='F' )
    r1 = np.empty( (2,3) , dtype=bool, order='F' )
    r2 = np.copy(r1)

    f1(x, a, r1)
    f2(x, a, r2)

    assert np.array_equal( r1, r2 )

def test_array_float_1d_complex_3d_expr(language):

    f1 = arrays.array_float_1d_complex_3d_expr
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [1.,2.,3.] )
    x2 = np.copy(x1)
    a  = np.array( [-1.,-2.,-3.] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_float_2d_C_complex_3d_expr(language):

    f1 = arrays.array_float_2d_C_complex_3d_expr
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]] )
    x2 = np.copy(x1)
    a  = np.array( [[-1.,-2.,-3.], [-4.,-5.,-6.]] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_float_2d_F_complex_3d_expr(language):

    f1 = arrays.array_float_2d_F_complex_3d_expr
    f2 = epyccel( f1 , language = language)

    x1 = np.array( [[ 1., 2., 3.], [4.,5.,6.]], order='F' )
    x2 = np.copy(x1)
    a  = np.array( [[-1.,-2.,-3.], [-4.,-5.,-6.]], order='F' )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

#==============================================================================
# TEST: 1D Stack ARRAYS OF REAL
#==============================================================================

def test_array_float_sum_stack_array(language):

    f1 = arrays.array_float_1d_sum_stack_array
    f2 = epyccel( f1 , language = language)
    x1 = f1()
    x2 = f2()
    assert np.equal( x1, x2 )

def test_array_float_div_stack_array(language):

    f1 = arrays.array_float_1d_div_stack_array
    f2 = epyccel( f1 , language = language)
    x1 = f1()
    x2 = f2()
    assert np.equal( x1, x2 )

def test_multiple_stack_array_1(language):

    f1 = arrays.multiple_stack_array_1
    f2 = epyccel(f1, language = language)
    assert np.allclose(f1(), f2(), rtol=RTOL, atol=ATOL)

def test_multiple_stack_array_2(language):

    f1 = arrays.multiple_stack_array_2
    f2 = epyccel(f1, language = language)
    assert np.allclose(f1(), f2(), rtol=RTOL, atol=ATOL)

#==============================================================================
# TEST: 2D Stack ARRAYS OF REAL
#==============================================================================

def test_array_float_sum_2d_stack_array(language):

    f1 = arrays.array_float_2d_sum_stack_array
    f2 = epyccel( f1 , language = language)
    x1 = f1()
    x2 = f2()
    assert np.equal( x1, x2 )

def test_array_float_div_2d_stack_array(language):

    f1 = arrays.array_float_2d_div_stack_array
    f2 = epyccel( f1 , language = language)
    x1 = f1()
    x2 = f2()
    assert np.equal( x1, x2 )

def test_multiple_2d_stack_array_1(language):

    f1 = arrays.multiple_2d_stack_array_1
    f2 = epyccel(f1, language = language)
    assert np.allclose(f1(), f2(), rtol=RTOL, atol=ATOL)

def test_multiple_2d_stack_array_2(language):

    f1 = arrays.multiple_2d_stack_array_2
    f2 = epyccel(f1, language = language)
    assert np.allclose(f1(), f2(), rtol=RTOL, atol=ATOL)

#==============================================================================
# TEST: Product and matrix multiplication
#==============================================================================
@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="prod not implemented in c"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("python", marks = pytest.mark.python)
    ]
)
def test_array_float_1d_1d_prod(language):
    f1 = arrays.array_float_1d_1d_prod
    f2 = epyccel( f1 , language = language)
    x1 = np.array([3.0, 2.0, 1.0])
    x2 = np.copy(x1)
    y1 = np.empty(3)
    y2 = np.empty(3)
    f1(x1, y1)
    f2(x2, y2)
    assert np.array_equal(y1, y2)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="matmul not implemented in c"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("python", marks = pytest.mark.python)
    ]
)
def test_array_float_2d_1d_matmul(language):
    f1 = arrays.array_float_2d_1d_matmul
    f2 = epyccel( f1 , language = language)
    A1 = np.ones([3, 2])
    A1[1,0] = 2
    A2 = np.copy(A1)
    x1 = np.ones([2])
    x2 = np.copy(x1)
    y1 = np.empty([3])
    y2 = np.empty([3])
    f1(A1, x1, y1)
    f2(A2, x2, y2)
    assert np.array_equal(y1, y2)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="matmul not implemented in c"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("python", marks = pytest.mark.python)
    ]
)
def test_array_float_2d_1d_matmul_creation(language):
    f1 = arrays.array_float_2d_1d_matmul_creation
    f2 = epyccel( f1 , language = language)
    A1 = np.ones([3, 2])
    A1[1,0] = 2
    A2 = np.copy(A1)
    x1 = np.ones([2])
    x2 = np.copy(x1)
    y1 = f1(A1, x1)
    y2 = f2(A2, x2)
    assert np.isclose(y1, y2)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="matmul not implemented in c"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("python", marks = pytest.mark.python)
    ]
)
def test_array_float_2d_1d_matmul_order_F_F(language):
    f1 = arrays.array_float_2d_1d_matmul_order_F
    f2 = epyccel( f1 , language = language)
    A1 = np.ones([3, 2], order='F')
    A1[1,0] = 2
    A2 = np.copy(A1)
    x1 = np.ones([2])
    x2 = np.copy(x1)
    y1 = np.empty([3])
    y2 = np.empty([3])
    f1(A1, x1, y1)
    f2(A2, x2, y2)
    assert np.array_equal(y1, y2)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="matmul not implemented in c"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("python", marks = pytest.mark.python)
    ]
)
def test_array_float_2d_2d_matmul(language):
    f1 = arrays.array_float_2d_2d_matmul
    f2 = epyccel( f1 , language = language)
    A1 = np.ones([3, 2])
    A1[1, 0] = 2
    A2 = np.copy(A1)
    B1 = np.ones([2, 3])
    B2 = np.copy(B1)
    C1 = np.empty([3,3])
    C2 = np.empty([3,3])
    f1(A1, B1, C1)
    f2(A2, B2, C2)
    assert np.array_equal(C1, C2)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="matmul not implemented in c"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("python", marks = pytest.mark.python)
    ]
)
def test_array_float_2d_2d_matmul_F_F_F_F(language):
    f1 = arrays.array_float_2d_2d_matmul_F_F
    f2 = epyccel( f1 , language = language)
    A1 = np.ones([3, 2], order='F')
    A1[1, 0] = 2
    A2 = np.copy(A1)
    B1 = np.ones([2, 3], order='F')
    B2 = np.copy(B1)
    C1 = np.empty([3,3], order='F')
    C2 = np.empty([3,3], order='F')
    f1(A1, B1, C1)
    f2(A2, B2, C2)
    assert np.array_equal(C1, C2)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="matmul not implemented in c"),
            pytest.mark.c]),
        pytest.param("fortran", marks = [
            pytest.mark.fortran,
            pytest.mark.skip(reason="Should fail as long as mixed order not supported, see #244")
            ]),
        pytest.param("python", marks = pytest.mark.python)
    ]
)
def test_array_float_2d_2d_matmul_mixorder(language):
    f1 = arrays.array_float_2d_2d_matmul_mixorder
    f2 = epyccel( f1 , language = language)
    A1 = np.ones([3, 2])
    A1[1, 0] = 2
    A2 = np.copy(A1)
    B1 = np.ones([2, 3], order = 'F')
    B2 = np.copy(B1)
    C1 = np.empty([3,3])
    C2 = np.empty([3,3])
    f1(A1, B1, C1)
    f2(A2, B2, C2)
    assert np.array_equal(C1, C2)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="matmul not implemented in c"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("python", marks = pytest.mark.python)
    ]
)
def test_array_float_2d_2d_matmul_operator(language):
    f1 = arrays.array_float_2d_2d_matmul_operator
    f2 = epyccel( f1 , language = language)
    A1 = np.ones([3, 2])
    A1[1, 0] = 2
    A2 = np.copy(A1)
    B1 = np.ones([2, 3])
    B2 = np.copy(B1)
    C1 = np.empty([3,3])
    C2 = np.empty([3,3])
    f1(A1, B1, C1)
    f2(A2, B2, C2)
    assert np.array_equal(C1, C2)

def test_array_float_loopdiff(language):
    f1 = arrays.array_float_loopdiff
    f2 = epyccel( f1 , language = language)
    x1 = np.ones(5)
    y1 = np.zeros(5)
    x2 = np.copy(x1)
    y2 = np.copy(y1)
    z1 = np.empty(5)
    z2 = np.empty(5)
    f1(x1, y1, z1)
    f2(x2, y2, z2)
    assert np.array_equal(z1, z2)

#==============================================================================
# TEST: keyword arguments
#==============================================================================
def test_array_kwargs_full(language):
    f1 = arrays.array_kwargs_full
    f2 = epyccel( f1 , language = language)
    assert f1() == f2()

def test_array_kwargs_ones(language):
    f1 = arrays.array_kwargs_ones
    f2 = epyccel( f1 , language = language)
    assert f1() == f2()

#==============================================================================
# TEST: Negative indexes
#==============================================================================

def test_constant_negative_index(language):
    from numpy.random import randint
    n = randint(2, 10)
    f1 = arrays.constant_negative_index
    f2 = epyccel( f1 , language = language)
    assert f1(n) == f2(n)

def test_almost_negative_index(language):
    from numpy.random import randint
    n = randint(2, 10)
    f1 = arrays.constant_negative_index
    f2 = epyccel( f1 , language = language)
    assert f1(n) == f2(n)

def test_var_negative_index(language):
    from numpy.random import randint
    n = randint(2, 10)
    idx = randint(-n,0)
    f1 = arrays.var_negative_index
    f2 = epyccel( f1 , language = language)
    assert f1(n,idx) == f2(n,idx)

def test_expr_negative_index(language):
    from numpy.random import randint
    n = randint(2, 10)
    idx1 = randint(-n,2*n)
    idx2 = randint(idx1,idx1+n+1)
    f1 = arrays.expr_negative_index
    f2 = epyccel( f1 , language = language)
    assert f1(n,idx1,idx2) == f2(n,idx1,idx2)

def test_multiple_negative_index(language):
    f1 = arrays.test_multiple_negative_index
    f2 = epyccel(f1, language = language)

    assert f1(-2, -1) == f2(-2, -1)

def test_multiple_negative_index_2(language):
    f1 = arrays.test_multiple_negative_index_2
    f2 = epyccel(f1, language = language)

    assert f1(-4, -2) == f2(-4, -2)

def test_multiple_negative_index_3(language):
    f1 = arrays.test_multiple_negative_index_3
    f2 = epyccel(f1, language = language)

    assert f1(-1, -1, -3) == f2(-1, -1, -3)

def test_argument_negative_index_1(language):
    a = arrays.a_1d

    f1 = arrays.test_argument_negative_index_1
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_argument_negative_index_2(language):
    a = arrays.a_1d

    f1 = arrays.test_argument_negative_index_2
    f2 = epyccel(f1, language = language)
    assert f1(a, a) == f2(a, a)

def test_c_order_argument_negative_index(language):
    a = np.random.randint(20, size=(3,4))

    f1 = arrays.test_c_order_argument_negative_index
    f2 = epyccel(f1, language = language)
    assert f1(a, a) == f2(a, a)

def test_f_order_argument_negative_index(language):
    a = np.array(np.random.randint(20, size=(3,4)), order='F')

    f1 = arrays.test_f_order_argument_negative_index
    f2 = epyccel(f1, language = language)
    assert f1(a, a) == f2(a, a)

#==============================================================================
# TEST: shape initialisation
#==============================================================================

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="randint not implemented in c"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_random_size(language):
    f1 = arrays.array_random_size
    f2 = epyccel( f1 , language = language)
    s1, s2 = f2()
    assert s1 == s2

def test_array_variable_size(language):
    f1 = arrays.array_variable_size
    f2 = epyccel( f1 , language = language)
    from numpy.random import randint
    n = randint(1, 10)
    m = randint(11,20)
    s1, s2 = f2(n,m)
    assert s1 == s2

#==============================================================================
# TEST : 1d array slices
#==============================================================================

def test_array_1d_slice_1(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_1
    f2 = epyccel(f1, language = language)

    assert f1(a) == f2(a)

def test_array_1d_slice_2(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_2
    f2 = epyccel(f1, language = language)

    assert f1(a) == f2(a)

def test_array_1d_slice_3(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_3
    f2 = epyccel(f1, language = language)

    assert f1(a) == f2(a)

def test_array_1d_slice_4(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_4
    f2 = epyccel(f1, language = language)

    assert f1(a) == f2(a)

def test_array_1d_slice_5(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_5
    f2 = epyccel(f1, language = language)

    assert f1(a) == f2(a)

def test_array_1d_slice_6(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_6
    f2 = epyccel(f1, language = language)

    assert f1(a) == f2(a)

def test_array_1d_slice_7(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_7
    f2 = epyccel(f1, language = language)

    assert f1(a) == f2(a)

def test_array_1d_slice_8(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_8
    f2 = epyccel(f1, language = language)

    assert f1(a) == f2(a)

def test_array_1d_slice_9(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_9
    f2 = epyccel(f1, language = language)

    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Array slicing does not work with negative variables in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_1d_slice_10(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_10
    f2 = epyccel(f1, language = language)

    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Array slicing does not work with negative variables in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_1d_slice_11(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_11
    f2 = epyccel(f1, language = language)

    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Array slicing does not work with negative variables in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_1d_slice_12(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_12
    f2 = epyccel(f1, language = language)

    assert f1(a) == f2(a)

#==============================================================================
# TEST : 2d array slices order F
#==============================================================================

def test_array_2d_F_slice_1(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_1
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_F_slice_2(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_2
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_F_slice_3(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_3
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_F_slice_4(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_4
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_F_slice_5(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_5
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_F_slice_6(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_6
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_F_slice_7(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_7
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_F_slice_8(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_8
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_F_slice_9(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_9
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_F_slice_10(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_10
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_F_slice_11(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_11
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_F_slice_12(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_12
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_F_slice_13(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_13
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_F_slice_14(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_14
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_F_slice_15(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_15
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_F_slice_16(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_16
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_F_slice_17(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_17
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_F_slice_18(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_18
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_F_slice_19(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_19
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_F_slice_20(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_20
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Array slicing does not work with negative variables in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_F_slice_21(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_21
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Array slicing does not work with negative variables in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_F_slice_22(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_22
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Array slicing does not work with negative variables in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_F_slice_23(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_23
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

#==============================================================================
# TEST : 2d array slices order C
#==============================================================================


def test_array_2d_C_slice_1(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_1
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_C_slice_2(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_2
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_C_slice_3(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_3
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_C_slice_4(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_4
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_C_slice_5(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_5
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_C_slice_6(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_6
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_C_slice_7(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_7
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_C_slice_8(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_8
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_C_slice_9(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_9
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_C_slice_10(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_10
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_C_slice_11(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_11
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_C_slice_12(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_12
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_C_slice_13(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_13
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_C_slice_14(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_14
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_C_slice_15(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_15
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_C_slice_16(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_16
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_C_slice_17(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_17
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_C_slice_18(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_18
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_C_slice_19(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_19
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_C_slice_20(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_20
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Array slicing does not work with negative variables in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_C_slice_21(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_21
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Array slicing does not work with negative variables in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_C_slice_22(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_22
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Array slicing does not work with negative variables in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_C_slice_23(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_23
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

#==============================================================================
# TEST : 1d array slices stride
#==============================================================================

def test_array_1d_slice_stride_1(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_stride_1
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)


@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_1d_slice_stride_2(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_stride_2
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)


def test_array_1d_slice_stride_3(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_stride_3
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)


def test_array_1d_slice_stride_4(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_stride_4
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)


def test_array_1d_slice_stride_5(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_stride_5
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)


@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_1d_slice_stride_6(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_stride_6
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)


def test_array_1d_slice_stride_7(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_stride_7
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)


@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_1d_slice_stride_8(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_stride_8
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_1d_slice_stride_9(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_stride_9
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)


@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_1d_slice_stride_10(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_stride_10
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_1d_slice_stride_11(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_stride_11
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)


@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_1d_slice_stride_12(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_stride_12
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_1d_slice_stride_13(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_stride_13
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_1d_slice_stride_14(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_stride_14
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)


@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_1d_slice_stride_15(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_stride_15
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_1d_slice_stride_16(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_stride_16
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_1d_slice_stride_17(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_stride_17
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_1d_slice_stride_18(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_stride_18
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_1d_slice_stride_19(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_stride_19
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_1d_slice_stride_20(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_stride_20
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_1d_slice_stride_21(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_stride_21
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_1d_slice_stride_22(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_stride_22
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_1d_slice_stride_23(language):
    a = arrays.a_1d

    f1 = arrays.array_1d_slice_stride_23
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

#==============================================================================
# TEST : 2d array slices stride order F
#==============================================================================

def test_array_2d_F_slice_stride_1(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_stride_1
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_F_slice_stride_2(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_stride_2
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_F_slice_stride_3(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_stride_3
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_F_slice_stride_4(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_stride_4
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_F_slice_stride_5(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_stride_5
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_F_slice_stride_6(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_stride_6
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_F_slice_stride_7(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_stride_7
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)


def test_array_2d_F_slice_stride_8(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_stride_8
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_F_slice_stride_9(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_stride_9
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_F_slice_stride_10(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_stride_10
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_F_slice_stride_11(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_stride_11
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_F_slice_stride_12(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_stride_12
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_F_slice_stride_13(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_stride_13
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_F_slice_stride_14(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_stride_14
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_F_slice_stride_15(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_stride_15
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_F_slice_stride_16(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_stride_16
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_F_slice_stride_17(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_stride_17
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_F_slice_stride_18(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_stride_18
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_F_slice_stride_19(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_stride_19
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_F_slice_stride_20(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_stride_20
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)


@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_F_slice_stride_21(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_stride_21
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_F_slice_stride_22(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_stride_22
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_F_slice_stride_23(language):
    a = arrays.a_2d_f

    f1 = arrays.array_2d_F_slice_stride_23
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

#==============================================================================
# TEST : 2d array slices stride order C
#==============================================================================

def test_array_2d_C_slice_stride_1(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_stride_1
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_C_slice_stride_2(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_stride_2
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_C_slice_stride_3(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_stride_3
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_C_slice_stride_4(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_stride_4
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_C_slice_stride_5(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_stride_5
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

def test_array_2d_C_slice_stride_6(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_stride_6
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_C_slice_stride_7(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_stride_7
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)


def test_array_2d_C_slice_stride_8(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_stride_8
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_C_slice_stride_9(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_stride_9
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_C_slice_stride_10(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_stride_10
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_C_slice_stride_11(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_stride_11
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_C_slice_stride_12(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_stride_12
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_C_slice_stride_13(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_stride_13
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_C_slice_stride_14(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_stride_14
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_C_slice_stride_15(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_stride_15
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_C_slice_stride_16(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_stride_16
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_C_slice_stride_17(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_stride_17
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_C_slice_stride_18(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_stride_18
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_C_slice_stride_19(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_stride_19
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_C_slice_stride_20(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_stride_20
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_C_slice_stride_21(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_stride_21
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_C_slice_stride_22(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_stride_22
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_array_2d_C_slice_stride_23(language):
    a = arrays.a_2d_c

    f1 = arrays.array_2d_C_slice_stride_23
    f2 = epyccel(f1, language = language)
    assert f1(a) == f2(a)

#==============================================================================
# TEST : arithmetic operations
#==============================================================================

def test_arrs_similar_shapes_0(language):
    f1 = arrays.arrs_similar_shapes_0
    f2 = epyccel(f1, language = language)
    assert f1() == f2()

def test_arrs_similar_shapes_1(language):
    f1 = arrays.arrs_similar_shapes_1
    f2 = epyccel(f1, language = language)
    assert f1() == f2()

def test_arrs_different_shapes_0(language):
    f1 = arrays.arrs_different_shapes_0
    f2 = epyccel(f1, language = language)
    assert f1() == f2()

def test_arrs_uncertain_shape_1(language):
    f1 = arrays.arrs_uncertain_shape_1
    f2 = epyccel(f1, language = language)
    assert f1() == f2()

def test_arrs_2d_similar_shapes_0(language):
    f1 = arrays.arrs_2d_similar_shapes_0
    f2 = epyccel(f1, language = language)
    assert f1() == f2()

def test_arrs_2d_different_shapes_0(language):
    f1 = arrays.arrs_2d_different_shapes_0
    f2 = epyccel(f1, language = language)
    assert f1() == f2()

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="Negative start of range does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_arrs_1d_negative_index_1(language):
    f1 = arrays.arrs_1d_negative_index_1
    f2 = epyccel(f1, language = language)
    assert np.allclose(f1(), f2(), rtol=RTOL, atol=ATOL)

def test_arrs_1d_negative_index_2(language):
    f1 = arrays.arrs_1d_negative_index_2
    f2 = epyccel(f1, language = language)
    assert np.allclose(f1(), f2(), rtol=RTOL, atol=ATOL)

def test_arrs_1d_int32_index(language):
    f1 = arrays.arrs_1d_int32_index
    f2 = epyccel(f1, language = language)
    assert f1() == f2()

def test_arrs_1d_int64_index(language):
    f1 = arrays.arrs_1d_int64_index
    f2 = epyccel(f1, language = language)
    assert f1() == f2()

def test_arr_tuple_slice_index(language):
    f1 = arrays.arr_tuple_slice_index
    f2 = epyccel(f1, language = language)

    r_python = f1(arrays.a_2d_c)
    r_pyccel = f2(arrays.a_2d_c)

    check_array_equal(r_python, r_pyccel)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_arrs_1d_negative_index_negative_step(language):
    f1 = arrays.arrs_1d_negative_index_negative_step
    f2 = epyccel(f1, language = language)
    assert np.allclose(f1(), f2(), rtol=RTOL, atol=ATOL)

def test_arrs_1d_negative_step_positive_step(language):
    f1 = arrays.arrs_1d_negative_step_positive_step
    f2 = epyccel(f1, language = language)
    assert np.allclose(f1(), f2(), rtol=RTOL, atol=ATOL)

@pytest.mark.parametrize( 'language', [
        pytest.param("c", marks = [
            pytest.mark.skip(reason="negative step does not work in c. See #1311"),
            pytest.mark.c]),
        pytest.param("fortran", marks = pytest.mark.fortran)
    ]
)
def test_arrs_2d_negative_index(language):
    f1 = arrays.arrs_2d_negative_index
    f2 = epyccel(f1, language = language)
    assert np.allclose(f1(), f2(), rtol=RTOL, atol=ATOL)

#==============================================================================
# TEST : NUMPY ARANGE
#==============================================================================

def test_numpy_arange_one_arg(language):
    f1 = arrays.arr_arange_1
    f2 = epyccel(f1, language = language)
    assert f1() == f2()

def test_numpy_arange_two_arg(language):
    f1 = arrays.arr_arange_2
    f2 = epyccel(f1, language = language)
    assert f1() == f2()

def test_numpy_arange_full_arg(language):
    f1 = arrays.arr_arange_3
    f2 = epyccel(f1, language = language)

    r_f1 = f1()
    r_f2 = f2()

    assert (type(r_f1[1]) is type(r_f2[1]))
    np.testing.assert_allclose(f1(), f2(), rtol=RTOL, atol=ATOL)

def test_numpy_arange_with_dtype(language):
    f1 = arrays.arr_arange_4
    f2 = epyccel(f1, language = language)
    assert f1() == f2()

def test_numpy_arange_negative_step(language):
    f1 = arrays.arr_arange_5
    f2 = epyccel(f1, language = language)

    r_f1 = f1()
    r_f2 = f2()

    assert (type(r_f1[1]) is type(r_f2[1]))
    np.testing.assert_allclose(f1(), f2(), rtol=RTOL, atol=ATOL)

def test_numpy_arange_negative_step_2(language):
    f1 = arrays.arr_arange_6
    f2 = epyccel(f1, language = language)

    r_f1 = f1()
    r_f2 = f2()

    assert (type(r_f1[1]) is type(r_f2[1]))
    np.testing.assert_allclose(f1(), f2(), rtol=RTOL, atol=ATOL)

def test_numpy_arange_into_slice(language):
    f1 = arrays.arr_arange_7
    f2 = epyccel(f1, language = language)
    n = randint(2, 10)
    m = randint(2, 10)
    x = np.array(100 * np.random.random((n, m)), dtype=int)
    x_expected = x.copy()
    f1(x_expected)
    f2(x)
    np.testing.assert_allclose(x, x_expected, rtol=RTOL, atol=ATOL)

def test_iterate_slice(language):
    f1 = arrays.iterate_slice
    f2 = epyccel(f1, language = language)
    i = randint(2, 10)
    assert f1(i) == f2(i)
##==============================================================================
## TEST NESTED ARRAYS INITIALIZATION WITH ORDER C
##==============================================================================

def test_array_float_nested_C_array_initialization(language):

    f1 = arrays.array_float_nested_C_array_initialization
    f2 = epyccel(f1, language = language)

    x  = np.random.random((3,2,4))
    y  = np.random.random((2,4))
    z  = np.random.random((2,4))
    a  = np.array([x, [y, z, z], x])

    x1 = np.zeros_like(a)
    x2 = np.zeros_like(a)

    f1(x, y, z, x1)
    f2(x, y, z, x2)

    assert np.array_equal(x1, x2)

def test_array_float_nested_C_array_initialization_2(language):
    f1 = arrays.array_float_nested_C_array_initialization_2
    f2 = epyccel(f1, language = language)

    a = np.random.random((2,2,3))
    e = np.random.random((2,3))
    f = np.random.random(3)
    nested = np.array([[e, [f, f]], a, [[f, f], [f, f]]])

    x1 = np.zeros_like(nested)
    x2 = np.zeros_like(nested)

    f1(a, e, f, x1)
    f2(a, e, f, x2)

    assert np.array_equal(x1, x2)

def test_array_float_nested_C_array_initialization_3(language):
    f1 = arrays.array_float_nested_C_array_initialization_3
    f2 = epyccel(f1, language = language)

    a = np.random.random((2,2,3))
    e = np.random.random((2,3))
    nested = np.array([[e, [[1., 2., 3.], [1., 2., 3.]]], a, [[[1., 2., 3.], [1., 2., 3.]], [[1., 2., 3.], [1., 2., 3.]]]], order="C")

    x1 = np.zeros_like(nested)
    x2 = np.zeros_like(nested)

    f1(a, e, x1)
    f2(a, e, x2)

    assert np.array_equal(x1, x2)

#==============================================================================
# NUMPY SUM
#==============================================================================

def test_arr_bool_sum(language):
    f1 = arrays.arr_bool_sum
    f2 = epyccel(f1, language = language)
    assert f1() == f2()
    assert isinstance(f1(), type(f2()))

def test_tuple_sum(language):
    f1 = arrays.tuple_sum
    f2 = epyccel(f1, language = language)
    assert f1() == f2()

#==============================================================================
# NUMPY LINSPACE
#==============================================================================

def test_multiple_np_linspace(language):
    f1 = arrays.multiple_np_linspace
    f2 = epyccel(f1, language = language)
    assert f1() == f2()

##==============================================================================
## TEST NESTED ARRAYS INITIALIZATION WITH ORDER F
##==============================================================================

def test_array_float_nested_F_array_initialization(language):
    f1 = arrays.array_float_nested_F_array_initialization
    f2 = epyccel(f1, language = language)

    x  = np.random.random((3,2,4))
    y  = np.random.random((2,4))
    z  = np.random.random((2,4))
    a  = np.array([x, [y, z, z], x], order="F")

    x1 = np.zeros_like(a)
    x2 = np.zeros_like(a)

    f1(x, y, z, x1)
    f2(x, y, z, x2)

    assert np.array_equal(x1, x2)

def test_array_float_nested_F_array_initialization_2(language):
    f1 = arrays.array_float_nested_F_array_initialization_2
    f2 = epyccel(f1, language = language)

    a = np.random.random((2,2,3))
    e = np.random.random((2,3))
    f = np.random.random(3)
    nested = np.array([[e, [f, f]], a, [[f, f], [f, f]]], order="F")

    x1 = np.zeros_like(nested)
    x2 = np.zeros_like(nested)

    f1(a, e, f, x1)
    f2(a, e, f, x2)

    assert np.array_equal(x1, x2)

def test_array_float_nested_F_array_initialization_3(language):
    f1 = arrays.array_float_nested_F_array_initialization_3
    f2 = epyccel(f1, language = language)

    a = np.random.random((2,2,3))
    e = np.random.random((2,3))
    nested = np.array([[e, [[1., 2., 3.], [1., 2., 3.]]], a, [[[1., 2., 3.], [1., 2., 3.]], [[1., 2., 3.], [1., 2., 3.]]]], order="F")

    x1 = np.zeros_like(nested)
    x2 = np.zeros_like(nested)

    f1(a, e, x1)
    f2(a, e, x2)

    assert np.array_equal(x1, x2)

##==============================================================================
## TEST SIMPLE ARRAY SLICING WITH ORDER C 1D
##==============================================================================

def test_array_view_steps_C_1D_1(language):
    a = arrays.a_1d

    f1 = arrays.array_view_steps_C_1D_1
    f2 = epyccel(f1, language = language)
    check_array_equal(f1(a), f2(a))

def test_array_view_steps_C_1D_2(language):
    a = arrays.a_1d

    f1 = arrays.array_view_steps_C_1D_2
    f2 = epyccel(f1, language = language)
    check_array_equal(f1(a), f2(a))

##==============================================================================
## TEST SIMPLE ARRAY SLICING WITH ORDER C 2D
##==============================================================================

def test_array_view_steps_C_2D_1(language):
    a = arrays.a_2d_c

    f1 = arrays.array_view_steps_C_2D_1
    f2 = epyccel(f1, language = language)
    check_array_equal(f1(a), f2(a))

def test_array_view_steps_C_2D_2(language):
    a = arrays.a_2d_c

    f1 = arrays.array_view_steps_C_2D_2
    f2 = epyccel(f1, language = language)
    check_array_equal(f1(a), f2(a))

def test_array_view_steps_C_2D_3(language):
    a = arrays.a_2d_c

    f1 = arrays.array_view_steps_C_2D_3
    f2 = epyccel(f1, language = language)
    check_array_equal(f1(a), f2(a))

##==============================================================================
## TEST ARRAY VIEW STEPS ARRAY INITIALIZATION ORDER F 1D
##==============================================================================

def test_array_view_steps_F_1D_1(language):
    a = arrays.a_1d_f

    f1 = arrays.array_view_steps_F_1D_1
    f2 = epyccel(f1, language = language)
    check_array_equal(f1(a), f2(a))

def test_array_view_steps_F_1D_2(language):
    a = arrays.a_1d_f

    f1 = arrays.array_view_steps_F_1D_2
    f2 = epyccel(f1, language = language)
    check_array_equal(f1(a), f2(a))

##==============================================================================
## TEST ARRAY VIEW STEPS ARRAY INITIALIZATION ORDER F 2D
##==============================================================================

def test_array_view_steps_F_2D_1(language):
    a = arrays.a_2d_f

    f1 = arrays.array_view_steps_F_2D_1
    f2 = epyccel(f1, language = language)
    check_array_equal(f1(a), f2(a))

def test_array_view_steps_F_2D_2(language):
    a = arrays.a_2d_f

    f1 = arrays.array_view_steps_F_2D_2
    f2 = epyccel(f1, language = language)
    check_array_equal(f1(a), f2(a))

def test_array_view_steps_F_2D_3(language):
    a = arrays.a_2d_f

    f1 = arrays.array_view_steps_F_2D_3
    f2 = epyccel(f1, language = language)
    check_array_equal(f1(a), f2(a))

#==============================================================================
# TEST: Array with ndmin argument
#==============================================================================

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = [
            pytest.mark.skip(reason=("Template results in wrong ordered arrays")),
            pytest.mark.python]
        )
    )
)
def test_array_ndmin_1(language):
    f1 = arrays.array_ndmin_1
    f2 = epyccel(f1, language = language)

    a = arrays.a_1d
    b = arrays.a_2d_c
    c = arrays.a_2d_c
    d = randint(low = iinfo(int).min, high = iinfo(int).max, dtype=int, size=(2,3,4))
    e = d.copy(order='F')

    check_array_equal(f1(a), f2(a))
    check_array_equal(f1(b), f2(b))
    check_array_equal(f1(c), f2(c))
    check_array_equal(f1(d), f2(d))
    check_array_equal(f1(e), f2(e))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = [
            pytest.mark.skip(reason=("Template results in wrong ordered arrays")),
            pytest.mark.python]
        )
    )
)
def test_array_ndmin_2(language):
    f1 = arrays.array_ndmin_2
    f2 = epyccel(f1, language = language)

    a = arrays.a_1d
    b = arrays.a_2d_c
    c = arrays.a_2d_c
    d = randint(low = iinfo(int).min, high = iinfo(int).max, dtype=int, size=(2,3,4))
    e = d.copy(order='F')

    check_array_equal(f1(a), f2(a))
    check_array_equal(f1(b), f2(b))
    check_array_equal(f1(c), f2(c))
    check_array_equal(f1(d), f2(d))
    check_array_equal(f1(e), f2(e))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = [
            pytest.mark.skip(reason=("Template results in wrong ordered arrays")),
            pytest.mark.python]
        )
    )
)
def test_array_ndmin_4(language):
    f1 = arrays.array_ndmin_4
    f2 = epyccel(f1, language = language)

    a = arrays.a_1d
    b = arrays.a_2d_c
    c = arrays.a_2d_c
    d = randint(low = iinfo(int).min, high = iinfo(int).max, dtype=int, size=(2,3,4))
    e = d.copy(order='F')

    check_array_equal(f1(a), f2(a))
    check_array_equal(f1(b), f2(b))
    check_array_equal(f1(c), f2(c))
    check_array_equal(f1(d), f2(d))
    check_array_equal(f1(e), f2(e))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = [
            pytest.mark.skip(reason=("Template results in wrong ordered arrays")),
            pytest.mark.python]
        )
    )
)
def test_array_ndmin_2_order(language):
    f1 = arrays.array_ndmin_2_order
    f2 = epyccel(f1, language = language)

    a = arrays.a_1d
    b = arrays.a_2d_c
    c = arrays.a_2d_c
    d = randint(low = iinfo(int).min, high = iinfo(int).max, dtype=int, size=(2,3,4))
    e = d.copy(order='F')

    check_array_equal(f1(a), f2(a))
    check_array_equal(f1(b), f2(b))
    check_array_equal(f1(c), f2(c))
    check_array_equal(f1(d), f2(d))
    check_array_equal(f1(e), f2(e))

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_bool_to_other_types_1(language):
    size = (2, 2)
    bl = randint(0, 2, size = size, dtype= bool)

    epyccel_func = epyccel(arrays.dtype_convert_to_int8, language=language)
    assert epyccel_func(bl) == arrays.dtype_convert_to_int8(bl)

    epyccel_func = epyccel(arrays.dtype_convert_to_bool, language=language)
    assert epyccel_func(bl) == arrays.dtype_convert_to_bool(bl)

    epyccel_func = epyccel(arrays.dtype_convert_to_int16, language=language)
    assert epyccel_func(bl) == arrays.dtype_convert_to_int16(bl)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_bool_to_other_types_2(language):
    size = (2, 2)
    bl = randint(0, 2, size = size, dtype= bool)

    epyccel_func = epyccel(arrays.dtype_convert_to_int32, language=language)
    assert epyccel_func(bl) == arrays.dtype_convert_to_int32(bl)

    epyccel_func = epyccel(arrays.dtype_convert_to_int64, language=language)
    assert epyccel_func(bl) == arrays.dtype_convert_to_int64(bl)

    epyccel_func = epyccel(arrays.dtype_convert_to_float32, language=language)
    assert epyccel_func(bl) == arrays.dtype_convert_to_float32(bl)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_bool_to_other_types_3(language):
    size = (2, 2)
    bl = randint(0, 2, size = size, dtype= bool)

    epyccel_func = epyccel(arrays.dtype_convert_to_float64, language=language)
    assert epyccel_func(bl) == arrays.dtype_convert_to_float64(bl)

    epyccel_func = epyccel(arrays.dtype_convert_to_cfloat, language=language)
    assert epyccel_func(bl) == arrays.dtype_convert_to_cfloat(bl)

    epyccel_func = epyccel(arrays.dtype_convert_to_cdouble, language=language)
    assert epyccel_func(bl) == arrays.dtype_convert_to_cdouble(bl)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_bool_to_other_types_4(language):
    size = (2, 2)
    bl = randint(0, 2, size = size, dtype= bool)

    epyccel_func = epyccel(arrays.dtype_convert_to_pyint, language=language)
    assert epyccel_func(bl) == arrays.dtype_convert_to_pyint(bl)

    epyccel_func = epyccel(arrays.dtype_convert_to_pyfloat, language=language)
    assert epyccel_func(bl) == arrays.dtype_convert_to_pyfloat(bl)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_int8_to_other_types_1(language):
    size = (2, 2)
    integer8 = randint(low = iinfo('int8').min, high = iinfo('int8').max , size = size, dtype=np.int8)

    epyccel_func = epyccel(arrays.dtype_convert_to_int8, language=language)
    assert epyccel_func(integer8) == arrays.dtype_convert_to_int8(integer8)

    epyccel_func = epyccel(arrays.dtype_convert_to_bool, language=language)
    assert epyccel_func(integer8) == arrays.dtype_convert_to_bool(integer8)

    epyccel_func = epyccel(arrays.dtype_convert_to_int16, language=language)
    assert epyccel_func(integer8) == arrays.dtype_convert_to_int16(integer8)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_int8_to_other_types_2(language):
    size = (2, 2)
    integer8 = randint(low = iinfo('int8').min, high = iinfo('int8').max , size = size, dtype=np.int8)

    epyccel_func = epyccel(arrays.dtype_convert_to_int32, language=language)
    assert epyccel_func(integer8) == arrays.dtype_convert_to_int32(integer8)

    epyccel_func = epyccel(arrays.dtype_convert_to_int64, language=language)
    assert epyccel_func(integer8) == arrays.dtype_convert_to_int64(integer8)

    epyccel_func = epyccel(arrays.dtype_convert_to_float32, language=language)
    assert epyccel_func(integer8) == arrays.dtype_convert_to_float32(integer8)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_int8_to_other_types_3(language):
    size = (2, 2)
    integer8 = randint(low = iinfo('int8').min, high = iinfo('int8').max , size = size, dtype=np.int8)

    epyccel_func = epyccel(arrays.dtype_convert_to_float64, language=language)
    assert epyccel_func(integer8) == arrays.dtype_convert_to_float64(integer8)

    epyccel_func = epyccel(arrays.dtype_convert_to_cfloat, language=language)
    assert epyccel_func(integer8) == arrays.dtype_convert_to_cfloat(integer8)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_int8_to_other_types_4(language):
    size = (2, 2)
    integer8 = randint(low = iinfo('int8').min, high = iinfo('int8').max , size = size, dtype=np.int8)

    epyccel_func = epyccel(arrays.dtype_convert_to_cdouble, language=language)
    assert epyccel_func(integer8) == arrays.dtype_convert_to_cdouble(integer8)

    epyccel_func = epyccel(arrays.dtype_convert_to_pyint, language=language)
    assert epyccel_func(integer8) == arrays.dtype_convert_to_pyint(integer8)

    epyccel_func = epyccel(arrays.dtype_convert_to_pyfloat, language=language)
    assert epyccel_func(integer8) == arrays.dtype_convert_to_pyfloat(integer8)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_int16_to_other_types_1(language):
    size = (2, 2)
    integer16 = randint(low = iinfo('int16').min, high = iinfo('int16').max , size = size, dtype=np.int16)

    epyccel_func = epyccel(arrays.dtype_convert_to_int8, language=language)
    assert epyccel_func(integer16) == arrays.dtype_convert_to_int8(integer16)

    epyccel_func = epyccel(arrays.dtype_convert_to_bool, language=language)
    assert epyccel_func(integer16) == arrays.dtype_convert_to_bool(integer16)

    epyccel_func = epyccel(arrays.dtype_convert_to_int16, language=language)
    assert epyccel_func(integer16) == arrays.dtype_convert_to_int16(integer16)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_int16_to_other_types_2(language):
    size = (2, 2)
    integer16 = randint(low = iinfo('int16').min, high = iinfo('int16').max , size = size, dtype=np.int16)

    epyccel_func = epyccel(arrays.dtype_convert_to_int32, language=language)
    assert epyccel_func(integer16) == arrays.dtype_convert_to_int32(integer16)

    epyccel_func = epyccel(arrays.dtype_convert_to_int64, language=language)
    assert epyccel_func(integer16) == arrays.dtype_convert_to_int64(integer16)

    epyccel_func = epyccel(arrays.dtype_convert_to_float32, language=language)
    assert epyccel_func(integer16) == arrays.dtype_convert_to_float32(integer16)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_int16_to_other_types_3(language):
    size = (2, 2)
    integer16 = randint(low = iinfo('int16').min, high = iinfo('int16').max , size = size, dtype=np.int16)

    epyccel_func = epyccel(arrays.dtype_convert_to_float64, language=language)
    assert epyccel_func(integer16) == arrays.dtype_convert_to_float64(integer16)

    epyccel_func = epyccel(arrays.dtype_convert_to_cfloat, language=language)
    assert epyccel_func(integer16) == arrays.dtype_convert_to_cfloat(integer16)

    epyccel_func = epyccel(arrays.dtype_convert_to_cdouble, language=language)
    assert epyccel_func(integer16) == arrays.dtype_convert_to_cdouble(integer16)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_int16_to_other_types_4(language):
    size = (2, 2)
    integer16 = randint(low = iinfo('int16').min, high = iinfo('int16').max , size = size, dtype=np.int16)

    epyccel_func = epyccel(arrays.dtype_convert_to_pyint, language=language)
    assert epyccel_func(integer16) == arrays.dtype_convert_to_pyint(integer16)

    epyccel_func = epyccel(arrays.dtype_convert_to_pyfloat, language=language)
    assert epyccel_func(integer16) == arrays.dtype_convert_to_pyfloat(integer16)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_int32_to_other_types_1(language):
    size = (2, 2)
    integer32 = randint(low = iinfo('int32').min, high = iinfo('int32').max , size = size, dtype=np.int32)

    epyccel_func = epyccel(arrays.dtype_convert_to_int8, language=language)
    assert epyccel_func(integer32) == arrays.dtype_convert_to_int8(integer32)

    epyccel_func = epyccel(arrays.dtype_convert_to_bool, language=language)
    assert epyccel_func(integer32) == arrays.dtype_convert_to_bool(integer32)

    epyccel_func = epyccel(arrays.dtype_convert_to_int16, language=language)
    assert epyccel_func(integer32) == arrays.dtype_convert_to_int16(integer32)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_int32_to_other_types_2(language):
    size = (2, 2)
    integer32 = randint(low = iinfo('int32').min, high = iinfo('int32').max , size = size, dtype=np.int32)

    epyccel_func = epyccel(arrays.dtype_convert_to_int32, language=language)
    assert epyccel_func(integer32) == arrays.dtype_convert_to_int32(integer32)

    epyccel_func = epyccel(arrays.dtype_convert_to_int64, language=language)
    assert epyccel_func(integer32) == arrays.dtype_convert_to_int64(integer32)

    epyccel_func = epyccel(arrays.dtype_convert_to_float32, language=language)
    assert epyccel_func(integer32) == arrays.dtype_convert_to_float32(integer32)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_int32_to_other_types_3(language):
    size = (2, 2)
    integer32 = randint(low = iinfo('int32').min, high = iinfo('int32').max , size = size, dtype=np.int32)

    epyccel_func = epyccel(arrays.dtype_convert_to_float64, language=language)
    assert epyccel_func(integer32) == arrays.dtype_convert_to_float64(integer32)

    epyccel_func = epyccel(arrays.dtype_convert_to_cfloat, language=language)
    assert epyccel_func(integer32) == arrays.dtype_convert_to_cfloat(integer32)

    epyccel_func = epyccel(arrays.dtype_convert_to_cdouble, language=language)
    assert epyccel_func(integer32) == arrays.dtype_convert_to_cdouble(integer32)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_int32_to_other_types_4(language):
    size = (2, 2)
    integer32 = randint(low = iinfo('int32').min, high = iinfo('int32').max , size = size, dtype=np.int32)

    epyccel_func = epyccel(arrays.dtype_convert_to_pyint, language=language)
    assert epyccel_func(integer32) == arrays.dtype_convert_to_pyint(integer32)

    epyccel_func = epyccel(arrays.dtype_convert_to_pyfloat, language=language)
    assert epyccel_func(integer32) == arrays.dtype_convert_to_pyfloat(integer32)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_int64_to_other_types_1(language):
    size = (2, 2)
    integer64 = randint(low = iinfo('int64').min, high = iinfo('int64').max , size = size, dtype=np.int64)

    epyccel_func = epyccel(arrays.dtype_convert_to_int8, language=language)
    assert epyccel_func(integer64) == arrays.dtype_convert_to_int8(integer64)

    epyccel_func = epyccel(arrays.dtype_convert_to_bool, language=language)
    assert epyccel_func(integer64) == arrays.dtype_convert_to_bool(integer64)

    epyccel_func = epyccel(arrays.dtype_convert_to_int16, language=language)
    assert epyccel_func(integer64) == arrays.dtype_convert_to_int16(integer64)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_int64_to_other_types_2(language):
    size = (2, 2)
    integer64 = randint(low = iinfo('int64').min, high = iinfo('int64').max , size = size, dtype=np.int64)

    epyccel_func = epyccel(arrays.dtype_convert_to_int32, language=language)
    assert epyccel_func(integer64) == arrays.dtype_convert_to_int32(integer64)

    epyccel_func = epyccel(arrays.dtype_convert_to_int64, language=language)
    assert epyccel_func(integer64) == arrays.dtype_convert_to_int64(integer64)

    epyccel_func = epyccel(arrays.dtype_convert_to_float32, language=language)
    assert epyccel_func(integer64) == arrays.dtype_convert_to_float32(integer64)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_int64_to_other_types_3(language):
    size = (2, 2)
    integer64 = randint(low = iinfo('int64').min, high = iinfo('int64').max , size = size, dtype=np.int64)

    epyccel_func = epyccel(arrays.dtype_convert_to_float64, language=language)
    assert epyccel_func(integer64) == arrays.dtype_convert_to_float64(integer64)

    epyccel_func = epyccel(arrays.dtype_convert_to_cfloat, language=language)
    assert epyccel_func(integer64) == arrays.dtype_convert_to_cfloat(integer64)

    epyccel_func = epyccel(arrays.dtype_convert_to_cdouble, language=language)
    assert epyccel_func(integer64) == arrays.dtype_convert_to_cdouble(integer64)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_int64_to_other_types_4(language):
    size = (2, 2)
    integer64 = randint(low = iinfo('int64').min, high = iinfo('int64').max , size = size, dtype=np.int64)

    epyccel_func = epyccel(arrays.dtype_convert_to_pyint, language=language)
    assert epyccel_func(integer64) == arrays.dtype_convert_to_pyint(integer64)

    epyccel_func = epyccel(arrays.dtype_convert_to_pyfloat, language=language)
    assert epyccel_func(integer64) == arrays.dtype_convert_to_pyfloat(integer64)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_float32_to_other_types_1(language):
    size = (2, 2)
    fl32 = uniform(finfo('float32').min / 2, finfo('float32').max / 2, size = size)
    fl32 = np.float32(fl32)

    epyccel_func = epyccel(arrays.dtype_convert_to_int8, language=language)
    assert epyccel_func(fl32) == arrays.dtype_convert_to_int8(fl32)

    epyccel_func = epyccel(arrays.dtype_convert_to_bool, language=language)
    assert epyccel_func(fl32) == arrays.dtype_convert_to_bool(fl32)

    epyccel_func = epyccel(arrays.dtype_convert_to_int16, language=language)
    assert epyccel_func(fl32) == arrays.dtype_convert_to_int16(fl32)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_float32_to_other_types_2(language):
    size = (2, 2)
    fl32 = uniform(finfo('float32').min / 2, finfo('float32').max / 2, size = size)
    fl32 = np.float32(fl32)

    epyccel_func = epyccel(arrays.dtype_convert_to_int32, language=language)
    assert epyccel_func(fl32) == arrays.dtype_convert_to_int32(fl32)

    epyccel_func = epyccel(arrays.dtype_convert_to_int64, language=language)
    assert epyccel_func(fl32) == arrays.dtype_convert_to_int64(fl32)

    epyccel_func = epyccel(arrays.dtype_convert_to_float32, language=language)
    assert epyccel_func(fl32) == arrays.dtype_convert_to_float32(fl32)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_float32_to_other_types_3(language):
    size = (2, 2)
    fl32 = uniform(finfo('float32').min / 2, finfo('float32').max / 2, size = size)
    fl32 = np.float32(fl32)

    epyccel_func = epyccel(arrays.dtype_convert_to_float64, language=language)
    assert epyccel_func(fl32) == arrays.dtype_convert_to_float64(fl32)

    epyccel_func = epyccel(arrays.dtype_convert_to_cfloat, language=language)
    assert epyccel_func(fl32) == arrays.dtype_convert_to_cfloat(fl32)

    epyccel_func = epyccel(arrays.dtype_convert_to_cdouble, language=language)
    assert epyccel_func(fl32) == arrays.dtype_convert_to_cdouble(fl32)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_float32_to_other_types_4(language):
    size = (2, 2)
    fl32 = uniform(finfo('float32').min / 2, finfo('float32').max / 2, size = size)
    fl32 = np.float32(fl32)

    epyccel_func = epyccel(arrays.dtype_convert_to_pyint, language=language)
    assert epyccel_func(fl32) == arrays.dtype_convert_to_pyint(fl32)

    epyccel_func = epyccel(arrays.dtype_convert_to_pyfloat, language=language)
    assert epyccel_func(fl32) == arrays.dtype_convert_to_pyfloat(fl32)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_float64_to_other_types_1(language):
    size = (2, 2)
    fl64 = uniform(finfo('float64').min / 2, finfo('float64').max / 2, size = size)

    epyccel_func = epyccel(arrays.dtype_convert_to_int8, language=language)
    assert epyccel_func(fl64) == arrays.dtype_convert_to_int8(fl64)

    epyccel_func = epyccel(arrays.dtype_convert_to_bool, language=language)
    assert epyccel_func(fl64) == arrays.dtype_convert_to_bool(fl64)

    epyccel_func = epyccel(arrays.dtype_convert_to_int16, language=language)
    assert epyccel_func(fl64) == arrays.dtype_convert_to_int16(fl64)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_float64_to_other_types_2(language):
    size = (2, 2)
    fl64 = uniform(finfo('float64').min / 2, finfo('float64').max / 2, size = size)

    epyccel_func = epyccel(arrays.dtype_convert_to_int32, language=language)
    assert epyccel_func(fl64) == arrays.dtype_convert_to_int32(fl64)

    epyccel_func = epyccel(arrays.dtype_convert_to_int64, language=language)
    assert epyccel_func(fl64) == arrays.dtype_convert_to_int64(fl64)

    epyccel_func = epyccel(arrays.dtype_convert_to_float32, language=language)
    assert epyccel_func(fl64) == arrays.dtype_convert_to_float32(fl64)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_float64_to_other_types_3(language):
    size = (2, 2)
    fl64 = uniform(finfo('float64').min / 2, finfo('float64').max / 2, size = size)

    epyccel_func = epyccel(arrays.dtype_convert_to_float64, language=language)
    assert epyccel_func(fl64) == arrays.dtype_convert_to_float64(fl64)

    epyccel_func = epyccel(arrays.dtype_convert_to_cfloat, language=language)
    assert epyccel_func(fl64) == arrays.dtype_convert_to_cfloat(fl64)

    epyccel_func = epyccel(arrays.dtype_convert_to_cdouble, language=language)
    assert epyccel_func(fl64) == arrays.dtype_convert_to_cdouble(fl64)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_float64_to_other_types_4(language):
    size = (2, 2)
    fl64 = uniform(finfo('float64').min / 2, finfo('float64').max / 2, size = size)

    epyccel_func = epyccel(arrays.dtype_convert_to_pyint, language=language)
    assert epyccel_func(fl64) == arrays.dtype_convert_to_pyint(fl64)

    epyccel_func = epyccel(arrays.dtype_convert_to_pyfloat, language=language)
    assert epyccel_func(fl64) == arrays.dtype_convert_to_pyfloat(fl64)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_complex64_to_other_types_1(language):
    size = (2, 2)
    cmplx128_from_float32 = uniform(low= finfo('float32').min / 2, high= finfo('float32').max / 2, size = size) + uniform(low=finfo('float32').min / 2, high=finfo('float32').max / 2, size = size) * 1j
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)

    epyccel_func = epyccel(arrays.dtype_convert_to_int8, language=language)
    assert epyccel_func(cmplx64) == arrays.dtype_convert_to_int8(cmplx64)

    epyccel_func = epyccel(arrays.dtype_convert_to_bool, language=language)
    assert epyccel_func(cmplx64) == arrays.dtype_convert_to_bool(cmplx64)

    epyccel_func = epyccel(arrays.dtype_convert_to_int16, language=language)
    assert epyccel_func(cmplx64) == arrays.dtype_convert_to_int16(cmplx64)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_complex64_to_other_types_2(language):
    size = (2, 2)
    cmplx128_from_float32 = uniform(low= finfo('float32').min / 2, high= finfo('float32').max / 2, size = size) + uniform(low=finfo('float32').min / 2, high=finfo('float32').max / 2, size = size) * 1j
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)

    epyccel_func = epyccel(arrays.dtype_convert_to_int32, language=language)
    assert epyccel_func(cmplx64) == arrays.dtype_convert_to_int32(cmplx64)

    epyccel_func = epyccel(arrays.dtype_convert_to_int64, language=language)
    assert epyccel_func(cmplx64) == arrays.dtype_convert_to_int64(cmplx64)

    epyccel_func = epyccel(arrays.dtype_convert_to_float32, language=language)
    assert epyccel_func(cmplx64) == arrays.dtype_convert_to_float32(cmplx64)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_complex64_to_other_types_3(language):
    size = (2, 2)
    cmplx128_from_float32 = uniform(low= finfo('float32').min / 2, high= finfo('float32').max / 2, size = size) + uniform(low=finfo('float32').min / 2, high=finfo('float32').max / 2, size = size) * 1j
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)

    epyccel_func = epyccel(arrays.dtype_convert_to_float64, language=language)
    assert epyccel_func(cmplx64) == arrays.dtype_convert_to_float64(cmplx64)

    epyccel_func = epyccel(arrays.dtype_convert_to_cfloat, language=language)
    assert epyccel_func(cmplx64) == arrays.dtype_convert_to_cfloat(cmplx64)

    epyccel_func = epyccel(arrays.dtype_convert_to_cdouble, language=language)
    assert epyccel_func(cmplx64) == arrays.dtype_convert_to_cdouble(cmplx64)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_complex64_to_other_types_4(language):
    size = (2, 2)
    cmplx128_from_float32 = uniform(low= finfo('float32').min / 2, high= finfo('float32').max / 2, size = size) + uniform(low=finfo('float32').min / 2, high=finfo('float32').max / 2, size = size) * 1j
    # the result of the last operation is a Python complex type which has 8 bytes in the alignment,
    # that's why we need to convert it to a numpy.complex64 the needed type.
    cmplx64 = np.complex64(cmplx128_from_float32)

    epyccel_func = epyccel(arrays.dtype_convert_to_pyint, language=language)
    assert epyccel_func(cmplx64) == arrays.dtype_convert_to_pyint(cmplx64)

    epyccel_func = epyccel(arrays.dtype_convert_to_pyfloat, language=language)
    assert epyccel_func(cmplx64) == arrays.dtype_convert_to_pyfloat(cmplx64)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_complex128_to_other_types_1(language):
    size = (2, 2)
    cmplx128 = uniform(low=finfo('float64').min / 2, high=finfo('float64').max / 2, size = size) + uniform(low=finfo('float64').min / 2, high=finfo('float64').max / 2, size = size) * 1j

    epyccel_func = epyccel(arrays.dtype_convert_to_int8, language=language)
    assert epyccel_func(cmplx128) == arrays.dtype_convert_to_int8(cmplx128)

    epyccel_func = epyccel(arrays.dtype_convert_to_bool, language=language)
    assert epyccel_func(cmplx128) == arrays.dtype_convert_to_bool(cmplx128)

    epyccel_func = epyccel(arrays.dtype_convert_to_int16, language=language)
    assert epyccel_func(cmplx128) == arrays.dtype_convert_to_int16(cmplx128)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_complex128_to_other_types_2(language):
    size = (2, 2)
    cmplx128 = uniform(low=finfo('float64').min / 2, high=finfo('float64').max / 2, size = size) + uniform(low=finfo('float64').min / 2, high=finfo('float64').max / 2, size = size) * 1j

    epyccel_func = epyccel(arrays.dtype_convert_to_int32, language=language)
    assert epyccel_func(cmplx128) == arrays.dtype_convert_to_int32(cmplx128)

    epyccel_func = epyccel(arrays.dtype_convert_to_int64, language=language)
    assert epyccel_func(cmplx128) == arrays.dtype_convert_to_int64(cmplx128)

    epyccel_func = epyccel(arrays.dtype_convert_to_float32, language=language)
    assert epyccel_func(cmplx128) == arrays.dtype_convert_to_float32(cmplx128)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_complex128_to_other_types_3(language):
    size = (2, 2)
    cmplx128 = uniform(low=finfo('float64').min / 2, high=finfo('float64').max / 2, size = size) + uniform(low=finfo('float64').min / 2, high=finfo('float64').max / 2, size = size) * 1j

    epyccel_func = epyccel(arrays.dtype_convert_to_float64, language=language)
    assert epyccel_func(cmplx128) == arrays.dtype_convert_to_float64(cmplx128)

    epyccel_func = epyccel(arrays.dtype_convert_to_cfloat, language=language)
    assert epyccel_func(cmplx128) == arrays.dtype_convert_to_cfloat(cmplx128)

    epyccel_func = epyccel(arrays.dtype_convert_to_cdouble, language=language)
    assert epyccel_func(cmplx128) == arrays.dtype_convert_to_cdouble(cmplx128)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_dtype_conversion_complex128_to_other_types_4(language):
    size = (2, 2)
    cmplx128 = uniform(low=finfo('float64').min / 2, high=finfo('float64').max / 2, size = size) + uniform(low=finfo('float64').min / 2, high=finfo('float64').max / 2, size = size) * 1j

    epyccel_func = epyccel(arrays.dtype_convert_to_pyint, language=language)
    assert epyccel_func(cmplx128) == arrays.dtype_convert_to_pyint(cmplx128)

    epyccel_func = epyccel(arrays.dtype_convert_to_pyfloat, language=language)
    assert epyccel_func(cmplx128) == arrays.dtype_convert_to_pyfloat(cmplx128)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_src_dest_array_diff_sizes_dtype_conversion(language):
    size = (1,2)
    integer16_1 = randint(low = iinfo('int16').min, high = iinfo('int16').max , size = size, dtype=np.int16)
    integer16_2 = randint(low = iinfo('int16').min, high = iinfo('int16').max , size = size, dtype=np.int16)
    integer16_3 = randint(low = iinfo('int16').min, high = iinfo('int16').max , size = size, dtype=np.int16)
    integer64_1 = randint(low = iinfo('int64').min, high = iinfo('int64').max , size = size, dtype=np.int64)
    integer64_2 = randint(low = iinfo('int64').min, high = iinfo('int64').max , size = size, dtype=np.int64)
    integer64_3 = randint(low = iinfo('int64').min, high = iinfo('int64').max , size = size, dtype=np.int64)
    b1 = randint(0, 2, size = size, dtype= bool)
    b2 = randint(0, 2, size = size, dtype= bool)
    b3 = randint(0, 2, size = size, dtype= bool)

    epyccel_func = epyccel(arrays.src_dest_diff_sizes_dtype_convert_to_bool, language=language)
    assert epyccel_func(integer64_1, integer64_2, integer64_3) == arrays.src_dest_diff_sizes_dtype_convert_to_bool(integer64_1, integer64_2, integer64_3)

    epyccel_func = epyccel(arrays.src_dest_diff_sizes_dtype_convert_to_float64, language=language)
    assert epyccel_func(integer16_1, integer16_2, integer16_3) == arrays.src_dest_diff_sizes_dtype_convert_to_float64(integer16_1, integer16_2, integer16_3)

    epyccel_func = epyccel(arrays.src_dest_diff_sizes_dtype_convert_to_complex128, language=language)
    assert epyccel_func(b1, b2, b3) == arrays.src_dest_diff_sizes_dtype_convert_to_complex128(b1, b2, b3)

@pytest.mark.parametrize( 'language', (
        pytest.param("fortran", marks = [
            pytest.mark.skip(reason=("Template results in compilation error")),
            pytest.mark.fortran]),
        pytest.param("c", marks = pytest.mark.c),
        pytest.param("python", marks = pytest.mark.python)
    )
)
def test_src_dest_array_diff_sizes_dtype_conversion_orderF(language):
    size = (1,2)
    integer16_1 = randint(low = iinfo('int16').min, high = iinfo('int16').max , size = size, dtype=np.int16)
    integer16_2 = randint(low = iinfo('int16').min, high = iinfo('int16').max , size = size, dtype=np.int16)
    integer16_3 = randint(low = iinfo('int16').min, high = iinfo('int16').max , size = size, dtype=np.int16)
    integer64_1 = randint(low = iinfo('int64').min, high = iinfo('int64').max , size = size, dtype=np.int64)
    integer64_2 = randint(low = iinfo('int64').min, high = iinfo('int64').max , size = size, dtype=np.int64)
    integer64_3 = randint(low = iinfo('int64').min, high = iinfo('int64').max , size = size, dtype=np.int64)
    b1 = randint(0, 2, size = size, dtype= bool)
    b2 = randint(0, 2, size = size, dtype= bool)
    b3 = randint(0, 2, size = size, dtype= bool)

    epyccel_func = epyccel(arrays.src_dest_diff_sizes_dtype_convert_to_bool_orderF, language=language)
    assert epyccel_func(integer64_1, integer64_2, integer64_3) == arrays.src_dest_diff_sizes_dtype_convert_to_bool_orderF(integer64_1, integer64_2, integer64_3)

    epyccel_func = epyccel(arrays.src_dest_diff_sizes_dtype_convert_to_float64_orderF, language=language)
    assert epyccel_func(integer16_1, integer16_2, integer16_3) == arrays.src_dest_diff_sizes_dtype_convert_to_float64_orderF(integer16_1, integer16_2, integer16_3)

    epyccel_func = epyccel(arrays.src_dest_diff_sizes_dtype_convert_to_complex128_orderF, language=language)
    assert epyccel_func(b1, b2, b3) == arrays.src_dest_diff_sizes_dtype_convert_to_complex128_orderF(b1, b2, b3)

#def teardown_module():
#    import os, glob
#    dirname  = os.path.dirname( arrays.__file__ )
#    pattern  = os.path.join( dirname, '__epyccel__*' )
#    filelist = glob.glob( pattern )
#    for f in filelist:
#        os.remove( f )
