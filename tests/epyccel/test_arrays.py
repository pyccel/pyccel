import pytest
import numpy as np

from pyccel.epyccel import epyccel
from modules        import arrays

#==============================================================================
# TEST: 1D ARRAYS OF INT
#==============================================================================

def test_array_int_1d_scalar_add():

    f1 = arrays.array_int_1d_scalar_add
    f2 = epyccel( f1 )

    x1 = np.array( [1,2,3], dtype=np.int32 )
    x2 = x1.copy()
    a = 5

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_1d_scalar_sub():

    f1 = arrays.array_int_1d_scalar_sub
    f2 = epyccel( f1 )

    x1 = np.array( [1,2,3], dtype=np.int32 )
    x2 = x1.copy()
    a = 5

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_1d_scalar_mul():

    f1 = arrays.array_int_1d_scalar_mul
    f2 = epyccel( f1 )

    x1 = np.array( [1,2,3], dtype=np.int32 )
    x2 = x1.copy()
    a = 5

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_1d_scalar_idiv():

    f1 = arrays.array_int_1d_scalar_idiv
    f2 = epyccel( f1 )

    x1 = np.array( [1,2,3], dtype=np.int32 )
    x2 = x1.copy()
    a = 5

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_1d_add():

    f1 = arrays.array_int_1d_add
    f2 = epyccel( f1 )

    x1 = np.array( [1,2,3], dtype=np.int32 )
    x2 = x1.copy()
    a  = np.array( [1,2,3], dtype=np.int32 )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_1d_sub():

    f1 = arrays.array_int_1d_sub
    f2 = epyccel( f1 )

    x1 = np.array( [1,2,3], dtype=np.int32 )
    x2 = x1.copy()
    a  = np.array( [1,2,3], dtype=np.int32 )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_1d_mul():

    f1 = arrays.array_int_1d_mul
    f2 = epyccel( f1 )

    x1 = np.array( [1,2,3], dtype=np.int32 )
    x2 = x1.copy()
    a  = np.array( [1,2,3], dtype=np.int32 )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_1d_idiv():

    f1 = arrays.array_int_1d_idiv
    f2 = epyccel( f1 )

    x1 = np.array( [1,2,3], dtype=np.int32 )
    x2 = x1.copy()
    a  = np.array( [1,2,3], dtype=np.int32 )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

#==============================================================================
# TEST: 2D ARRAYS OF INT WITH C ORDERING
#==============================================================================

def test_array_int_2d_C_scalar_add():

    f1 = arrays.array_int_2d_C_scalar_add
    f2 = epyccel( f1 )

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32 )
    x2 = x1.copy()
    a = 5

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_2d_C_scalar_sub():

    f1 = arrays.array_int_2d_C_scalar_sub
    f2 = epyccel( f1 )

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32 )
    x2 = x1.copy()
    a = 5

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_2d_C_scalar_mul():

    f1 = arrays.array_int_2d_C_scalar_mul
    f2 = epyccel( f1 )

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32 )
    x2 = x1.copy()
    a = 5

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_2d_C_scalar_idiv():

    f1 = arrays.array_int_2d_C_scalar_idiv
    f2 = epyccel( f1 )

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32 )
    x2 = x1.copy()
    a = 5

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_2d_C_add():

    f1 = arrays.array_int_2d_C_add
    f2 = epyccel( f1 )

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32 )
    x2 = x1.copy()
    a  = np.array( [[-1,-2,-3], [-4,-5,-6]], dtype=np.int32 )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_2d_C_sub():

    f1 = arrays.array_int_2d_C_sub
    f2 = epyccel( f1 )

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32 )
    x2 = x1.copy()
    a  = np.array( [[-1,-2,-3], [-4,-5,-6]], dtype=np.int32 )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_2d_C_mul():

    f1 = arrays.array_int_2d_C_mul
    f2 = epyccel( f1 )

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32 )
    x2 = x1.copy()
    a  = np.array( [[-1,-2,-3], [-4,-5,-6]], dtype=np.int32 )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_2d_C_idiv():

    f1 = arrays.array_int_2d_C_idiv
    f2 = epyccel( f1 )

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32 )
    x2 = x1.copy()
    a  = np.array( [[-1,-2,-3], [-4,-5,-6]], dtype=np.int32 )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

#==============================================================================
# TEST: 2D ARRAYS OF INT WITH F ORDERING
#==============================================================================

def test_array_int_2d_F_scalar_add():

    f1 = arrays.array_int_2d_F_scalar_add
    f2 = epyccel( f1 )

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32 )
    x2 = x1.copy()
    a = 5

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_2d_F_scalar_sub():

    f1 = arrays.array_int_2d_F_scalar_sub
    f2 = epyccel( f1 )

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32 )
    x2 = x1.copy()
    a = 5

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_2d_F_scalar_mul():

    f1 = arrays.array_int_2d_F_scalar_mul
    f2 = epyccel( f1 )

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32 )
    x2 = x1.copy()
    a = 5

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_2d_F_scalar_idiv():

    f1 = arrays.array_int_2d_F_scalar_idiv
    f2 = epyccel( f1 )

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32 )
    x2 = x1.copy()
    a = 5

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_2d_F_add():

    f1 = arrays.array_int_2d_F_add
    f2 = epyccel( f1 )

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32 )
    x2 = x1.copy()
    a  = np.array( [[-1,-2,-3], [-4,-5,-6]], dtype=np.int32 )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_2d_F_sub():

    f1 = arrays.array_int_2d_F_sub
    f2 = epyccel( f1 )

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32 )
    x2 = x1.copy()
    a  = np.array( [[-1,-2,-3], [-4,-5,-6]], dtype=np.int32 )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_2d_F_mul():

    f1 = arrays.array_int_2d_F_mul
    f2 = epyccel( f1 )

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32 )
    x2 = x1.copy()
    a  = np.array( [[-1,-2,-3], [-4,-5,-6]], dtype=np.int32 )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_int_2d_F_idiv():

    f1 = arrays.array_int_2d_F_idiv
    f2 = epyccel( f1 )

    x1 = np.array( [[1,2,3], [4,5,6]], dtype=np.int32 )
    x2 = x1.copy()
    a  = np.array( [[-1,-2,-3], [-4,-5,-6]], dtype=np.int32 )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

#==============================================================================
# TEST: 1D ARRAYS OF REAL
#==============================================================================

def test_array_real_1d_scalar_add():

    f1 = arrays.array_real_1d_scalar_add
    f2 = epyccel( f1 )

    x1 = np.array( [1.,2.,3.] )
    x2 = x1.copy()
    a = 5.

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_real_1d_scalar_sub():

    f1 = arrays.array_real_1d_scalar_sub
    f2 = epyccel( f1 )

    x1 = np.array( [1.,2.,3.] )
    x2 = x1.copy()
    a = 5.

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_real_1d_scalar_mul():

    f1 = arrays.array_real_1d_scalar_mul
    f2 = epyccel( f1 )

    x1 = np.array( [1.,2.,3.] )
    x2 = x1.copy()
    a = 5.

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_real_1d_scalar_div():

    f1 = arrays.array_real_1d_scalar_div
    f2 = epyccel( f1 )

    x1 = np.array( [1.,2.,3.] )
    x2 = x1.copy()
    a = 5.

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_real_1d_add():

    f1 = arrays.array_real_1d_add
    f2 = epyccel( f1 )

    x1 = np.array( [1.,2.,3.] )
    x2 = x1.copy()
    a  = np.array( [1.,2.,3.] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_real_1d_sub():

    f1 = arrays.array_real_1d_sub
    f2 = epyccel( f1 )

    x1 = np.array( [1.,2.,3.] )
    x2 = x1.copy()
    a  = np.array( [1.,2.,3.] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_real_1d_mul():

    f1 = arrays.array_real_1d_mul
    f2 = epyccel( f1 )

    x1 = np.array( [1.,2.,3.] )
    x2 = x1.copy()
    a  = np.array( [1.,2.,3.] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_real_1d_div():

    f1 = arrays.array_real_1d_div
    f2 = epyccel( f1 )

    x1 = np.array( [1.,2.,3.] )
    x2 = x1.copy()
    a  = np.array( [1.,2.,3.] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

#==============================================================================
# TEST: 2D ARRAYS OF REAL WITH C ORDERING
#==============================================================================

def test_array_real_2d_C_scalar_add():

    f1 = arrays.array_real_2d_C_scalar_add
    f2 = epyccel( f1 )

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]] )
    x2 = x1.copy()
    a = 5.

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_real_2d_C_scalar_sub():

    f1 = arrays.array_real_2d_C_scalar_sub
    f2 = epyccel( f1 )

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]] )
    x2 = x1.copy()
    a = 5.

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_real_2d_C_scalar_mul():

    f1 = arrays.array_real_2d_C_scalar_mul
    f2 = epyccel( f1 )

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]] )
    x2 = x1.copy()
    a = 5.

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_real_2d_C_scalar_div():

    f1 = arrays.array_real_2d_C_scalar_div
    f2 = epyccel( f1 )

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]] )
    x2 = x1.copy()
    a = 5.

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_real_2d_C_add():

    f1 = arrays.array_real_2d_C_add
    f2 = epyccel( f1 )

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]] )
    x2 = x1.copy()
    a  = np.array( [[-1.,-2.,-3.], [-4.,-5.,-6.]] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_real_2d_C_sub():

    f1 = arrays.array_real_2d_C_sub
    f2 = epyccel( f1 )

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]] )
    x2 = x1.copy()
    a  = np.array( [[-1.,-2.,-3.], [-4.,-5.,-6.]] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_real_2d_C_mul():

    f1 = arrays.array_real_2d_C_mul
    f2 = epyccel( f1 )

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]] )
    x2 = x1.copy()
    a  = np.array( [[-1.,-2.,-3.], [-4.,-5.,-6.]] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_real_2d_C_div():

    f1 = arrays.array_real_2d_C_div
    f2 = epyccel( f1 )

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]] )
    x2 = x1.copy()
    a  = np.array( [[-1.,-2.,-3.], [-4.,-5.,-6.]] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

#==============================================================================
# TEST: 2D ARRAYS OF REAL WITH F ORDERING
#==============================================================================

def test_array_real_2d_F_scalar_add():

    f1 = arrays.array_real_2d_F_scalar_add
    f2 = epyccel( f1 )

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]] )
    x2 = x1.copy()
    a = 5.

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_real_2d_F_scalar_sub():

    f1 = arrays.array_real_2d_F_scalar_sub
    f2 = epyccel( f1 )

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]] )
    x2 = x1.copy()
    a = 5.

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_real_2d_F_scalar_mul():

    f1 = arrays.array_real_2d_F_scalar_mul
    f2 = epyccel( f1 )

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]] )
    x2 = x1.copy()
    a = 5.

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_real_2d_F_scalar_div():

    f1 = arrays.array_real_2d_F_scalar_div
    f2 = epyccel( f1 )

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]] )
    x2 = x1.copy()
    a = 5.

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_real_2d_F_add():

    f1 = arrays.array_real_2d_F_add
    f2 = epyccel( f1 )

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]] )
    x2 = x1.copy()
    a  = np.array( [[-1.,-2.,-3.], [-4.,-5.,-6.]] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_real_2d_F_sub():

    f1 = arrays.array_real_2d_F_sub
    f2 = epyccel( f1 )

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]] )
    x2 = x1.copy()
    a  = np.array( [[-1.,-2.,-3.], [-4.,-5.,-6.]] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_real_2d_F_mul():

    f1 = arrays.array_real_2d_F_mul
    f2 = epyccel( f1 )

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]] )
    x2 = x1.copy()
    a  = np.array( [[-1.,-2.,-3.], [-4.,-5.,-6.]] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

def test_array_real_2d_F_div():

    f1 = arrays.array_real_2d_F_div
    f2 = epyccel( f1 )

    x1 = np.array( [[1.,2.,3.], [4.,5.,6.]] )
    x2 = x1.copy()
    a  = np.array( [[-1.,-2.,-3.], [-4.,-5.,-6.]] )

    f1(x1, a)
    f2(x2, a)

    assert np.array_equal( x1, x2 )

#==============================================================================
# TEST: 1D Stack ARRAYS OF REAL
#==============================================================================

def test_array_real_sum_stack_array():

    f1 = arrays.array_real_1d_sum_stack_array
    f2 = epyccel( f1 )
    x1 = f1()
    x2 = f2()
    assert np.equal( x1, x2 )

def test_array_real_div_stack_array():

    f1 = arrays.array_real_1d_div_stack_array
    f2 = epyccel( f1 )
    x1 = f1()
    x2 = f2()
    assert np.equal( x1, x2 )

##==============================================================================
## CLEAN UP GENERATED FILES AFTER RUNNING TESTS
##==============================================================================
#
#def teardown_module():
#    import os, glob
#    dirname  = os.path.dirname( arrays.__file__ )
#    pattern  = os.path.join( dirname, '__epyccel__*' )
#    filelist = glob.glob( pattern )
#    for f in filelist:
#        os.remove( f )
