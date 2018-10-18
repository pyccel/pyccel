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

    x = np.array( [1,2,3] )
    a = 5

    assert np.array_equal( f1(x,a), f2(x,a) )

# ...
def test_array_int_1d_scalar_sub():

    f1 = arrays.array_int_1d_scalar_sub
    f2 = epyccel( f1 )

    x = np.array( [1,2,3] )
    a = 5

    assert np.array_equal( f1(x,a), f2(x,a) )

# ...
def test_array_int_1d_scalar_mul():

    f1 = arrays.array_int_1d_scalar_mul
    f2 = epyccel( f1 )
 
    x = np.array( [1,2,3] )
    a = 5

    assert np.array_equal( f1(x,a), f2(x,a) )
 
# ...
def test_array_int_1d_scalar_div():

    f1 = arrays.array_int_1d_scalar_div
    f2 = epyccel( f1 )
 
    x = np.array( [1,2,3] )
    a = 5

    assert np.allclose( f1(x,a), f2(x,a), rtol=1e-15, atol=1e-15 )
 
# ...
def test_array_int_1d_scalar_idiv():

    f1 = arrays.array_int_1d_scalar_idiv
    f2 = epyccel( f1 )
 
    x = np.array( [10,-12,13] )
    a = 5

    assert np.array_equal( f1(x,a), f2(x,a) )
 
# ...
def test_array_int_1d_scalar_mod():

    f1 = arrays.array_int_1d_scalar_mod
    f2 = epyccel( f1 )
 
    x = np.array( [10,-12,13] )
    a = 5

    assert np.array_equal( f1(x,a), f2(x,a) )

# ...
def test_array_int_1d_add():

    f1 = arrays.array_int_1d_add
    f2 = epyccel( f1 )
 
    x = np.array( [1,2,3] )
    y = np.array( [7,5,3] )

    assert np.array_equal( f1(x,y), f2(x,y) )

# ...
def test_array_int_1d_sub():

    f1 = arrays.array_int_1d_sub
    f2 = epyccel( f1 )

    x = np.array( [1,2,3] )
    y = np.array( [7,5,3] )

    assert np.array_equal( f1(x,y), f2(x,y) )

# ...
def test_array_int_1d_mul():

    f1 = arrays.array_int_1d_mul
    f2 = epyccel( f1 )

    x = np.array( [1,2,3] )
    y = np.array( [7,5,3] )

    assert np.array_equal( f1(x,y), f2(x,y) )

# ...
def test_array_int_1d_div():

    f1 = arrays.array_int_1d_div
    f2 = epyccel( f1 )

    x = np.array( [1,2,3] )
    y = np.array( [7,5,3] )

    assert np.array_equal( f1(x,y), f2(x,y) )

# ...
def test_array_int_1d_idiv():

    f1 = arrays.array_int_1d_idiv
    f2 = epyccel( f1 )

    x = np.array( [1,2,3] )
    y = np.array( [7,5,3] )

    assert np.array_equal( f1(x,y), f2(x,y) )

# ...
def test_array_int_1d_mod():

    f1 = arrays.array_int_1d_mod
    f2 = epyccel( f1 )

    x = np.array( [1,2,3] )
    y = np.array( [7,5,3] )

    assert np.array_equal( f1(x,y), f2(x,y) )

#==============================================================================
# TEST: 2D ARRAYS OF INT, with C ordering
#==============================================================================

def test_array_int_2d_C_scalar_add():

    f1 = arrays.array_int_2d_C_scalar_add
    f2 = epyccel( f1 )

    x = np.array( [[1,2,3],[4,5,6]] )
    a = 7

    assert np.array_equal( f1(x,a), f2(x,a) )

# ...
def test_array_int_2d_C_scalar_sub():

    f1 = arrays.array_int_2d_C_scalar_sub
    f2 = epyccel( f1 )

    x = np.array( [[1,2,3],[4,5,6]] )
    a = 7

    assert np.array_equal( f1(x,a), f2(x,a) )

# ...
def test_array_int_2d_C_scalar_mul():

    f1 = arrays.array_int_2d_C_scalar_mul
    f2 = epyccel( f1 )

    x = np.array( [[1,2,3],[4,5,6]] )
    a = 3

    assert np.array_equal( f1(x,a), f2(x,a) )

# ...
def test_array_int_2d_C_scalar_div():

    f1 = arrays.array_int_2d_C_scalar_div
    f2 = epyccel( f1 )

    x = np.array( [[1,2,3],[4,5,6]] )
    a = 3

    assert np.allclose( f1(x,a), f2(x,a), rtol=1e-15, atol=1e-15 )

# ...
def test_array_int_2d_C_scalar_idiv():

    f1 = arrays.array_int_2d_C_scalar_idiv
    f2 = epyccel( f1 )

    x = np.array( [[1,2,3],[4,5,6]] )
    a = 3

    assert np.array_equal( f1(x,a), f2(x,a) )

# ...
def test_array_int_2d_C_scalar_mod():

    f1 = arrays.array_int_2d_C_scalar_mod
    f2 = epyccel( f1 )

    x = np.array( [[1,2,3],[4,5,6]] )
    a = 3

    assert np.array_equal( f1(x,a), f2(x,a) )

# ...
def test_array_int_2d_C_add():

    f1 = arrays.array_int_2d_C_add
    f2 = epyccel( f1 )

    x = np.array( [[1,2,3],[4,5,6]] )
    y = np.array( [[9,8,7],[6,5,4]] )

    assert np.array_equal( f1(x,y), f2(x,y) )

# ...
def test_array_int_2d_C_sub():

    f1 = arrays.array_int_2d_C_sub
    f2 = epyccel( f1 )

    x = np.array( [[1,2,3],[4,5,6]] )
    y = np.array( [[9,8,7],[6,5,4]] )

    assert np.array_equal( f1(x,y), f2(x,y) )

# ...
def test_array_int_2d_C_mul():

    f1 = arrays.array_int_2d_C_mul
    f2 = epyccel( f1 )

    x = np.array( [[1,2,3],[4,5,6]] )
    y = np.array( [[9,8,7],[6,5,4]] )

    assert np.array_equal( f1(x,y), f2(x,y) )

# ...
def test_array_int_2d_C_div():

    f1 = arrays.array_int_2d_C_div
    f2 = epyccel( f1 )

    x = np.array( [[1,2,3],[4,5,6]] )
    y = np.array( [[9,8,7],[6,5,4]] )

    assert np.allclose( f1(x,y), f2(x,y), rtol=1e-15, atol=1e-15 )

# ...
def test_array_int_2d_C_idiv():

    f1 = arrays.array_int_2d_C_idiv
    f2 = epyccel( f1 )

    x = np.array( [[1,2,3],[4,5,6]] )
    y = np.array( [[9,8,7],[6,5,4]] )

    assert np.array_equal( f1(x,y), f2(x,y) )

# ...
def test_array_int_2d_C_mod():

    f1 = arrays.array_int_2d_C_mod
    f2 = epyccel( f1 )

    x = np.array( [[1,2,3],[4,5,6]] )
    y = np.array( [[9,8,7],[6,5,4]] )

    assert np.array_equal( f1(x,y), f2(x,y) )

#==============================================================================
# TEST: 2D ARRAYS OF INT, with Fortran ordering
#==============================================================================

def test_array_int_2d_F_scalar_add():

    f1 = arrays.array_int_2d_F_scalar_add
    f2 = epyccel( f1 )

    x = np.array( [[1,2,3],[4,5,6]], order='F' )
    a = 7

    assert np.array_equal( f1(x,a), f2(x,a) )

# ...
def test_array_int_2d_F_scalar_sub():

    f1 = arrays.array_int_2d_F_scalar_sub
    f2 = epyccel( f1 )

    x = np.array( [[1,2,3],[4,5,6]], order='F' )
    a = 7

    assert np.array_equal( f1(x,a), f2(x,a) )

# ...
def test_array_int_2d_F_scalar_mul():

    f1 = arrays.array_int_2d_F_scalar_mul
    f2 = epyccel( f1 )

    x = np.array( [[1,2,3],[4,5,6]], order='F' )
    a = 3

    assert np.array_equal( f1(x,a), f2(x,a) )

# ...
def test_array_int_2d_F_scalar_div():

    f1 = arrays.array_int_2d_F_scalar_div
    f2 = epyccel( f1 )

    x = np.array( [[1,2,3],[4,5,6]], order='F' )
    a = 3

    assert np.allclose( f1(x,a), f2(x,a), rtol=1e-15, atol=1e-15 )

# ...
def test_array_int_2d_F_scalar_idiv():

    f1 = arrays.array_int_2d_F_scalar_idiv
    f2 = epyccel( f1 )

    x = np.array( [[1,2,3],[4,5,6]], order='F' )
    a = 3

    assert np.array_equal( f1(x,a), f2(x,a) )

# ...
def test_array_int_2d_F_scalar_mod():

    f1 = arrays.array_int_2d_F_scalar_mod
    f2 = epyccel( f1 )

    x = np.array( [[1,2,3],[4,5,6]], order='F' )
    a = 3

    assert np.array_equal( f1(x,a), f2(x,a) )

# ...
def test_array_int_2d_F_add():

    f1 = arrays.array_int_2d_F_add
    f2 = epyccel( f1 )

    x = np.array( [[1,2,3],[4,5,6]], order='F' )
    y = np.array( [[9,8,7],[6,5,4]], order='F' )

    assert np.array_equal( f1(x,y), f2(x,y) )

# ...
def test_array_int_2d_F_sub():

    f1 = arrays.array_int_2d_F_sub
    f2 = epyccel( f1 )

    x = np.array( [[1,2,3],[4,5,6]], order='F' )
    y = np.array( [[9,8,7],[6,5,4]], order='F' )

    assert np.array_equal( f1(x,y), f2(x,y) )

# ...
def test_array_int_2d_F_mul():

    f1 = arrays.array_int_2d_F_mul
    f2 = epyccel( f1 )

    x = np.array( [[1,2,3],[4,5,6]], order='F' )
    y = np.array( [[9,8,7],[6,5,4]], order='F' )

    assert np.array_equal( f1(x,y), f2(x,y) )

# ...
def test_array_int_2d_F_div():

    f1 = arrays.array_int_2d_F_div
    f2 = epyccel( f1 )

    x = np.array( [[1,2,3],[4,5,6]], order='F' )
    y = np.array( [[9,8,7],[6,5,4]], order='F' )

    assert np.allclose( f1(x,y), f2(x,y), rtol=1e-15, atol=1e-15 )

# ...
def test_array_int_2d_F_idiv():

    f1 = arrays.array_int_2d_F_idiv
    f2 = epyccel( f1 )

    x = np.array( [[1,2,3],[4,5,6]], order='F' )
    y = np.array( [[9,8,7],[6,5,4]], order='F' )

    assert np.array_equal( f1(x,y), f2(x,y) )

# ...
def test_array_int_2d_F_mod():

    f1 = arrays.array_int_2d_F_mod
    f2 = epyccel( f1 )

    x = np.array( [[1,2,3],[4,5,6]], order='F' )
    y = np.array( [[9,8,7],[6,5,4]], order='F' )

    assert np.array_equal( f1(x,y), f2(x,y) )

#==============================================================================
# CLEAN UP GENERATED FILES AFTER RUNNING TESTS
#==============================================================================

def teardown_module():
    import os, glob
    dirname  = os.path.dirname( arrays.__file__ )
    pattern  = os.path.join( dirname, '__epyccel__*' )
    filelist = glob.glob( pattern )
    for f in filelist:
        os.remove( f )
