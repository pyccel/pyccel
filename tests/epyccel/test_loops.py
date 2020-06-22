import pytest
import numpy as np
import platform

from pyccel.epyccel import epyccel
from modules        import loops
from conftest       import *

@pytest.fixture( params=[
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.c]
        )
    ],
    scope='module'
)
def language(request):
    return request.param

#==============================================================================

def test_sum_natural_numbers(language):
    f1 = loops.sum_natural_numbers
    f2 = epyccel( f1, language = language, verbose = True )
    assert f1( 42 ) == f2( 42 )

def test_factorial(language):
    f1 = loops.factorial
    f2 = epyccel( f1, language = language, verbose = True )
    assert f1( 11 ) == f2( 11 )

def test_fibonacci(language):
    f1 = loops.fibonacci
    f2 = epyccel( f1, language = language, verbose = True )
    assert f1( 42 ) == f2( 42 )

def test_double_loop(language):
    f1 = loops.double_loop
    f2 = epyccel( f1, language = language, verbose = True )
    assert f1( 2 ) == f2( 2 )

def test_double_loop_on_2d_array_C():

    f1 = loops.double_loop_on_2d_array_C
    f2 = epyccel( f1 )

    x = np.zeros( (11,4), dtype=int )
    y = np.ones ( (11,4), dtype=int )

    f1( x )
    f2( y )
    assert np.array_equal( x, y )

def test_double_loop_on_2d_array_F():

    f1 = loops.double_loop_on_2d_array_F
    f2 = epyccel( f1 )

    x = np.zeros( (11,4), dtype=int, order='F' )
    y = np.ones ( (11,4), dtype=int, order='F' )

    f1( x )
    f2( y )
    assert np.array_equal( x, y )

def test_product_loop_on_2d_array_C():

    f1 = loops.product_loop_on_2d_array_C
    f2 = epyccel( f1 )

    x = np.zeros( (11,4), dtype=int )
    y = np.ones ( (11,4), dtype=int )

    f1( x )
    f2( y )
    assert np.array_equal( x, y )

def test_product_loop_on_2d_array_F():

    f1 = loops.product_loop_on_2d_array_F
    f2 = epyccel( f1 )

    x = np.zeros( (11,4), dtype=int, order='F' )
    y = np.ones ( (11,4), dtype=int, order='F' )

    f1( x )
    f2( y )
    assert np.array_equal( x, y )

def test_map_on_1d_array():

    f1 = loops.map_on_1d_array
    f2 = epyccel( f1 )

    z = np.arange( 7 )

    assert np.array_equal( f1(z), f2(z) )

def test_enumerate_on_1d_array():

    f1 = loops.enumerate_on_1d_array
    f2 = epyccel( f1 )

    z = np.arange( 7 )

    assert np.array_equal( f1(z), f2(z) )

@pytest.mark.parametrize( 'lang', (
        pytest.param("fortran", marks = pytest.mark.fortran),
        pytest.param("c", marks = [
            pytest.mark.xfail(reason="zip and functional for not implemented"),
            pytest.mark.c]
        )
    )
)
def test_zip_prod(lang):

    f1 = loops.zip_prod
    f2 = epyccel( f1, language = lang )

    assert np.array_equal( f1(10), f2(10) )

def test_loop_on_real_array():

    f1 = loops.product_loop_on_real_array
    f2 = epyccel( f1 )

    z1 = np.ones(11)
    out1 = np.empty_like(z1)
    z2 = z1.copy()
    out2 = out1.copy()

    f1(z1, out1)
    f2(z2, out2)

    assert np.array_equal( out1, out2 )

##==============================================================================
## CLEAN UP GENERATED FILES AFTER RUNNING TESTS
##==============================================================================
#
#def teardown_module():
#    import os, glob
#    dirname  = os.path.dirname( loops.__file__ )
#    pattern  = os.path.join( dirname, '__epyccel__*' )
#    filelist = glob.glob( pattern )
#    for f in filelist:
#        os.remove( f )
